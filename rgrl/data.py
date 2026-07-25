"""Resolve a source dataset, logical episode selection, and target schema.

configs/data/*.yaml describe only the source: where it lives, what's in it
(shape_meta), and how its keys map onto the target vocabulary (rename).
configs/policy/*.yaml describe the target shape_meta a policy trains
against. `load_store` is where the two meet: it diffs them
(rgrl.conversion.plan_transforms), converts once into an rgrl store under a
content-addressed cache directory, and reuses that cache on every
subsequent call with the same (source, target, conversion code).

Cache identity uses an explicit conversion schema version plus source
metadata, resolved selection contents, and target schema. Construction happens
under a per-cache file lock in a temporary sibling directory and is published
with one atomic rename, so concurrent jobs and interrupted conversions cannot
expose a partial cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import fcntl
import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rgrl.conversion import plan_transforms
from rgrl.store import EPISODE_VIEW_FILE, EpisodeStore, write_sidecars

log = logging.getLogger(__name__)

_MANIFEST_FILE = "cache_manifest.json"
_EXPECTED_FILES = ["meta.json", "rgrl_meta.json", "stats.json", _MANIFEST_FILE]
_CONVERSION_SCHEMA_VERSION = 2


def _to_plain(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def _hash_document(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _store_task_episodes(store_path: Path) -> dict[str, list[int]]:
    """Read the store's tasks.json into task name -> sorted physical episode ids."""
    tasks_path = store_path / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"generative selection needs {tasks_path} (task label per episode); not found"
        )
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode, task in json.loads(tasks_path.read_text()).items():
        by_task[str(task)].append(int(episode))
    return {task: sorted(ids) for task, ids in by_task.items()}


def _store_curation(store_path: Path) -> dict[str, dict[int, int]]:
    """Read the store's optional curation.json: the success curation baked into
    the store as per-task {unsuccessful_episode -> successful substitute}. It is
    applied to a draw only where the unsuccessful episode is actually drawn, so
    disjoint draws (base vs follow-up) never collide. Missing file -> no curation
    (a manifest whose correct resolution depends on curation guards that with
    expected_sha256, which then fails loudly rather than training on raw draws)."""
    path = store_path / "curation.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {task: {int(o): int(n) for o, n in swaps.items()} for task, swaps in raw.items()}


def _load_store_meta(
    store_path: Path, cache: dict[str, tuple[dict, dict]]
) -> tuple[dict[str, list[int]], dict[str, dict[int, int]]]:
    """(task_episodes, curation) for a store, read once per resolve() even across
    a parent chain, and validated: every curation source/target must be a real
    episode of its own task (the cross-task/typo guard the old generator had)."""
    key = str(store_path)
    if key in cache:
        return cache[key]
    task_episodes = _store_task_episodes(store_path)
    curation = _store_curation(store_path)
    for task, swaps in curation.items():
        if task not in task_episodes:
            raise ValueError(f"{store_path}/curation.json references unknown task {task!r}")
        pool = set(task_episodes[task])
        for old, new in swaps.items():
            if old not in pool:
                raise ValueError(
                    f"{store_path}/curation.json source {old} is not an episode of "
                    f"task {task!r} (stale or mistyped)"
                )
            if new not in pool:
                raise ValueError(
                    f"{store_path}/curation.json target {new} is not an episode of "
                    f"task {task!r} (a substitute must stay in-task)"
                )
    cache[key] = (task_episodes, curation)
    return cache[key]


def _episode_ids_sha256(episode_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(episode_ids).encode()).hexdigest()


def _seeded_choice(episodes: list[int], count: int, seed: int) -> list[int]:
    """`count` episodes drawn without replacement from `episodes`, seeded and
    sorted. count >= len keeps them all. The RNG contract (np.random.default_rng
    + choice/replace=False) is fixed: it defines every checked-in selection."""
    values = np.asarray(episodes)
    if count >= len(values):
        return sorted(values.tolist())
    return sorted(np.random.default_rng(seed).choice(values, size=count, replace=False).tolist())


def _seeded_continuation(episodes: list[int], used: set[int], count: int, seed: int) -> list[int]:
    """Next `count` episodes for the same task, disjoint from `used` (the raw
    draws of prior selections) -- this is "two selections for one task = sampling
    without replacement". Uses permutation of the remainder so it is stable
    regardless of how many were taken before."""
    remaining = np.asarray([e for e in episodes if e not in used])
    if count >= len(remaining):
        return sorted(remaining.tolist())
    return sorted(np.random.default_rng(seed).permutation(remaining)[:count].tolist())


def _first_n(episodes: list[int], used: set[int], count: int) -> list[int]:
    """First `count` episodes by ascending id, skipping any already drawn by a
    prior same-task rule. Deterministic and *stable under store growth*: new
    episodes are appended to the store with higher ids, so they land at the end
    of the sorted pool and never displace the first `count`. `episodes` arrives
    sorted from _store_task_episodes."""
    return [e for e in episodes if e not in used][:count]


def _apply_curation(draw: list[int], swaps: dict[int, int]) -> list[int]:
    """Substitute unsuccessful episodes present in this draw with their
    successful replacements."""
    result = set(draw)
    for old, new in swaps.items():
        if old not in result:
            continue  # this swap belongs to a different (disjoint) draw
        if new in result:
            raise ValueError(f"curation target {new} already present in the draw")
        result.discard(old)
        result.add(new)
    return sorted(result)


def _resolve_selection(selection: dict, store_path: Path | None, name: str = "selection") -> list[int]:
    """Resolve an inline selection block (from a data config's `selection:` key)
    to its episode ids.

    Generative form -- {method?, seed?, select: [{task, count}, ...]} -- draws
    `count` episodes per rule from the store's tasks.json. Two methods:

      method: random (default) -- seeded uniform draw (np.random.default_rng +
        choice/replace=False). Reproducible for a fixed store but NOT stable
        under store growth: adding episodes to a task changes the pool size, so
        the same seed selects a different subset. Needs `seed`.
      method: first_n -- the first `count` episodes by ascending id. Stable
        under store growth (new episodes get higher ids and append past the
        first `count`). Deterministic, so it takes no `seed`.

    Listing the same task more than once continues without replacement, so a
    follow-up split is just the baseline rules plus one extra same-task rule (no
    cross-file parent needed). Success curation (store curation.json) is applied
    per draw. Explicit form -- {episode_ids: [...]} -- is the escape hatch for an
    externally-curated set no rule can regenerate. Optional `count` and
    `expected_sha256` pin the resolved set."""
    if not isinstance(selection, dict):
        raise TypeError(f"{name}: selection must be a mapping, got {type(selection).__name__}")

    if "episode_ids" in selection:
        if "select" in selection or "seed" in selection:
            raise ValueError(f"{name}: use either episode_ids or seed/select, not both")
        # An explicit set is "exactly these episodes, in this order"; preserve
        # order but require uniqueness up front (a repeated id is an authoring
        # mistake, and the EpisodeView would reject it anyway).
        episode_ids = [int(e) for e in selection["episode_ids"]]
    else:
        if "select" not in selection:
            raise ValueError(f"{name}: generative selection needs a 'select' list")
        method = selection.get("method", "random")
        if method not in ("random", "first_n"):
            raise ValueError(f"{name}: unknown selection method {method!r} (random | first_n)")
        if method == "random" and "seed" not in selection:
            raise ValueError(f"{name}: method=random needs a 'seed'")
        if method == "first_n" and "seed" in selection:
            raise ValueError(f"{name}: method=first_n is deterministic and takes no 'seed'")
        if store_path is None:
            raise ValueError(
                f"{name}: generative selection needs the source store_path "
                "(for tasks.json/curation.json) but none was provided"
            )
        seed = int(selection["seed"]) if method == "random" else 0
        task_episodes, curation = _load_store_meta(store_path, {})
        raw_used: dict[str, set[int]] = defaultdict(set)
        ids: list[int] = []
        for rule in selection["select"]:
            task, count = rule["task"], int(rule["count"])
            if task not in task_episodes:
                raise ValueError(f"{name}: unknown task {task!r}")
            pool = task_episodes[task]
            # A repeated task continues without replacement (skips prior draws).
            if method == "first_n":
                raw = _first_n(pool, raw_used[task], count)
            elif raw_used[task]:
                raw = _seeded_continuation(pool, raw_used[task], count, seed)
            else:
                raw = _seeded_choice(pool, count, seed)
            raw_used[task] |= set(raw)
            ids += _apply_curation(raw, curation.get(task, {}))
        episode_ids = sorted(ids)

    # Runtime episode-uniqueness guard: every resolved selection must map to
    # distinct physical episodes. Duplicates would double-weight an episode's
    # frames in training. Generative draws are disjoint by construction, but a
    # curation substitute that a later same-task continuation happened to redraw
    # would slip through -- catch it here rather than silently double-counting
    # (or leaving a count/sha mismatch to explain).
    if len(episode_ids) != len(set(episode_ids)):
        counts = Counter(episode_ids)
        dupes = sorted(e for e, c in counts.items() if c > 1)
        raise ValueError(f"{name}: resolved selection has duplicate episodes {dupes}")

    if selection.get("count") is not None and int(selection["count"]) != len(episode_ids):
        raise ValueError(
            f"{name}: declares count={selection['count']} but resolves to "
            f"{len(episode_ids)} episodes"
        )
    # Audit lock: pin the exact resolved set. Catches a resolution that silently
    # drifted -- a missing/edited store curation.json, a numpy RNG stream change
    # across versions (NEP 19), or an edited seed/count -- rather than training on
    # a different dataset than every prior run. Update it deliberately. It is
    # optional: you cannot know it when authoring a config (it takes resolving
    # the recipe against the store), so it is a create-then-pin step -- resolve
    # unlocked first, then paste the hash (see `python -m rgrl.pin_selection`).
    expected = selection.get("expected_sha256")
    actual = _episode_ids_sha256(episode_ids)
    if expected is None:
        log.info(
            "[selection] %s resolved %d episodes, unlocked (sha256 %s); "
            "pin it with `expected_sha256: %r` to guard against silent drift",
            name, len(episode_ids), actual, actual,
        )
    elif expected != actual:
        raise ValueError(
            f"{name}: resolved to episode ids sha256 {actual}, not the declared "
            f"expected_sha256 {expected}. The resolved set changed (missing/edited "
            "store curation.json, a numpy RNG version change, or an edited "
            "seed/count). Re-pin with `python -m rgrl.pin_selection` only if this "
            "change is intended."
        )
    return episode_ids


def resolve_source_config(source_cfg) -> dict:
    """Resolve Hydra input, replacing the inline `selection:` block with the
    resolved episode ids. Only the resolved ids enter `source` (and thus the
    conversion-cache and checkpoint fingerprints) -- the selection's authoring
    form (seed, per-task counts, notes) is deliberately not hashed, so editing a
    comment without changing which episodes it resolves to reuses the cache."""
    source = _to_plain(source_cfg)
    selection = source.pop("selection", None)
    if selection is not None and source.get("episode_ids") is not None:
        raise ValueError("data config cannot set both selection and episode_ids")
    if selection is not None:
        store_path = source.get("store_path")
        source["episode_ids"] = _resolve_selection(
            selection,
            Path(store_path) if store_path else None,
            name=str(source.get("name", "selection")),
        )
    return source


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source_cfg: dict, source_kind: str) -> dict:
    """Fingerprint authoritative metadata without hashing multi-GB videos."""
    if source_cfg.get("source_fingerprint"):
        return {"declared": source_cfg["source_fingerprint"]}
    source_path = Path(
        source_cfg["store_path"] if source_kind == "store" else source_cfg["path"]
    )
    if source_kind == "hdf5":
        stat = source_path.stat()
        return {
            "path": str(source_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    files = {}
    for name in ("meta.json", "episode_ends.npy", "episodes.json", "tasks.json"):
        path = source_path / name
        if path.exists():
            files[name] = _sha256_file(path)
    return {"path": str(source_path.resolve()), "metadata_sha256": files}


def _manifest_document(source_cfg: dict, target_shape_meta: dict, source_kind: str) -> dict:
    # sample_weights_path names a training-time artifact (a curation-score
    # sweep will point it at a different npy every run), not a property of
    # the converted data itself — excluding it from the hash means sweeping
    # scores reuses the same cached store instead of reconverting.
    hashed_source_cfg = {k: v for k, v in source_cfg.items() if k != "sample_weights_path"}
    doc = {
        "conversion_schema_version": _CONVERSION_SCHEMA_VERSION,
        "source_cfg": hashed_source_cfg,
        "source_identity": _source_identity(source_cfg, source_kind),
        "target_shape_meta": target_shape_meta,
    }
    doc["hash"] = _hash_document(doc)
    return doc


@contextmanager
def _cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    """Flush every file and directory under `path` to stable storage. Without
    this, the atomic os.replace() can publish a cache whose file contents are
    still only in the page cache, so a power loss right after publish would
    leave existing-but-truncated meta/stats behind (which the existence-only
    verify would then accept, silently corrupting normalization)."""
    for child in sorted(path.rglob("*")):
        if child.is_dir():
            _fsync_directory(child)
        elif child.is_file():
            fd = os.open(child, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_directory(path)


def _is_short_prefix_collision(cache_dir: Path, expected_doc: dict) -> bool:
    """True iff `cache_dir` holds a cache built for a *different full hash* —
    i.e. a genuine 32-bit short-prefix collision between two distinct configs
    sharing a name. Distinct from tampering (same full hash, edited content),
    which _verify_cache must still surface as an error rather than route around.
    On a missing/unreadable manifest, defer to the normal path (returns False)."""
    manifest = cache_dir / _MANIFEST_FILE
    if not manifest.exists():
        return False
    try:
        on_disk = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return False
    return on_disk.get("hash") != expected_doc["hash"]


def _verify_cache(cache_dir: Path, expected_doc: dict) -> None:
    missing = [f for f in _EXPECTED_FILES if not (cache_dir / f).exists()]
    if expected_doc["source_cfg"].get("episode_ids") is not None:
        if not (cache_dir / EPISODE_VIEW_FILE).exists():
            missing.append(EPISODE_VIEW_FILE)
    if missing:
        raise RuntimeError(
            f"cache dir {cache_dir} exists but is missing {missing} — likely an "
            "interrupted conversion. Not touching it automatically: delete it "
            "manually to force reconversion."
        )
    on_disk = json.loads((cache_dir / _MANIFEST_FILE).read_text())
    if on_disk != expected_doc:
        raise RuntimeError(
            f"cache dir {cache_dir} exists but its manifest doesn't match the current "
            "source config, target shape_meta, and conversion code (either a hash "
            "prefix collision, or the directory was modified after being written). "
            "Not touching it automatically: delete it manually to force reconversion."
        )


def load_store(
    source_cfg,
    target_shape_meta,
    cache_root: str | Path = "data/cache",
    stats_workers: int | None = None,
) -> EpisodeStore:
    """Convert (if needed) and open the store for `source_cfg` against
    `target_shape_meta`. Safe to call repeatedly (e.g. once at train time,
    once at eval time) — later calls with identical inputs hit the cache.

    Two source kinds: `path` (a robomimic HDF5 file, converted into a fresh
    rgrl zarr store) or `store_path` (a dataset already in a robo_dataset
    native format, e.g. a DROID VideoStore — no re-encode needed, see
    `_convert_from_store`).

    `stats_workers` sets the normalizer-fit process count at conversion time
    (its sole source is the caller, i.e. the `stats_workers` Hydra config).
    None means one process per key, capped at the CPU count. Only matters on a
    cache miss."""
    source_cfg = resolve_source_config(source_cfg)
    target_shape_meta = _to_plain(target_shape_meta)
    source_kind = "store" if "store_path" in source_cfg else "hdf5"
    doc = _manifest_document(source_cfg, target_shape_meta, source_kind)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    # Short 32-bit suffix keeps directory names readable. Two distinct manifests
    # sharing the same name could in principle collide on it; if the short dir
    # already holds a *different* manifest, fall back to the full digest so the
    # colliding identity stays cacheable instead of erroring forever.
    cache_dir = cache_root / f"{source_cfg['name']}-{doc['hash'][:8]}"
    if cache_dir.exists() and _is_short_prefix_collision(cache_dir, doc):
        cache_dir = cache_root / f"{source_cfg['name']}-{doc['hash']}"
    lock_path = cache_root / f".{cache_dir.name}.lock"

    with _cache_lock(lock_path):
        if cache_dir.exists():
            _verify_cache(cache_dir, doc)
            log.info(f"[load_store] cache hit {cache_dir}")
            return EpisodeStore(cache_dir, target_shape_meta)

        tmp_dir = cache_root / f".{cache_dir.name}.{uuid.uuid4().hex}.tmp"
        try:
            if source_kind == "store":
                log.info(
                    f"[load_store] cache miss {cache_dir} -- linking "
                    f"{source_cfg['store_path']}"
                )
                _convert_from_store(source_cfg, target_shape_meta, tmp_dir, stats_workers)
            else:
                log.info(
                    f"[load_store] cache miss {cache_dir} -- converting "
                    f"{source_cfg['path']}"
                )
                _convert(source_cfg, target_shape_meta, tmp_dir, stats_workers)
            (tmp_dir / _MANIFEST_FILE).write_text(
                json.dumps(doc, indent=1, sort_keys=True)
            )
            _verify_cache(tmp_dir, doc)
            _fsync_tree(tmp_dir)
            os.replace(tmp_dir, cache_dir)
            _fsync_directory(cache_root)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    return EpisodeStore(cache_dir, target_shape_meta)


def _convert(
    source_cfg: dict, target_shape_meta: dict, out: Path, num_workers: int | None = None
) -> None:
    from robo_dataset.backends.hdf5 import RobomimicHDF5Store
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter
    from robo_dataset.convert import convert as robo_convert

    source_meta = source_cfg["shape_meta"]
    rename = source_cfg.get("rename", {})
    transform, image_keys, source_key_of = plan_transforms(source_meta, target_shape_meta, rename)

    src = RobomimicHDF5Store(source_cfg["path"], obs_keys=list(source_meta["obs"].keys()))

    target_obs_keys = list(target_shape_meta["obs"].keys())
    key_shapes = {k: tuple(target_shape_meta["obs"][k]["shape"]) for k in target_obs_keys}
    key_shapes["action"] = tuple(target_shape_meta["action"]["shape"])
    key_dtypes = {
        k: (src.frame_dtype(source_key_of[k]) if k in image_keys else np.dtype(np.float32))
        for k in target_obs_keys
    }
    key_dtypes["action"] = np.dtype(np.float32)

    dst = ZarrStoreWriter.create(str(out), key_shapes=key_shapes, key_dtypes=key_dtypes, image_keys=image_keys)

    stats_accum: dict[str, list[np.ndarray]] = {k: [] for k in key_shapes if k not in image_keys}

    def transform_and_collect(ep: dict) -> dict:
        out_ep = transform(ep)
        for k, acc in stats_accum.items():
            acc.append(out_ep[k])
        return out_ep

    robo_convert(src, dst, transform=transform_and_collect, verbose=True)
    # Hand the per-episode chunk lists straight to write_sidecars, which
    # concatenates and frees one key at a time -- no second all-keys copy.

    with h5py.File(source_cfg["path"], "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"]) if "env_args" in f["data"].attrs else {}
    # A dataset's own env_args can be recorded against a newer robosuite than
    # the one installed here (e.g. lite_physics / nested "BASIC" controller
    # configs from robosuite>=1.5, unsupported by the 1.4.1 pin this repo
    # needs for OSC controller/mujoco compatibility elsewhere) -- let the data
    # config override individual env_meta keys (e.g. env_kwargs) rather than
    # forcing every hdf5 to already match the installed robosuite's schema.
    env_meta = {**env_meta, **source_cfg.get("env_meta", {})}
    env_meta["abs_action"] = source_cfg.get("abs_action", False)

    write_sidecars(
        out,
        stats_keys=stats_accum,
        env_meta=env_meta,
        source=str(source_cfg["path"]),
        fps=20.0,
        norm_method=source_cfg.get("norm_method", "minmax"),
        num_workers=num_workers,
    )


def _convert_from_store(
    source_cfg: dict, target_shape_meta: dict, out: Path, num_workers: int | None = None
) -> None:
    """Adapt a dataset already in a robo_dataset-native format (zarr, video,
    lerobot, ...) for rgrl.

    Default (no `episode_ids`): no per-frame rewrite -- the on-disk files
    are symlinked as-is into the cache dir (image keys the target doesn't
    declare, e.g. a dropped camera, are simply never read — EpisodeStore
    only touches keys present in `target_shape_meta`) and only rgrl's
    stats/env_meta sidecars are computed fresh.

    `episode_ids`, if given, creates a zero-copy EpisodeView: the physical
    store is still symlinked as-is, while episode_view.json records the
    logical-to-source mapping. Overlapping DROID subsets therefore share the
    original videos and never decode/re-encode them during cache creation.

    Because the default path rewrites nothing, `rename` and channel_order/
    quat_convention conversions aren't supported (they'd require rewriting
    the bytes on disk) — only an exact-name, exact-convention subset of the
    source is representable this way, `episode_ids` or not. Raise rather
    than silently ignore a declared conversion that won't actually happen."""
    from robo_dataset.store_factory import open_store

    store_path = Path(source_cfg["store_path"])
    source_meta = source_cfg["shape_meta"]
    rename = source_cfg.get("rename", {})
    if rename:
        raise NotImplementedError(
            "_convert_from_store can't rename keys without rewriting the store "
            f"(rename={rename}) — use the `path` (HDF5) source kind instead"
        )
    _transform, image_keys, source_key_of = plan_transforms(source_meta, target_shape_meta, rename)
    for k in image_keys:
        src_co = source_meta["obs"][k].get("channel_order", "rgb")
        tgt_co = target_shape_meta["obs"][k].get("channel_order", "rgb")
        if src_co != tgt_co:
            raise NotImplementedError(
                f"_convert_from_store can't flip channel_order without rewriting the store "
                f"({k}: {src_co} -> {tgt_co})"
            )

    src = open_store(str(store_path))
    episode_ids = source_cfg.get("episode_ids")

    out.mkdir(parents=True, exist_ok=True)
    # ZarrStore reads episode_ends.npy as a required top-level file;
    # video-backed stores keep the same information in meta.json. A logical
    # EpisodeView is applied by rgrl.store.EpisodeStore after opening these
    # physical links.
    for entry in ("meta.json", "episode_ends.npy", "videos", "data"):
        src_entry = store_path / entry
        if src_entry.exists():
            (out / entry).symlink_to(src_entry.resolve())

    if episode_ids is None:
        dst = src
    else:
        from robo_dataset.views import EpisodeView

        dst = EpisodeView(src, list(episode_ids))
        (out / EPISODE_VIEW_FILE).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": str(store_path.resolve()),
                    "episode_ids": list(dst.episode_ids),
                },
                indent=1,
            )
        )

    n = int(dst.episode_ends[-1]) if len(dst.episode_ends) else 0
    stats_keys = {
        k: dst.get_sequence(k, 0, n).astype(np.float32)
        for k in target_shape_meta["obs"]
        if k not in image_keys
    }
    stats_keys["action"] = dst.get_sequence("action", 0, n).astype(np.float32)

    meta_path = store_path / "meta.json"
    fps = json.loads(meta_path.read_text()).get("fps") if meta_path.exists() else None

    write_sidecars(
        out,
        stats_keys=stats_keys,
        # Unlike _convert, there's no source hdf5 to pull env_args from here —
        # a caller whose store needs eval (a real robomimic env_meta, not
        # just abs_action) must declare it explicitly on the data config.
        env_meta={**source_cfg.get("env_meta", {}), "abs_action": source_cfg.get("abs_action", False)},
        source=str(store_path),
        fps=fps,
        norm_method=source_cfg.get("norm_method", "minmax"),
        num_workers=num_workers,
    )
