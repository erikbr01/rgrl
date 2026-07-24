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


def _load_selection_manifest(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Resolve a versioned selection, including optional parent + additions."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"selection manifest parent cycle: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"selection manifest not found: {path}")
    doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(doc, dict):
        raise TypeError(f"selection manifest {path} must contain a mapping")
    if doc.get("schema_version") != 1:
        raise ValueError(
            f"selection manifest {path} has unsupported schema_version "
            f"{doc.get('schema_version')!r}"
        )

    parent = None
    base_ids: list[int] = []
    if doc.get("parent"):
        parent_path = Path(doc["parent"])
        if not parent_path.suffix:
            parent_path = parent_path.with_suffix(path.suffix)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent = _load_selection_manifest(parent_path, (*seen, path))
        base_ids = list(parent["episode_ids"])

    if "episode_ids" in doc and ("add_episode_ids" in doc or parent is not None):
        raise ValueError(
            f"selection manifest {path} must use either episode_ids or parent/add_episode_ids"
        )
    if "episode_ids" in doc:
        episode_ids = list(doc["episode_ids"])
    else:
        episode_ids = base_ids + list(doc.get("add_episode_ids", []))
    if doc.get("ordering", "preserve") == "sorted":
        episode_ids = sorted(episode_ids)
    elif doc.get("ordering", "preserve") != "preserve":
        raise ValueError(f"selection manifest {path} has unknown ordering")
    if doc.get("count") is not None and int(doc["count"]) != len(episode_ids):
        raise ValueError(
            f"selection manifest {path} declares count={doc['count']} but resolves "
            f"to {len(episode_ids)} episodes"
        )

    resolved = {
        "schema_version": 1,
        "name": doc.get("name", path.stem),
        "source": doc.get("source", {}),
        "provenance": doc.get("provenance", {}),
        "episode_ids": episode_ids,
        "parent_fingerprint": parent["fingerprint"] if parent else None,
        "raw": doc,
    }
    resolved["fingerprint"] = _hash_document(resolved)
    return resolved


def resolve_source_config(source_cfg) -> dict:
    """Resolve Hydra input and replace selection_path with immutable contents."""
    source = _to_plain(source_cfg)
    selection_path = source.pop("selection_path", None)
    if selection_path is not None and source.get("episode_ids") is not None:
        raise ValueError("data config cannot set both selection_path and episode_ids")
    if selection_path is not None:
        selection = _load_selection_manifest(Path(selection_path))
        expected_source = selection.get("source", {}).get("dataset_id")
        actual_source = source.get("dataset_id")
        if expected_source and actual_source and expected_source != actual_source:
            raise ValueError(
                f"selection expects dataset_id={expected_source!r}, data config "
                f"declares {actual_source!r}"
            )
        source["episode_ids"] = selection["episode_ids"]
        source["selection_manifest"] = selection
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

    `stats_workers` sets the normalizer-fit thread pool at conversion time;
    None falls back to the RGRL_STATS_WORKERS env var, then to one thread per
    key (capped at the CPU count). Only matters on a cache miss."""
    if stats_workers is None:
        env_workers = os.environ.get("RGRL_STATS_WORKERS")
        stats_workers = int(env_workers) if env_workers else None
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
