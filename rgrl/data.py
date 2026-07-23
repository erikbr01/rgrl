"""Resolve a data config (the SOURCE dataset) plus a policy's target
shape_meta into a ready-to-train EpisodeStore, converting and caching the
result on demand.

configs/data/*.yaml describe only the source: where it lives, what's in it
(shape_meta), and how its keys map onto the target vocabulary (rename).
configs/policy/*.yaml describe the target shape_meta a policy trains
against. `load_store` is where the two meet: it diffs them
(rgrl.conversion.plan_transforms), converts once into an rgrl store under a
content-addressed cache directory, and reuses that cache on every
subsequent call with the same (source, target, conversion code).

The cache key intentionally covers more than the two configs: the *code*
that performs the conversion is hashed too (via inspect.getsource), so a
change to plan_transforms, this module, or the robo_dataset primitives they
call also invalidates the cache — a config-only hash would let a behavior
change silently reuse data converted under the old behavior. A cache hit is
re-verified against a manifest written at conversion time; if a cache
directory exists but doesn't match (or is missing files, e.g. an
interrupted conversion), load_store raises rather than deleting or
reconverting — a mismatched cache is treated as something for a human to
look at, not something to silently repair.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rgrl.conversion import plan_transforms
from rgrl.store import EpisodeStore, write_sidecars

log = logging.getLogger(__name__)

_MANIFEST_FILE = "cache_manifest.json"
_EXPECTED_FILES = ["meta.json", "rgrl_meta.json", "stats.json", _MANIFEST_FILE]


def _to_plain(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def _code_hash(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def _conversion_code_hashes(source_kind: str) -> dict[str, str]:
    """Source of every function/class whose behavior affects conversion
    output. A robo_dataset submodule update that changes any of these also
    invalidates the cache, even though no rgrl config changed."""
    targets = {
        "rgrl.conversion.plan_transforms": plan_transforms,
        "rgrl.data.load_store": load_store,
        "rgrl.store.write_sidecars": write_sidecars,
    }
    if source_kind == "hdf5":
        from robo_dataset.backends.hdf5 import RobomimicHDF5Store
        from robo_dataset.backends.zarr_backend import ZarrStoreWriter
        from robo_dataset.convert import convert as robo_convert

        targets["rgrl.data._convert"] = _convert
        targets["robo_dataset.backends.hdf5.RobomimicHDF5Store"] = RobomimicHDF5Store
        targets["robo_dataset.backends.zarr_backend.ZarrStoreWriter"] = ZarrStoreWriter
        targets["robo_dataset.convert.convert"] = robo_convert
    else:
        from robo_dataset.backends.video import VideoStoreWriter
        from robo_dataset.backends.zarr_backend import ZarrStoreWriter
        from robo_dataset.convert import convert as robo_convert
        from robo_dataset.store_factory import open_store

        targets["rgrl.data._convert_from_store"] = _convert_from_store
        targets["rgrl.data._convert_episode_subset"] = _convert_episode_subset
        targets["robo_dataset.store_factory.open_store"] = open_store
        targets["robo_dataset.convert.convert"] = robo_convert
        targets["robo_dataset.backends.video.VideoStoreWriter"] = VideoStoreWriter
        targets["robo_dataset.backends.zarr_backend.ZarrStoreWriter"] = ZarrStoreWriter
    return {name: _code_hash(obj) for name, obj in targets.items()}


def _manifest_document(source_cfg: dict, target_shape_meta: dict, source_kind: str) -> dict:
    # sample_weights_path names a training-time artifact (a curation-score
    # sweep will point it at a different npy every run), not a property of
    # the converted data itself — excluding it from the hash means sweeping
    # scores reuses the same cached store instead of reconverting.
    hashed_source_cfg = {k: v for k, v in source_cfg.items() if k != "sample_weights_path"}
    doc = {
        "source_cfg": hashed_source_cfg,
        "target_shape_meta": target_shape_meta,
        "code_hashes": _conversion_code_hashes(source_kind),
    }
    doc["hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def _verify_cache(cache_dir: Path, expected_doc: dict) -> None:
    missing = [f for f in _EXPECTED_FILES if not (cache_dir / f).exists()]
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


def load_store(source_cfg, target_shape_meta, cache_root: str | Path = "data/cache") -> EpisodeStore:
    """Convert (if needed) and open the store for `source_cfg` against
    `target_shape_meta`. Safe to call repeatedly (e.g. once at train time,
    once at eval time) — later calls with identical inputs hit the cache.

    Two source kinds: `path` (a robomimic HDF5 file, converted into a fresh
    rgrl zarr store) or `store_path` (a dataset already in a robo_dataset
    native format, e.g. a DROID VideoStore — no re-encode needed, see
    `_convert_from_store`)."""
    source_cfg = _to_plain(source_cfg)
    target_shape_meta = _to_plain(target_shape_meta)
    source_kind = "store" if "store_path" in source_cfg else "hdf5"
    doc = _manifest_document(source_cfg, target_shape_meta, source_kind)
    cache_dir = Path(cache_root) / f"{source_cfg['name']}-{doc['hash'][:8]}"

    if cache_dir.exists():
        _verify_cache(cache_dir, doc)
        log.info(f"[load_store] cache hit {cache_dir}")
        return EpisodeStore(cache_dir, target_shape_meta)

    if source_kind == "store":
        log.info(f"[load_store] cache miss {cache_dir} -- linking {source_cfg['store_path']}")
        _convert_from_store(source_cfg, target_shape_meta, cache_dir)
    else:
        log.info(f"[load_store] cache miss {cache_dir} -- converting {source_cfg['path']}")
        _convert(source_cfg, target_shape_meta, cache_dir)
    (cache_dir / _MANIFEST_FILE).write_text(json.dumps(doc, indent=1, sort_keys=True))
    return EpisodeStore(cache_dir, target_shape_meta)


def _convert(source_cfg: dict, target_shape_meta: dict, out: Path) -> None:
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
    stats_keys = {k: np.concatenate(v, axis=0) for k, v in stats_accum.items()}

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
        out, stats_keys=stats_keys, env_meta=env_meta, source=str(source_cfg["path"]), fps=20.0
    )


def _convert_from_store(source_cfg: dict, target_shape_meta: dict, out: Path) -> None:
    """Adapt a dataset already in a robo_dataset-native format (zarr, video,
    lerobot, ...) for rgrl.

    Default (no `episode_ids`): no per-frame rewrite -- the on-disk files
    are symlinked as-is into the cache dir (image keys the target doesn't
    declare, e.g. a dropped camera, are simply never read — EpisodeStore
    only touches keys present in `target_shape_meta`) and only rgrl's
    stats/env_meta sidecars are computed fresh.

    `episode_ids`, if given (a list of source episode indices, in the order
    to keep them): builds a fresh store containing only those episodes,
    renumbered contiguously from 0 -- see `_convert_episode_subset`. Every
    store format requires contiguous 0-based episodes, so an arbitrary
    subset can't be symlinked the way the whole store can; this re-encodes
    via robo_dataset.convert.convert instead of hand-reconstructing a
    format's on-disk invariants (episode counts, boundaries, ...) here.

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
    if episode_ids is None:
        # ZarrStore reads episode_ends.npy as a required top-level file;
        # video-backed stores keep the same information in meta.json.
        for entry in ("meta.json", "episode_ends.npy", "videos", "data"):
            src_entry = store_path / entry
            if src_entry.exists():
                (out / entry).symlink_to(src_entry.resolve())
        dst = src
    else:
        _convert_episode_subset(src, out, target_shape_meta, image_keys, list(episode_ids))
        dst = open_store(str(out))

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
    )


def _convert_episode_subset(
    src, out: Path, target_shape_meta: dict, image_keys: list[str], episode_ids: list[int]
) -> None:
    """Write a fresh store containing only `episode_ids`, in that order,
    renumbered contiguously from 0 -- the reproducible, auditable way to
    pin a training subset out of a larger shared store (e.g. one DROID
    store holding many tasks): the exact indices live in the data config,
    not a random/percentage sample recomputed differently every run."""
    from robo_dataset.convert import convert as robo_convert

    bad = [i for i in episode_ids if i < 0 or i >= src.n_episodes]
    if bad:
        raise ValueError(f"episode_ids {bad} out of range for a {src.n_episodes}-episode store")

    keys = list(target_shape_meta["obs"].keys()) + ["action"]
    key_shapes = {k: src.frame_shape(k) for k in keys}
    key_dtypes = {k: src.frame_dtype(k) for k in keys}

    if image_keys:
        from robo_dataset.backends.video import VideoStoreWriter

        dst = VideoStoreWriter.create(
            str(out),
            key_shapes=key_shapes,
            key_dtypes=key_dtypes,
            n_episodes=len(episode_ids),
            image_keys=image_keys,
        )
    else:
        from robo_dataset.backends.zarr_backend import ZarrStoreWriter

        dst = ZarrStoreWriter.create(str(out), key_shapes=key_shapes, key_dtypes=key_dtypes)

    robo_convert(src, dst, keys=keys, episode_indices=episode_ids, verbose=True)
