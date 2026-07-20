"""Integration tests for rgrl.data.load_store: conversion, caching, and
cache-invalidation safety (manifest mismatch / incomplete cache both raise
instead of silently deleting or reconverting)."""

import copy
import json

import h5py
import numpy as np
import pytest

from rgrl.data import load_store


@pytest.fixture
def synthetic_hdf5(tmp_path):
    """A minimal robomimic-format HDF5: 2 episodes, one lowdim key, one image
    key, actions, and an env_args attr (mirrors third_party/robo_dataset's
    own test fixture layout)."""
    path = tmp_path / "source.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Test", "env_kwargs": {}})
        for i, t in enumerate([5, 4]):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs.create_dataset("state", data=rng.normal(size=(t, 3)).astype(np.float64))
            obs.create_dataset("cam", data=rng.integers(0, 255, size=(t, 4, 4, 3), dtype=np.uint8))
            demo.create_dataset("actions", data=rng.normal(size=(t, 2)).astype(np.float64))
    return str(path)


def _source_cfg(hdf5_path: str) -> dict:
    return {
        "name": "synth",
        "path": hdf5_path,
        "abs_action": False,
        "rename": {},
        "shape_meta": {
            "obs": {
                "state": {"shape": [3], "type": "low_dim"},
                "cam": {"shape": [4, 4, 3], "type": "rgb", "channel_order": "rgb"},
            },
            "action": {"shape": [2]},
        },
    }


def test_load_store_converts_then_hits_cache(synthetic_hdf5, tmp_path):
    source_cfg = _source_cfg(synthetic_hdf5)
    target_meta = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"

    store = load_store(source_cfg, target_meta, cache_root=cache_root)
    assert store.meta.num_episodes == 2
    assert store.meta.num_frames == 9
    assert set(store.meta.keys) == {"state", "cam", "action"}

    manifest = store.path / "cache_manifest.json"
    mtime_before = manifest.stat().st_mtime_ns

    store2 = load_store(source_cfg, target_meta, cache_root=cache_root)
    assert store2.path == store.path
    assert manifest.stat().st_mtime_ns == mtime_before  # untouched -> cache hit, no reconversion


def test_different_target_converts_and_uses_a_different_cache_dir(synthetic_hdf5, tmp_path):
    source_cfg = _source_cfg(synthetic_hdf5)
    cache_root = tmp_path / "cache"
    target_rgb = source_cfg["shape_meta"]
    target_bgr = copy.deepcopy(target_rgb)
    target_bgr["obs"]["cam"]["channel_order"] = "bgr"

    store_rgb = load_store(source_cfg, target_rgb, cache_root=cache_root)
    store_bgr = load_store(source_cfg, target_bgr, cache_root=cache_root)

    assert store_rgb.path != store_bgr.path
    np.testing.assert_array_equal(store_bgr.load("cam"), store_rgb.load("cam")[..., ::-1])


def test_tampered_manifest_raises_without_deleting_cache(synthetic_hdf5, tmp_path):
    source_cfg = _source_cfg(synthetic_hdf5)
    target_meta = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"
    store = load_store(source_cfg, target_meta, cache_root=cache_root)

    manifest_path = store.path / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_cfg"]["path"] = "/tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="manifest"):
        load_store(source_cfg, target_meta, cache_root=cache_root)
    assert manifest_path.exists()  # not deleted or repaired


def test_incomplete_cache_raises_without_deleting(synthetic_hdf5, tmp_path):
    source_cfg = _source_cfg(synthetic_hdf5)
    target_meta = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"
    store = load_store(source_cfg, target_meta, cache_root=cache_root)

    stats_path = store.path / "stats.json"
    stats_path.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        load_store(source_cfg, target_meta, cache_root=cache_root)
    assert not stats_path.exists()  # not silently regenerated


@pytest.fixture
def synthetic_store(tmp_path):
    """A minimal robo_dataset zarr store on disk (2 episodes, one lowdim
    key) — the store_path source kind's `_convert_from_store` symlinks this
    in directly rather than rewriting it, so the fixture must be a real
    on-disk ZarrStoreWriter output, not an in-memory EpisodeStore."""
    from robo_dataset.backends.zarr_backend import ZarrStoreWriter

    path = tmp_path / "src_store"
    rng = np.random.default_rng(0)
    w = ZarrStoreWriter.create(
        str(path),
        key_shapes={"state": (3,), "action": (2,)},
        key_dtypes={"state": np.float32, "action": np.float32},
    )
    for length in (5, 4):
        w.put_episode(
            {
                "state": rng.normal(size=(length, 3)).astype(np.float32),
                "action": rng.normal(size=(length, 2)).astype(np.float32),
            }
        )
    w.finalize()
    return str(path)


def _store_source_cfg(store_path: str, env_meta: dict | None = None) -> dict:
    cfg = {
        "name": "synth_store",
        "store_path": store_path,
        "abs_action": True,
        "rename": {},
        "shape_meta": {
            "obs": {"state": {"shape": [3], "type": "low_dim"}},
            "action": {"shape": [2]},
        },
    }
    if env_meta is not None:
        cfg["env_meta"] = env_meta
    return cfg


def test_convert_from_store_merges_declared_env_meta(synthetic_store, tmp_path):
    # A store built outside rgrl (e.g. by another repo's dataset writer) has
    # no robomimic env_args of its own to pull from — the caller must declare
    # env_meta explicitly on the data config for eval to be able to construct
    # the sim env at all.
    env_meta = {"env_name": "TwoArmTransport", "type": 1, "env_kwargs": {"foo": "bar"}}
    source_cfg = _store_source_cfg(synthetic_store, env_meta=env_meta)
    target_meta = source_cfg["shape_meta"]
    store = load_store(source_cfg, target_meta, cache_root=tmp_path / "cache")
    assert store.meta.num_episodes == 2
    assert store.meta.num_frames == 9
    assert store.meta.env_meta == {**env_meta, "abs_action": True}


def test_convert_from_store_without_env_meta_falls_back_to_abs_action_only(synthetic_store, tmp_path):
    source_cfg = _store_source_cfg(synthetic_store)  # no env_meta declared
    target_meta = source_cfg["shape_meta"]
    store = load_store(source_cfg, target_meta, cache_root=tmp_path / "cache")
    assert store.meta.env_meta == {"abs_action": True}


def test_sample_weights_path_does_not_bust_cache(synthetic_store, tmp_path):
    source_cfg = _store_source_cfg(synthetic_store)
    target_meta = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"

    store = load_store(source_cfg, target_meta, cache_root=cache_root)
    manifest = store.path / "cache_manifest.json"
    mtime_before = manifest.stat().st_mtime_ns

    weighted_cfg = {**source_cfg, "sample_weights_path": "/some/scores.npy"}
    store2 = load_store(weighted_cfg, target_meta, cache_root=cache_root)
    assert store2.path == store.path
    assert manifest.stat().st_mtime_ns == mtime_before  # no reconversion triggered
