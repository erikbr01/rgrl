"""Integration tests for rgrl.data.load_store: conversion, caching, and
cache-invalidation safety (manifest mismatch / incomplete cache both raise
instead of silently deleting or reconverting)."""

import copy
import json
from pathlib import Path

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


def test_failed_conversion_never_publishes_a_partial_cache(
    synthetic_hdf5, tmp_path, monkeypatch
):
    import rgrl.data as data_module

    source_cfg = _source_cfg(synthetic_hdf5)
    cache_root = tmp_path / "cache"
    original_convert = data_module._convert

    def fail_after_partial_write(source, target, out):
        out.mkdir(parents=True)
        (out / "partial").write_text("injected")
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(data_module, "_convert", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected"):
        load_store(source_cfg, source_cfg["shape_meta"], cache_root=cache_root)
    assert not [path for path in cache_root.iterdir() if path.is_dir()]
    assert not list(cache_root.glob("*.tmp"))

    monkeypatch.setattr(data_module, "_convert", original_convert)
    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=cache_root)
    assert store.meta.num_frames == 9


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


def _zarr_store_source_cfg(store_path: str, env_meta: dict | None = None) -> dict:
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
    source_cfg = _zarr_store_source_cfg(synthetic_store, env_meta=env_meta)
    target_meta = source_cfg["shape_meta"]
    store = load_store(source_cfg, target_meta, cache_root=tmp_path / "cache")
    assert store.meta.num_episodes == 2
    assert store.meta.num_frames == 9
    assert store.meta.env_meta == {**env_meta, "abs_action": True}


def test_convert_from_store_without_env_meta_falls_back_to_abs_action_only(synthetic_store, tmp_path):
    source_cfg = _zarr_store_source_cfg(synthetic_store)  # no env_meta declared
    target_meta = source_cfg["shape_meta"]
    store = load_store(source_cfg, target_meta, cache_root=tmp_path / "cache")
    assert store.meta.env_meta == {"abs_action": True}


def test_sample_weights_path_does_not_bust_cache(synthetic_store, tmp_path):
    source_cfg = _zarr_store_source_cfg(synthetic_store)
    target_meta = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"

    store = load_store(source_cfg, target_meta, cache_root=cache_root)
    manifest = store.path / "cache_manifest.json"
    mtime_before = manifest.stat().st_mtime_ns

    weighted_cfg = {**source_cfg, "sample_weights_path": "/some/scores.npy"}
    store2 = load_store(weighted_cfg, target_meta, cache_root=cache_root)
    assert store2.path == store.path
    assert manifest.stat().st_mtime_ns == mtime_before  # no reconversion triggered


def test_selection_manifest_parent_additions_resolve_to_zero_copy_view(
    synthetic_store, tmp_path
):
    selection_dir = tmp_path / "selections"
    selection_dir.mkdir()
    (selection_dir / "base.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "base",
                "source": {"dataset_id": "synthetic"},
                "ordering": "preserve",
                "count": 1,
                "episode_ids": [1],
            }
        )
    )
    child_path = selection_dir / "child.json"
    child_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "child",
                "source": {"dataset_id": "synthetic"},
                "parent": "base.json",
                "ordering": "preserve",
                "count": 2,
                "add_episode_ids": [0],
            }
        )
    )
    source_cfg = _zarr_store_source_cfg(synthetic_store)
    source_cfg["dataset_id"] = "synthetic"
    source_cfg["selection_path"] = str(child_path)

    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")
    assert store.episode_ids == (1, 0)
    assert store.meta.num_episodes == 2
    assert (store.path / "data").is_symlink()

    from robo_dataset.store_factory import open_store

    source = open_store(synthetic_store)
    first_length = source.episode_length(1)
    np.testing.assert_array_equal(
        store.load("state")[:first_length],
        source.get_frames(1, "state", 0, first_length),
    )


@pytest.fixture
def video_store_path(tmp_path):
    """A small robo_dataset VideoStore (4 episodes, one image key, one
    lowdim key, actions) -- exercises the store_path (episode_ids
    subsetting) source kind, unlike synthetic_hdf5's `path` source kind."""
    pytest.importorskip("av", reason="PyAV needed to build a VideoStore fixture (rgrl's `video` extra)")
    from robo_dataset.backends.video import VideoStoreWriter

    path = tmp_path / "source_store"
    ep_lengths = [5, 4, 6, 3]
    rng = np.random.default_rng(0)
    writer = VideoStoreWriter.create(
        str(path),
        key_shapes={"state": (3,), "cam": (8, 8, 3), "action": (2,)},
        key_dtypes={"state": np.float32, "cam": np.uint8, "action": np.float32},
        n_episodes=len(ep_lengths),
        image_keys=["cam"],
    )
    for ep_idx, t in enumerate(ep_lengths):
        writer.put_episode(
            ep_idx,
            {
                "state": rng.normal(size=(t, 3)).astype(np.float32),
                "cam": rng.integers(0, 255, size=(t, 8, 8, 3), dtype=np.uint8),
                "action": rng.normal(size=(t, 2)).astype(np.float32),
            },
        )
    writer.finalize()
    return str(path)


def _video_store_source_cfg(store_path: str, episode_ids=None) -> dict:
    cfg = {
        "name": "synth_store",
        "store_path": store_path,
        "abs_action": False,
        "rename": {},
        "shape_meta": {
            "obs": {
                "state": {"shape": [3], "type": "low_dim"},
                "cam": {"shape": [8, 8, 3], "type": "rgb", "channel_order": "rgb"},
            },
            "action": {"shape": [2]},
        },
    }
    if episode_ids is not None:
        cfg["episode_ids"] = episode_ids
    return cfg


def test_load_store_from_store_path_whole_store_unchanged(video_store_path, tmp_path):
    """No episode_ids: existing symlink-the-whole-store behavior."""
    source_cfg = _video_store_source_cfg(video_store_path)
    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")
    assert store.meta.num_episodes == 4
    assert store.meta.num_frames == 5 + 4 + 6 + 3


def test_episode_ids_selects_and_reindexes_a_reproducible_subset(video_store_path, tmp_path):
    """episode_ids keeps exactly the given source episodes, in that order,
    renumbered from 0 -- both a correctness check (right frames end up in
    the right place) and a reproducibility check (stats reflect only the
    subset, not the whole source store)."""
    from robo_dataset.store_factory import open_store

    src = open_store(video_store_path)
    episode_ids = [2, 0]  # subset, deliberately out of order and skipping episodes 1, 3
    source_cfg = _video_store_source_cfg(video_store_path, episode_ids=episode_ids)

    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")
    assert store.meta.num_episodes == len(episode_ids)
    assert store.meta.num_frames == sum(src.episode_length(i) for i in episode_ids)
    # Selection is metadata-only: the cache points at the original physical
    # store and carries a logical episode mapping, with no re-encoded videos.
    assert (store.path / "videos").is_symlink()
    assert (store.path / "data").is_symlink()
    view_doc = json.loads((store.path / "episode_view.json").read_text())
    assert view_doc["episode_ids"] == episode_ids
    assert (store.path / "videos").resolve() == (Path(video_store_path) / "videos").resolve()

    # dst episode 0 must be src episode 2's data (order preserved, not sorted)
    src_ep0_len = src.episode_length(2)
    np.testing.assert_allclose(
        store.load("state")[:src_ep0_len], src.get_frames(2, "state", 0, src_ep0_len)
    )

    # stats computed over just the 2-episode subset, not all 4 source episodes
    subset_frames = sum(src.episode_length(i) for i in episode_ids)
    all_frames = int(src.episode_ends[-1])
    assert subset_frames < all_frames  # sanity: the subset really is smaller
    subset_state = np.concatenate(
        [src.get_frames(i, "state", 0, src.episode_length(i)) for i in episode_ids]
    )
    stats = json.loads((store.path / "stats.json").read_text())
    np.testing.assert_allclose(stats["state"]["stats"]["min"], subset_state.min(axis=0), rtol=1e-5)


def test_episode_ids_out_of_range_raises(video_store_path, tmp_path):
    source_cfg = _video_store_source_cfg(video_store_path, episode_ids=[0, 99])
    with pytest.raises(ValueError, match="out of range"):
        load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")


@pytest.mark.parametrize(
    ("episode_ids", "message"),
    [([], "at least one"), ([0, 0], "unique"), ([0, 1.5], "integers")],
)
def test_episode_ids_are_validated(video_store_path, tmp_path, episode_ids, message):
    source_cfg = _video_store_source_cfg(video_store_path, episode_ids=episode_ids)
    with pytest.raises((TypeError, ValueError), match=message):
        load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")


def test_short_prefix_collision_detected_by_full_hash(tmp_path):
    # A genuine 32-bit prefix collision (different full hash) is distinguishable
    # from an in-place tamper (same full hash, edited content); only the former
    # should route to a fallback directory. Tampering stays a hard error.
    from rgrl.data import _MANIFEST_FILE, _is_short_prefix_collision

    d = tmp_path / "cache_dir"
    d.mkdir()
    (d / _MANIFEST_FILE).write_text(json.dumps({"hash": "aaaa", "payload": 1}))
    assert _is_short_prefix_collision(d, {"hash": "bbbb"}) is True   # different config
    assert _is_short_prefix_collision(d, {"hash": "aaaa"}) is False  # same identity
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _is_short_prefix_collision(empty, {"hash": "aaaa"}) is False
