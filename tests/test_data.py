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

    def fail_after_partial_write(source, target, out, num_workers=None):
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


def test_explicit_inline_selection_resolves_to_zero_copy_view(synthetic_store, tmp_path):
    """An inline explicit-episode_ids selection (the escape hatch for externally
    curated sets) resolves to exactly those episodes, in order, as a zero-copy
    EpisodeView."""
    source_cfg = _zarr_store_source_cfg(synthetic_store)
    source_cfg["dataset_id"] = "synthetic"
    source_cfg["selection"] = {"count": 2, "episode_ids": [1, 0]}

    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")
    assert store.episode_ids == (1, 0)  # caller order preserved
    assert store.meta.num_episodes == 2
    assert (store.path / "data").is_symlink()

    from robo_dataset.store_factory import open_store

    source = open_store(synthetic_store)
    first_length = source.episode_length(1)
    np.testing.assert_array_equal(
        store.load("state")[:first_length],
        source.get_frames(1, "state", 0, first_length),
    )


def _write_generative_store(tmp_path):
    """A directory with just the tasks.json (+curation.json) a generative
    selection reads -- enough to unit-test resolution without a full store."""
    store = tmp_path / "gen_store"
    store.mkdir(parents=True, exist_ok=True)
    # 6 'alpha' + 4 'beta' episodes, interleaved so a sorted pool != draw order.
    tasks = {str(e): ("alpha" if e % 5 < 3 else "beta") for e in range(10)}
    (store / "tasks.json").write_text(json.dumps(tasks))
    return store


_ALPHA = [e for e in range(10) if e % 5 < 3]


def test_generative_selection_is_deterministic_and_reproducible(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    sel = {"seed": 7, "select": [{"task": "alpha", "count": 3}]}
    first = _resolve_selection(sel, store)
    second = _resolve_selection(sel, store)
    assert first == second  # deterministic
    assert len(first) == 3
    assert set(first) <= set(_ALPHA)  # only alpha episodes


def test_generative_seed_changes_the_draw(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)

    def draw(seed):
        return _resolve_selection({"seed": seed, "select": [{"task": "alpha", "count": 2}]}, store)

    assert draw(1) != draw(2)


def test_repeated_task_continues_without_replacement(tmp_path):
    """Listing a task twice in one select list continues the same seeded draw
    without replacement -- the mechanism the plus* follow-up configs use."""
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    once = _resolve_selection({"seed": 5, "select": [{"task": "alpha", "count": 3}]}, store)
    twice = _resolve_selection(
        {"seed": 5, "select": [{"task": "alpha", "count": 3}, {"task": "alpha", "count": 3}]},
        store,
    )
    assert set(once) <= set(twice)  # the second draw is additive
    assert set(twice) == set(_ALPHA)  # all 6 alpha episodes, no overlap
    assert len(twice) == len(set(twice)) == 6


def test_generative_curation_swaps_unsuccessful_episodes(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    sel = {"seed": 7, "select": [{"task": "alpha", "count": 3}]}
    drawn = _resolve_selection(sel, store)
    # Bake a curation swapping a drawn alpha episode for another alpha episode.
    victim = drawn[0]
    substitute = next(e for e in _ALPHA if e not in drawn)
    (store / "curation.json").write_text(json.dumps({"alpha": {str(victim): substitute}}))
    curated = _resolve_selection(sel, store)
    assert victim not in curated
    assert substitute in curated


def test_generative_selection_validation_errors(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)

    with pytest.raises(ValueError, match="unknown task"):
        _resolve_selection({"seed": 1, "select": [{"task": "nope", "count": 1}]}, store)
    with pytest.raises(ValueError, match="count"):
        _resolve_selection(
            {"seed": 1, "count": 99, "select": [{"task": "alpha", "count": 2}]}, store
        )
    with pytest.raises(ValueError, match="store_path"):
        _resolve_selection({"seed": 1, "select": [{"task": "alpha", "count": 1}]}, None)
    with pytest.raises(ValueError, match="either episode_ids or seed/select"):
        _resolve_selection({"episode_ids": [0], "seed": 1, "select": []}, store)


def test_first_n_is_stable_under_store_growth(tmp_path):
    """method=first_n takes the first `count` episodes by id, so appending new
    (higher-id) episodes to the store never changes the selection -- unlike the
    seeded random draw, whose pool size changes."""
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    sel = {"method": "first_n", "select": [{"task": "alpha", "count": 3}]}
    before = _resolve_selection(sel, store)
    assert before == [0, 1, 2]  # first three alpha ids

    # Grow the store: two more 'alpha' episodes with higher ids.
    tasks = json.loads((store / "tasks.json").read_text())
    tasks["100"] = "alpha"
    tasks["101"] = "alpha"
    (store / "tasks.json").write_text(json.dumps(tasks))
    assert _resolve_selection(sel, store) == before  # unchanged under growth

    # Contrast: the seeded random draw is NOT stable across the same growth.
    rnd = {"seed": 42, "select": [{"task": "alpha", "count": 3}]}
    store2 = _write_generative_store(tmp_path / "b")
    small = _resolve_selection(rnd, store2)
    t = json.loads((store2 / "tasks.json").read_text())
    t["100"] = "alpha"
    t["101"] = "alpha"
    (store2 / "tasks.json").write_text(json.dumps(t))
    assert _resolve_selection(rnd, store2) != small


def test_first_n_continuation_takes_the_next_n(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    ids = _resolve_selection(
        {"method": "first_n", "select": [{"task": "alpha", "count": 3}, {"task": "alpha", "count": 2}]},
        store,
    )
    # first 3 alpha = [0, 1, 2], next 2 = [5, 6]; disjoint, still first-by-id.
    assert ids == [0, 1, 2, 5, 6]


def test_resolved_selection_rejects_duplicate_episodes(tmp_path):
    """The runtime uniqueness guard fires for any resolved selection with a
    repeated episode (here via an explicit list); generative draws are disjoint
    by construction, but the guard also catches a curation-substitute redraw."""
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    with pytest.raises(ValueError, match="duplicate episodes"):
        _resolve_selection({"episode_ids": [0, 1, 0]}, store)


def test_selection_method_validation(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    with pytest.raises(ValueError, match="unknown selection method"):
        _resolve_selection({"method": "nope", "seed": 1, "select": [{"task": "alpha", "count": 1}]}, store)
    with pytest.raises(ValueError, match="method=random needs a 'seed'"):
        _resolve_selection({"select": [{"task": "alpha", "count": 1}]}, store)
    with pytest.raises(ValueError, match="takes no 'seed'"):
        _resolve_selection(
            {"method": "first_n", "seed": 1, "select": [{"task": "alpha", "count": 1}]}, store
        )


def test_selection_provenance_edits_do_not_change_resolved_ids(tmp_path):
    """The resolved source (hence the conversion cache / resume fingerprint)
    keys on episode ids only: editing free-text notes while keeping the same
    (seed, select) must not change what resolves."""
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    body = {"seed": 7, "select": [{"task": "alpha", "count": 3}]}
    assert (
        _resolve_selection(body, store)
        == _resolve_selection({**body, "notes": "totally different text"}, store)
    )


def test_expected_sha256_audit_lock_catches_drift(tmp_path):
    """The expected_sha256 lock turns a silently-changed resolution (missing
    curation, numpy RNG drift, edited seed/count) into a hard error."""
    from rgrl.data import _resolve_selection, _episode_ids_sha256

    store = _write_generative_store(tmp_path)
    body = {"seed": 7, "select": [{"task": "alpha", "count": 3}]}
    good = _resolve_selection(body, store)
    assert _resolve_selection({**body, "expected_sha256": _episode_ids_sha256(good)}, store) == good
    with pytest.raises(ValueError, match="expected_sha256"):
        _resolve_selection({**body, "expected_sha256": "deadbeef"}, store)


def test_curation_must_reference_in_task_episodes(tmp_path):
    from rgrl.data import _resolve_selection

    store = _write_generative_store(tmp_path)
    sel = {"seed": 7, "select": [{"task": "alpha", "count": 3}]}
    beta_episode = next(e for e in range(10) if e % 5 >= 3)  # belongs to 'beta'
    # Target crosses tasks (substitute must stay in-task).
    (store / "curation.json").write_text(json.dumps({"alpha": {"0": beta_episode}}))
    with pytest.raises(ValueError, match="stay in-task"):
        _resolve_selection(sel, store)
    # Source is not a real episode of the task (stale/mistyped).
    (store / "curation.json").write_text(json.dumps({"alpha": {"999": 1}}))
    with pytest.raises(ValueError, match="not an episode"):
        _resolve_selection(sel, store)


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


def test_generative_selection_builds_a_unique_episode_dataset(video_store_path, tmp_path):
    """End-to-end episode uniqueness at construction time: a generative selection
    -> load_store -> dataset windows. Every logical episode maps to a distinct
    physical episode, no frame is reused (num_frames == sum of the selected
    episodes' lengths), and no observation window straddles an episode boundary."""
    from robo_dataset.store_factory import open_store
    from rgrl.dataset import Windows, _split_bounds, _windows_index_table

    # Tag the 4 source episodes (lengths 5, 4, 6, 3) with two tasks, then pick
    # one per task plus a second 'A' -- a continuation, so uniqueness across the
    # repeated task is exercised at build time, not just in resolution.
    (Path(video_store_path) / "tasks.json").write_text(
        json.dumps({"0": "A", "1": "A", "2": "B", "3": "B"})
    )
    source_cfg = _video_store_source_cfg(video_store_path)
    source_cfg["selection"] = {
        "method": "first_n",
        "select": [
            {"task": "A", "count": 1},
            {"task": "B", "count": 1},
            {"task": "A", "count": 1},  # continuation -> next 'A' by id
        ],
    }
    store = load_store(source_cfg, source_cfg["shape_meta"], cache_root=tmp_path / "cache")

    phys = list(store.episode_ids)
    assert phys == [0, 1, 2]  # A:0, B:2, then next A:1 -- resorted logical order
    assert len(phys) == len(set(phys)) == store.meta.num_episodes  # unique episodes

    src = open_store(video_store_path)
    # No frame reused: total frames == sum over the DISTINCT selected episodes.
    assert store.meta.num_frames == sum(src.episode_length(i) for i in phys)

    # No window crosses an episode boundary in the constructed dataset.
    bounds = _split_bounds(store, np.arange(store.meta.num_episodes))
    obs, _act, _pad = _windows_index_table(bounds, Windows(obs_horizon=2, horizon=4))
    frame_ep = {f: ep for ep, (lo, hi) in enumerate(bounds.tolist()) for f in range(lo, hi)}
    assert all(len({frame_ep[int(f)] for f in row}) == 1 for row in obs.numpy())


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


def test_stats_workers_do_not_change_stats(synthetic_hdf5, tmp_path):
    # Parallel per-key fitting must produce byte-identical stats.
    source_cfg = _source_cfg(synthetic_hdf5)
    target_meta = source_cfg["shape_meta"]
    serial = load_store(source_cfg, target_meta, cache_root=tmp_path / "c1", stats_workers=1)
    parallel = load_store(source_cfg, target_meta, cache_root=tmp_path / "c4", stats_workers=4)
    a = json.loads((serial.path / "stats.json").read_text())
    b = json.loads((parallel.path / "stats.json").read_text())
    assert a == b and a  # non-empty and identical


def test_norm_method_is_part_of_cache_identity(synthetic_hdf5, tmp_path):
    # Changing the normalization method must rebuild the cache, never silently
    # reuse stats fit with a different method.
    source_cfg = _source_cfg(synthetic_hdf5)
    target = source_cfg["shape_meta"]
    cache_root = tmp_path / "cache"
    mm = load_store({**source_cfg, "norm_method": "minmax"}, target, cache_root=cache_root)
    zs = load_store({**source_cfg, "norm_method": "zscore"}, target, cache_root=cache_root)
    assert mm.path != zs.path
    assert json.loads((mm.path / "stats.json").read_text())["action"]["method"] == "minmax"
    assert json.loads((zs.path / "stats.json").read_text())["action"]["method"] == "zscore"
