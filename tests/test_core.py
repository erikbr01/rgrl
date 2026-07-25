"""Fast CPU sanity tests: window indexing, normalization, crops, UNet shapes."""

import json
import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from robo_dataset.backends.zarr_backend import ZarrStoreWriter
from robo_dataset.normalization import fit_key
from robo_dataset.transforms.gpu_batch import CenterCrop, RandomCrop

from rgrl.dataset import CpuSequenceDataset, GpuSequenceDataset, Windows, build_dataset
from rgrl.inference.checkpoint import InferenceConfig, load_policy
from rgrl.inference.legacy_checkpoint import legacy_shape_meta
from rgrl.nets.unet1d import ConditionalUnet1D
from rgrl.normalize import Normalizer
from rgrl.policies.base import ObsEncoder
from rgrl.policies.diffusion import DiffusionPolicy
from rgrl.policies.dit_flow import DiTFlowPolicy
from rgrl.policies.flow import FlowPolicy
from rgrl.store import EpisodeStore, KeySpec, StoreMeta, write_sidecars
from rgrl.train_loop import TrainConfig, seed_everything, train
from rgrl.transforms import (
    CropTransform,
    DataPipeline,
    ImagePreprocessTransform,
    ImageToTensorTransform,
    ResizeTransform,
)


@pytest.fixture
def tiny_store(tmp_path):
    rng = np.random.default_rng(0)
    key_shapes = {"state": (3,), "action": (2,)}
    key_dtypes = {"state": np.float32, "action": np.float32}
    w = ZarrStoreWriter.create(str(tmp_path / "store"), key_shapes=key_shapes, key_dtypes=key_dtypes)
    episodes = []
    for length in (7, 5, 9):
        ep = {
            "state": rng.normal(size=(length, 3)).astype(np.float32),
            "action": rng.normal(size=(length, 2)).astype(np.float32),
        }
        w.put_episode(ep)
        episodes.append(ep)
    w.finalize()
    write_sidecars(
        tmp_path / "store",
        stats_keys={k: np.concatenate([ep[k] for ep in episodes]) for k in key_shapes},
        source="test",
    )
    shape_meta = {
        "obs": {"state": {"shape": [3], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    return EpisodeStore(tmp_path / "store", shape_meta)


def test_windowing_clamps_and_pads(tiny_store):
    ds = GpuSequenceDataset(tiny_store, Windows(obs_horizon=2, horizon=4), torch.device("cpu"))
    assert ds.num_samples == 7 + 5 + 9

    # first sample of episode 0 (t=0): obs indices clamp to episode start
    assert ds.obs_idx[0].tolist() == [0, 0]
    # action window starts at first obs frame: t-oh+1 = -1 -> clamped, padded
    assert ds.act_idx[0].tolist() == [0, 0, 1, 2]
    assert ds.act_pad[0].tolist() == [True, False, False, False]

    # last sample of episode 0 (t=6): actions clamp at episode end (frame 6)
    assert ds.act_idx[6].tolist() == [5, 6, 6, 6]
    assert ds.act_pad[6].tolist() == [False, False, True, True]

    # first sample of episode 1 (global frame 7) never reaches into episode 0
    assert ds.obs_idx[7].tolist() == [7, 7]
    assert ds.act_idx[7].min().item() == 7


def test_gather_matches_direct_indexing(tiny_store):
    ds = GpuSequenceDataset(tiny_store, Windows(2, 4), torch.device("cpu"))
    batch = ds.gather(torch.tensor([0, 8, 20]))
    assert batch["state"].shape == (3, 2, 3)
    assert batch["action"].shape == (3, 4, 2)
    raw = ds.tensors["state"]
    assert torch.equal(batch["state"][1], raw[ds.obs_idx[8]])


def test_image_window_preserves_nhwc_and_current_action_alignment():
    """The first action a policy returns is prediction slot obs_horizon-1.

    That slot must align with the newest image in the observation window,
    including at episode boundaries. This is the image-policy version of the
    generic index-table check above and guards the complete gather contract.
    """
    n = 4
    images = np.stack(
        [np.full((3, 5, 3), t, dtype=np.uint8) for t in range(n)]
    )
    actions = np.stack(
        [np.array([t, -t], dtype=np.float32) for t in range(n)]
    )

    class ImageStore:
        meta = StoreMeta(
            episode_ends=[n],
            keys={
                "camera": KeySpec([3, 5, 3], "uint8", "image", channel_order="rgb"),
                "action": KeySpec([2], "float32", "action"),
            },
        )

        @staticmethod
        def episode_bounds():
            return np.array([[0, n]], dtype=np.int64)

        @staticmethod
        def load(key, episode_range=None):
            assert episode_range == (0, 1)
            return images if key == "camera" else actions

    ds = GpuSequenceDataset(ImageStore(), Windows(2, 4), torch.device("cpu"))
    batch = ds.gather(torch.tensor([0, 2, 3]))

    assert batch["camera"].shape == (3, 2, 3, 5, 3)
    assert batch["camera"].dtype == torch.uint8
    assert batch["camera"].is_contiguous()
    # Front fill at t=0; normal consecutive context at t=2; never CHW.
    torch.testing.assert_close(
        batch["camera"][0, :, 0, 0, 0], torch.tensor([0, 0], dtype=torch.uint8)
    )
    torch.testing.assert_close(
        batch["camera"][1, :, 0, 0, 0], torch.tensor([1, 2], dtype=torch.uint8)
    )
    # Prediction slot 1 corresponds to the current image/action at each anchor.
    torch.testing.assert_close(batch["action"][:, 1], torch.from_numpy(actions[[0, 2, 3]]))


def test_val_split_loads_only_its_own_episodes_not_a_full_store_copy(tiny_store):
    """Regression test: train and val must each load only their own split's
    frames, not a full duplicate copy of the store -- previously
    GpuSequenceDataset.__init__ called store.load(key) unconditionally
    (every episode, regardless of split), so a train+val pair together
    loaded the whole store onto the device TWICE. With 1 val episode out
    of 3 (lengths 7, 5, 9 -> 21 total frames), train should hold 12 frames
    and val should hold 9 -- 21 total, not 42."""
    train = GpuSequenceDataset(
        tiny_store, Windows(2, 4), torch.device("cpu"), val_episodes=1, val_split="train"
    )
    val = GpuSequenceDataset(
        tiny_store, Windows(2, 4), torch.device("cpu"), val_episodes=1, val_split="val"
    )
    assert train.tensors["state"].shape[0] == 7 + 5
    assert val.tensors["state"].shape[0] == 9
    assert train.num_samples == 7 + 5
    assert val.num_samples == 9

    # val's data must be episode 2's (the last episode) actual frames, not
    # episode 0's or garbage from a wrong offset -- get it independently
    # from the store and compare.
    expected = tiny_store.load("state", episode_range=(2, 3))
    np.testing.assert_array_equal(val.tensors["state"].numpy(), expected)

    # val's windowing must be entirely local (0-based), never referencing
    # frames outside its own 9-frame tensor -- a stale absolute offset
    # would index past the end (or into the wrong data) instead of raising.
    assert val.obs_idx.max().item() < 9
    assert val.act_idx.max().item() < 9
    batch = val.gather(torch.tensor([0]))
    np.testing.assert_array_equal(batch["state"][0, -1].numpy(), expected[0])


def test_cpu_sequence_dataset_batches_and_gather_are_structurally_correct(tiny_store):
    """CpuSequenceDataset (real DataLoader, spawn worker processes) must
    produce the same shapes/dtypes/keys GpuSequenceDataset does for the
    same store -- not bit-identical sampling (DataLoader's own shuffle
    replaces the custom epoch_perm scheme for this mode, intentionally),
    just structurally correct, and it must actually complete several
    batches through real worker processes without hanging."""
    windows = Windows(2, 4)
    gpu_ds = GpuSequenceDataset(tiny_store, windows, torch.device("cpu"))
    cpu_ds = CpuSequenceDataset(tiny_store, windows, torch.device("cpu"), num_workers=2)
    assert cpu_ds.num_samples == gpu_ds.num_samples == 7 + 5 + 9

    batches = cpu_ds.batches(batch_size=4, seed=0)
    for _ in range(3):  # several batches -- exercises the persistent worker pool repeatedly
        batch = next(batches)
        assert set(batch.keys()) == {"state", "action", "action_is_pad"}
        assert batch["state"].shape == (4, 2, 3)
        assert batch["action"].shape == (4, 4, 2)
        assert batch["action_is_pad"].shape == (4, 4)
        assert batch["state"].dtype == torch.float32
        assert batch["action_is_pad"].dtype == torch.bool

    # gather() (the val-evaluation path) covers arbitrary explicit sample_ids
    # directly, no DataLoader involved.
    gathered = cpu_ds.gather(torch.tensor([0, 8, 20]))
    assert gathered["state"].shape == (3, 2, 3)
    assert gathered["action"].shape == (3, 4, 2)
    np.testing.assert_array_equal(gathered["state"].numpy(), gpu_ds.gather(torch.tensor([0, 8, 20]))["state"].numpy())


def test_build_dataset_stays_gpu_resident_on_cpu_device_regardless_of_threshold(tiny_store):
    """pin_fallback_gb only ever matters for a CUDA target -- CPU-target
    training has no VRAM to run out of, so build_dataset must always
    return the plain GpuSequenceDataset there, even with a threshold of 0."""
    ds = build_dataset(tiny_store, Windows(2, 4), torch.device("cpu"), pin_fallback_gb=0.0, dataloader_workers=4)
    assert isinstance(ds, GpuSequenceDataset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_build_dataset_picks_cpu_dataset_past_the_gb_threshold_on_cuda():
    """The actual dispatch condition build_dataset exists for: a CUDA
    target whose split exceeds pin_fallback_gb gets CpuSequenceDataset
    (dataloader_workers=0 or not -- the threshold alone decides the mode,
    see build_dataset's docstring); comfortably under it stays
    GpuSequenceDataset."""
    import tempfile

    from robo_dataset.backends.zarr_backend import ZarrStoreWriter

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        key_shapes = {"state": (3,), "action": (2,)}
        key_dtypes = {"state": np.float32, "action": np.float32}
        w = ZarrStoreWriter.create(f"{tmp}/store", key_shapes=key_shapes, key_dtypes=key_dtypes)
        ep = {"state": rng.normal(size=(5, 3)).astype(np.float32), "action": rng.normal(size=(5, 2)).astype(np.float32)}
        w.put_episode(ep)
        w.finalize()
        write_sidecars(f"{tmp}/store", stats_keys={k: ep[k] for k in key_shapes}, source="test")
        shape_meta = {"obs": {"state": {"shape": [3], "type": "low_dim"}}, "action": {"shape": [2]}}
        store = EpisodeStore(f"{tmp}/store", shape_meta)

        device = torch.device("cuda")
        under = build_dataset(store, Windows(2, 4), device, pin_fallback_gb=1e6, dataloader_workers=0)
        over = build_dataset(store, Windows(2, 4), device, pin_fallback_gb=0.0, dataloader_workers=0)
        assert isinstance(under, GpuSequenceDataset)
        assert isinstance(over, CpuSequenceDataset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_side_stream_prefetch_records_main_stream_lifetime():
    from robo_dataset.prefetch import DataPrefetcher

    class Source(torch.utils.data.Dataset):
        def __len__(self):
            return 128

        def __getitem__(self, index):
            return {"value": torch.full((4096,), float(index))}

    loader = torch.utils.data.DataLoader(
        Source(),
        batch_size=1,
        pin_memory=True,
        num_workers=0,
    )
    prefetch = DataPrefetcher(loader, device=torch.device("cuda"))
    assert prefetch.strategy == "stream"

    results = []
    for index, batch in enumerate(prefetch):
        # Queue use on the main stream, then let the generator advance and
        # release/reuse prior side-stream allocations before synchronizing.
        results.append((batch["value"] * 3).sum())
        torch.empty_like(batch["value"])
        assert index < len(loader)
    actual = torch.stack(results).cpu()
    expected = torch.arange(128, dtype=torch.float32) * (4096 * 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_epoch_perm_reproducible(tiny_store):
    ds = GpuSequenceDataset(tiny_store, Windows(2, 4), torch.device("cpu"))
    assert torch.equal(ds.epoch_perm(1, 5), ds.epoch_perm(1, 5))
    assert not torch.equal(ds.epoch_perm(1, 5), ds.epoch_perm(1, 6))


def test_weighted_epoch_perm_deterministic_and_extreme_skew(tiny_store):
    n = tiny_store.meta.num_frames
    weights = torch.zeros(n)
    weights[0] = 1.0  # only frame 0 has nonzero weight -> every draw is forced to index 0
    ds = GpuSequenceDataset(
        tiny_store, Windows(2, 4), torch.device("cpu"), sample_weights=weights
    )
    assert torch.equal(ds.epoch_perm(1, 5), ds.epoch_perm(1, 5))
    assert (ds.epoch_perm(1, 5) == 0).all()  # every draw must be the one nonzero-weight sample


def test_weighted_epoch_perm_biases_toward_high_weight(tiny_store):
    n = tiny_store.meta.num_frames
    weights = torch.ones(n)
    weights[0] = 100.0
    ds = GpuSequenceDataset(
        tiny_store, Windows(2, 4), torch.device("cpu"), sample_weights=weights
    )
    assert torch.equal(ds.epoch_perm(1, 5), ds.epoch_perm(1, 5))
    assert not torch.equal(ds.epoch_perm(1, 5), ds.epoch_perm(1, 6))  # not degenerate -> varies
    counts = torch.zeros(ds.num_samples)
    for epoch in range(20):
        counts += torch.bincount(ds.epoch_perm(0, epoch), minlength=ds.num_samples)
    assert (counts[0] / counts.sum()).item() > 0.5  # well above uniform (1/n)


def test_sample_weights_validation(tiny_store):
    n = tiny_store.meta.num_frames
    with pytest.raises(ValueError, match="length"):
        GpuSequenceDataset(
            tiny_store, Windows(2, 4), torch.device("cpu"), sample_weights=torch.ones(n - 1)
        )
    with pytest.raises(ValueError, match="finite"):
        GpuSequenceDataset(
            tiny_store, Windows(2, 4), torch.device("cpu"),
            sample_weights=torch.full((n,), float("nan")),
        )
    with pytest.raises(ValueError, match="finite"):
        bad = torch.ones(n)
        bad[0] = -1.0
        GpuSequenceDataset(tiny_store, Windows(2, 4), torch.device("cpu"), sample_weights=bad)
    with pytest.raises(ValueError, match="sum to 0"):
        GpuSequenceDataset(
            tiny_store, Windows(2, 4), torch.device("cpu"), sample_weights=torch.zeros(n)
        )


def test_normalizer_roundtrip():
    arr = np.array([[-2.0, 0.0, 5.0], [2.0, 1.0, 5.0]], dtype=np.float32)  # last dim constant
    n = Normalizer(fit_key(arr, "minmax"))
    x = torch.tensor([[1.0, 0.25, 5.0], [-2.0, 1.0, 5.0]])
    y = n(x)
    assert y.min() >= -1 and y.max() <= 1
    assert torch.allclose(y[:, 2], torch.zeros(2))  # constant dim -> 0
    assert torch.allclose(n.inverse(y), x, atol=1e-6)


def test_crops():
    # ObsEncoder's exact usage: (B, T, C, H, W), pad=0 (crop within the image,
    # no padding), per_timestep=True (independent offset per obs frame).
    x = torch.arange(2 * 2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 2, 3, 8, 8)
    rc = RandomCrop(["img"], size=(5, 5), pad=0, per_timestep=True)
    torch.manual_seed(0)
    out = rc({"img": x})["img"]
    assert out.shape == (2, 2, 3, 5, 5)
    # crop content must be a contiguous window of the source at each (B, T)
    for b in range(2):
        for t in range(2):
            src = x[b, t, 0]
            found = any(
                torch.equal(out[b, t, 0], src[i : i + 5, j : j + 5])
                for i in range(4)
                for j in range(4)
            )
            assert found

    cc = CenterCrop(["img"], size=(8, 8))
    assert cc({"img": x})["img"].shape == (2, 2, 3, 8, 8)  # no-op at native size
    cc5 = CenterCrop(["img"], size=(5, 5))
    out1 = cc5({"img": x})["img"]
    out2 = cc5({"img": x})["img"]
    assert torch.equal(out1, out2)  # deterministic


def test_uint8_first_image_preprocess_matches_float_first_pipeline():
    image = torch.arange(3 * 2 * 8 * 8 * 3, dtype=torch.int64)
    image = image.remainder(256).to(torch.uint8).reshape(3, 2, 8, 8, 3)
    old = DataPipeline(
        [
            ImageToTensorTransform(["img"]),
            ResizeTransform(["img"], size=(8, 8)),
            CropTransform(["img"], size=(5, 5), pad=0, per_timestep=True),
        ]
    )
    new = ImagePreprocessTransform(
        ["img"],
        native_hw=(8, 8),
        crop=CropTransform(["img"], size=(5, 5), pad=0, per_timestep=True),
    )

    old.train()
    new.train()
    torch.manual_seed(11)
    expected = old({"img": image})["img"]
    torch.manual_seed(11)
    actual = new({"img": image})["img"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    old.eval()
    new.eval()
    expected = old({"img": image})["img"]
    actual = new({"img": image})["img"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Non-native live input takes the resize-first compatibility path.
    live = torch.randint(0, 256, (2, 2, 10, 10, 3), dtype=torch.uint8)
    old_live = DataPipeline(
        [
            ImageToTensorTransform(["img"]),
            ResizeTransform(["img"], size=(8, 8)),
            CropTransform(["img"], size=(5, 5), pad=0, per_timestep=True),
        ]
    ).eval()
    new_live = ImagePreprocessTransform(
        ["img"],
        native_hw=(8, 8),
        crop=CropTransform(["img"], size=(5, 5), pad=0, per_timestep=True),
    ).eval()
    torch.testing.assert_close(
        new_live({"img": live})["img"],
        old_live({"img": live})["img"],
        rtol=0,
        atol=0,
    )


def _multicamera_store_metadata(tiny_store):
    keys = {
        "state": tiny_store.meta.keys["state"],
        "camera.left": KeySpec([32, 32, 3], "uint8", "image"),
        "camera.right": KeySpec([32, 32, 3], "uint8", "image"),
    }
    return SimpleNamespace(
        meta=SimpleNamespace(keys=keys),
        stats={"state": tiny_store.stats["state"]},
    )


def test_shared_image_encoder_is_opt_in_and_shape_compatible(tiny_store):
    store = _multicamera_store_metadata(tiny_store)
    separate = ObsEncoder(
        store, obs_horizon=2, target_hw=(32, 32), num_kp=4,
        share_image_encoder=False,
    )
    shared = ObsEncoder(
        store, obs_horizon=2, target_hw=(32, 32), num_kp=4,
        share_image_encoder=True,
    )
    batch = {
        "state": torch.randn(2, 2, 3),
        "camera.left": torch.rand(2, 2, 3, 32, 32),
        "camera.right": torch.rand(2, 2, 3, 32, 32),
    }
    assert separate(batch).shape == shared(batch).shape == (2, separate.out_dim)
    assert sum(p.numel() for p in shared.parameters()) < sum(
        p.numel() for p in separate.parameters()
    )
    assert not any(key.startswith("shared_encoder.") for key in separate.state_dict())
    assert any(key.startswith("shared_encoder.") for key in shared.state_dict())


def test_observation_module_name_collisions_fail_fast(tiny_store):
    store = SimpleNamespace(
        meta=SimpleNamespace(
            keys={
                "joint.position": tiny_store.meta.keys["state"],
                "joint_position": tiny_store.meta.keys["state"],
            }
        ),
        stats={
            "joint.position": tiny_store.stats["state"],
            "joint_position": tiny_store.stats["state"],
        },
    )
    with pytest.raises(ValueError, match="collide"):
        ObsEncoder(store, obs_horizon=2)


def test_unet_shapes():
    net = ConditionalUnet1D(input_dim=7, global_cond_dim=20, down_dims=(32, 64))
    x = torch.randn(4, 16, 7)
    out = net(x, torch.randint(0, 100, (4,)), torch.randn(4, 20))
    assert out.shape == x.shape


def test_diffusion_policy_sample_timesteps_buffer(tiny_store):
    # rgrl/inference/cuda_graph.py needs these precomputed once, GPU-buffer-
    # backed rather than an ephemeral per-call CPU tensor (see
    # DiffusionPolicy.__init__) -- confirm the precomputed schedule matches
    # the formula it replaced, and predict_action still works end to end.
    policy = DiffusionPolicy(
        tiny_store, obs_horizon=2, horizon=4, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[],
        train_timesteps=20, inference_steps=5, sampler="ddim",
    )
    ratio = 20 // 5
    expected = (torch.arange(5) * ratio).flip(0).tolist()
    assert policy._sample_timesteps_cpu == expected
    assert torch.equal(policy._sample_timesteps, torch.tensor(expected))

    for sampler in ("ddim", "ddpm"):
        policy.cfg.sampler = sampler
        batch = {"state": torch.randn(3, 2, 3)}
        action = policy.predict_action(batch)
        assert action.shape == (3, 2, 2)
        assert torch.isfinite(action).all()


def test_flow_policy_rejects_nonpositive_inference_steps(tiny_store):
    with pytest.raises(ValueError, match="inference_steps must be positive"):
        FlowPolicy(tiny_store, inference_steps=0, data_transforms=[])


@pytest.mark.parametrize(
    ("policy_name", "policy_class", "specific_kwargs"),
    [
        (
            "diffusion",
            DiffusionPolicy,
            {"train_timesteps": 20, "inference_steps": 5, "sampler": "ddim"},
        ),
        (
            "flow",
            FlowPolicy,
            {"inference_steps": 5, "timestep_scale": 1000},
        ),
    ],
)
def test_legacy_checkpoint_loader_uses_declared_store_schema(
    tiny_store,
    tmp_path,
    policy_name,
    policy_class,
    specific_kwargs,
):
    # Legacy checkpoints omitted shape_meta and therefore require the old
    # store's explicit image_keys sidecar (empty for this state-only fixture).
    rgrl_meta_path = tiny_store.path / "rgrl_meta.json"
    rgrl_meta = json.loads(rgrl_meta_path.read_text())
    rgrl_meta["image_keys"] = []
    rgrl_meta_path.write_text(json.dumps(rgrl_meta))

    kwargs = {
        "obs_horizon": 2,
        "horizon": 4,
        "n_action_steps": 2,
        "down_dims": [8, 16],
        "num_kp": 4,
        **specific_kwargs,
    }
    original = policy_class(tiny_store, data_transforms=[], **kwargs)
    ckpt_path = tmp_path / f"legacy-{policy_name}.pt"
    torch.save(
        {
            "model": original.state_dict(),
            "ema": original.state_dict(),
            "step": 123,
            "config": {
                "store": str(tiny_store.path),
                "policy": policy_name,
                policy_name: kwargs,
                "crop_hw": None,
            },
        },
        ckpt_path,
    )

    loaded, store, step = load_policy(
        ckpt_path,
        InferenceConfig(compile=False),
        torch.device("cpu"),
    )
    assert step == 123
    assert store.meta.keys == tiny_store.meta.keys
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)

    batch = {"state": torch.randn(2, 2, 3)}
    torch.manual_seed(17)
    expected = original.eval().predict_action(batch)
    torch.manual_seed(17)
    actual = loaded.predict_action(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_schema_refuses_to_guess_modalities(tiny_store):
    with pytest.raises(ValueError, match="no image_keys declaration"):
        legacy_shape_meta(tiny_store.path)


def test_legacy_schema_preserves_declared_image_modality(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "keys": ["camera", "state", "action"],
                "shapes": {
                    "camera": [32, 48, 3],
                    "state": [4],
                    "action": [2],
                },
            }
        )
    )
    (tmp_path / "rgrl_meta.json").write_text(json.dumps({"image_keys": ["camera"]}))
    (tmp_path / "stats.json").write_text(json.dumps({"state": {}, "action": {}}))

    assert legacy_shape_meta(tmp_path) == {
        "obs": {
            "camera": {
                "shape": [32, 48, 3],
                "type": "rgb",
                "channel_order": "rgb",
            },
            "state": {"shape": [4], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }


def _rtc_policy_checks(policy):
    """Shared RTC contract: returns (B, P + n_action_steps, act_dim), the
    first P actions repeat the prefix exactly (inpainting clamps them clean
    at the end -- only the normalize/inverse roundtrip in between), and an
    over-long prefix that can't fit in the trained horizon raises."""
    batch = {"state": torch.randn(3, 2, 3)}
    prefix = torch.randn(3, 4, 2)
    out = policy.predict_action_rtc(batch, prefix)
    assert out.shape == (3, 4 + 2, 2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :4], prefix, atol=1e-5, rtol=1e-5)

    # obs offset (1) + prefix (6) + n_action_steps (2) > horizon (8)
    with pytest.raises(ValueError, match="too long"):
        policy.predict_action_rtc(batch, torch.randn(3, 6, 2))


def test_diffusion_policy_rtc(tiny_store):
    policy = DiffusionPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[],
        train_timesteps=20, inference_steps=5, sampler="ddim",
    )
    for sampler in ("ddim", "ddpm"):
        policy.cfg.sampler = sampler
        _rtc_policy_checks(policy)


def test_flow_policy_rtc(tiny_store):
    policy = FlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[], inference_steps=5,
    )
    _rtc_policy_checks(policy)


def _trained_rtc_checks(policy):
    """Train-time RTC contract: loss trains (finite, grads flow to the
    denoiser), plain predict_action works (all-zero mask / plain t), rtc
    prefix comes back exactly, and a prefix beyond the trained
    max_action_prefix raises."""
    batch = {
        "state": torch.randn(4, 2, 3),
        "action": torch.randn(4, 8, 2),
    }
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    net = policy.unet if hasattr(policy, "unet") else policy.net
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    policy.zero_grad()

    policy.eval()
    obs = {"state": torch.randn(3, 2, 3)}
    action = policy.predict_action(obs)
    assert action.shape == (3, 2, 2)
    assert torch.isfinite(action).all()

    prefix = torch.randn(3, 3, 2)
    out = policy.predict_action_rtc(obs, prefix)
    assert out.shape == (3, 3 + 2, 2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :3], prefix, atol=1e-5, rtol=1e-5)

    # P=4 > trained max_action_prefix=3, even though it fits the horizon
    with pytest.raises(ValueError, match="max_action_prefix"):
        policy.predict_action_rtc(obs, torch.randn(3, 4, 2))


def test_diffusion_policy_trained_rtc(tiny_store):
    policy = DiffusionPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[],
        train_timesteps=20, inference_steps=5, sampler="ddim",
        max_action_prefix=3,
    )
    # mask channel: unet input is act_dim + 1
    assert policy.unet.final[-1].out_channels == 2
    _trained_rtc_checks(policy)


def test_flow_policy_trained_rtc(tiny_store):
    policy = FlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[], inference_steps=5,
        max_action_prefix=3,
    )
    _trained_rtc_checks(policy)


def test_dit_flow_policy_trained_rtc(tiny_store):
    policy = DiTFlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        hidden_size=32, depth=2, num_heads=4, num_kp=4, data_transforms=[],
        inference_steps=5, max_action_prefix=3,
    )
    _trained_rtc_checks(policy)


def test_dit_flow_policy_without_prefix_training(tiny_store):
    # max_action_prefix=0: plain flow matching + inpainting-fallback rtc
    policy = DiTFlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        hidden_size=32, depth=2, num_heads=4, num_kp=4, data_transforms=[],
        inference_steps=5, max_action_prefix=0,
    )
    batch = {"state": torch.randn(4, 2, 3), "action": torch.randn(4, 8, 2)}
    assert torch.isfinite(policy.compute_loss(batch))
    policy.eval()
    obs = {"state": torch.randn(3, 2, 3)}
    prefix = torch.randn(3, 4, 2)
    out = policy.predict_action_rtc(obs, prefix)
    assert out.shape == (3, 4 + 2, 2)
    torch.testing.assert_close(out[:, :4], prefix, atol=1e-5, rtol=1e-5)


def test_rtc_matches_plain_predict_shapes(tiny_store):
    # plain predict_action is untouched by the _sample refactor
    policy = DiffusionPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, data_transforms=[],
        train_timesteps=20, inference_steps=5, sampler="ddim",
    )
    action = policy.predict_action({"state": torch.randn(3, 2, 3)})
    assert action.shape == (3, 2, 2)
    assert torch.isfinite(action).all()


class _TinyRandomLossPolicy(nn.Module):
    """Small trainer fixture whose loss consumes Torch RNG like flow matching."""

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(obs_horizon=2, horizon=4)
        self.net = nn.Linear(2 * 3, 4 * 2)
        self.data_transforms_pipeline = None

    def _loss_sum_weight(self, batch):
        prediction = self.net(batch["state"].flatten(1)).reshape(-1, 4, 2)
        target = batch["action"] + 0.1 * torch.randn_like(batch["action"])
        err = (prediction - target).square()
        return err.sum(), err.numel()

    def compute_loss(self, batch):
        total, weight = self._loss_sum_weight(batch)
        return total / weight

    def loss(self, batch):
        return self.compute_loss(batch)

    def loss_terms(self, batch):
        return self._loss_sum_weight(batch)


def _tiny_train_cfg(out, **overrides):
    values = {
        "out": out,
        "steps": 6,
        "batch_size": 4,
        "lr": 1e-3,
        "warmup_steps": 1,
        "seed": 123,
        "val_seed": 456,
        "val_episodes": 1,
        "val_every": 2,
        "log_every": 100,
        "save_every": 3,
        "resume": "auto",
        "compile": False,
        "bf16": False,
        "device": "cpu",
        "profile": False,
    }
    values.update(overrides)
    return TrainConfig(**values)


def _train_tiny(tiny_store, cfg, run_config=None):
    seed_everything(cfg.seed)
    policy = _TinyRandomLossPolicy()
    path = train(
        cfg,
        policy,
        tiny_store,
        {"_target_": "tests._TinyRandomLossPolicy"},
        {"name": "tiny"},
        {"obs": {"state": {"shape": [3]}}, "action": {"shape": [2]}},
        {"kind": "uniform", "weights_path": None, "replacement": False},
        run_config=run_config,
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def test_full_run_config_is_recorded(tiny_store, tmp_path):
    # The entire Hydra config is logged verbatim, not a reconstructed subset:
    # an arbitrary top-level knob (`stats_workers`) and every group survive.
    out = tmp_path / "run"
    cfg = _tiny_train_cfg(out)
    run_config = {
        "data": {"name": "tiny", "episodes": [0, 1, 2]},
        "target": {"obs": {"state": {"shape": [3]}}, "action": {"shape": [2]}},
        "policy": {"_target_": "tests._TinyRandomLossPolicy"},
        "training": {"steps": 6, "seed": 123},
        "sampling": {"kind": "uniform"},
        "out": str(out),
        "stats_workers": 7,
    }
    checkpoint = _train_tiny(tiny_store, cfg, run_config=run_config)
    saved = json.loads((out / "config.json").read_text())
    # Every launched-with key is present, including ones train() never sees
    # directly.
    assert saved["stats_workers"] == 7
    assert saved["training"] == {"steps": 6, "seed": 123}
    assert saved["data"] == {"name": "tiny", "episodes": [0, 1, 2]}
    # Inference contract keys (checkpoint self-containment) survive the merge.
    assert saved["policy"] == {"_target_": "tests._TinyRandomLossPolicy"}
    assert saved["target"] == {"obs": {"state": {"shape": [3]}}, "action": {"shape": [2]}}
    assert "env_meta" in saved
    assert "resolved_training" in saved
    # config.json and the checkpoint carry the same config.
    assert checkpoint["config"] == saved


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_seed_everything_controls_policy_initialization():
    seed_everything(9)
    first = _TinyRandomLossPolicy()
    seed_everything(9)
    second = _TinyRandomLossPolicy()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cpu_and_resident_batch_sequences_resume_exactly(tiny_store):
    windows = Windows(2, 4)
    resident = GpuSequenceDataset(tiny_store, windows, torch.device("cpu"))
    cpu = CpuSequenceDataset(tiny_store, windows, torch.device("cpu"), num_workers=0)

    resident_batches = resident.batches(4, seed=7)
    cpu_batches = cpu.batches(4, seed=7)
    expected = []
    for _ in range(5):
        resident_batch = next(resident_batches)
        cpu_batch = next(cpu_batches)
        torch.testing.assert_close(resident_batch["state"], cpu_batch["state"], rtol=0, atol=0)
        torch.testing.assert_close(resident_batch["action"], cpu_batch["action"], rtol=0, atol=0)
        expected.append(resident_batch)

    resumed_resident = next(resident.batches(4, seed=7, start_batch=3))
    resumed_cpu = next(cpu.batches(4, seed=7, start_batch=3))
    torch.testing.assert_close(resumed_resident["state"], expected[3]["state"], rtol=0, atol=0)
    torch.testing.assert_close(resumed_cpu["state"], expected[3]["state"], rtol=0, atol=0)


def test_tiny_or_invalid_splits_fail_instead_of_hanging(tiny_store):
    with pytest.raises(ValueError, match="val_episodes"):
        GpuSequenceDataset(
            tiny_store,
            Windows(2, 4),
            torch.device("cpu"),
            val_episodes=tiny_store.meta.num_episodes,
        )
    ds = GpuSequenceDataset(tiny_store, Windows(2, 4), torch.device("cpu"))
    with pytest.raises(ValueError, match="needs at least"):
        next(ds.batches(ds.num_samples + 1, seed=0))


def test_validation_frequency_does_not_change_training_rng(tiny_store, tmp_path):
    with_validation = _train_tiny(
        tiny_store, _tiny_train_cfg(tmp_path / "with_validation", val_every=2)
    )
    without_validation = _train_tiny(
        tiny_store, _tiny_train_cfg(tmp_path / "without_validation", val_every=1000)
    )
    _assert_nested_equal(with_validation["model"], without_validation["model"])
    _assert_nested_equal(with_validation["ema"], without_validation["ema"])
    _assert_nested_equal(with_validation["optimizer"], without_validation["optimizer"])
    _assert_nested_equal(with_validation["rng"], without_validation["rng"])


def test_interrupted_resume_is_bit_exact(tiny_store, tmp_path, monkeypatch):
    import rgrl.train_loop as train_loop

    uninterrupted = _train_tiny(tiny_store, _tiny_train_cfg(tmp_path / "uninterrupted"))

    original_save = train_loop._atomic_save_checkpoint

    class SimulatedCrash(RuntimeError):
        pass

    def save_then_crash(out, step, payload):
        path = original_save(out, step, payload)
        if step == 3:
            raise SimulatedCrash("injected immediately after durable checkpoint")
        return path

    monkeypatch.setattr(train_loop, "_atomic_save_checkpoint", save_then_crash)
    interrupted_cfg = _tiny_train_cfg(tmp_path / "interrupted")
    with pytest.raises(SimulatedCrash):
        _train_tiny(tiny_store, interrupted_cfg)
    monkeypatch.setattr(train_loop, "_atomic_save_checkpoint", original_save)

    resumed = _train_tiny(tiny_store, interrupted_cfg)
    for key in ("model", "ema", "optimizer", "scheduler", "rng"):
        _assert_nested_equal(uninterrupted[key], resumed[key])
    assert resumed["step"] == uninterrupted["step"] == 6
    assert (tmp_path / "interrupted" / "latest.pt").is_symlink()
    assert not list((tmp_path / "interrupted").glob(".*.tmp"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_resume_is_bit_exact(tiny_store, tmp_path, monkeypatch):
    # Regression: the checkpoint is loaded with map_location=device, which moves
    # the saved RNG ByteTensors onto the GPU. torch.{,cuda.}set_rng_state require
    # CPU ByteTensors, so a CUDA resume used to die in _restore_rng_state with
    # "RNG state must be a torch.ByteTensor". This exercises the map_location=cuda
    # path the CPU-device resume tests never hit -- and, like its CPU sibling,
    # asserts the resumed run is *bit-identical* to an uninterrupted one, so a
    # coerced-but-wrong RNG restore (e.g. a zeroed/shallow-copied state that still
    # reaches step 6) is caught rather than passing silently.
    import rgrl.train_loop as train_loop

    uninterrupted = _train_tiny(tiny_store, _tiny_train_cfg(tmp_path / "uninterrupted", device="cuda"))

    original_save = train_loop._atomic_save_checkpoint

    class SimulatedCrash(RuntimeError):
        pass

    def save_then_crash(out, step, payload):
        path = original_save(out, step, payload)
        if step == 3:
            raise SimulatedCrash("injected immediately after durable checkpoint")
        return path

    monkeypatch.setattr(train_loop, "_atomic_save_checkpoint", save_then_crash)
    interrupted_cfg = _tiny_train_cfg(tmp_path / "interrupted", device="cuda")
    with pytest.raises(SimulatedCrash):
        _train_tiny(tiny_store, interrupted_cfg)
    monkeypatch.setattr(train_loop, "_atomic_save_checkpoint", original_save)

    resumed = _train_tiny(tiny_store, interrupted_cfg)
    for key in ("model", "ema", "optimizer", "scheduler", "rng"):
        _assert_nested_equal(uninterrupted[key], resumed[key])
    assert resumed["step"] == uninterrupted["step"] == 6


def test_completed_and_incompatible_resumes_are_guarded(tiny_store, tmp_path):
    out = tmp_path / "guarded"
    cfg = _tiny_train_cfg(out, steps=2, save_every=2)
    completed = _train_tiny(tiny_store, cfg)
    latest = out / "latest.pt"
    checkpoint = latest.resolve()
    before = (latest.lstat().st_mtime_ns, checkpoint.stat().st_mtime_ns)

    completed_again = _train_tiny(tiny_store, cfg)
    assert completed_again["step"] == completed["step"] == 2
    assert before == (latest.lstat().st_mtime_ns, checkpoint.stat().st_mtime_ns)

    with pytest.raises(RuntimeError, match="beyond requested target"):
        _train_tiny(tiny_store, _tiny_train_cfg(out, steps=1))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _train_tiny(tiny_store, _tiny_train_cfg(out, steps=2, lr=2e-3))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _train_tiny(tiny_store, _tiny_train_cfg(out, steps=2, warmup_steps=0))


def test_write_sidecars_streaming_and_parallel_match_full_array(tmp_path):
    # Fitting must be identical whether stats come in as one array or as
    # per-episode chunks, and independent of the worker count.
    import json

    rng = np.random.default_rng(0)
    chunks = {
        "state": [rng.normal(size=(7, 3)).astype(np.float32) for _ in range(5)],
        "action": [rng.normal(size=(7, 2)).astype(np.float32) for _ in range(5)],
    }
    full = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    def run(name, stats_keys, **kw):
        d = tmp_path / name
        d.mkdir()
        # write_sidecars consumes the mapping, so hand it a fresh copy
        payload = {
            k: (v.copy() if isinstance(v, np.ndarray) else [c.copy() for c in v])
            for k, v in stats_keys.items()
        }
        write_sidecars(d, stats_keys=payload, source="t", **kw)
        return json.loads((d / "stats.json").read_text())

    ref = run("full", full)
    assert run("chunked", chunks, num_workers=1) == ref
    assert run("parallel", chunks, num_workers=4) == ref


@pytest.mark.parametrize("method", ["minmax", "zscore", "minmax_p99"])
def test_write_sidecars_norm_method_is_persisted(tmp_path, method):
    import json

    rng = np.random.default_rng(0)
    d = tmp_path / method
    d.mkdir()
    write_sidecars(
        d,
        stats_keys={"action": rng.normal(size=(64, 2)).astype(np.float32)},
        source="t",
        norm_method=method,
    )
    stats = json.loads((d / "stats.json").read_text())["action"]
    assert stats["method"] == method
    assert len(stats["scale"]) == 2 and len(stats["offset"]) == 2


def test_norm_methods_produce_different_fits(tmp_path):
    import json

    rng = np.random.default_rng(1)
    arr = rng.normal(size=(200, 2)).astype(np.float32)
    fits = {}
    for method in ("minmax", "zscore", "minmax_p99"):
        d = tmp_path / method
        d.mkdir()
        write_sidecars(d, stats_keys={"action": arr.copy()}, source="t", norm_method=method)
        fits[method] = json.loads((d / "stats.json").read_text())["action"]["scale"]
    assert fits["minmax"] != fits["zscore"] != fits["minmax_p99"]


def test_write_sidecars_rejects_unknown_norm_method(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    with pytest.raises(ValueError, match="norm_method"):
        write_sidecars(d, stats_keys={"action": np.zeros((3, 2), np.float32)}, norm_method="bogus")
