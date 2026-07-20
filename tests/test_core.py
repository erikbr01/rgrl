"""Fast CPU sanity tests: window indexing, normalization, crops, UNet shapes."""

import numpy as np
import pytest
import torch

from robo_dataset.backends.zarr_backend import ZarrStoreWriter
from robo_dataset.normalization import fit_key
from robo_dataset.transforms.gpu_batch import CenterCrop, RandomCrop

from rgrl.dataset import GpuSequenceDataset, Windows
from rgrl.nets.unet1d import ConditionalUnet1D
from rgrl.normalize import Normalizer
from rgrl.policies.diffusion import DiffusionPolicy
from rgrl.policies.dit_flow import DiTFlowPolicy
from rgrl.policies.flow import FlowPolicy
from rgrl.store import EpisodeStore, write_sidecars


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
        down_dims=(8, 16), num_kp=4, image_hw=None,
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
        down_dims=(8, 16), num_kp=4, image_hw=None,
        train_timesteps=20, inference_steps=5, sampler="ddim",
    )
    for sampler in ("ddim", "ddpm"):
        policy.cfg.sampler = sampler
        _rtc_policy_checks(policy)


def test_flow_policy_rtc(tiny_store):
    policy = FlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, image_hw=None, inference_steps=5,
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
        down_dims=(8, 16), num_kp=4, image_hw=None,
        train_timesteps=20, inference_steps=5, sampler="ddim",
        max_action_prefix=3,
    )
    # mask channel: unet input is act_dim + 1
    assert policy.unet.final[-1].out_channels == 2
    _trained_rtc_checks(policy)


def test_flow_policy_trained_rtc(tiny_store):
    policy = FlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        down_dims=(8, 16), num_kp=4, image_hw=None, inference_steps=5,
        max_action_prefix=3,
    )
    _trained_rtc_checks(policy)


def test_dit_flow_policy_trained_rtc(tiny_store):
    policy = DiTFlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        hidden_size=32, depth=2, num_heads=4, num_kp=4, image_hw=None,
        inference_steps=5, max_action_prefix=3,
    )
    _trained_rtc_checks(policy)


def test_dit_flow_policy_without_prefix_training(tiny_store):
    # max_action_prefix=0: plain flow matching + inpainting-fallback rtc
    policy = DiTFlowPolicy(
        tiny_store, obs_horizon=2, horizon=8, n_action_steps=2,
        hidden_size=32, depth=2, num_heads=4, num_kp=4, image_hw=None,
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
        down_dims=(8, 16), num_kp=4, image_hw=None,
        train_timesteps=20, inference_steps=5, sampler="ddim",
    )
    action = policy.predict_action({"state": torch.randn(3, 2, 3)})
    assert action.shape == (3, 2, 2)
    assert torch.isfinite(action).all()
