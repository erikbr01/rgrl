"""Fast CPU sanity tests: window indexing, normalization, crops, UNet shapes."""

import numpy as np
import pytest
import torch

from rgrl.dataset import GpuSequenceDataset, Windows
from rgrl.nets.encoders import center_crop, random_crop
from rgrl.nets.unet1d import ConditionalUnet1D
from rgrl.normalize import MinMaxNormalizer
from rgrl.store import EpisodeStore, StoreWriter


@pytest.fixture
def tiny_store(tmp_path):
    w = StoreWriter(tmp_path / "store")
    rng = np.random.default_rng(0)
    for length in (7, 5, 9):
        w.add_episode(
            {
                "state": rng.normal(size=(length, 3)).astype(np.float32),
                "action": rng.normal(size=(length, 2)).astype(np.float32),
            },
            kinds={"state": "state", "action": "action"},
        )
    w.finalize(source="test")
    return EpisodeStore(tmp_path / "store")


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


def test_minmax_normalizer_roundtrip():
    stats = {"min": [-2.0, 0.0, 5.0], "max": [2.0, 1.0, 5.0]}  # last dim constant
    n = MinMaxNormalizer(stats, 3)
    x = torch.tensor([[1.0, 0.25, 5.0], [-2.0, 1.0, 5.0]])
    y = n(x)
    assert y.min() >= -1 and y.max() <= 1
    assert torch.allclose(y[:, 2], torch.zeros(2))  # constant dim -> 0
    assert torch.allclose(n.inverse(y), x, atol=1e-6)


def test_crops():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    out = random_crop(x, (5, 5), generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 3, 5, 5)
    # crop content must be a contiguous window of the source
    src_rows = x[0, 0]
    found = any(
        torch.equal(out[0, 0], src_rows[i : i + 5, j : j + 5]) for i in range(4) for j in range(4)
    )
    assert found
    assert center_crop(x, (8, 8)).shape == (2, 3, 8, 8)


def test_unet_shapes():
    net = ConditionalUnet1D(input_dim=7, global_cond_dim=20, down_dims=(32, 64))
    x = torch.randn(4, 16, 7)
    out = net(x, torch.randint(0, 100, (4,)), torch.randn(4, 20))
    assert out.shape == x.shape
