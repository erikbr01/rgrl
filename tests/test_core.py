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
