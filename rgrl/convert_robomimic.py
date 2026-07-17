"""Convert a robomimic HDF5 dataset into an EpisodeStore. Runs once.

Obs keys are auto-detected: uint8 3-dim per-frame arrays are images, the rest
are low-dim state. Pass --obs-keys to select a subset explicitly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import tyro

from rgrl.store import KIND_ACTION, KIND_IMAGE, KIND_STATE, StoreWriter


def convert(
    hdf5_path: Path,
    out: Path,
    obs_keys: list[str] | None = None,
    abs_action: bool = False,
) -> None:
    """Convert robomimic demos to a flat-array store.

    Args:
        hdf5_path: robomimic dataset (e.g. lift/ph/image.hdf5)
        out: output store directory
        obs_keys: obs subkeys to keep (default: all under obs/)
        abs_action: record in meta that actions are absolute (informational)
    """
    t0 = time.time()
    writer = StoreWriter(out)
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_meta = json.loads(data.attrs["env_args"]) if "env_args" in data.attrs else {}
        # demo_0, demo_1, ... sorted numerically
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        first = data[demos[0]]["obs"]
        keys = obs_keys if obs_keys is not None else sorted(first.keys())
        kinds = {}
        for k in keys:
            d = first[k]
            kinds[k] = KIND_IMAGE if (d.dtype == np.uint8 and d.ndim == 4) else KIND_STATE
        kinds["action"] = KIND_ACTION
        print(f"{len(demos)} episodes; obs keys: { {k: kinds[k] for k in keys} }")

        for i, name in enumerate(demos):
            demo = data[name]
            arrays = {k: demo["obs"][k][:] for k in keys}
            arrays["action"] = demo["actions"][:].astype(np.float32)
            writer.add_episode(arrays, kinds)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(demos)} episodes")

    env_meta["abs_action"] = abs_action
    writer.finalize(env_meta=env_meta, source=str(hdf5_path), fps=20.0)
    print(f"wrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    tyro.cli(convert)
