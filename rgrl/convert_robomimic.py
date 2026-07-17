"""Convert a robomimic HDF5 dataset into an rgrl store. Runs once.

Storage/conversion is robo_dataset's job (RobomimicHDF5Store -> ZarrStoreWriter
via its streaming convert()); this script only adds rgrl's sidecars (kind
classification, robomimic env_meta, normalization stats — see rgrl/store.py).

Obs keys are auto-detected: uint8 3-dim per-frame arrays are images, the rest
are low-dim state. Pass --obs-keys to select a subset explicitly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import tyro
from robo_dataset.backends.hdf5 import RobomimicHDF5Store
from robo_dataset.backends.zarr_backend import ZarrStoreWriter
from robo_dataset.convert import convert

from rgrl.store import write_sidecars

log = logging.getLogger(__name__)


def convert_dataset(
    hdf5_path: Path,
    out: Path,
    obs_keys: list[str] | None = None,
    abs_action: bool = False,
) -> None:
    """Convert robomimic demos to an rgrl store.

    Args:
        hdf5_path: robomimic dataset (e.g. lift/ph/image.hdf5)
        out: output store directory
        obs_keys: obs subkeys to keep (default: all under obs/)
        abs_action: record in meta that actions are absolute (informational)
    """
    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        env_meta = json.loads(data.attrs["env_args"]) if "env_args" in data.attrs else {}
        obs_keys = obs_keys if obs_keys is not None else sorted(data["demo_0"]["obs"].keys())

    src = RobomimicHDF5Store(str(hdf5_path), obs_keys=obs_keys)
    image_keys = [k for k in obs_keys if src.frame_dtype(k) == np.uint8 and len(src.frame_shape(k)) == 3]
    log.info(f"{src.n_episodes} episodes; image keys: {image_keys}")

    # robomimic hdf5 stores lowdim obs/actions as float64; the model's buffers
    # are float32, so cast at write time rather than carry float64 through the
    # store (avoids a dtype mismatch at every forward pass).
    def _cast_f32(ep: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: (v if k in image_keys else v.astype(np.float32)) for k, v in ep.items()}

    dst = ZarrStoreWriter.create(
        str(out),
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: (src.frame_dtype(k) if k in image_keys else np.dtype(np.float32)) for k in src.keys},
        image_keys=image_keys,
    )
    convert(src, dst, transform=_cast_f32, verbose=True)

    stats_keys = {k: src.get_sequence(k, 0, src.n_frames) for k in src.keys if k not in image_keys}
    env_meta["abs_action"] = abs_action
    write_sidecars(out, image_keys=image_keys, stats_keys=stats_keys, env_meta=env_meta, source=str(hdf5_path), fps=20.0)
    log.info(f"wrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    tyro.cli(convert_dataset)
