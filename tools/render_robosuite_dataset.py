#!/usr/bin/env python3
"""Render camera observations into a Robomimic-compatible HDF5 dataset.

This is intentionally implemented against modern Robosuite and MuJoCo
directly. It does not import Robomimic or the retired mujoco_py backend.

Example (ToolHang PH, two cameras, 84x84):

    uv run --extra eval python tools/render_robosuite_dataset.py \
      data/robomimic/tool_hang/ph/demo_v141.hdf5 \
      data/robomimic/tool_hang/ph/image_84.hdf5 \
      --camera-names sideview robot0_eye_in_hand --height 84 --width 84
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from rgrl.inference.envs.robomimic import _env_kwargs, _frame, _make_env

_DEFAULT_STATE_KEYS = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]


def _episode_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**63 - 1, name


def _decode_name(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _set_episode_model(env, model_xml: str) -> None:
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(model_xml))
    env.sim.reset()


def _observe_state(env, state: np.ndarray) -> dict:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return env._get_observations(force_update=True)


def _copy_filtered_masks(source: h5py.File, output: h5py.File, demos: list[str]) -> None:
    if "mask" not in source:
        return
    allowed = set(demos)
    target = output.create_group("mask")
    for key, dataset in source["mask"].items():
        selected = [_decode_name(name) for name in dataset[()] if _decode_name(name) in allowed]
        target.create_dataset(key, data=np.asarray(selected, dtype="S"))


def render_dataset(
    source_path: Path,
    output_path: Path,
    camera_names: list[str],
    image_hw: tuple[int, int],
    state_keys: list[str],
    n_episodes: int | None,
    compression: str | None,
    overwrite: bool,
) -> None:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists (pass --overwrite to replace it)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")

    image_keys = [f"{name}_image" for name in camera_names]
    env = None
    try:
        with h5py.File(source_path, "r") as source:
            env_meta = json.loads(source["data"].attrs["env_args"])
            demos = sorted(source["data"].keys(), key=_episode_sort_key)
            if n_episodes is not None:
                demos = demos[:n_episodes]
            if not demos:
                raise ValueError("no episodes selected")

            env = _make_env(env_meta, image_keys, image_hw, abs_action=False)
            # Dataset playback reloads each demonstration's recorded model
            # XML. Use a hard reset here so task-side object references are
            # rebuilt from that model; rollout workers disable hard resets
            # only after constructing their one fixed evaluation model.
            env.hard_reset = True
            output_meta = copy.deepcopy(env_meta)
            output_meta["env_version"] = __import__("robosuite").__version__
            output_meta["env_kwargs"] = _env_kwargs(
                env_meta,
                image_keys,
                image_hw,
                abs_action=False,
            )
            output_meta["env_kwargs"].pop("render_gpu_device_id", None)

            with h5py.File(tmp_path, "w") as output:
                output_data = output.create_group("data")
                total = 0
                for demo in tqdm(demos, desc="Rendering", unit="episode"):
                    source_demo = source["data"][demo]
                    states = np.asarray(source_demo["states"])
                    model_xml = _decode_name(source_demo.attrs["model_file"])
                    _set_episode_model(env, model_xml)

                    observations = {key: [] for key in [*state_keys, *image_keys]}
                    for state in states:
                        raw = _observe_state(env, state)
                        frame = _frame(raw, image_keys, state_keys)
                        for key, value in frame.items():
                            observations[key].append(value)

                    target_demo = output_data.create_group(demo)
                    target_demo.create_dataset("actions", data=source_demo["actions"][()])
                    target_demo.create_dataset("states", data=states)
                    for key in ("rewards", "dones"):
                        if key in source_demo:
                            target_demo.create_dataset(key, data=source_demo[key][()])
                    obs_group = target_demo.create_group("obs")
                    for key, values in observations.items():
                        obs_group.create_dataset(
                            key,
                            data=np.stack(values),
                            compression=compression,
                        )
                    for key, value in source_demo.attrs.items():
                        target_demo.attrs[key] = value
                    target_demo.attrs["num_samples"] = len(states)
                    total += len(states)

                _copy_filtered_masks(source, output, demos)
                output_data.attrs["total"] = total
                output_data.attrs["env_args"] = json.dumps(output_meta, indent=2)
                output.flush()

        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(f"rendered {len(demos)} episodes / {total} frames -> {output_path}")
    finally:
        if env is not None:
            env.close()
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="raw Robosuite HDF5")
    parser.add_argument("output", type=Path, help="output HDF5")
    parser.add_argument("--camera-names", nargs="+", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--state-keys", nargs="+", default=_DEFAULT_STATE_KEYS)
    parser.add_argument("--n-episodes", type=int)
    parser.add_argument("--compression", choices=["none", "lzf", "gzip"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.n_episodes is not None and args.n_episodes <= 0:
        parser.error("--n-episodes must be positive")

    render_dataset(
        source_path=args.source,
        output_path=args.output,
        camera_names=args.camera_names,
        image_hw=(args.height, args.width),
        state_keys=args.state_keys,
        n_episodes=args.n_episodes,
        compression=None if args.compression == "none" else args.compression,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
