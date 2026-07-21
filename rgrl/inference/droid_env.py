"""Real-world DROID env behind the generic Env contract (rgrl/inference/runner.py).

Wraps `droid.robot_env.RobotEnv` with:
  - observation preprocessing (BGR/BGRA -> RGB, matching the DROID training
    data convention -- robo_dataset.backends.droid.DroidStore's MP4 decode
    path does the identical conversion)
  - `dry_run` mode for smoke-testing the control loop without hardware

Obs key names match robo_dataset's DroidStore / real trained checkpoints
exactly: joint_positions, cartesian_position, gripper_position,
wrist_image_left, wrist_image_right, exterior_image_1_left,
exterior_image_2_left (cartesian_position is xyz + euler, not a quaternion --
per DroidStore's VectorSchema). Naming this way from the start means, unlike
the diffusion_policy reference this was ported from, no key-alias table is
needed: the env already emits the names a checkpoint's shape_meta expects.

num_envs is always 1 -- one physical robot. The `droid` package (Franka/ZED
driver stack) is imported lazily so this module is importable anywhere;
hardware code paths raise a clear error when it's absent. Dropped from the
reference: save_droid_trajectory (DROID-format trajectory + mp4
persistence) -- that's data collection, not inference.
"""

from __future__ import annotations

import cv2
import numpy as np

_DROID_HINT = (
    "requires the DROID robot stack ('droid' package + ZED/Franka drivers), "
    "present on the robot workstation"
)


class DroidEnv:
    """Env protocol implementation for one physical DROID robot.

    action_space: 'joint_velocity', 'cartesian_velocity', or
    'joint_position' (a velocity wrapper: the absolute target is converted to
    joint_velocity via (target - current_pos) * control_hz each step).
    max_joint_velocity, if set, scales the whole 7-vector down (preserving
    direction) rather than clipping components independently, so Cartesian
    motion direction is unchanged, just slower.
    """

    # Camera serials matching the lab rig (from droid/misc/parameters.py in the reference).
    DEFAULT_WRIST_SERIAL = "14313307"
    DEFAULT_EXT1_SERIAL = "36716034"
    DEFAULT_EXT2_SERIAL = "37617599"

    num_envs = 1

    def __init__(
        self,
        action_space: str = "joint_velocity",
        control_hz: float = 15.0,
        position_gain: float = 1.0,
        max_joint_velocity: float | None = None,
        wrist_serial: str = DEFAULT_WRIST_SERIAL,
        ext1_serial: str = DEFAULT_EXT1_SERIAL,
        ext2_serial: str = DEFAULT_EXT2_SERIAL,
        dry_run: bool = False,
    ) -> None:
        if action_space not in ("joint_velocity", "cartesian_velocity", "joint_position"):
            raise ValueError(f"Unknown action_space: {action_space!r}")

        self._action_space_name = action_space
        self._control_hz = control_hz
        self._position_gain = position_gain
        self._max_joint_velocity = max_joint_velocity
        self.wrist_serial = wrist_serial
        self.ext1_serial = ext1_serial
        self.ext2_serial = ext2_serial
        self.dry_run = dry_run

        # 7-DOF Franka: cartesian = 6+1=7, joint = 7+1=8.
        self.action_dim = 7 if "cartesian" in action_space else 8

        self._robot_env = None
        self._last_obs: dict[str, np.ndarray] | None = None

        if not dry_run:
            self._init_robot()

    @property
    def _robot_action_space(self) -> str:
        # joint_position is implemented as a velocity wrapper on top of joint_velocity
        return "joint_velocity" if self._action_space_name == "joint_position" else self._action_space_name

    def _init_robot(self) -> None:
        try:
            import droid.robot_env as robot_env_mod
        except ImportError as e:
            raise ImportError(f"importing 'droid.robot_env' failed -- this {_DROID_HINT}") from e
        camera_kwargs = {
            self.wrist_serial: {"image": True},
            self.ext1_serial: {"image": True},
            self.ext2_serial: {"image": True},
        }
        self._robot_env = robot_env_mod.RobotEnv(
            action_space=self._robot_action_space, camera_kwargs=camera_kwargs, do_reset=True,
        )

    # ------------------------------------------------------------------
    # Env protocol
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            self._robot_env.reset()
            obs = self._get_obs()
        self._last_obs = obs
        return {k: v[None] for k, v in obs.items()}

    def step(
        self, actions: np.ndarray, active: np.ndarray | None = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        active = np.ones(1, dtype=bool) if active is None else active
        if not active[0]:
            frame = {k: v[None] for k, v in self._last_obs.items()}
            return frame, np.zeros(1, np.float32), np.zeros(1, dtype=bool), {}

        action = actions[0]
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            robot_action = self._to_robot_action(action, self._last_obs or {})
            self._robot_env.update_robot(
                robot_action,
                action_space=self._robot_action_space,
                gripper_action_space="position",
                blocking=False,
            )
            obs = self._get_obs()
        self._last_obs = obs
        frame = {k: v[None] for k, v in obs.items()}
        return frame, np.zeros(1, np.float32), np.zeros(1, dtype=bool), {}

    def close(self) -> None:
        pass  # ZED cameras / RobotEnv have no explicit close in current droid versions

    # ------------------------------------------------------------------
    # Action space conversion
    # ------------------------------------------------------------------

    def _to_robot_action(self, action: np.ndarray, current_obs: dict) -> np.ndarray:
        """Convert a policy action to the robot's native action space."""
        if self._action_space_name == "cartesian_velocity":
            return action
        if self._action_space_name != "joint_position":
            return self._clip_joint_velocity(action)
        current_joints = current_obs.get("joint_positions", np.zeros(7))
        velocity = (action[:7] - current_joints) * self._control_hz * self._position_gain
        velocity = self._clip_joint_velocity(velocity)
        return np.concatenate([velocity, action[7:8]], axis=0)

    def _clip_joint_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Scale the 7 joint-velocity components (not the trailing gripper
        one) down so the largest magnitude doesn't exceed max_joint_velocity,
        preserving their direction — clipping each independently would
        distort the Cartesian direction of motion via the Jacobian."""
        if self._max_joint_velocity is None:
            return velocity
        velocity = np.array(velocity, dtype=np.float32, copy=True)
        peak = np.abs(velocity[:7]).max()
        if peak > self._max_joint_velocity:
            velocity[:7] *= self._max_joint_velocity / peak
        return velocity

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        raw = self._robot_env.get_observation()
        return self._extract_observation(raw)

    def _extract_observation(self, raw: dict) -> dict[str, np.ndarray]:
        image_obs = raw.get("image", {})
        robot_state = raw.get("robot_state", {})
        obs: dict[str, np.ndarray] = {
            "joint_positions": np.array(robot_state.get("joint_positions", np.zeros(7)), dtype=np.float32),
            "gripper_position": np.atleast_1d(
                np.array(robot_state.get("gripper_position", 0.0), dtype=np.float32)
            ),
            "cartesian_position": np.array(robot_state.get("cartesian_position", np.zeros(6)), dtype=np.float32),
        }
        for serial, key, view in (
            (self.wrist_serial, "wrist_image_left", "left"),
            (self.wrist_serial, "wrist_image_right", "right"),
            (self.ext1_serial, "exterior_image_1_left", "left"),
            (self.ext2_serial, "exterior_image_2_left", "left"),
        ):
            img = image_obs.get(f"{serial}_{view}")
            if img is not None:
                obs[key] = self._preprocess_image(img)
        return obs

    @staticmethod
    def _preprocess_image(img: np.ndarray) -> np.ndarray:
        # Only convert colour space — resizing is a policy-server-side concern.
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1].copy()  # BGR -> RGB
        return img

    def _dummy_obs(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng()
        return {
            "joint_positions": rng.random(7).astype(np.float32),
            "gripper_position": rng.random(1).astype(np.float32),
            "cartesian_position": rng.random(6).astype(np.float32),
            "wrist_image_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "wrist_image_right": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_1_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_2_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        }
