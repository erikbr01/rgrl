"""Fast, hardware-free tests for rgrl/inference: wire format, obs windowing,
and Runner's batch/done-mask/chunking mechanics against a mock Env + PolicyBase
(no robosuite/mujoco/droid needed — those are exercised live, not in CI)."""

import numpy as np
import pytest
import torch
from torch import nn

from rgrl.inference.activations import ActivationCapture
from rgrl.inference.obs_window import RollingObsWindow
from rgrl.inference.policy_client import PolicyBase
from rgrl.inference.runner import InferenceMode, Runner
from rgrl.inference.wire import decode_batch, encode_batch


def test_wire_roundtrip():
    arrays = {
        "state": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "image": np.arange(2 * 2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 2, 4, 4, 3),
    }
    decoded = decode_batch(encode_batch(arrays))
    assert decoded.keys() == arrays.keys()
    for k in arrays:
        np.testing.assert_array_equal(decoded[k], arrays[k])
        assert decoded[k].dtype == arrays[k].dtype


def test_rolling_obs_window_front_fills_and_stacks():
    w = RollingObsWindow(obs_horizon=3)
    w.push({"state": np.full((2, 4), 1.0, dtype=np.float32)})
    window = w.window()
    assert window["state"].shape == (2, 3, 4)
    np.testing.assert_allclose(window["state"], 1.0)  # front-filled: all 3 slots are the first obs

    w.push({"state": np.full((2, 4), 2.0, dtype=np.float32)})
    window = w.window()
    np.testing.assert_allclose(window["state"][:, 0], 1.0)
    np.testing.assert_allclose(window["state"][:, 1], 1.0)
    np.testing.assert_allclose(window["state"][:, 2], 2.0)  # newest pushed obs is the last slot


def test_rolling_obs_window_reset_clears_frontfill():
    w = RollingObsWindow(obs_horizon=2)
    w.push({"state": np.full((1, 2), 1.0, dtype=np.float32)})
    w.reset()
    w.push({"state": np.full((1, 2), 5.0, dtype=np.float32)})
    np.testing.assert_allclose(w.window()["state"], 5.0)


class MockEnv:
    """Deterministic Env: obs["state"] tracks each slot's own step count.
    done_at_step (B,), if given, marks a slot done once its count reaches it.
    step_calls counts individual (not batched) active-slot step invocations,
    mirroring what the old per-env eval_robomimic.py loop counted."""

    def __init__(self, num_envs: int, action_dim: int, done_at_step=None):
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.done_at_step = done_at_step
        self.step_calls = 0
        self._t = None
        self._done = None

    def reset(self):
        self._t = np.zeros(self.num_envs, dtype=int)
        self._done = np.zeros(self.num_envs, dtype=bool)
        return self._obs()

    def _obs(self):
        return {"state": np.stack([np.full(4, t, dtype=np.float32) for t in self._t])}

    def step(self, actions, active=None):
        active = np.ones(self.num_envs, dtype=bool) if active is None else np.asarray(active)
        assert actions.shape == (self.num_envs, self.action_dim)
        for i in range(self.num_envs):
            if active[i]:
                self._t[i] += 1
                self.step_calls += 1
        if self.done_at_step is not None:
            self._done = self._done | (self._t >= self.done_at_step)
        return self._obs(), np.zeros(self.num_envs, dtype=np.float32), self._done.copy(), {}


class MockPolicy(PolicyBase):
    """Returns a fixed-value action chunk of the requested batch shape."""

    def __init__(self, action_dim: int, n_action_steps: int):
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_submits = 0
        self._pending = None

    def reset(self):
        self._pending = None

    def submit(self, obs_batch, instruction=None):
        self.n_submits += 1
        self.last_instruction = instruction
        b = next(iter(obs_batch.values())).shape[0]
        self._pending = np.ones((b, self.n_action_steps, self.action_dim), dtype=np.float32)

    def get(self):
        assert self._pending is not None, "get() called before submit()"
        out, self._pending = self._pending, None
        return out

    @property
    def pending(self):
        return False


def test_sync_mode_stops_stepping_done_envs_and_chunks_correctly():
    env = MockEnv(num_envs=2, action_dim=3, done_at_step=np.array([2, 100]))
    policy = MockPolicy(action_dim=3, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None,
        open_loop_horizon=4, max_timesteps=10,
    )
    record = runner.run_batch()

    assert record.dones.tolist() == [True, False]  # env 0 hit done_at_step=2; env 1 never reaches 100
    assert record.n_steps == 10  # runs to max_timesteps since env 1 never finishes
    assert record.n_new_chunks == 3  # requeries every 4 steps over 10 steps: 0,4,8
    assert env.step_calls == 2 + 10  # env 0 stops being stepped after t=2; env 1 runs the full 10


def test_async_chunk_mode_batches_and_masks_done():
    env = MockEnv(num_envs=2, action_dim=3, done_at_step=np.array([3, 100]))
    policy = MockPolicy(action_dim=3, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.ASYNC_CHUNK, control_hz=None,
        open_loop_horizon=4, max_timesteps=10,
    )
    record = runner.run_batch()

    assert record.dones.tolist() == [True, False]
    assert record.n_steps == 10
    assert env.step_calls == 3 + 10


def test_async_streaming_mode_runs_to_completion_with_pacing():
    env = MockEnv(num_envs=1, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.ASYNC_STREAMING,
        control_hz=200.0,  # fast pacing so the test stays quick
        policy_hz=100.0,
        max_timesteps=20,
    )
    record = runner.run_batch()

    assert record.n_steps == 20
    assert record.dones.tolist() == [False]
    assert record.n_new_chunks >= 1


def test_sync_mode_with_no_pacing_matches_bulk_eval_shape():
    """control_hz=None (today's bulk statistical-eval case): no sleeping,
    same mechanics as the real-time-paced case otherwise."""
    env = MockEnv(num_envs=4, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=8)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None,
        open_loop_horizon=8, max_timesteps=16,
    )
    record = runner.run_batch()
    assert record.num_envs == 4
    assert record.n_steps == 16
    assert record.n_new_chunks == 2  # requery every 8 steps over 16 steps


def test_instruction_broadcasts_scalar_to_every_batch_item():
    env = MockEnv(num_envs=3, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None, max_timesteps=4)
    runner.run_batch(instruction="pick up the cup")
    np.testing.assert_array_equal(policy.last_instruction, np.array(["pick up the cup"] * 3, dtype=object))


def test_instruction_accepts_one_per_batch_item():
    env = MockEnv(num_envs=2, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None, max_timesteps=4)
    runner.run_batch(instruction=["pick up the cup", "close the drawer"])
    assert list(policy.last_instruction) == ["pick up the cup", "close the drawer"]


def test_instruction_wrong_length_raises():
    env = MockEnv(num_envs=3, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None, max_timesteps=4)
    with pytest.raises(ValueError, match="length-3"):
        runner.run_batch(instruction=["only one"])


def test_instruction_defaults_to_none():
    env = MockEnv(num_envs=2, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None, max_timesteps=4)
    runner.run_batch()
    assert policy.last_instruction is None


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(4, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.enc(x))


def test_activation_capture_hooks_named_submodule():
    policy = _TinyPolicy()
    capture = ActivationCapture(policy, ["enc"])
    x = torch.randn(3, 4)
    policy(x)
    acts = capture.pop()
    assert set(acts) == {"enc"}
    assert acts["enc"].shape == (3, 8)
    # cleared after pop(): a second pop with no intervening forward is empty
    assert capture.pop() == {}


def test_activation_capture_unknown_layer_raises():
    policy = _TinyPolicy()
    with pytest.raises(ValueError, match="not a submodule"):
        ActivationCapture(policy, ["does_not_exist"])


def test_activation_capture_remove_stops_hooking():
    policy = _TinyPolicy()
    capture = ActivationCapture(policy, ["enc"])
    capture.remove()
    policy(torch.randn(2, 4))
    assert capture.pop() == {}
