"""Fast, hardware-free tests for rgrl/inference: wire format, obs windowing,
and Runner's batch/done-mask/chunking mechanics against a mock Env + PolicyBase
(no robosuite/mujoco/droid needed — those are exercised live, not in CI)."""

import numpy as np
import pytest
import torch
from torch import nn

from rgrl.inference.activations import ActivationCapture
from rgrl.inference.envs.robomimic import _env_kwargs, _frame, _task_success
from rgrl.inference.obs_window import RollingObsWindow
from rgrl.inference.policy_client import HttpPolicy, LocalPolicy, PolicyBase, WebSocketPolicy
from rgrl.inference.rollout_cli import build_policy
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


def test_robosuite_frame_uses_dataset_key_names_and_camera_orientation():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    raw = {
        "object-state": np.array([1.0, 2.0], dtype=np.float64),
        "robot0_eef_pos": np.array([3.0, 4.0, 5.0], dtype=np.float64),
        "sideview_image": image,
    }
    frame = _frame(
        raw,
        image_keys=["sideview_image"],
        state_keys=["object", "robot0_eef_pos"],
    )

    np.testing.assert_array_equal(frame["object"], [1.0, 2.0])
    np.testing.assert_array_equal(frame["robot0_eef_pos"], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(frame["sideview_image"], image[::-1])
    assert frame["object"].dtype == np.float32
    assert frame["sideview_image"].dtype == np.uint8
    assert frame["sideview_image"].flags.c_contiguous


def test_robosuite_kwargs_are_host_local_and_do_not_mutate_metadata(monkeypatch):
    class EglProbe:
        @staticmethod
        def get_available_devices():
            return [3]

    monkeypatch.setitem(__import__("sys").modules, "egl_probe", EglProbe)
    env_meta = {
        "type": 1,
        "env_kwargs": {
            "render_gpu_device_id": 7,
            "controller_configs": {"control_delta": True},
        },
    }
    kwargs = _env_kwargs(
        env_meta,
        image_keys=["sideview_image"],
        image_hw=(84, 96),
        abs_action=True,
    )

    assert kwargs["render_gpu_device_id"] == 3
    assert kwargs["camera_names"] == ["sideview"]
    assert kwargs["camera_heights"] == 84
    assert kwargs["camera_widths"] == 96
    assert kwargs["controller_configs"]["control_delta"] is False
    assert env_meta["env_kwargs"]["render_gpu_device_id"] == 7
    assert env_meta["env_kwargs"]["controller_configs"]["control_delta"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ({"task": True, "stage": False}, True)],
)
def test_robosuite_task_success_accepts_native_result_shapes(value, expected):
    class Env:
        def _check_success(self):
            return value

    assert _task_success(Env()) is expected


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
    """Returns a distinguishable action chunk: every action in the chunk from
    submit #s at within-chunk position j has value 100*s + j. Honors the RTC
    contract — a submitted action_prefix comes back verbatim at the chunk's
    head, with n_action_steps fresh actions after it. Records every observed
    obs and the obs window (last 2 observes) present at each submit."""

    def __init__(self, action_dim: int, n_action_steps: int):
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_submits = 0
        self.n_observes = 0
        self.prefixes = []  # the action_prefix (or None) of every submit, in order
        self.submit_windows = []  # [prev_obs, latest_obs] at each submit
        self._recent = []
        self._pending = None

    def reset(self):
        self._pending = None
        self._recent = []

    def observe(self, obs_batch):
        self.n_observes += 1
        self._latest_obs = obs_batch
        self._recent = (self._recent + [obs_batch])[-2:]

    def submit(self, instruction=None, action_prefix=None):
        self.n_submits += 1
        self.last_instruction = instruction
        self.prefixes.append(None if action_prefix is None else np.array(action_prefix))
        self.submit_windows.append(list(self._recent))
        b = next(iter(self._latest_obs.values())).shape[0]
        fresh = np.stack(
            [np.full((b, self.action_dim), 100 * self.n_submits + j, dtype=np.float32)
             for j in range(self.n_action_steps)],
            axis=1,
        )
        if action_prefix is not None:
            fresh = np.concatenate([np.asarray(action_prefix, dtype=np.float32), fresh], axis=1)
        self._pending = fresh

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


def test_policy_observes_every_step_and_windows_are_consecutive():
    """Regression test for the bug that zeroed tool_hang eval: the policy
    must see EVERY control step's obs (Runner calls observe() per step), so
    an obs window holds consecutive frames [o_{t-1}, o_t] — not frames one
    whole chunk apart, which is what pushing obs only at submit time
    produced (trained on consecutive frames, that silently breaks rollouts)."""
    env = MockEnv(num_envs=1, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.SYNC, control_hz=None,
        open_loop_horizon=4, max_timesteps=12,
    )
    runner.run_batch()
    # reset obs + one per step
    assert policy.n_observes == 1 + 12
    # every submit's window is [o_{t-1}, o_t]: consecutive step counters
    # (MockEnv obs value == its step count), except the first (front-fill)
    for window in policy.submit_windows[1:]:
        prev, latest = window[0]["state"][0, 0], window[1]["state"][0, 0]
        assert latest - prev == 1.0, (prev, latest)


class RecordingEnv(MockEnv):
    """MockEnv that also records every stepped action batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = []

    def step(self, actions, active=None):
        self.actions.append(np.array(actions))
        return super().step(actions, active)


def test_rtc_requires_async_chunk_mode():
    env = MockEnv(num_envs=1, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    with pytest.raises(ValueError, match="ASYNC_CHUNK"):
        Runner(env=env, policy=policy, mode=InferenceMode.SYNC, rtc=True)


def test_rtc_sends_queued_actions_as_prefix_and_drops_them_from_the_reply():
    env = RecordingEnv(num_envs=1, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.ASYNC_CHUNK, control_hz=None,
        open_loop_horizon=4, replan_step=0, max_timesteps=8, rtc=True,
    )
    record = runner.run_batch()
    assert record.n_steps == 8
    assert record.n_new_chunks == 2

    # First submit has no prefix; each replan sends exactly the still-queued
    # actions of the executing chunk (submit #s produces values 100*s + j).
    assert policy.prefixes[0] is None
    np.testing.assert_array_equal(policy.prefixes[1][0, :, 0], [101, 102, 103])
    np.testing.assert_array_equal(policy.prefixes[2][0, :, 0], [201, 202, 203])

    # Executed actions: chunk 1 in full, then chunk 2's *fresh* actions only
    # — the prefix repeated at chunk 2's head must have been dropped, never
    # re-executed.
    executed = [a[0, 0] for a in env.actions]
    assert executed == [100, 101, 102, 103, 200, 201, 202, 203]


def test_rtc_with_late_replan_sends_shorter_prefix():
    env = RecordingEnv(num_envs=1, action_dim=2)
    policy = MockPolicy(action_dim=2, n_action_steps=4)
    runner = Runner(
        env=env, policy=policy, mode=InferenceMode.ASYNC_CHUNK, control_hz=None,
        open_loop_horizon=4, replan_step=2, max_timesteps=8, rtc=True,
    )
    runner.run_batch()
    # replan at i=2 of a 4-step chunk -> one queued action left -> P=1
    assert policy.prefixes[1].shape == (1, 1, 2)
    np.testing.assert_array_equal(policy.prefixes[1][0, :, 0], [103])
    executed = [a[0, 0] for a in env.actions]
    assert executed == [100, 101, 102, 103, 200, 201, 202, 203]


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


# rollout_cli.build_policy: the one dispatch shared by eval.py, run_realtime.py,
# and collect_rollouts.py -- ws(s):// / http(s):// / ckpt / default-to-local-server.
# None of these constructors connect eagerly (HttpPolicy/WebSocketPolicy just store
# config; LocalPolicy is the one exception, so its branch uses a real dry ckpt=None
# case plus a monkeypatch for the ckpt-given case).


def test_build_policy_ws_url_returns_websocket_policy():
    policy = build_policy(ckpt=None, server="ws://gpubox:8000", inference=None, external_camera="ext2")
    assert isinstance(policy, WebSocketPolicy)


def test_build_policy_wss_url_returns_websocket_policy():
    policy = build_policy(ckpt=None, server="wss://gpubox:8000", inference=None)
    assert isinstance(policy, WebSocketPolicy)


def test_build_policy_http_url_returns_http_policy():
    policy = build_policy(ckpt=None, server="http://gpubox:5001", inference=None)
    assert isinstance(policy, HttpPolicy)
    assert policy._url == "http://gpubox:5001"


def test_build_policy_no_ckpt_no_server_defaults_to_local_policy_server():
    policy = build_policy(ckpt=None, server=None, inference=None)
    assert isinstance(policy, HttpPolicy)
    assert policy._url == "http://127.0.0.1:5001"


def test_build_policy_ckpt_given_returns_local_policy(monkeypatch):
    captured = {}

    class _FakeCfg:
        obs_horizon = 2

    class _FakePolicy:
        cfg = _FakeCfg()

    def fake_load_policy(ckpt_path, inference, device):
        captured["ckpt_path"] = ckpt_path
        captured["device"] = device
        return _FakePolicy(), object(), 0

    monkeypatch.setattr("rgrl.inference.checkpoint.load_policy", fake_load_policy)
    policy = build_policy(ckpt="runs/x/latest.pt", server=None, inference=None, device="cpu")
    assert isinstance(policy, LocalPolicy)
    assert str(captured["ckpt_path"]) == "runs/x/latest.pt"
    assert captured["device"] == torch.device("cpu")
