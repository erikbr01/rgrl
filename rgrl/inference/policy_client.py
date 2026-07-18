"""Policy backends for rgrl inference: a uniform submit/get interface,
batch-aware (leading dimension B — 1 for a single real robot, N for N
parallel sim envs run in lockstep).

  HttpPolicy      — talks to rgrl.inference.policy_server over HTTP
  LocalPolicy     — loads a checkpoint in-process (no server; same-box
                    testing, or the default in-process path
                    rgrl/eval_robomimic.py uses)
  WebSocketPolicy — talks to an openpi websocket server (pi0 / pi0.5) —
                    a different *kind* of policy than rgrl trains, not an
                    alternate transport for the same one; see its docstring

All three expose `.last_activations` (None, or {layer_name: array} — see
rgrl/inference/activations.py) after get(), populated only if
InferenceConfig.capture_layers was set (HttpPolicy/LocalPolicy) — always
None for WebSocketPolicy, which doesn't run rgrl's own policy code.

Heavy/optional imports (requests, torch, openpi_client) happen lazily so
this module imports cleanly everywhere.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rgrl.inference.activations import ACTIVATION_KEY_PREFIX
from rgrl.inference.obs_window import RollingObsWindow
from rgrl.inference.wire import ACTION_PREFIX_KEY, decode_batch, encode_batch


class PolicyBase(ABC):
    """Uniform inference interface: observe (every control step) -> submit
    (non-blocking) -> get (blocking).

    observe/submit are separate calls because obs *history* and inference
    *timing* run at different rates: policies trained on obs_horizon
    consecutive frames need every control step's observation to build a
    correct window, while inference fires only at replan points (every
    chunk, or at policy_hz). Folding the two into one call is exactly the
    bug that motivated this split — a window pushed only at submit time
    holds frames one chunk apart (e.g. [o_{t-8}, o_t]) where training used
    consecutive frames ([o_{t-1}, o_t]), which silently destroys rollout
    performance on precise tasks.
    """

    last_activations: dict[str, np.ndarray] | None = None

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode/batch."""

    @abstractmethod
    def observe(self, obs_batch: dict[str, np.ndarray]) -> None:
        """Feed the newest per-step observation (one array per obs key,
        each (B, *key_shape)). Call after env.reset() and after every
        env.step() — the Runner does this. Cheap; never blocks."""

    @abstractmethod
    def submit(
        self,
        instruction: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
    ) -> None:
        """Kick off inference asynchronously on the observations accumulated
        via observe(). Does not block.

        instruction: (B,) array of per-item language prompts for VLA-style
        policies (WebSocketPolicy) — batch-shaped like everything else here,
        not a single Runner-wide constant; ignored by backends running
        rgrl's own (non-language-conditioned) policies.
        action_prefix: real-time chunking (Runner(rtc=True)) — (B, P,
        action_dim) unnormalized actions already committed to execute while
        this inference runs. The policy conditions on them and returns a
        chunk whose first P actions repeat the prefix; the Runner drops
        those P before executing. Not supported by WebSocketPolicy (openpi
        servers do any RTC of their own server-side).
        """

    @abstractmethod
    def get(self) -> np.ndarray:
        """Block until the last submitted inference completes.

        Returns the action chunk, shape (B, T, action_dim).
        """

    def poll(self) -> np.ndarray | None:
        """Return the action chunk if inference is already done, else None."""
        if not self.pending and self._has_result():
            return self.get()
        return None

    @property
    @abstractmethod
    def pending(self) -> bool:
        """True while a submitted call has not yet completed."""

    def _has_result(self) -> bool:
        return False


class _FuturePolicy(PolicyBase):
    """Shared single-worker-executor future plumbing."""

    def __init__(self, thread_name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._future: Future | None = None

    def reset(self) -> None:
        self._future = None

    def _submit_fn(self, fn, *args) -> None:
        self._future = self._executor.submit(fn, *args)

    def get(self) -> np.ndarray:
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        chunk = self._future.result()  # re-raises if inference failed
        self._future = None
        return chunk

    @property
    def pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def _has_result(self) -> bool:
        return self._future is not None and self._future.done()


class HttpPolicy(_FuturePolicy):
    """Client for rgrl.inference.policy_server.

    On first submit it fetches /meta to learn the checkpoint's obs_horizon
    and assembles obs windows client-side via RollingObsWindow — one uniform
    path for image and state keys alike, since rgrl's shape_meta already
    describes every key the same way (no image/lowdim branch).
    """

    def __init__(self, url: str = "http://127.0.0.1:5001", timeout: float = 30.0) -> None:
        super().__init__("rgrl_http_policy")
        self._url = url.rstrip("/")
        self._timeout = timeout
        self.meta: dict | None = None
        self._window: RollingObsWindow | None = None
        # observe() runs on the control loop; submit() may run on a Runner
        # streaming thread — guard the shared window.
        self._window_lock = threading.Lock()

    def ensure_meta(self) -> dict:
        """Fetch /meta now if it hasn't been already — lets a caller learn
        the checkpoint's obs layout (e.g. to build a matching env) before
        the first submit(), instead of only on first use."""
        if self.meta is None:
            self._fetch_meta()
        return self.meta

    def _fetch_meta(self) -> None:
        import requests

        resp = requests.get(f"{self._url}/meta", timeout=self._timeout)
        resp.raise_for_status()
        self.meta = resp.json()
        self._window = RollingObsWindow(int(self.meta["obs_horizon"]))

    def reset(self) -> None:
        super().reset()
        if self._window is not None:
            self._window.reset()

    def observe(self, obs_batch: dict[str, np.ndarray]) -> None:
        self.ensure_meta()
        with self._window_lock:
            self._window.push(obs_batch)

    def submit(
        self,
        instruction: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
    ) -> None:
        if self._window is None or not len(self._window):
            raise RuntimeError("call observe() with the current obs before submit()")
        with self._window_lock:
            arrays = self._window.window()
        if action_prefix is not None:
            arrays = {**arrays, ACTION_PREFIX_KEY: np.asarray(action_prefix, dtype=np.float32)}
        self._submit_fn(self._post, encode_batch(arrays))

    def get(self) -> np.ndarray:
        action, self.last_activations = self._get_with_activations()
        return action

    def _get_with_activations(self) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        arrays = self._future.result()
        self._future = None
        action = arrays["action"]
        activations = {
            k[len(ACTIVATION_KEY_PREFIX):]: v
            for k, v in arrays.items()
            if k.startswith(ACTIVATION_KEY_PREFIX)
        }
        return action, (activations or None)

    def _post(self, data: bytes) -> dict[str, np.ndarray]:
        import requests

        resp = requests.post(
            f"{self._url}/infer",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return decode_batch(resp.content)


class LocalPolicy(_FuturePolicy):
    """Run a checkpoint in-process (no server) — same-box testing, or a
    deployment where server/env venv separation isn't needed.

    Inference runs in a single background worker so this behaves identically
    to HttpPolicy from the caller's perspective (async submit/get).
    """

    def __init__(self, checkpoint: str, device: str = "cpu", inference=None) -> None:
        super().__init__("rgrl_local_policy")
        import torch

        from rgrl.inference.activations import ActivationCapture
        from rgrl.inference.checkpoint import InferenceConfig, load_policy

        if inference is not None and inference.cuda_graph and inference.capture_layers:
            raise ValueError(
                "cuda_graph + capture_layers is not supported: forward hooks only fire "
                "during the one-time graph capture, not on later graph replays, so "
                "captured activations would silently go stale after the first call."
            )
        self._torch = torch
        self._device = torch.device(device)
        self._window_lock = threading.Lock()  # observe() on control loop vs submit() on streaming thread
        inference = inference or InferenceConfig()
        self.policy, self.store, self.step = load_policy(Path(checkpoint), inference, self._device)
        self._window = RollingObsWindow(self.policy.cfg.obs_horizon)
        self._autocast = torch.autocast(
            device_type=self._device.type, dtype=torch.bfloat16, enabled=inference.bf16
        )
        self._capture = (
            ActivationCapture(self.policy, inference.capture_layers)
            if inference.capture_layers
            else None
        )
        # `self.policy` stays the raw nn.Module (callers read .cfg/.store off
        # it — see run_realtime.py); `self._predict` is what _infer() actually
        # calls, so the graphed path is additive, not a replacement in place.
        self._predict = self.policy
        if inference.cuda_graph:
            if self._device.type != "cuda":
                raise ValueError("cuda_graph requires device='cuda:*'")
            from rgrl.inference.cuda_graph import GraphedPolicy

            self._predict = GraphedPolicy(self.policy)

    def reset(self) -> None:
        super().reset()
        self._window.reset()

    def observe(self, obs_batch: dict[str, np.ndarray]) -> None:
        with self._window_lock:
            self._window.push(obs_batch)

    def submit(
        self,
        instruction: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
    ) -> None:
        if not len(self._window):
            raise RuntimeError("call observe() with the current obs before submit()")
        with self._window_lock:
            arrays = self._window.window()
        batch = {k: self._torch.as_tensor(v, device=self._device) for k, v in arrays.items()}
        prefix = (
            None
            if action_prefix is None
            else self._torch.as_tensor(np.asarray(action_prefix, dtype=np.float32), device=self._device)
        )
        self._submit_fn(self._infer, batch, prefix)

    def get(self) -> np.ndarray:
        action, self.last_activations = self._future_result()
        return action

    def _future_result(self) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        if self._future is None:
            raise RuntimeError("call submit() before get()")
        result = self._future.result()
        self._future = None
        return result

    def _infer(self, batch: dict, prefix=None) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        with self._torch.no_grad(), self._autocast:
            if prefix is None:
                action = self._predict.predict_action(batch)
            else:
                action = self._predict.predict_action_rtc(batch, prefix)
        activations = self._capture.pop() if self._capture is not None else None
        return action.float().cpu().numpy(), (activations or None)


class WebSocketPolicy(_FuturePolicy):
    """Client for an openpi websocket server (pi0 / pi0.5).

    Unlike HttpPolicy, this runs a *different kind* of policy than rgrl
    trains — not an alternate transport for the same checkpoint. openpi/pi0
    policies consume the current single-frame observation (no obs_horizon
    window) plus a language instruction, so this backend bypasses
    RollingObsWindow entirely and forwards obs_batch[key][0] (B is always 1
    — one physical robot) directly, matching the wire format
    openpi_client.websocket_client_policy.WebsocketClientPolicy expects.

    State/camera key names default to rgrl.inference.droid_env.DroidEnv's
    native obs keys (joint_positions, gripper_position, wrist_image_left,
    exterior_image_1_left/2_left) so no key-remapping is needed when pairing
    this with DroidEnv, same as HttpPolicy.
    """

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8000",
        external_camera: str = "ext1",
        wrist_key: str = "wrist_image_left",
        ext_key: str | None = None,
        state_keys: tuple[str, ...] = ("joint_positions", "gripper_position"),
    ) -> None:
        super().__init__("rgrl_ws_policy")
        self._url = url
        self._ext_key = ext_key or (
            "exterior_image_1_left" if external_camera == "ext1" else "exterior_image_2_left"
        )
        self._wrist_key = wrist_key
        self._state_keys = tuple(state_keys)
        self._client = None
        self._latest_obs: dict[str, np.ndarray] | None = None

    def _get_client(self):
        if self._client is None:
            from openpi_client import websocket_client_policy

            rest = self._url.removeprefix("ws://").removeprefix("wss://")
            host, _, port = rest.partition(":")
            self._client = websocket_client_policy.WebsocketClientPolicy(host, int(port))
        return self._client

    def observe(self, obs_batch: dict[str, np.ndarray]) -> None:
        # openpi/pi0 consumes only the current frame — keep just the latest.
        self._latest_obs = obs_batch

    def submit(
        self,
        instruction: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
    ) -> None:
        if action_prefix is not None:
            # openpi's wire format has no client-supplied prefix (pi0.5-style
            # RTC happens server-side), and Runner(rtc=True) would then drop
            # real actions from a reply that never contained the prefix.
            raise ValueError("WebSocketPolicy does not support action_prefix -- run without rtc=True")
        if self._latest_obs is None:
            raise RuntimeError("call observe() with the current obs before submit()")
        prompt = "" if instruction is None else str(instruction[0])  # one robot: drop the batch dim
        self._submit_fn(self._infer, self._build_request(self._latest_obs, prompt))

    def _build_request(self, obs_batch: dict[str, np.ndarray], prompt: str) -> dict:
        from openpi_client import image_tools

        obs = {k: v[0] for k, v in obs_batch.items()}  # one robot: drop the batch dim
        request: dict = {"prompt": prompt}
        for key in self._state_keys:
            if key in obs:
                request[f"observation/{key}"] = np.asarray(obs[key], dtype=np.float32)
        ext_img = obs.get(self._ext_key)
        wrist_img = obs.get(self._wrist_key)
        if ext_img is not None:
            request["observation/exterior_image_1_left"] = image_tools.resize_with_pad(ext_img, 224, 224)
        if wrist_img is not None:
            request["observation/wrist_image_left"] = image_tools.resize_with_pad(wrist_img, 224, 224)
        return request

    def _infer(self, request: dict) -> np.ndarray:
        result = self._get_client().infer(request)
        actions = np.asarray(result["actions"], dtype=np.float32)  # (T, action_dim)
        return actions[None]  # -> (1, T, action_dim), matching PolicyBase's batched contract
