"""Generic, real-time-capable rollout runner: wraps an Env + PolicyBase and
decides how inference is paced relative to env stepping.

Every mode operates on a *batch* of B episodes run in lockstep — B=1 for a
single real robot or a single real-time-paced sim, B=N for N parallel sim
envs run for statistical throughput. `Env.step()` takes an `active` mask so
a batch item that's already done stops being stepped (and its obs freezes)
without changing the batch's static shape — this is what
rgrl/eval_robomimic.py's bespoke loop already relied on (skip re-stepping
finished envs, keep a stable batch shape for torch.compile) and is preserved
here as a first-class part of the Env contract instead of being reinvented
per call site.

Three inference modes
----------------------
SYNC
    Inference blocks the control loop. At every open_loop_horizon boundary,
    submit + get() before executing the next chunk. `control_hz` is optional
    in *every* mode, including this one: set it for wall-clock-paced
    real-time envs (a real robot, or a sim paced to match deployment
    timing); leave it None to run the batch as fast as possible (today's
    bulk statistical eval — B parallel envs, no pacing).

ASYNC_CHUNK
    At step `replan_step` of each chunk (0 = default, start of chunk),
    submit inference for the *next* chunk. Execute the current chunk while
    inference runs in the background; block for the new chunk only at the
    chunk boundary.

    With `rtc=True` (real-time chunking, after https://github.com/amazon-far/abc
    and the Physical Intelligence RTC paper): each replan additionally sends
    the still-queued actions of the current chunk as an `action_prefix`, so
    the policy conditions its new chunk on the actions that will execute
    during inference latency (inference-time inpainting — see
    predict_action_rtc in rgrl/policies/diffusion.py) and continues them
    smoothly instead of jumping at the chunk boundary. The returned chunk
    repeats the prefix at its head; the Runner drops those P actions before
    executing. Requires the policy's trained horizon to have room past
    n_action_steps for the prefix: P = open_loop_horizon - 1 - replan_step
    must be ≤ horizon - (obs_horizon-1) - n_action_steps.

ASYNC_STREAMING
    A dedicated background thread calls submit -> get in a loop, always
    using the latest observation, throttled to at most `policy_hz`. A
    freshly completed chunk only preempts the one currently executing once
    `min_commit_steps` of it have run. The foreground loop reads from a
    lock-protected buffer at `control_hz`; if the buffer is empty (policy
    slower than the loop), applies `stale_action_policy`.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from rgrl.inference.policy_client import PolicyBase

log = logging.getLogger(__name__)


class Env(Protocol):
    num_envs: int
    action_dim: int

    def reset(self) -> dict[str, np.ndarray]:
        """Returns an obs dict; each array shaped (B, *key_shape)."""

    def step(
        self, actions: np.ndarray, active: Optional[np.ndarray] = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        """actions: (B, action_dim). active: (B,) bool, default all-True.

        Returns (obs_batch, rewards (B,), dones (B,), info). Batch items
        where active is False are not actually stepped — their obs and done
        flag are returned unchanged from the previous call.
        """


class InferenceMode(enum.Enum):
    SYNC = "sync"
    ASYNC_CHUNK = "async_chunk"
    ASYNC_STREAMING = "async_streaming"


class StaleActionPolicy(enum.Enum):
    HOLD_LAST = "hold_last"
    ZERO_ON_STALE = "zero_on_stale"


@dataclass
class BatchRecord:
    num_envs: int
    n_steps: int
    n_new_chunks: int
    n_stale_steps: int
    dones: np.ndarray  # (B,) bool — task success per batch item (real robot: always False)
    avg_chunk_latency_ms: float
    chunk_latencies_ms: list = field(default_factory=list)
    # Streaming mode only: leading steps taken before the first chunk arrived.
    n_warmup_steps: int = 0


Profiler = Callable[[str], "object"]  # name -> context manager, e.g. rgrl.profiler.SectionProfiler


def _noop_profiler(_name: str):
    return nullcontext()


class Runner:
    """Runs one batch of B episodes with a configurable inference mode."""

    def __init__(
        self,
        env: Env,
        policy: PolicyBase,
        mode: InferenceMode = InferenceMode.SYNC,
        control_hz: float | None = None,
        stale_action_policy: StaleActionPolicy = StaleActionPolicy.HOLD_LAST,
        open_loop_horizon: int = 8,
        max_timesteps: int = 400,
        policy_hz: float | None = None,
        min_commit_steps: int = 0,
        replan_step: int = 0,
        rtc: bool = False,
        profiler: Profiler | None = None,
    ) -> None:
        if rtc and mode != InferenceMode.ASYNC_CHUNK:
            raise ValueError(
                "rtc=True requires mode=ASYNC_CHUNK: real-time chunking conditions the "
                "next chunk on actions executing during inference latency, which only "
                "exists when replanning overlaps execution at fixed chunk boundaries."
            )
        self.env = env
        self.policy = policy
        self.mode = mode
        self.control_hz = control_hz
        self.stale_action_policy = stale_action_policy
        self.open_loop_horizon = open_loop_horizon
        self.max_timesteps = max_timesteps
        self.policy_hz = policy_hz
        self.min_commit_steps = min_commit_steps
        self.replan_step = replan_step
        self.rtc = rtc
        self._profiler = profiler or _noop_profiler

    def run_batch(
        self,
        instruction: str | list[str] | None = None,
        stop_event: threading.Event | None = None,
    ) -> BatchRecord:
        """instruction: language prompt(s) for VLA-style policies
        (WebSocketPolicy); ignored by backends running rgrl's own policies.
        A single string broadcasts to every batch item; a length-B sequence
        gives each item its own instruction (e.g. B parallel real-time sims
        each pursuing a different task) — batch-shaped like every other
        per-item value here, rather than one Runner-wide constant."""
        obs = self.env.reset()
        self.policy.reset()
        self.policy.observe(obs)
        instr = self._normalize_instruction(instruction)
        if self.mode == InferenceMode.SYNC:
            return self._run_sync(instr, stop_event)
        if self.mode == InferenceMode.ASYNC_CHUNK:
            return self._run_async_chunk(instr, stop_event)
        return self._run_async_streaming(instr, stop_event)

    def _normalize_instruction(self, instruction: str | list[str] | None) -> np.ndarray | None:
        if instruction is None:
            return None
        b = self.env.num_envs
        if isinstance(instruction, str):
            return np.array([instruction] * b, dtype=object)
        arr = np.asarray(instruction, dtype=object)
        if arr.shape != (b,):
            raise ValueError(f"instruction must be a string or a length-{b} sequence, got shape {arr.shape}")
        return arr

    # ------------------------------------------------------------------
    # SYNC mode
    # ------------------------------------------------------------------

    def _run_sync(
        self, instruction: np.ndarray | None, stop_event: threading.Event | None
    ) -> BatchRecord:
        B = self.env.num_envs
        dt = 1.0 / self.control_hz if self.control_hz else None
        done = np.zeros(B, dtype=bool)
        chunk: np.ndarray | None = None
        chunk_idx = 0
        n_chunks = 0
        latencies: list[float] = []
        total_steps = 0

        for _ in range(self.max_timesteps):
            if stop_event is not None and stop_event.is_set():
                break
            if done.all():
                break
            t0 = time.time()

            if chunk is None or chunk_idx >= self.open_loop_horizon or chunk_idx >= chunk.shape[1]:
                t_inf = time.time()
                with self._profiler("policy"):
                    self.policy.submit(instruction)
                    chunk = self.policy.get()  # (B, T, action_dim)
                latencies.append((time.time() - t_inf) * 1000)
                chunk_idx = 0
                n_chunks += 1

            actions = chunk[:, chunk_idx]
            chunk_idx += 1
            with self._profiler("env"):
                obs, _, dones_step, _ = self.env.step(actions, active=~done)
            self.policy.observe(obs)
            done = done | dones_step
            total_steps += 1

            if dt is not None:
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        return self._make_record(B, total_steps, n_chunks, 0, latencies, done)

    # ------------------------------------------------------------------
    # ASYNC_CHUNK mode
    # ------------------------------------------------------------------

    def _run_async_chunk(
        self, instruction: np.ndarray | None, stop_event: threading.Event | None
    ) -> BatchRecord:
        B = self.env.num_envs
        dt = 1.0 / self.control_hz if self.control_hz else None
        n_chunks = 0
        latencies: list[float] = []
        done = np.zeros(B, dtype=bool)

        t_submit = time.time()
        with self._profiler("policy"):
            self.policy.submit(instruction)
        sent_prefix_len = 0  # P of the request currently in flight (rtc only)

        total_steps = 0
        t_next = time.time()
        while total_steps < self.max_timesteps and not done.all():
            if stop_event is not None and stop_event.is_set():
                break
            with self._profiler("policy"):
                chunk = self.policy.get()  # (B, T, action_dim); rtc: (B, P + T, action_dim)
            latencies.append((time.time() - t_submit) * 1000)
            n_chunks += 1
            if sent_prefix_len:
                # rtc: the chunk's first P actions repeat the prefix we sent
                # — they're already executing/executed, start after them.
                chunk = chunk[:, sent_prefix_len:]

            horizon = min(self.open_loop_horizon, chunk.shape[1], self.max_timesteps - total_steps)
            replan_step = min(self.replan_step, horizon - 1)

            for i in range(horizon):
                if stop_event is not None and stop_event.is_set():
                    break
                if dt is not None:
                    now = time.time()
                    if t_next > now:
                        time.sleep(t_next - now)
                    t_next += dt

                with self._profiler("env"):
                    obs, _, dones_step, _ = self.env.step(chunk[:, i], active=~done)
                self.policy.observe(obs)
                done = done | dones_step
                total_steps += 1
                if done.all():
                    break

                if i == replan_step:
                    prefix = None
                    sent_prefix_len = 0
                    if self.rtc and i + 1 < horizon:
                        # Actions committed but not yet executed — exactly
                        # what runs between this obs and the chunk boundary.
                        prefix = chunk[:, i + 1 : horizon]
                        sent_prefix_len = prefix.shape[1]
                    t_submit = time.time()
                    with self._profiler("policy"):
                        self.policy.submit(instruction, prefix)

        return self._make_record(B, total_steps, n_chunks, 0, latencies, done)

    # ------------------------------------------------------------------
    # ASYNC_STREAMING mode
    # ------------------------------------------------------------------

    def _run_async_streaming(
        self, instruction: np.ndarray | None, stop_event: threading.Event | None
    ) -> BatchRecord:
        B = self.env.num_envs
        dt = 1.0 / self.control_hz if self.control_hz else None

        lock = threading.Lock()
        buffer_ref: list = [None]
        chunk_step_ref: list = [0]
        last_action_ref: list = [None]
        pending_ref: list = [None]
        stop = threading.Event()
        latencies: list[float] = []
        n_chunks_ref = [0]

        min_submit_period = 1.0 / self.policy_hz if self.policy_hz else 0.0

        def _inference_thread() -> None:
            while not stop.is_set():
                t0 = time.time()
                try:
                    # the main loop feeds every step's obs via policy.observe();
                    # submit() snapshots the current window (thread-safe in the
                    # policy backends)
                    with self._profiler("policy"):
                        self.policy.submit(instruction)
                        new_chunk = self.policy.get()
                except Exception as e:  # noqa: BLE001 — keep the loop alive
                    log.warning("inference error: %s", e)
                    continue
                lat_ms = (time.time() - t0) * 1000
                with lock:
                    pending_ref[0] = new_chunk
                    n_chunks_ref[0] += 1
                    latencies.append(lat_ms)
                remaining = min_submit_period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)

        thread = threading.Thread(target=_inference_thread, daemon=True, name="rgrl_infer")
        thread.start()

        n_stale = 0
        total_steps = 0
        done = np.zeros(B, dtype=bool)
        step_has_chunk: list[bool] = []

        try:
            for _ in range(self.max_timesteps):
                if stop_event is not None and stop_event.is_set():
                    break
                if done.all():
                    break
                t0 = time.time()

                with lock:
                    if pending_ref[0] is not None and (
                        buffer_ref[0] is None
                        or chunk_step_ref[0] >= buffer_ref[0].shape[1]
                        or chunk_step_ref[0] >= self.min_commit_steps
                    ):
                        buffer_ref[0] = pending_ref[0]
                        chunk_step_ref[0] = 0
                        pending_ref[0] = None
                    chunk = buffer_ref[0]
                    idx = chunk_step_ref[0]
                step_has_chunk.append(chunk is not None)

                if chunk is not None and idx < chunk.shape[1]:
                    actions = chunk[:, idx]
                    with lock:
                        chunk_step_ref[0] += 1
                    last_action_ref[0] = actions
                else:
                    if self.stale_action_policy == StaleActionPolicy.HOLD_LAST and last_action_ref[0] is not None:
                        actions = last_action_ref[0]
                    else:
                        actions = np.zeros((B, self.env.action_dim), dtype=np.float32)
                    if last_action_ref[0] is not None:
                        n_stale += 1

                with self._profiler("env"):
                    new_obs, _, dones_step, _ = self.env.step(actions, active=~done)
                self.policy.observe(new_obs)
                done = done | dones_step
                total_steps += 1

                if dt is not None:
                    elapsed = time.time() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
        finally:
            stop.set()
            thread.join(timeout=5.0)

        n_warmup_steps = next((i for i, has in enumerate(step_has_chunk) if has), len(step_has_chunk))
        return self._make_record(
            B, total_steps, n_chunks_ref[0], n_stale, latencies, done, n_warmup_steps
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _make_record(
        num_envs: int,
        n_steps: int,
        n_chunks: int,
        n_stale: int,
        latencies: list[float],
        dones: np.ndarray,
        n_warmup_steps: int = 0,
    ) -> BatchRecord:
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        return BatchRecord(
            num_envs=num_envs,
            n_steps=n_steps,
            n_new_chunks=n_chunks,
            n_stale_steps=n_stale,
            dones=dones,
            avg_chunk_latency_ms=avg_lat,
            chunk_latencies_ms=list(latencies),
            n_warmup_steps=n_warmup_steps,
        )
