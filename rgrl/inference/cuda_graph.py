"""CUDA-graph-accelerated inference: wraps `policy.predict_action` in a
lazily-captured `torch.cuda.CUDAGraph`, eliminating Python/launch overhead
across the whole call — the encoder forward, every diffusion/flow sampling
step, and the final unnormalize.

Composes with `InferenceConfig.compile`: `load_policy` (rgrl/inference/
checkpoint.py) compiles the UNet with `fullgraph=True` but *without*
`mode="reduce-overhead"` when `cuda_graph` is also set, since that mode
manages its own internal CUDA graph (inductor's cudagraph-trees) and can't
be captured inside this module's outer one (confirmed: raises
cudaErrorStreamCaptureUnsupported). Compile-without-reduce-overhead still
gets Inductor's kernel fusion; wrapping the *whole* call here on top gets
the cross-step launch-overhead elimination too — this is ABC's own "single
fullgraph compile, then CUDA graphs on top" strategy (https://abc.bot/),
adapted to rgrl's simpler self-contained predict_action.

This only works because `DiffusionPolicy`/`FlowPolicy.predict_action` are
self-contained (normalize, encode, sample, unnormalize all inside one call,
no external per-step Python state) and graph-safe as written: no operation
inside references an ephemeral, per-call CPU tensor via a host-to-device
copy (the one place that used to — DiffusionPolicy's per-step diffusion
timestep — was refactored in rgrl/policies/diffusion.py::DiffusionPolicy to
use a buffer-backed GPU tensor precomputed once at construction, precisely
so its memory address stays valid across every future graph replay; buffer
lookups also had to switch to plain-int indexing, since indexing with a
0-dim *tensor* triggers a different sync-requiring code path that capture
also rejects). `torch.randn`/`torch.randn_like` calls inside the captured
region are handled correctly by `torch.cuda.graph`'s own Philox RNG state
management — each replay advances the offset and produces fresh random
draws.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

_Signature = dict  # {key: (shape: tuple[int, ...], dtype: torch.dtype)}


class GraphedPolicy:
    """Drop-in replacement for `policy.predict_action` wherever it's called
    today (LocalPolicy._infer, policy_server.py's /infer handler) — same
    call signature, same return value, backed by a captured CUDA graph
    after the first call.

    Captures lazily, keyed by the first batch's per-key (shape, dtype)
    signature. A later call with a *different* signature raises rather than
    silently falling back to eager or re-capturing: a real robot getting a
    silent latency cliff (fallback) or unbounded memory growth (repeated
    capture) is worse than a clear error. Construct a separate
    GraphedPolicy per distinct batch shape if you genuinely need more than
    one (uncommon — a given deployment's batch size is normally fixed for
    its whole lifetime, same assumption num_envs/obs_horizon already rely
    on elsewhere in rgrl/inference).
    """

    def __init__(self, policy: nn.Module, warmup_steps: int = 5, replay_warmups: int = 2):
        """warmup_steps: also what triggers torch.compile to finish
        compiling, if the wrapped policy's UNet is compiled — give it
        enough iterations (5, matching the reference's own choice) rather
        than just enough for cuDNN/cuBLAS algorithm selection."""
        self.policy = policy
        self.warmup_steps = warmup_steps
        self.replay_warmups = replay_warmups
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: dict[str, torch.Tensor] | None = None
        self._static_output: torch.Tensor | None = None
        self._signature: _Signature | None = None
        # Separate graph for the RTC path (predict_action_rtc) — a distinct
        # captured program with its own static prefix input, mirroring the
        # ABC reference's FastRTCInferenceGraph alongside FastInferenceGraph.
        # Both coexist: an RTC deployment's first chunk of each episode has
        # no prefix (plain graph), every later chunk does (rtc graph).
        self._rtc_graph: torch.cuda.CUDAGraph | None = None
        self._rtc_static_batch: dict[str, torch.Tensor] | None = None
        self._rtc_static_prefix: torch.Tensor | None = None
        self._rtc_static_output: torch.Tensor | None = None
        self._rtc_signature: tuple | None = None

    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sig = {k: (tuple(v.shape), v.dtype) for k, v in batch.items()}
        if self._graph is None:
            log.info("capturing CUDA graph for batch signature %s", sig)
            self._capture(batch, sig)
        elif sig != self._signature:
            raise RuntimeError(
                f"GraphedPolicy was captured for batch signature {self._signature}, "
                f"got {sig} -- construct a separate GraphedPolicy per distinct batch shape."
            )
        for k, v in batch.items():
            self._static_batch[k].copy_(v)
        self._graph.replay()
        return self._static_output

    def _capture(self, batch: dict[str, torch.Tensor], sig: _Signature) -> None:
        self._signature = sig
        self._static_batch = {k: v.clone() for k, v in batch.items()}

        # Warm up on a side stream first: cuDNN/cuBLAS algorithm selection
        # and any lazy one-time init must happen *before* capture, or it'll
        # get baked into the graph incorrectly (or capture will just fail).
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_steps):
                self.policy.predict_action(self._static_batch)
        torch.cuda.current_stream().wait_stream(stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.policy.predict_action(self._static_batch)

        for _ in range(self.replay_warmups):
            self._graph.replay()
        torch.cuda.synchronize()

    def predict_action_rtc(self, batch: dict[str, torch.Tensor], action_prefix: torch.Tensor) -> torch.Tensor:
        """Graphed counterpart of the policy's predict_action_rtc (see
        rgrl/policies/diffusion.py) — same contract, captured separately
        from the plain path since the prefix is an additional static input
        and the captured program differs (the inpainting clamps)."""
        sig = (
            {k: (tuple(v.shape), v.dtype) for k, v in batch.items()},
            (tuple(action_prefix.shape), action_prefix.dtype),
        )
        if self._rtc_graph is None:
            log.info("capturing RTC CUDA graph for signature %s", sig)
            self._capture_rtc(batch, action_prefix, sig)
        elif sig != self._rtc_signature:
            raise RuntimeError(
                f"GraphedPolicy's RTC graph was captured for signature {self._rtc_signature}, "
                f"got {sig} -- construct a separate GraphedPolicy per distinct batch/prefix shape."
            )
        for k, v in batch.items():
            self._rtc_static_batch[k].copy_(v)
        self._rtc_static_prefix.copy_(action_prefix)
        self._rtc_graph.replay()
        return self._rtc_static_output

    def _capture_rtc(self, batch: dict[str, torch.Tensor], action_prefix: torch.Tensor, sig: tuple) -> None:
        self._rtc_signature = sig
        self._rtc_static_batch = {k: v.clone() for k, v in batch.items()}
        self._rtc_static_prefix = action_prefix.clone()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_steps):
                self.policy.predict_action_rtc(self._rtc_static_batch, self._rtc_static_prefix)
        torch.cuda.current_stream().wait_stream(stream)

        self._rtc_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._rtc_graph):
            self._rtc_static_output = self.policy.predict_action_rtc(
                self._rtc_static_batch, self._rtc_static_prefix
            )

        for _ in range(self.replay_warmups):
            self._rtc_graph.replay()
        torch.cuda.synchronize()
