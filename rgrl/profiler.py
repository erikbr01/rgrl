"""Always-on pipeline profiling.

Every log window reports where step time went (data / forward / backward /
optim / ema), measured with CUDA events so there are no mid-step syncs. For
deep dives, `trace_steps` captures a torch.profiler chrome trace viewable in
chrome://tracing or https://ui.perfetto.dev.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class SectionProfiler:
    """Usage:
        prof = SectionProfiler(device)
        with prof("data"): batch = ...
        with prof("forward"): loss = ...
        ...
        breakdown, timings = prof.report_and_reset()   # after torch.cuda.synchronize()
        print(breakdown)
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = enabled and device.type == "cuda"
        self._events: dict[str, list[tuple]] = defaultdict(list)
        self._wall: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._order: list[str] = []

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        if name not in self._order:
            self._order.append(name)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._wall[name] += time.perf_counter() - t0
            self._counts[name] += 1
            self._events[name].append((start, end))

    def avg_wall_ms(self, name: str) -> float:
        """Per-call CPU wall time (ms) for `name` so far, without resetting —
        call before report_and_reset() if you need both."""
        return 1e3 * self._wall[name] / max(1, self._counts[name])

    def report_and_reset(self) -> tuple[str, dict[str, float]]:
        """Call after a torch.cuda.synchronize(). Per-step averages:
        'fwd 6.1|5.9ms 47%' means 6.1ms GPU-stream time, 5.9ms CPU wall time
        spent enqueueing that section. gpu >> wall: GPU-bound (normal for
        fwd/bwd). wall ~ gpu in a section that should be instant: the CPU is
        the bottleneck there.

        Also returns {section: avg_gpu_ms} — the same numbers the string
        formats, for callers that want to log them (e.g. to wandb) rather
        than just print them."""
        if not self.enabled or not self._events:
            return "", {}
        totals = {
            name: sum(s.elapsed_time(e) for s, e in pairs) / len(pairs)
            for name, pairs in self._events.items()
        }
        grand = sum(totals.values()) or 1.0
        parts = [
            f"{name} {totals[name]:.1f}|{1e3 * self._wall[name] / self._counts[name]:.1f}ms"
            f" {100 * totals[name] / grand:.0f}%"
            for name in self._order
            if name in totals
        ]
        self._events.clear()
        self._wall.clear()
        self._counts.clear()
        return " | ".join(parts), totals


@contextmanager
def chrome_trace(out_path, active_steps: int):
    """Capture a chrome trace of `active_steps` steps. Yields the profiler;
    call .step() once per training step inside the block."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=active_steps),
        on_trace_ready=lambda p: p.export_chrome_trace(str(out_path)),
        with_stack=False,
    ) as prof:
        yield prof
