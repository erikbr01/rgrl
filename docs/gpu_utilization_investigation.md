# GPU utilization investigation — `prepare_tray_baseline_dit`

**Context.** During the `dit_flow` training run on `prepare_tray_baseline`
(dual ResNet18 image encoders + 8-block DiT, batch 256, `image_hw=[160,288]`,
pinned-CPU data fallback since the 26.35GB store doesn't fit on GPU),
`nvidia-smi` showed ~85-91% raw GPU utilization but noticeably lower
"effective" utilization (SM occupancy inside the kernels). This doc is the
record of tracking down why, without changing batch size (ruled out
up front — the known lever, but off the table for this run).

All benchmarking below ran on GPU 1, isolated from the live training job on
GPU 0, which continued unaffected throughout.

## Summary

| Hypothesis | Verdict | Magnitude |
|---|---|---|
| Pipeline/orchestration has idle bubbles | **Ruled out** | GPU-stream time already ~99% of wall-clock per step |
| `channels_last` memory format | **No benefit** | Neutral under compile, slightly worse in eager |
| `torch.compile(mode="max-autotune")` | **Not promising** | cuDNN already ~4700x faster than any Triton conv candidate; drags in CUDA graphs internally |
| CUDA graphs | **Previously rejected** (different arch) | -6%, but measured on a GPU-resident 1D-conv lowdim policy, not this run — reasoning (no idle time, AdamW `capturable` slowdown) still applies |
| Fused attention | **Already in place** | `F.scaled_dot_product_attention`, nothing to fix |
| "data" section's inflated wall-time | **Profiler artifact, not real cost** | `idx.cpu()` sync waits on unrelated queued GPU work, but that wait is already hidden behind real compute — properly-controlled fix only bought ~1% |
| H2D transfer bandwidth | **Real, but already maxed** | 12.25 GB/s achieved vs. 12.39 GB/s measured ceiling |
| CPU gather thread count | **Real headroom, doesn't matter** | 16→32 threads: 5.33→5.95 GB/s (~12%), but gather isn't the bottleneck either |
| CPU-pinned data path vs. true GPU-resident | **Real, quantified cost** | **+5.7%** throughput, isolated and controlled |
| Prefetch: background thread only | **Real but GIL-limited** | +1.2%, correctness-verified |
| Prefetch: thread + dedicated CUDA stream | **Better, still GIL-limited** | +2.4%, correctness-verified |

**Bottom line.** The 90%-raw/60%-effective gap is fundamentally per-kernel SM
occupancy — the dual ResNet18 encoders and the DiT (only 16 tokens/sequence)
don't hand each kernel launch enough parallel work at this batch size to
saturate every SM, even though the GPU is almost never idle *between*
kernels. Batch size is the only verified lever for that (established
separately, on a different architecture, but the "no idle time, GPU already
saturated" mechanism was independently reconfirmed here) and is out of scope
for this run.

Separately, and independently of the above: needing the pinned-CPU fallback
at all (because this dataset doesn't fit on GPU, unlike smaller lowdim runs)
costs a real, measured **~5.7%**, of which **~2.4%** is recoverable with a
tested, correct prefetcher. That 2.4% fix has **not been applied** — it's
real implementation work (integrating into `GpuSequenceDataset.batches()`/
`gather()`, handling epoch boundaries, the val split, and the `gpu_resident`
case correctly) for a modest gain, and the decision to build it is still
open.

## 1. The pipeline has no idle time to reclaim

Per-step profiler breakdown (GPU-stream time, i.e. actual kernel execution):

```
data 45.8ms | fwd 63.0ms | bwd 129.2ms | opt 2.1ms | ema 0.8ms   (sum ≈ 240.9ms)
```

Measured wall-clock per step at 4.1 it/s ≈ 243.9ms. That's ~99% accounted
for — no meaningful scheduling bubble between kernels for anything at the
pipeline level to close.

## 2. `channels_last` — tested, no benefit

```python
import torch
from rgrl.nets.encoders import ResNet18Encoder

device = "cuda:1"
image_hw = (160, 288)
batch = 512  # B=256 * obs_horizon=2

def bench(memory_format, compiled):
    enc = ResNet18Encoder(image_hw, num_kp=32).to(device)
    if memory_format is not None:
        enc = enc.to(memory_format=memory_format)
    fwd = torch.compile(enc) if compiled else enc
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def make_input():
        x = torch.rand(batch, 3, *image_hw, device=device)
        return x.to(memory_format=memory_format) if memory_format else x

    for _ in range(3):
        x = make_input()
        with autocast:
            out = fwd(x)
        out.sum().backward()
    torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(30):
        x = make_input()
        with autocast:
            out = fwd(x)
            loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()
    print(f"{memory_format} compiled={compiled}: {(time.time()-t0)/30*1000:.2f} ms/iter")

bench(None, False)             # contiguous, eager
bench(torch.channels_last, False)
bench(None, True)              # contiguous, compiled
bench(torch.channels_last, True)
```

Results: `165.45 / 185.22 / 84.70 / 85.18` ms/iter. `channels_last` is worse
in eager, a wash under compile — Inductor already picks its own optimal
internal layout regardless of input format for this net.

## 3. `max-autotune` — not promising

Ran the same encoder through `torch.compile(enc, mode="max-autotune")`.
Inductor's own autotune log settles the question: for the stem conv, the
native cuDNN kernel (`0.0009 ms`) beats the best Triton-generated candidate
(`4.31 ms`) by ~4700x. For matmuls, candidate configs cluster within a few
percent of each other. There's nothing better to find, and `max-autotune`
uses CUDA graphs internally (visible in a `CUDAGraphs`-related crash when a
test script omitted `zero_grad` between iterations) — which independently
brings in the next finding.

## 4. CUDA graphs — prior finding doesn't transfer directly, but the reasoning does

`rgrl/train_loop.py`'s docstring already documents: whole-step capture
(gather + fwd + bwd + fused AdamW) measured **-6%** vs. the eager loop. That
benchmark used **GPU-resident data** (no `.cpu()` sync) on a **different,
much lighter architecture** ("small conv1d kernels", a lowdim 1D-conv UNet) —
not this run's dual ResNet18 + DiT with a pinned-CPU fallback. The number
doesn't transfer cleanly across architectures. But its *reasoning* — eager's
CPU run-ahead already saturates the GPU (no idle time for graphs to
eliminate), and graph capture forces fused AdamW onto its slower
`capturable` kernel path — was independently reconfirmed for *this* run in
§1 above (99% GPU-stream utilization, no idle bubbles).

## 5. The "data" section's wall-time is a profiler attribution artifact

`GpuSequenceDataset.take()`'s pinned-CPU fallback does:

```python
out = src[idx.cpu().reshape(-1)].to(self.device, non_blocking=True)
```

`idx.cpu()` has no `non_blocking` argument — it's a blocking D2H copy.
Reproduction of what that actually costs:

```python
import torch, time
torch.cuda.set_device(1)
device = torch.device("cuda:1")

def queue_busy_work():
    a = torch.randn(8192, 8192, device=device)
    b = torch.randn(8192, 8192, device=device)
    for _ in range(40):
        a = a @ b
    return a

idx = torch.randint(0, 1000, (512,), device=device)

torch.cuda.synchronize(device)
t0 = time.time(); idx.cpu(); print(f"idle: {(time.time()-t0)*1000:.2f} ms")

torch.cuda.synchronize(device)
queue_busy_work()
t0 = time.time(); idx.cpu(); print(f"after queuing work: {(time.time()-t0)*1000:.2f} ms")
```

Result: **0.11 ms idle vs. 721 ms with unrelated queued GPU work** — `.cpu()`
without `non_blocking` waits for the *entire* stream backlog, not just its
own producer op. This is why the profiler shows the "data" section's wall
time (~175-180ms) far exceeding its own GPU-stream time (~44-46ms): the CPU
thread is blocked catching up to the *previous* step's still-draining
fwd/bwd, and that wait gets attributed to "data".

**But it doesn't cost real throughput.** A properly-controlled A/B test with
a realistic backward pass in the loop (not just discarded busy-work) showed
only **+1.0%** difference between the current blocking call and a
non-blocking version — most of that wait is already hidden behind real GPU
compute, not creating idle GPU time. (A `.item()`-forced-read variant of the
non-blocking fix was tested and passed a 200-trial correctness stress test,
but was **not shipped** — no documented PyTorch guarantee was found for why
it's safe, only strong empirical support, and 1% wasn't worth the residual
uncertainty on a live multi-day run.)

## 6. H2D bandwidth — real limit, already at the ceiling

```python
import torch, time
torch.cuda.set_device(1)
device = torch.device("cuda:1")

size_mb = 512
x = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8).pin_memory()
for _ in range(3):
    x.to(device, non_blocking=True)
torch.cuda.synchronize(device)
t0 = time.time()
for _ in range(10):
    x.to(device, non_blocking=True)
torch.cuda.synchronize(device)
dt = (time.time() - t0) / 10
print(f"sequential H2D: {size_mb/1024/dt:.2f} GB/s")
```

Result: **12.39 GB/s** sequential ceiling, vs. **12.25 GB/s** achieved by the
real random-gather H2D copy — already at the hardware limit (consistent with
a PCIe Gen3 x16-ish link). Not a lever.

## 7. CPU gather thread count — real headroom, doesn't matter

```python
import torch, time
N_FRAMES, H, W, C = 76232, 160, 288, 3
src = torch.randint(0, 255, (N_FRAMES, H, W, C), dtype=torch.uint8).pin_memory()
idx = torch.randint(0, N_FRAMES, (512,))

for threads in (1, 2, 4, 8, 16, 32):
    torch.set_num_threads(threads)
    for _ in range(3):
        src[idx]
    t0 = time.time()
    for _ in range(30):
        src[idx]
    dt = (time.time() - t0) / 30
    nbytes = 512 * H * W * C
    print(f"threads={threads:3d}: {nbytes/1e9/dt:.2f} GB/s")
```

Result: `1.31 / 2.25 / 3.51 / 4.93 / 5.33 / 5.95` GB/s for 1→32 threads. The
default (16 threads) leaves ~12% on the table vs. using all 32 cores — a
real, previously-unverified finding — but irrelevant here since the gather
isn't the throughput bottleneck (see §5, §8).

## 8. CPU-pinned data path vs. true GPU-resident — the real, quantified cost

Isolates *only* the data-placement variable: same realistic compute load
(includes a real backward pass), same batch shape, only varying whether
data needs to move from CPU or is already resident on GPU.

```python
import torch, time

torch.cuda.set_device(1)
device = torch.device("cuda:1")
BATCH, OBS_HORIZON, H, W, C = 256, 2, 160, 288, 3

N_FRAMES_CPU = 76232
src_cpu = torch.randint(0, 255, (N_FRAMES_CPU, H, W, C), dtype=torch.uint8).pin_memory()
N_FRAMES_GPU = 4000  # smaller so it fits alongside the compute tensors -- isolates "needs H2D" from "store size"
src_gpu = torch.randint(0, 255, (N_FRAMES_GPU, H, W, C), dtype=torch.uint8, device=device)

obs_idx_cpu_src = torch.randint(0, N_FRAMES_CPU, (BATCH, OBS_HORIZON), device=device)
obs_idx_gpu_src = torch.randint(0, N_FRAMES_GPU, (BATCH, OBS_HORIZON), device=device)

def compute_step(batch_gpu):
    a = torch.randn(8192, 8192, device=device, requires_grad=True)
    b = torch.randn(8192, 8192, device=device)
    x = a
    for _ in range(11):
        x = x @ b
    (x.sum() + batch_gpu.float().sum() * 0).backward()
    return x.detach()

def gather_pinned_cpu(sample_ids):
    out = None
    for _ in range(2):  # 2 camera keys
        idx = obs_idx_cpu_src[sample_ids].reshape(-1)
        out = src_cpu[idx.cpu()].to(device, non_blocking=True)
    return out

def gather_gpu_resident(sample_ids):
    out = None
    for _ in range(2):
        idx = obs_idx_gpu_src[sample_ids].reshape(-1)
        out = src_gpu[idx]
    return out

def run_loop(gather_fn, label):
    sample_ids = torch.randint(0, BATCH, (BATCH,), device=device)
    for _ in range(3):
        compute_step(gather_fn(sample_ids))
    torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(20):
        compute_step(gather_fn(sample_ids))
    torch.cuda.synchronize(device)
    dt = (time.time() - t0) / 20
    print(f"{label}: {dt*1000:.2f} ms/iter")
    return dt

dt_pinned = run_loop(gather_pinned_cpu, "CPU-pinned")
dt_gpu = run_loop(gather_gpu_resident, "GPU-resident")
print(f"difference: {100*(dt_pinned-dt_gpu)/dt_pinned:+.1f}%")
```

Result: **477.01ms vs. 449.74ms → +5.7%**. This is the real, unavoidable
(given the dataset's size relative to GPU memory) cost of the pinned-CPU
fallback — genuinely a bandwidth/data-movement cost, not an artifact.

## 9. Prefetching — the promising, not-yet-shipped fix

`policy_graphs/third_party/diffusion_policy`'s
`train_diffusion_unet_hybrid_workspace.py` already has a proven pattern for
this (`DataPrefetcher`, lines 39-68): a dedicated `torch.cuda.Stream` +
`wait_stream` + `record_stream`, overlapping batch N+1's H2D transfer with
batch N's compute. It assumes a `DataLoader` with worker processes already
did the CPU-side gather, though — `rgrl`'s `GpuSequenceDataset` has no
DataLoader/worker-process model at all, so the CPU-bound gather itself also
needs to move off the main thread.

**Attempt 1 — background thread only** (no dedicated stream):

```python
from concurrent.futures import ThreadPoolExecutor

class Prefetcher:
    def __init__(self, gather_fn, sample_ids):
        self._gather_fn, self._sample_ids = gather_fn, sample_ids
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = self._executor.submit(self._gather_fn, self._sample_ids)

    def next_batch(self):
        batch = self._future.result()
        self._future = self._executor.submit(self._gather_fn, self._sample_ids)
        return batch
```

Result: **+1.2%**, correctness-verified (0/100 stress test with a known
index pattern). Smaller than hoped — the background thread's H2D copy
wasn't assigned to a dedicated stream, so it likely serialized into the
same default stream as compute, giving away most of the overlap.

**Attempt 2 — background thread + dedicated CUDA stream** (combining both
patterns):

```python
from concurrent.futures import ThreadPoolExecutor
import torch

def gather_cpu_only(sample_ids, obs_idx_src, src_cpu):
    """CPU-bound portion only -- runs in the background thread."""
    idx = obs_idx_src[sample_ids].reshape(-1)
    return src_cpu[idx.cpu()]  # stays on CPU (pinned)

class StreamPrefetcher:
    """Background thread does the CPU gather; the H2D copy for the result
    is issued on a dedicated CUDA stream (mirrors diffusion_policy's
    DataPrefetcher) so it doesn't serialize into the main compute stream."""

    def __init__(self, sample_ids, gather_cpu_fn, device):
        self.sample_ids = sample_ids
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._gather_cpu_fn = gather_cpu_fn
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._gpu_batch = None
        self._kick_off_cpu_gather()

    def _kick_off_cpu_gather(self):
        self._cpu_future = self._executor.submit(self._gather_cpu_fn, self.sample_ids)

    def _stage_h2d(self):
        cpu_batch = self._cpu_future.result()  # blocks only if not ready yet
        with torch.cuda.stream(self.stream):
            self._gpu_batch = cpu_batch.to(self.device, non_blocking=True)
        self._kick_off_cpu_gather()

    def next_batch(self):
        if self._gpu_batch is None:
            self._stage_h2d()
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._gpu_batch
        batch.record_stream(torch.cuda.current_stream())  # required: tells the
        # allocator this stream now depends on the buffer too, so it won't be
        # freed/reused while still in flight from the prefetch stream's copy.
        self._gpu_batch = None
        self._stage_h2d()
        return batch
```

Result: **+2.4%**, correctness-verified (0/100 stress test). Roughly double
the thread-only version — about 42% of the theoretical 5.7% ceiling. The
remainder is most likely still GIL contention on the CPU-bound gather itself
(a background *thread* still competes for the GIL with the main thread's
Python-level orchestration; only true multiprocessing — like the reference's
actual `DataLoader` workers — would close that further, at meaningfully
higher implementation cost).

**Status: not implemented in `rgrl/dataset.py`.** This is a real,
correctness-verified 2.4% win, but integrating it into
`GpuSequenceDataset.batches()`/`gather()` properly (handling epoch
boundaries, the val split, and correctly no-op'ing for the `gpu_resident`
case) is nontrivial implementation work on a codebase that deliberately
keeps this file simple. Decision to build it is still open.

## What was *not* tried

- **Multiprocessing-based prefetch** (true parallelism for the CPU-bound
  gather, no GIL contention) — would likely close more of the remaining gap
  after §9's stream-based fix, but is a bigger change (inter-process
  shared-memory handoff, worker lifecycle, cleanup on crash/restart) and
  wasn't built or measured.
- **`py-spy` sampling profiler** on the live process — blocked by missing
  passwordless sudo; worked around via targeted micro-benchmarks instead.
