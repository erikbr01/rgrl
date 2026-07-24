# Architecture and training-throughput review and implementation report

Date: 2026-07-23
Reviewed branch: `weighted-sampling`, after merging `origin/main` at `93675bc`
Merge commit: `dfd184a`

## Implementation update

This document began as the pre-change repository audit. The detailed sections
below preserve the reasoning and original failure modes; this section records
what was implemented and measured afterward. Measurements used one NVIDIA RTX
4090, PyTorch 2.11.0+cu128, and the working tree based on `dfd184a`.
The `robo_dataset` work is captured in local submodule commit `64bb0a5`
(based on `923891d`).

### What changed

| Area | Implemented outcome |
|---|---|
| Experiment state | Policy initialization is seeded at the application boundary. Checkpoints now contain model, EMA, optimizer, scheduler, all RNG states, exact batch cursor, sample count, W&B identity, schema version, and a compatibility fingerprint. |
| Checkpoint safety | A checkpoint is serialized once, fsynced, atomically renamed, and published through an atomically replaced relative `latest.pt` symlink. Completed/lower-target runs cannot overwrite progress; extensions are explicit. |
| Legacy inference | Pre-shape-meta diffusion and flow checkpoints load through an isolated compatibility module that requires and validates the original store sidecars. It never guesses image modalities from dtype. |
| Validation | Validation uses an isolated fixed RNG stream, pools a true element-weighted mean (accumulating the squared-error sum and valid-element count, so the variable prefix denominator and a short final batch are weighted correctly), validates split sizes, and does not change subsequent training updates. |
| Recipe semantics | Training length, warmup, logging, validation, and checkpoint cadence are all expressed in optimizer steps (a sample-budget front-end was tried and dropped as redundant; `batch_size` sets data-per-step). The big-batch recipe uses 1/4 the steps of the bs=256 default so both see the same data. |
| Sampling | Uniform/weighted sampling is a separate Hydra group. Resident and shared-memory modes use the same deterministic epoch order and resume from the exact next batch. |
| Episode subsets | DROID subsets are generative `(task, seed, count)` manifests resolved against the store's `tasks.json` (not hand-listed IDs); same-task selections continue one seeded draw without replacement, and success curation lives in the store's `curation.json`. An explicit-`episode_ids` escape hatch covers externally-curated sets. `EpisodeView` maps the resolved logical episodes to the original physical store without copying or re-encoding videos. |
| Conversion cache | Cache identity uses an explicit schema, source metadata fingerprint, the resolved episode IDs (not the manifest's authoring form, so editing provenance reuses the cache), and target schema. Per-key locks, temporary siblings, validation, directory fsync, and atomic publish prevent partial caches. |
| Data placement | `auto`, `cuda_resident`, and `shared_memory` modes are explicit. Auto placement considers current free VRAM, data bytes, estimated model/optimizer/EMA state, and a configured reserve, then logs its decision. |
| CPU/shared-memory path | Window tables and batch gathers are vectorized. PyTorch's batched `__getitems__` avoids per-sample dictionaries/default collation. A producer thread overlaps the next pinned host gather with side-stream H2D, with allocator-safe recursive `record_stream`. |
| Policy boundary | The trainer calls `policy.loss(raw_batch)`. Policies own image preprocessing for both training and inference, so preprocessing is included in the compiled loss boundary. |
| Image work | The common native-image path crops uint8 NHWC before float/NCHW expansion. An opt-in shared camera ResNet concatenates camera batches into one forward. Key-sanitization collisions now fail fast and camera transform order is deterministic. |
| Flow matching | Rectified flow is the default training policy. DROID launchers use DiT flow matching. Diffusion remains supported for old checkpoints and controlled comparisons. |
| Robosuite integration | Simulation now uses modern Robosuite/MuJoCo directly. The legacy Robomimic runtime dependency and its `mujoco_py` import workaround were removed. A direct, atomic raw-state-to-image renderer preserves Robomimic-compatible HDF5 metadata and filters subset masks to episodes actually emitted. |
| Launch/config hygiene | Five DROID scripts now call one generic flow-matching launcher. Generic inference defaults no longer carry a diffusion-specific name. Development dependencies and Ruff correctness checks were added (run locally; the GitHub CI workflow was intentionally removed as unnecessary for this repo). |

### Attributed performance results

These are isolated measurements, not additive speedup claims:

| Change / comparison | Before | After | Effect |
|---|---:|---:|---:|
| ToolHang state flow, physical batch | 23.4k samples/s at 256 | 42.3k samples/s at 1024 | **1.81×** samples/s |
| Section profiler, ToolHang state flow | 23.4k samples/s enabled | 23.4k samples/s disabled | No measurable steady-state cost |
| CPU ToolHang batch construction | 81.5k samples/s, per-sample + default collate | 2.974M samples/s, batched gather | **36.5×** batch-source throughput; the model remains the end-to-end limit |
| DROID-shaped 3-camera preprocessing, B=64, T=2, 180×320→76×76 | 1.658 ms, float full frame then crop | 0.222 ms, uint8 crop then float | **7.5×** transform speed; peak extra allocation 337.5→39.6 MiB |
| Real DROID compiled DiT flow, B=64, shared host data | 2.6k samples/s, synchronous host gather | 4.0k samples/s, producer-thread overlap | **1.52×** end-to-end; data wait fell from about 34% to effectively zero GPU-stream time |
| Same DROID run with two DataLoader workers | 2.4k samples/s | 4.0k with zero workers + producer thread | Worker IPC was slower for these 66 MB raw image batches |
| DROID observation encoder, B=32, T=2, 3 cameras | 33.58M params, 11.61 ms fwd+bwd | 11.19M params, 8.62 ms fwd+bwd | **66.7%** fewer encoder params and **25.8%** faster encoder step; modeling change |
| ToolHang compiled B=1 inference | DDIM-10: 12.847 ms; DDIM-16: 20.508 ms | Flow Euler-10: 12.602 ms | Flow is 1.9% faster at equal network calls and 38.5% faster than the 16-step preset |
| ToolHang image flow, B=64, two 84×84 cameras | Compile-affected first 100-step window: 0.4k samples/s | Steady state: 3.8k samples/s | Data wait was effectively zero; model fwd/bwd dominated |
| ToolHang image flow, physical batch | 3.8k samples/s at 64 | 6.4–6.5k samples/s at 256 | **1.71×** samples/s; full 7.68M-sample run reached 80% success |
| ToolHang image flow, batch 256→512 probe | 6.5k samples/s | 7.1k samples/s | Only **1.09×** further; the short equal-sample probe learned materially more slowly, so batch 512 was not promoted |
| Flow Euler timestep preallocation, compiled B=1 | 11.520 ms/call with `torch.full` | 11.528 ms/call with a saved buffer | No measurable effect; extra machinery was removed |

The real DROID set-table selection is 23.79 GB. An intentional
`cuda_resident` test exhausted a 24 GB card at the first Adam update. The same
configuration under `data_mode=auto` kept the 23.12 GB training split in
shared host memory, placed the 0.66 GB validation split on GPU, and completed
DiT flow training plus full validation. A separate compiled 200-step run
completed against all 45,879 logical frames.

Logical subset cache creation was also tested against the actual DROID source:
the 120-episode baseline resolved to 45,879 frames and the 180-episode
plus-left view to 70,128 frames. Each new cache contains about 24 KB of
metadata/symlinks, rather than copied encoded media. `rgrl.data` resolves the
generative manifests directly: every set-table selection reproduces its exact
episode set from source `tasks.json` using seed 42 and the store's
`curation.json` success swaps (regression-tested in `tests/test_data.py`).

### End-to-end Robosuite integration

Two real ToolHang PH paths were exercised through the public entry points:

- **State policy:** compiled rectified-flow training completed the full
  7.68M-sample / 7,500-step batch-1024 recipe, including 15 periodic
  validations and three atomic checkpoints. Steady state was 42.2–42.3k
  samples/s. The final EMA checkpoint then achieved **45/50 successes (90%)**
  in the simulator. Repeating all 50 episodes through the new direct
  Robosuite adapter produced the same 45/50 result.
- **Image policy:** `tools/render_robosuite_dataset.py` rendered all 200 real
  PH episodes (95,962 frames) from raw simulator states using `sideview` and
  `robot0_eye_in_hand` at 84×84. The non-privileged target uses those cameras
  plus proprioception and excludes `object`. A compiled rectified-flow run
  completed 7.68M samples / 30,000 batch-256 updates, including conversion,
  GPU-resident loading, fixed-noise validation, three atomic checkpoints,
  checkpoint reload, and 50 live image rollouts. Steady state was 6.4–6.5k
  samples/s. Success on the same first 20 seeds rose from **8/20 (40%)** at
  2.56M samples to **15/20 (75%)** at 5.12M and **16/20 (80%)** at 7.68M;
  the final 50-episode result was **40/50 (80%)**. The earlier 19,200-sample
  subset smoke test's 0/5 was therefore undertraining, not evidence of a
  broken image pipeline.

The direct renderer was checked against the old wrapper output on all 681
frames of one episode. Both camera tensors were byte-identical; low-dimensional
values agreed within 5.93e-8 (the direct tool stores float32). After dependency
resynchronization, neither `robomimic` nor `mujoco_py` was installed, and a
live rollout still completed against Robosuite 1.4.1 + MuJoCo 2.3.7.

### Image-pipeline correctness audit

End-to-end execution alone does not establish semantic correctness. The image
path was therefore checked at each boundary:

| Boundary | Check | Result |
|---|---|---|
| Raw state → rendered HDF5 | Replayed 15 recorded states spanning demos 0, 7, and 15 through modern Robosuite | Both camera tensors and every requested low-dimensional scalar were exact |
| Rendered HDF5 → training cache | Exhaustively compared all six policy keys in all 200 episodes | **95,962/95,962 frames exact**; zero unequal camera, proprioception, or action scalars |
| Episode windows | Checked start/interior/end anchors in every subset episode and added a synthetic image regression test | Observation is `[t-1,t]`; prediction slot `obs_horizon-1` is action `t`; padding never crosses episodes; uint8 NHWC is preserved |
| Image preprocessing | Compared actual ToolHang pixels against the former float-first pipeline and explicit center crop | In eager mode, seeded training random crops and evaluation center crops were bit-exact; output is float NCHW in `[0,1]`. Under the default `torch.compile`, the crop's RNG now runs inside the compiled loss and is not bit-identical to the former eager crop (it is still a valid, run-to-run-deterministic seeded crop) |
| Camera convention | Compared the direct renderer to the former official wrapper on a complete 681-frame episode | Camera bytes were identical, including vertical orientation |
| Shared inference stack | Loaded the known-good legacy Lift diffusion image checkpoint through the current loader and direct simulator adapter | **49/50 (98%)**, matching its historical result |
| Current flow + image stack | Trained a fresh 1.28M-sample Lift FlowPolicy and evaluated its EMA checkpoint | **50/50 (100%)** |
| ToolHang task-specific behavior | Matched-seed checkpoints at 2.56M, 5.12M, and 7.68M samples | **40% → 75% → 80%** on the common first 20 seeds; final **80% over 50** |

The audit found one real regression: old checkpoints predated the self-contained
`target` schema and failed with `KeyError: target`. The compatibility path now
recognizes only the old diffusion/flow format, requires `meta.json`,
`rgrl_meta.json`, and `stats.json`, reconstructs the exact historical crop,
and rejects stores without an explicit `image_keys` declaration. Current
checkpoints remain fully self-contained and do not open training data.

Two interpretation limits matter:

- The reference Diffusion Policy ToolHang recipe renders 240×240 and crops to
  116×116; this throughput-oriented experiment renders 84×84 and crops to
  76×76. Reaching 80% shows that 84×84 is viable, not that it is quality-optimal.
  A matched 84-vs-240 resolution ablation is the next image-quality experiment.
- Fixed-noise flow validation loss was not behaviorally calibrated: it rose
  late in both image runs even though Lift achieved 100% and ToolHang improved
  from 40% to 80%. It is useful for deterministic numerical comparisons, but
  rollout success remains the model-selection metric.

### Validation status and remaining limits

- `uv run --extra dev ruff check .`: passed.
- `CUDA_VISIBLE_DEVICES=0 uv run --all-extras pytest tests -q`: **88 passed**,
  including CUDA prefetch lifetime stress, bit-exact interruption/resume,
  validation RNG isolation, logical view/cache failure recovery, weighted
  sampling, crop equivalence, image/action temporal alignment, strict legacy
  checkpoint migration, and shared-encoder compatibility tests.
- `CUDA_VISIBLE_DEVICES=0 uv run --all-extras pytest
  third_party/robo_dataset/tests -q`: **202 passed, 15 skipped**, including
  direct `EpisodeView` contract/mapping tests.
- Weighted flow matching completed a 300-step ToolHang train/validation smoke
  with ESS logging and an atomic resumable checkpoint.
- Real DROID eager and compiled DiT-flow training completed. DROID robot
  rollout was intentionally not attempted on this machine.

The shared camera encoder has only throughput/shape validation here; it still
requires a DROID task-quality ablation before deployment. Likewise, the
historical batch-1024 rollout result is encouraging but not a replacement for
multi-seed evaluation. DDP, compact per-batch window descriptors, streaming
normalizer fitting, and stratified manifest-owned train/validation splits
remain worthwhile follow-ups.

### Reproducing the measurements

The isolated batch/preprocessing comparisons are committed as a benchmark
rather than encoded as flaky unit-test thresholds:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tools/benchmark_hotpaths.py \
  --benchmark all --iterations 300 --out hotpaths.json
```

The ToolHang image source can be reproduced without Robomimic or `mujoco_py`:

```bash
MUJOCO_GL=egl uv run --extra eval python tools/render_robosuite_dataset.py \
  data/robomimic/tool_hang/ph/demo_v141.hdf5 \
  data/robomimic/tool_hang/ph/image_84.hdf5 \
  --camera-names sideview robot0_eye_in_hand --height 84 --width 84
```

The steady-state ToolHang comparison uses the same compiled flow policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  policy=flow data=tool_hang_state target=tool_hang_state \
  training.steps=1200 \
  training.val_episodes=0 training.log_every=200 training.save_every=1200 \
  training.profile=false training.resume=never out=/tmp/toolhang_flow_bs256

CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  policy=flow data=tool_hang_state target=tool_hang_state training=bs1024 \
  training.steps=1200 training.val_episodes=0 \
  training.log_every=200 training.save_every=1200 \
  training.profile=false training.resume=never out=/tmp/toolhang_flow_bs1024
```

For DROID, `scripts/train_droid_flow.sh` owns the production-shaped recipe.
Set `DROID_SHARE_IMAGE_ENCODER=true` only for the explicitly opt-in shared
backbone ablation.

## Pre-implementation executive summary

The remaining review records the repository state before the implementation
above. It is retained because it explains the design choices and distinguishes
measured bottlenecks from attractive but previously disproven optimizations.

This repository has a good performance-oriented core. It is small enough to
understand, the GPU-resident fast path is genuinely effective, and the code
already uses the important PyTorch primitives: BF16, `torch.compile`, fused
AdamW, foreach EMA, TF32, and scaled-dot-product attention. The existing GPU
investigation is unusually useful because it rules out several attractive but
unproductive optimization directions.

The highest-value work is:

1. **Make experiments trustworthy before optimizing further.** Seeding happens
   after policy construction, resume omits optimizer/scheduler/RNG/sampler
   state, validation changes the training RNG stream, and a completed run can
   overwrite its latest checkpoint with a lower configured step. These can
   silently change or corrupt experiments.
2. **Promote large-batch training to a first-class recipe.** A workspace run
   increased throughput from about 22.6k to 38.7k samples/s at batch 1024,
   about **1.71×**, while holding the sample budget constant. Its 50-episode
   success rate was 0.94 versus 0.86 for batch 256. The evaluation is too small
   to establish a quality improvement, but it shows no obvious regression.
3. **Represent episode subsets as logical views and versioned selection
   manifests, not copied stores and hundreds of YAML integers.** The current
   set-table selections are internally consistent, but every subset can
   re-encode and duplicate videos. A view can expose selected episodes with
   contiguous logical indexing while reading the original store in place.
4. **Consolidate the duplicate data stacks.** `rgrl/dataset.py` reimplements
   responsibilities that already exist in the `robo_dataset` submodule:
   sequence sampling, shared-memory storage, dataset access, and asynchronous
   prefetch. `robo_dataset` should own storage, views, sampling, and transfer;
   `rgrl` should own its batch contract, policies, and trainer.
5. **For image training, optimize the actual limiting work.** The recorded
   profile shows an almost continuously busy GPU with insufficient per-kernel
   occupancy, not orchestration gaps. The promising levers are larger effective
   kernels/batches, cheaper or shared visual encoders, and fused/reordered image
   preprocessing. Channels-last, max-autotune, PCIe tuning, and whole-step CUDA
   graphs have already been measured as neutral or worse.

## Review scope and verification

I reviewed the training entry point and loop, data conversion/cache layer,
window datasets and sampling, all three policy families, transforms and
encoders, checkpoint loading, profiling, Hydra configurations, launch scripts,
tests, the `robo_dataset` submodule, existing profiler notes, and representative
run artifacts.

Validation performed:

- Merged `origin/main` into the working branch and resolved its overlap with
  weighted sampling.
- Ran `uv run --with pytest pytest tests/ -q`: **62 tests passed**.
- Audited the configured episode selections:
  - `prepare_tray_baseline`: 120 unique, sorted episode IDs.
  - `set_table_baseline`: 120 unique, sorted episode IDs.
  - Each set-table follow-up selection: 180 unique, sorted IDs.
  - Each follow-up is exactly the 120-episode baseline plus 60 episodes.
  - The three 60-episode additions are pairwise disjoint.
- Compared the recorded batch-256 and batch-1024 tool-hang runs and the prior
  GPU-utilization investigation.

This was a review, not a performance rerun. Statements below are labeled as
measured when backed by artifacts in this workspace; other speedups are
hypotheses that should be benchmarked.

## What is already strong

- The fast path in [`rgrl/dataset.py`](../rgrl/dataset.py) keeps a split on the
  accelerator and gathers a full batch without a per-sample `DataLoader`.
  Low-dimensional run logs show effectively zero data time.
- [`rgrl/train_loop.py`](../rgrl/train_loop.py) is compact and uses the right
  high-impact primitives rather than introducing a large training framework.
- Normalization state is stored as module buffers, so inference checkpoints do
  not require a separate normalizer artifact.
- The policy interface is small and the diffusion/flow implementations are
  readable.
- Profiling is treated as an engineering feature rather than an afterthought.
  [`docs/gpu_utilization_investigation.md`](gpu_utilization_investigation.md)
  contains controlled measurements and correctly distinguishes wall-time
  attribution from GPU-stream time.
- Data conversion is cached and its intended schema is explicit.
- The code has focused unit tests, and all of them pass after the merge.

The goal should be to preserve this small, legible core while making ownership
boundaries and experiment state more rigorous.

## Priority map

| Priority | Recommendation | Expected impact | Confidence | Effort |
|---|---|---:|---:|---:|
| P0 | Seed before constructing the policy | Experiment correctness | High | Small |
| P0 | Make checkpoint/resume complete and atomic | Correctness and recoverability | High | Medium |
| P0 | Prevent completed-run checkpoint regression | Prevents silent corruption | High | Small |
| P0 | Isolate validation RNG and validate nonempty splits | Correctness/reproducibility | High | Small |
| P1 | Make sample budget, not steps, the primary recipe unit | Cleaner scaling across batch/subsets | High | Small–medium |
| P1 | Promote and sweep larger batches | **1.71× measured** on one lowdim recipe | High for that recipe | Small |
| P1 | Add logical episode views and selection manifests | Eliminates subset copies; much cleaner data model | High | Medium |
| P1 | Consolidate sampling/prefetch into `robo_dataset` | Less duplicate code; better CPU path | High | Medium–large |
| P1 | Vectorize CPU batch gathering and overlap H2D | Up to a few percent on large image data | Medium | Medium |
| P1 | Move preprocessing inside the compiled policy loss | Fewer launches/allocations; cleaner API | Medium | Medium |
| P2 | Optional shared camera encoder/batched camera forward | Potentially material image-training gain | Medium | Medium |
| P2 | Make profiling sampled/optional in production | Small hot-loop saving | Medium | Small |
| P2 | Generic launcher and compositional experiment configs | Maintainability | High | Medium |
| P3 | DDP for aggregate throughput | Multi-GPU scaling | Medium | Large |

## P0: fix experiment correctness first

### 1. The configured seed does not seed policy initialization

[`train.py`](../train.py#L22) loads the store and constructs the policy before
[`rgrl/train_loop.py`](../rgrl/train_loop.py#L119) calls
`torch.manual_seed(cfg.seed)`. The seed controls later sampling but not initial
weights. Two fresh invocations with the same config can therefore start from
different models.

Seed Python, NumPy, Torch CPU, and CUDA at the application boundary before any
randomized object is constructed. A small `seed_everything(seed)` bootstrap
called before `instantiate(cfg.policy, ...)` is enough; the trainer should not
be the first owner of global initialization.

### 2. Resume is weight loading, not training-state resume

The checkpoint currently saves model, EMA, config, and step
([`rgrl/train_loop.py`](../rgrl/train_loop.py#L149)). It does not save:

- optimizer state, including Adam moments;
- scheduler state;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- data-sampler generator and exact sample cursor;
- AMP/scaler state if FP16 is ever added;
- W&B run identity.

The CPU `DataLoader` path explicitly ignores `start_epoch`
([`rgrl/dataset.py`](../rgrl/dataset.py#L276)). The GPU path resumes at an
estimated epoch boundary rather than the batch position inside that epoch.
Consequently, the current "reproducible and resumable" claim is too strong.

Introduce an explicit `TrainingState`:

```text
step, samples_seen
model, ema
optimizer, scheduler
rng: python, numpy, torch_cpu, torch_cuda[]
sampler: epoch, offset, generator_state
run_id, config_fingerprint, checkpoint_schema_version
```

Restore it before creating the next iterator. Add `resume: never | auto |
required` so a run cannot silently inherit state just because its output
directory exists.

### 3. A completed run can regress and overwrite `latest.pt`

If a loaded checkpoint's `start_step` is already greater than or equal to
`cfg.steps`, the training loop is skipped, but the unconditional final
`save(cfg.steps)` still runs
([`rgrl/train_loop.py`](../rgrl/train_loop.py#L302)). Reusing an output
directory after lowering the configured step count can replace `latest.pt`
with a lower step label.

This is particularly plausible here because scripts on `main` changed some
recipes from 500k to 200k steps while keeping their run directory names.

On resume:

- fail clearly if `target_steps < checkpoint.step`;
- return without writing if they are equal;
- require an explicit "extend" mode for a larger target;
- compare a saved config fingerprint and reject incompatible data, model, or
  optimizer changes.

### 4. Checkpoint writes are non-atomic and serialized twice

`latest.pt` is overwritten directly and a kept checkpoint serializes the same
payload again. A crash during the first write can destroy the recovery point.

Write the payload once to a temporary file in the output directory, flush and
atomically rename it to `checkpoint_<step>.pt`, then atomically update
`latest.pt` as a symlink or hard link. Keep at least one previous valid
checkpoint. This both removes duplicate serialization/I/O and makes the
watchdog strategy reliable.

### 5. Validation changes subsequent training randomness

Every policy loss samples random noise and times; see, for example,
[`rgrl/policies/diffusion.py`](../rgrl/policies/diffusion.py#L116) and
[`rgrl/policies/dit_flow.py`](../rgrl/policies/dit_flow.py#L86). Validation
uses the same global Torch RNG stream. Changing `val_every`, validation batch
size, or validation-set size therefore changes all later training noise.

Use a separate validation generator or run validation under `fork_rng` with a
fixed validation seed and restore the training RNG state afterward. For useful
comparisons, keep validation noise fixed across checkpoints.

The reported validation loss also averages batch means, giving a small final
batch the same weight as a full batch. Accumulate a loss sum and sample/element
count instead.

### 6. Validate tiny and empty datasets

Both batch iterators are infinite outer loops over a drop-last inner loop. If a
train split has fewer samples than `batch_size`, they yield nothing forever.
Likewise, `val_episodes >= num_episodes` can empty the train split.

Validate at construction:

- `0 <= val_episodes < num_episodes`;
- both requested splits contain samples;
- either `num_samples >= batch_size` or replacement/small-batch behavior is
  explicitly enabled;
- episode IDs are nonempty, unique integers in range.

Also guard `torch.cuda.synchronize()` by device type so CPU training can log.

### 7. Trace mode is incompatible with resume

Trace mode loops from step zero regardless of `start_step`
([`rgrl/train_loop.py`](../rgrl/train_loop.py#L243)), advancing optimizer,
scheduler, and EMA with incorrect step values. Make tracing cover
`range(start_step, start_step + warmup + active)` and decide explicitly whether
profile warmup steps count as training. The simplest semantics are that every
optimizer update increments the single global step.

## A cleaner target architecture

The repository currently has blurred ownership between `rgrl` and its
`robo_dataset` dependency. The following split keeps the application small:

```text
DatasetSource (HDF5 / Zarr / Video)
    |
    v
EpisodeView (selection manifest, logical episode mapping)
    |
    v
SequenceSampler (windows, padding, split and sample policy)
    |
    +--> CudaResidentBatchSource
    |
    +--> SharedMemory/StreamingBatchSource --> async device prefetch
                                              |
                                              v
Policy.loss(raw_batch)
    preprocess -> encode -> objective
                                              |
                                              v
Trainer (state machine, optimizer, validation, checkpoint manager)
```

Recommended ownership:

- **`robo_dataset`:** physical stores, logical episode views, sequence/window
  indexing, split masks, sampling, shared/memory-mapped storage, `DataLoader`
  integration, and accelerator prefetch.
- **`rgrl`:** target schema adapter, fitted-normalizer modules, policy
  preprocessing/loss/inference, training state, evaluation, and experiment
  configuration.

`rgrl.store.EpisodeStore` and `robo_dataset`'s store terminology are easy to
confuse. Rename the `rgrl` wrapper to something that states its extra role,
such as `TrainingStore` or `DatasetSpec`, if the compatibility cost is
acceptable.

## Clean episode subsets without copied data

### Current design

`episode_ids` is embedded directly in data YAML. For native stores,
[`rgrl/data.py`](../rgrl/data.py#L281) writes a new contiguous store and
re-encodes videos because physical store episode numbering is contiguous.
This gives reproducible IDs, which is good, but it duplicates overlapping
subsets and makes selection provenance a comment plus a long list.

The baseline and three 180-episode set-table supersets can therefore store the
same image bytes several times. It also makes changing a selection a conversion
operation rather than a metadata operation.

### Proposed design: `EpisodeView`

Add a read-only store adapter in `robo_dataset`:

```python
view = EpisodeView(
    source=store,
    episode_ids=[579, 582, ...],
)
```

It should:

- expose `n_episodes == len(episode_ids)`;
- expose logical cumulative `episode_ends`;
- map logical episode/frame ranges to source episode/frame ranges;
- forward image decode and state reads to the source store;
- preserve the configured order without requiring contiguous physical IDs;
- compute or cache normalization statistics for the selected logical view;
- expose a stable fingerprint derived from the source dataset ID and selection
  manifest hash.

The training sampler only needs logical episode boundaries and range reads; it
does not require the selected episodes to be physically rewritten. For a
GPU-resident load, selected episodes can be concatenated once in memory. For
the CPU/streaming path, reads remain against the original store.

### Generative selections (implemented)

The long integer lists — and the separate `configs/selections/*.json` files —
are gone. A data config carries an inline `selection:` block, a generative
recipe `rgrl/data.py` resolves at load time against the store's `tasks.json`:

```yaml
# configs/data/set_table_plusright.yaml
selection:
  seed: 42
  select:
    - {task: "prepare plate - milk right",  count: 60}
    - {task: "prepare plate - milk middle", count: 60}
    - {task: "prepare plate - milk right",  count: 60}   # +60 more, continued
  count: 180
  expected_sha256: "..."
```

Semantics: rules are drawn in order, and **listing a task again continues the
same seeded draw without replacement** — so a follow-up split is just the
baseline rules plus one extra same-task rule, fully self-contained (no
cross-file parent). The success curation (unsuccessful→successful episode swaps)
is baked into the store's `curation.json`, validated to stay in-task. An
explicit-`episode_ids` escape hatch remains for externally-curated sets with no
regenerable rule (none of the current configs need it). `count` is a
human-readable check and
`expected_sha256` is an audit lock over the resolved set — it turns a silently
changed resolution (missing curation, numpy RNG drift, edited seed/count) into a
hard error. The intended algebra (baseline inclusion, exactly 60 additions,
disjoint follow-ups, exact reproduction of the historical sets) is executable in
`tests/test_data.py` rather than documentary.

### Separate selection from sampling

`sample_weights_path` is a training-time sampler input, but it currently lives
inside the data-source config and is manually excluded from the conversion
cache hash ([`rgrl/data.py`](../rgrl/data.py#L87)). That exception is a sign
that source data and sampling policy are mixed.

Introduce a `SamplingConfig`:

```yaml
sampling:
  kind: weighted       # uniform | weighted
  weights: selections/transport_scores_v3.npy
  replacement: true
  epoch_samples: null  # null means number of source samples
```

Data source, logical selection, train/validation split, and sample distribution
then become independent, composable concepts.

## Training-throughput findings

### 1. Large batches are the strongest demonstrated lever

The tool-hang UMI artifacts compare equal sample budgets:

| Run | Batch | Steps | Final recorded rate | Approx. samples/s | 50-episode success |
|---|---:|---:|---:|---:|---:|
| `th_umi_base` | 256 | 30,000 | 88.3 it/s | 22.6k | 0.86 |
| `th_umi_bs1024_linlr` | 1,024 | 7,500 | 37.8 it/s | 38.7k | 0.94 |
| `th_umi_bs1024_sqrtlr` | 1,024 | 7,500 | 37.8 it/s | 38.7k | 0.92 |

Batch 1024 is therefore about **1.71× faster in samples/s** on this workload.
The success-rate sample is too small to distinguish 0.86, 0.92, and 0.94
reliably, but it is enough to justify a larger controlled sweep.

Promote the untracked `configs/training/bs1024.yaml` experiment into a reviewed
recipe after deciding its LR rule. Automate a batch/LR sweep and report:

- samples/s after compile warmup;
- peak allocated and reserved VRAM;
- time to a fixed number of samples;
- validation curve at fixed sample counts;
- multi-seed task success with confidence intervals.

For image DiT workloads that cannot directly fit a larger batch, first reduce
visual-encoder/preprocessing memory. Gradient accumulation preserves optimizer
batch semantics but does not increase per-kernel parallelism, so it will not
solve the occupancy problem by itself.

### 2. Express budgets in samples, not steps

The repository currently adjusts steps manually when either the dataset or
batch size changes. That is easy to get wrong and makes recipes hard to compare.

Make `train_samples` (or `samples_seen`) the primary budget:

```text
steps = ceil(train_samples / global_batch_size)
warmup_steps = ceil(warmup_samples / global_batch_size)
```

Log and checkpoint `samples_seen`. Express validation and checkpoint cadence in
samples as well. Dataset-size experiments can additionally specify epochs, but
sample count should be the cross-recipe comparison unit.

### 3. The CPU image path can be cleaner and modestly faster

The prior controlled investigation measured:

- GPU-resident versus pinned-CPU data: **+5.7%** throughput;
- background thread plus side CUDA stream prefetch: **+2.4%**;
- real H2D bandwidth: 12.25 GB/s versus a 12.39 GB/s measured ceiling.

This bounds expectations. The CPU path is worth cleaning up, but it is not
hiding a 2× gain.

Current `CpuSequenceDataset` performs one Python `__getitem__`, dictionary
construction, and several small indexed reads per sample, then default-collates
the batch. Replace this with batch-level gathering:

- let the sampler produce a tensor/list of batch IDs;
- gather all observation/action indices in one operation;
- perform one indexed read per key for the whole batch;
- keep images `uint8` through pinned transfer;
- prefetch the next batch on a side stream.

The vendored [`robo_dataset/prefetch.py`](../third_party/robo_dataset/robo_dataset/prefetch.py)
already provides the side-stream structure, but fix and stress-test tensor
lifetime handling before adopting it: after a side-stream transfer, recursively
call `record_stream(current_stream)` on the yielded tensors so the caching
allocator cannot reuse their storage while main-stream kernels still consume
it.

Expose a clear data-placement strategy:

```text
data_mode: auto | cuda_resident | shared_memory | streaming
device_memory_reserve_gb: ...
workers: ...
prefetch_factor: ...
```

`pin_fallback_gb` is a store-size threshold, not an actual memory-fit model. It
ignores policy parameters, EMA, optimizer state, activations, Inductor
workspaces, and the batch. In `auto`, estimate these or choose from current free
VRAM with an explicit reserve; always log the decision and estimates.

### 4. Remove the Python window-index construction

[`rgrl/dataset.py`](../rgrl/dataset.py#L64) loops once per frame in Python and
materializes separate `int64` observation and action tables plus a pad table.
At large scale this adds startup time and significant memory. With the common
2-observation/16-action horizon it is roughly 160 bytes per sample before
tensor data, or about 1.6 GB at 10 million samples.

`robo_dataset.SequenceSampler` already builds compact sequence descriptors
using Numba and supports episode/sample masks. Extend it to expose the pad mask
and use:

```text
sequence_length = horizon
pad_before = obs_horizon - 1
pad_after = horizon - obs_horizon
key_first_k[observation_key] = obs_horizon
```

Alternatively use compact `int32` anchor/episode descriptors and derive the
small index grids vectorially per batch. Either is preferable to maintaining
two competing window implementations.

### 5. Fuse or reorder image preprocessing

The policy transform pipeline currently runs in the trainer, outside the
compiled loss ([`rgrl/train_loop.py`](../rgrl/train_loop.py#L227)).
`ImageToTensor` casts the full native image to float before later crops.

Benchmark a single compiled `policy.loss(raw_batch)` that:

1. crops `uint8` NHWC images before conversion where the crop permits it;
2. converts directly to BF16 NCHW and normalizes;
3. combines cameras into fewer launches;
4. executes preprocessing and the loss under one compiled entry point.

For three 180×320 cameras cropped to 160×288, cropping before type expansion
avoids converting pixels that are immediately discarded. A fused kernel may
also remove intermediate float32 buffers and permutations. Measure numerical
and augmentation equivalence, compile time, data-section time, and total
samples/s.

The unconditional resize to the already-native size
([`rgrl/transforms.py`](../rgrl/transforms.py#L152)) is a no-op during training
and still adds traversal/dictionary overhead. Separate live-input resizing from
known-shape training preprocessing.

### 6. Consider a shared camera encoder as an opt-in architecture

`ObsEncoder` constructs a separate ResNet18 for every camera
([`rgrl/policies/base.py`](../rgrl/policies/base.py#L61)). If camera semantics
allow weight sharing, concatenate `B × T × cameras` and run one shared encoder.
This:

- creates a larger, more efficient encoder call;
- reduces parameters, gradients, Adam state, and EMA memory;
- may free enough VRAM for a larger batch.

It is a model change, not a semantics-preserving optimization, so retain the
current per-camera option and compare task success. A middle ground is shared
early layers plus camera-specific adapters/embeddings.

### 7. Sample profiler instrumentation in production

With profiling enabled, every step creates ten CUDA events for the five
sections ([`rgrl/profiler.py`](../rgrl/profiler.py#L35)). This is valuable
during development, but its production cost is not measured.

Benchmark `profile=false` and, if nontrivial, collect section timings only for
one log window every N windows. Keep the first-class profiling behavior without
paying for it on every long run.

### 8. Do not revisit already-disproven optimizations without new evidence

For the recorded image DiT workload:

- GPU-stream work accounts for about 99% of wall time, so there is no large
  Python scheduling bubble.
- Channels-last was 84.70 versus 85.18 ms/iteration under compile: neutral.
- Max-autotune found no useful convolution alternative.
- Whole-step CUDA graphs were slower on the prior lowdim architecture and the
  image profile shows little idle launch time to recover.
- H2D already reaches the measured link ceiling.

Focus on batch/kernel size, model/preprocessing work, and the bounded CPU-path
overlap instead.

### 9. Multi-GPU comes after single-process state is explicit

DDP is a reasonable future route to aggregate throughput, especially for image
models, but it should follow the `TrainingState` and `BatchSource` cleanup.
Then rank/world-aware sampling, global sample accounting, validation, and
checkpoint ownership have clear homes. Do not add `accelerate` merely to hide
these semantics.

## Extensibility and code quality

### Put preprocessing behind the policy boundary

Training manually applies `policy.data_transforms_pipeline`, whereas
`predict_action` applies it internally. A new trainer or caller can therefore
pass a raw batch to `compute_loss` incorrectly.

Give policies symmetric public methods:

```python
loss = policy.loss(raw_batch)
actions = policy.predict(raw_observation)
```

Keep `_loss_preprocessed` and `_predict_preprocessed` private if useful. Compile
the public loss wrapper. The trainer should not know whether a policy uses
images, crops, normalization, prefix conditioning, diffusion, or flow.

### Reuse small policy components instead of duplicating policy logic

Diffusion, flow, and DiT flow repeat configuration fields, prefix-mask
sampling, transform setup, normalization, and inference entry-point patterns.
Extract small value objects and functions:

- `ObservationSpec` / `ActionSpec`;
- `PrefixConditioningConfig` and `sample_prefix_mask`;
- `Preprocessor`;
- `SamplingSchedule`.

Avoid a deep policy inheritance hierarchy. Composition preserves the
repository's current readability.

### Avoid sanitized-key collisions

`ObsEncoder` stores modules using `k.replace(".", "_")`. Keys `a.b` and `a_b`
would collide silently. Use a `ModuleList` aligned with the original sorted
key list, or generate opaque internal IDs while retaining an explicit mapping.

In `robo_dataset`'s image transforms, retain image keys as an ordered tuple
rather than a `set`; set iteration can assign random augmentations to cameras
in a process-hash-dependent order.

### Make the conversion cache content-addressed and atomic

The cache hashes selected function source via `inspect.getsource`. Formatting
or comment changes invalidate it, while mutations to a dataset at the same
path do not. It also writes directly into the final cache directory, and an
interrupted conversion requires manual deletion.

Use:

- a stable conversion schema/version;
- source dataset ID plus content/metadata fingerprint;
- target schema and selected episode-view hash;
- a per-cache lock;
- conversion into a temporary sibling directory;
- validation, then atomic rename into place.

Validate symlink targets, not only the presence of the symlink path. Replace
conversion-time list-and-concatenate statistics
([`rgrl/data.py`](../rgrl/data.py#L172)) with a streaming accumulator to avoid
holding a second copy of all low-dimensional data.

### Use compositional, portable configs

Data configs contain absolute `/mnt/ssdB/...` and `/home/erbauer/...` paths,
and the set-table configs repeat the full shape metadata. Define named dataset
sources with `${oc.env:RGRL_DATA_ROOT}` (or a small local source registry), a
shared DROID schema config, and selection overrides.

The five near-identical DiT shell scripts and watchdogs should become:

- one base experiment config;
- small selection/seed overrides;
- one generic `train_with_restart.sh` or scheduler job.

Keep machine-specific GPU assignment outside committed experiment definitions.

### Treat data splits as first-class artifacts

Validation is currently the last N episodes. Since selection IDs are sorted,
that can correlate validation with collection time, task, scene, or robot
conditions. Record a seeded and preferably stratified split in the selection
manifest. Evaluation should be stable even if the physical store is reordered.

### Repair documentation drift

Examples found during review:

- `README.md` refers to nonexistent `rgrl/schema.py`.
- Its dataset description says "pinned-CPU fallback", while the implementation
  is now a shared-memory `CpuSequenceDataset` plus `DataLoader`.
- The top comment in `rgrl/inference/checkpoint.py` says `env_meta` is lost,
  while the later implementation and training checkpoint now preserve it.
- Some script comments describe fewer cameras or older step budgets than the
  current configurations.

Documentation close to fast-changing research code should state invariants and
ownership; measured numbers should include the commit, hardware, recipe, and
date.

## Testing and performance regression strategy

There are no train-loop tests, CI workflow, linter, formatter, type-checker, or
committed performance benchmark harness visible in the repository. Add CPU
tests for:

- same seed gives identical initial weights and first N losses;
- uninterrupted versus save/resume training produces identical state;
- completed-run handling never regresses `latest.pt`;
- incompatible config resume fails;
- trace plus resume preserves monotonically increasing steps;
- validation does not change the training RNG stream;
- empty/tiny datasets fail promptly rather than hanging;
- weighted sampling works in both resident and CPU modes;
- selection-manifest validation and parent/addition relations;
- atomic cache/checkpoint recovery after injected failure.

Add lightweight CI for the CPU suite plus Ruff/format checking. Put pytest and
development tools in a `dev` optional dependency group rather than requiring
`--with pytest` ad hoc.

GPU performance should be a separate, explicit benchmark rather than a flaky
unit-test threshold. Record JSON containing:

```text
commit, GPU, CUDA/PyTorch
recipe and resolved config hash
compile warmup time
samples/s median and distribution
section timings
peak allocated/reserved VRAM
data mode and transfer bandwidth
```

Track at least a lowdim resident case and a three-camera CPU/shared-memory
case. Compare fixed sample counts, not fixed steps.

## Suggested implementation sequence

### Phase 0: correctness guardrails

1. Move seeding before policy construction.
2. Add completed-run and dataset-size validation.
3. Isolate validation RNG and compute a sample-weighted validation mean.
4. Add explicit resume modes and config compatibility checks.
5. Make checkpoints atomic and save full training/sampler/RNG state.
6. Add focused trainer tests for all of the above.

### Phase 1: make the demonstrated speedup the default workflow

1. Introduce `train_samples`, `warmup_samples`, and `samples_seen`.
2. Add a reproducible batch/LR sweep command.
3. Promote the batch-1024 lowdim recipe after multi-seed evaluation.
4. Make profiler sampling configurable and benchmark its overhead.

### Phase 2: clean data ownership and subsets

1. Add source fingerprints and versioned selection manifests.
2. Implement `EpisodeView` in `robo_dataset`.
3. Move split and sampling policy out of source configs.
4. Replace `rgrl`'s Python window table with the shared sampler.
5. Make cache construction locked and atomic.

### Phase 3: optimize the large-image path

1. Add vectorized batch gathering.
2. Fix `record_stream` handling and integrate side-stream prefetch.
3. Move preprocessing inside compiled `policy.loss`.
4. Benchmark uint8 crop/fused conversion and camera batching.
5. Evaluate an optional shared visual encoder and use freed memory for larger
   batches.

### Phase 4: scale out if needed

Add DDP with global sample accounting, rank-aware deterministic sampling, and
single-writer atomic checkpoints. Only do this if aggregate throughput is more
valuable than retaining the current single-process simplicity.

## Bottom line

The repository does not need a framework rewrite. Its strongest qualities are
its directness and its measured, specialized GPU-resident path. The elegant
next version would preserve those while introducing three explicit boundaries:

1. immutable data source versus logical selection versus sampling policy;
2. raw batch source versus policy-owned preprocessing/loss;
3. policy weights versus complete, versioned training state.

With those boundaries in place, the code becomes easier to extend and the
performance work becomes more targeted. In the short term, large batches are
the only demonstrated major throughput win; in the image path, removing
duplicate/copy-heavy data machinery and reducing visual preprocessing/encoder
work are the most credible next experiments.
