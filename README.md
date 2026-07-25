# rgrl — really good robot learning

Minimal imitation-learning stack built for iteration speed. Think lerobot's
core features without the framework: a flat training loop you read top to
bottom, named hydra configs instead of a processor-pipeline framework, no
accelerate/diffusers.

## Numbers (RTX 4090, PyTorch 2.11.0+cu128)

| run | throughput |
|---|---:|
| ToolHang state flow, compiled, batch 256 | 91.3 it/s ≈ 23.4k samples/s |
| ToolHang state flow, compiled, batch 1024 | 41.3 it/s ≈ 42.3k samples/s |
| DROID set-table DiT flow, shared camera encoder + shared-host data, batch 64 | ≈ 4.0k samples/s |

Robomimic-scale stores stay GPU-resident and a batch is one fancy-index
gather. Larger DROID stores use vectorized shared-memory gathers with pinned
H2D on a side stream; a producer thread overlaps host gathering with the
current update without multiprocessing large image batches.

## Design

- **Storage** (`third_party/robo_dataset`, a standalone git submodule): all
  backends, chunking, and conversion live there — rgrl doesn't reimplement a
  store format. `rgrl/store.py` is a thin wrapper adding only what
  robo_dataset doesn't own: kind classification (image/state/action),
  robomimic env_meta, and fitted normalizers, as small sidecar files next to
  the robo_dataset zarr store. `rgrl/convert_robomimic.py` streams a robomimic
  HDF5 dataset through it once (200-demo image dataset in ~7 s).
- **Dataset** (`rgrl/dataset.py`): explicit `auto | cuda_resident |
  shared_memory` placement; vectorized windowing and batch gathering with
  edge-replication + pad masks. Sampling is a pure function of seed, epoch,
  and absolute batch cursor in every mode.
- **Selections**: a data config's inline `selection:` block is a generative
  `(task, count)` recipe resolved against the store's `tasks.json`, not a
  hand-listed set of integer IDs. Two draw methods: `random` (default; a seeded
  uniform draw — reproducible for a fixed store, but the subset changes if you
  add data to a task) and `first_n` (the first `count` episodes by id — stable
  under store growth, since new episodes get higher ids and append past the
  first `count`). Listing the same task twice continues the same draw without
  replacement (that's how the `set_table_plus*` follow-up splits add 60 disjoint
  episodes), and the success curation (unsuccessful→successful episode swaps) is
  baked into the store's `curation.json`. An explicit `episode_ids` escape hatch
  remains for externally-curated sets. An optional `expected_sha256` pins the
  resolved set (a later silent change — edited `curation.json`, a numpy RNG
  version bump, an edited seed/count — then errors instead of quietly training
  on a different dataset); you don't hand-write it — author the recipe unlocked,
  then `uv run python scripts/pin_selection.py data=<name>` prints the line to
  paste. `robo_dataset.EpisodeView` reads the resolved episodes from the
  original store without re-encoding or copying videos; only the resolved IDs
  enter the conversion-cache identity.
- **Normalization**: fitting (method, per-key stats) is
  `robo_dataset.normalization`'s job, done once at conversion time.
  `rgrl/normalize.py` wraps the fitted `KeyNormalizer`'s (scale, offset) as
  `nn.Module` buffers, so they travel with the checkpoint's `state_dict` —
  checkpoints are fully self-contained, no separate normalizer artifacts.
- **Policies** (`rgrl/policies/`): plain modules with
  `loss(raw_batch)` / `predict_action(raw_batch)`. Rectified flow is the
  default; diffusion remains supported. Policies own preprocessing, fitted
  normalization, encoders, objectives, and inference.
- **Train loop** (`rgrl/train_loop.py`): fast by default — bf16 autocast,
  `torch.compile`, TF32, fused AdamW, `_foreach` EMA, sample-based budgets,
  exact deterministic resume, and atomic complete-state checkpoints.
- **Profiling is first-class** (`rgrl/profiler.py`): every log line shows
  GPU-stream and CPU-wall time per pipeline section
  (`data 0.1|0.6ms 0% | fwd 3.4|4.6ms 24% | ...`); `--trace-steps N` writes a
  chrome trace. If something gets slow, the log says where.
- **Config** (`configs/`): Hydra, composed from groups —
  `data` (which store), `policy` (architecture + training-time schedule),
  `sampling` (uniform/weighted), `training` (optimizer/performance),
  `inference` (deployment overrides), and environment/target schemas.
  `experiment/*` presets (`+experiment=set_table_baseline`) compose a full
  run — data/target/policy/training + wandb + out — in one file, so a launch
  is `train.py +experiment=<name>` with no magic-number CLI args.
- **Outputs**: every run gets a self-contained directory with resolved config,
  immutable step checkpoints, an atomic relative `latest.pt` symlink, and
  logs/evaluations.

## Use

```bash
# one-time: convert a robomimic hdf5 into a store
uv run python -m rgrl.convert_robomimic \
    --hdf5-path .../lift/ph/image.hdf5 --out data/stores/lift_ph_image \
    --obs-keys agentview_image robot0_eye_in_hand_image robot0_eef_pos robot0_eef_quat robot0_gripper_qpos

# train rectified flow (the default); length/cadence are in optimizer steps
scripts/train.sh data=tool_hang_state target=tool_hang_state \
  training.steps=30000

# large-batch state recipe (1/4 the steps -> same data as the bs=256 default)
scripts/train.sh data=tool_hang_state target=tool_hang_state training=bs1024

# diffusion remains available for compatibility/ablations
scripts/train.sh policy=diffusion_umi

# normalization method is fit once at conversion and baked into the store cache
# (part of its identity, so changing it rebuilds): minmax (default) | zscore
# (mean/std) | minmax_p99 (1st/99th pct). Per-key fits run in parallel processes;
# stats_workers sets the count (default: one per key, capped at CPU).
scripts/train.sh data=tool_hang_state target=tool_hang_state \
  data.norm_method=zscore stats_workers=8

# weighted flow sampling is independent of dataset conversion/cache identity
scripts/train.sh sampling=weighted sampling.weights_path=/path/to/weights.npy

# evaluate a checkpoint in the robomimic env (needs: uv sync --extra eval)
scripts/eval.sh ckpt=runs/lift_image/latest.pt env=robomimic_lift

# Diffusion samplers are an inference-time choice; they remain sweepable
# without retraining:
scripts/eval.sh ckpt=runs/x/latest.pt inference.sampler=ddpm inference.inference_steps=100
scripts/eval.sh ckpt=runs/x/latest.pt inference=diffusion_dp env=robomimic_tool_hang
```

`scripts/train.sh`/`scripts/eval.sh` are thin wrappers over `uv run python
train.py`/`eval.py`, runnable from any directory. `train.py`/`eval.py` are
themselves the hydra entrypoints if you'd rather call them directly.

Adapting to a new benchmark = one converter (HDF5/whatever → store) + one
`data`/`env` config pair. Nothing else changes.

## Tests

```bash
uv run --extra dev ruff check .
uv run --extra dev pytest tests/ -q
```

See [`docs/architecture_and_training_review.md`](docs/architecture_and_training_review.md)
for the architecture rationale, exact-resume semantics, and attributed
ToolHang/DROID benchmarks.
