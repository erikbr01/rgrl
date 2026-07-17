# rgrl — really good robot learning

Minimal imitation-learning stack built for iteration speed. Think lerobot's
core features without the framework: a flat training loop you read top to
bottom, named hydra configs instead of a processor-pipeline framework, no
accelerate/diffusers.

## Numbers (RTX 4090, robomimic lift ph)

| run | throughput | wall time |
|---|---|---|
| state policy (63.5M diffusion UNet, bs 256) | ~91 it/s ≈ 23k samples/s | 10k steps in ~1.8 min |
| image policy (2× ResNet-18 + UNet, 63.5M, bs 64) | ~59 it/s ≈ 3.8k samples/s | 20k steps in ~5.6 min |

Data loading is 0% of step time: stores live on the GPU; a batch is one
fancy-index gather. No DataLoader, no workers, no per-sample decode.

## Design

- **Storage** (`third_party/robo_dataset`, a standalone git submodule): all
  backends, chunking, and conversion live there — rgrl doesn't reimplement a
  store format. `rgrl/store.py` is a thin wrapper adding only what
  robo_dataset doesn't own: kind classification (image/state/action),
  robomimic env_meta, and fitted normalizers, as small sidecar files next to
  the robo_dataset zarr store. `rgrl/convert_robomimic.py` streams a robomimic
  HDF5 dataset through it once (200-demo image dataset in ~7 s).
- **Dataset** (`rgrl/dataset.py`): whole store loaded onto the GPU (pinned-CPU
  fallback for big data); windowing with edge-replication clamping + pad
  masks, precomputed as index tables. Sampling is a pure function of
  `(seed, epoch)`.
- **Normalization**: fitting (method, per-key stats) is
  `robo_dataset.normalization`'s job, done once at conversion time.
  `rgrl/normalize.py` wraps the fitted `KeyNormalizer`'s (scale, offset) as
  `nn.Module` buffers, so they travel with the checkpoint's `state_dict` —
  checkpoints are fully self-contained, no separate normalizer artifacts.
- **Policies** (`rgrl/policies/`): plain nn.Modules with `compute_loss(batch)`
  / `predict_action(batch)`. Diffusion (DDPM train, DDIM or DDPM sample; the
  scheduler is 40 lines, no diffusers) and rectified flow matching share the
  same conditional UNet and obs encoder.
- **Train loop** (`rgrl/train_loop.py`): fast by default — bf16 autocast,
  `torch.compile`, TF32, fused AdamW, `_foreach` EMA, step-based training.
- **Profiling is first-class** (`rgrl/profiler.py`): every log line shows
  GPU-stream and CPU-wall time per pipeline section
  (`data 0.1|0.6ms 0% | fwd 3.4|4.6ms 24% | ...`); `--trace-steps N` writes a
  chrome trace. If something gets slow, the log says where.
- **Config** (`configs/`): hydra, composed from five groups —
  `data` (which store), `policy` (architecture + training-time schedule),
  `training` (optimizer/perf, policy-independent), `inference` (eval-time
  sampler/perf overrides, policy-specific — the policy itself always loads
  from the checkpoint), `env` (rollout settings, policy-independent — the
  same env config works for any checkpoint trained on that task). Every
  group's yaml carries a `_target_`; `hydra.utils.instantiate()` builds the
  real dataclass, no manual field dispatch. See `rgrl/schema.py`.
- **Outputs**: every run gets one self-contained experiment directory —
  `runs/<name>/{config.json, latest.pt, train.log, evals/<label>/{config.json,
  summary.json, eval.log}}`. Eval labels default to a config-derived slug
  (e.g. `ddim_10`, `ddpm_100`), so sweeping samplers against one checkpoint
  accumulates results instead of overwriting them.

## Use

```bash
# one-time: convert a robomimic hdf5 into a store
uv run python -m rgrl.convert_robomimic \
    --hdf5-path .../lift/ph/image.hdf5 --out data/stores/lift_ph_image \
    --obs-keys agentview_image robot0_eye_in_hand_image robot0_eef_pos robot0_eef_quat robot0_gripper_qpos

# train (all groups/flags: uv run python train.py --help)
scripts/train.sh data=lift_image training.steps=20000 training.batch_size=64

# flow matching instead of diffusion
scripts/train.sh policy=flow

# scheduling presets: dp = DDPM 100 train/100 sample (Chi et al.),
# umi = DDIM 50 train/16 sample. Default policy is DDIM 100/10.
scripts/train.sh policy=diffusion_umi

# evaluate a checkpoint in the robomimic env (needs: uv sync --extra eval)
scripts/eval.sh ckpt=runs/lift_image/latest.pt env=robomimic_lift

# samplers are an inference-time choice — sweep them on a trained checkpoint
# without retraining; each combination lands in its own evals/<label>/:
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
uv run --with pytest pytest tests/ -q
```
