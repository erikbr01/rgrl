# rgrl — really good robot learning

Minimal imitation-learning stack built for iteration speed. Think lerobot's
core features without the framework: ~1.5k lines, flat training loop you read
top to bottom, dataclass configs, no hydra/accelerate/diffusers.

## Numbers (RTX 4090, robomimic lift ph)

| run | throughput | wall time |
|---|---|---|
| state policy (63.5M diffusion UNet, bs 256) | ~71 it/s ≈ 18k samples/s | 10k steps in ~2.5 min |
| image policy (2× ResNet-18 + UNet, 89M, bs 64) | ~45 it/s ≈ 2.9k samples/s | 20k steps in ~7.5 min |

Data loading is 0% of step time: stores are raw uint8/float32 arrays that live
on the GPU; a batch is one fancy-index gather. No DataLoader, no workers, no
per-sample decode.

## Design

- **Store** (`rgrl/store.py`): a dataset is a directory of flat `.npy` arrays
  plus `meta.json` (episode boundaries, key kinds, env metadata) and
  `stats.json`. Converters pay all cost once (`rgrl/convert_robomimic.py`:
  200-demo image dataset in 9 s).
- **Dataset** (`rgrl/dataset.py`): whole store on GPU (pinned-CPU fallback for
  big data); windowing with edge-replication clamping + pad masks, precomputed
  as index tables. Sampling is a pure function of `(seed, epoch)`.
- **Policies** (`rgrl/policies/`): plain nn.Modules with `compute_loss(batch)`
  / `predict_action(batch)`. Normalization stats are registered buffers —
  checkpoints are self-contained. Diffusion (DDPM train, DDIM sample; the
  scheduler is 40 lines, no diffusers) and rectified flow matching share the
  same conditional UNet and obs encoder.
- **Train loop** (`rgrl/train_loop.py`): fast by default — bf16 autocast,
  `torch.compile`, TF32, fused AdamW, `_foreach` EMA, step-based training.
- **Profiling is first-class** (`rgrl/profiler.py`): every log line shows
  GPU-stream and CPU-wall time per pipeline section
  (`data 0.1|0.6ms 0% | fwd 3.4|4.6ms 24% | ...`); `--trace-steps N` writes a
  chrome trace. If something gets slow, the log says where.

## Use

```bash
# one-time: convert a robomimic hdf5 into a store
uv run python -m rgrl.convert_robomimic \
    --hdf5-path .../lift/ph/image.hdf5 --out data/stores/lift_ph_image \
    --obs-keys agentview_image robot0_eye_in_hand_image robot0_eef_pos robot0_eef_quat robot0_gripper_qpos

# train (all flags: uv run python train.py --help)
uv run python train.py --store data/stores/lift_ph_image --out runs/lift_image \
    --steps 20000 --batch-size 64

# flow matching instead of diffusion
uv run python train.py --store ... --policy flow

# scheduling presets: dp = DDPM 100 train/100 sample (Chi et al.),
# umi = DDIM 50 train/16 sample. Default is DDIM 100/10.
uv run python train.py --store ... --preset umi

# samplers are an inference-time choice — sweep them on a trained checkpoint:
uv run python eval.py runs/x/latest.pt --sampler ddpm --inference-steps 100

# evaluate a checkpoint in the robomimic env (needs: uv sync --extra eval)
uv run python eval.py --ckpt runs/lift_image/latest.pt --episodes 50
```

Adapting to a new benchmark = one converter (HDF5/whatever → store) + one eval
runner. Nothing else changes.

## Tests

```bash
uv run --with pytest pytest tests/ -q
```
