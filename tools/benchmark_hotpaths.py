"""Reproduce isolated CPU-batching and DROID preprocessing benchmarks.

Examples:
    uv run python tools/benchmark_hotpaths.py --benchmark data
    CUDA_VISIBLE_DEVICES=0 uv run python tools/benchmark_hotpaths.py \
        --benchmark preprocess --out benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from rgrl.data import load_store
from rgrl.dataset import CpuSequenceDataset, Windows
from rgrl.transforms import (
    CropTransform,
    DataPipeline,
    ImagePreprocessTransform,
    ImageToTensorTransform,
    ResizeTransform,
)


def benchmark_data(batch_size: int, iterations: int) -> dict:
    """Compare the old per-sample collation shape with batched __getitems__."""
    store = load_store(
        OmegaConf.load("configs/data/tool_hang_state.yaml"),
        OmegaConf.load("configs/target/tool_hang_state.yaml"),
    )
    dataset = CpuSequenceDataset(
        store,
        Windows(obs_horizon=2, horizon=16),
        torch.device("cpu"),
        num_workers=0,
    )

    class PerSampleAdapter(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            return dataset[index]

    available = len(dataset) // batch_size
    warmup = min(20, max(1, available // 10))
    measured = min(iterations, available - warmup)
    if measured <= 0:
        raise ValueError(
            f"dataset has only {available} full batches at batch_size={batch_size}"
        )

    def measure(loader: DataLoader) -> dict:
        iterator = iter(loader)
        for _ in range(warmup):
            next(iterator)
        start = time.perf_counter()
        samples = 0
        for _ in range(measured):
            batch = next(iterator)
            samples += batch["action"].shape[0]
        elapsed = time.perf_counter() - start
        return {
            "batches_per_second": measured / elapsed,
            "samples_per_second": samples / elapsed,
        }

    vectorized = measure(
        DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=lambda batch: batch,
        )
    )
    per_sample = measure(
        DataLoader(PerSampleAdapter(), batch_size=batch_size, drop_last=True)
    )
    return {
        "batch_size": batch_size,
        "iterations": measured,
        "vectorized": vectorized,
        "per_sample_default_collate": per_sample,
        "speedup": (
            vectorized["samples_per_second"] / per_sample["samples_per_second"]
        ),
    }


def benchmark_preprocess(batch_size: int, iterations: int) -> dict:
    """Compare full-frame float conversion with uint8-first crop on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("preprocess benchmark requires CUDA")
    device = torch.device("cuda")
    keys = (
        "wrist_image_left",
        "wrist_image_right",
        "exterior_image_1_left",
    )
    # Reuse one allocation across keys: kernels and output allocation still run
    # once per camera, while the benchmark does not waste VRAM on duplicate
    # random inputs.
    raw = torch.randint(
        0,
        256,
        (batch_size, 2, 180, 320, 3),
        dtype=torch.uint8,
        device=device,
    )
    batch = {key: raw for key in keys}
    old = DataPipeline(
        [
            ImageToTensorTransform(list(keys)),
            ResizeTransform(list(keys), (180, 320)),
            CropTransform(list(keys), (76, 76), pad=0, per_timestep=True),
        ]
    ).to(device).train()
    new = ImagePreprocessTransform(
        list(keys),
        (180, 320),
        CropTransform(list(keys), (76, 76), pad=0, per_timestep=True),
    ).to(device).train()

    def measure(module) -> dict:
        for _ in range(10):
            output = module(batch)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = module(batch)
        end.record()
        torch.cuda.synchronize()
        milliseconds = start.elapsed_time(end) / iterations
        del output

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = module(batch)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - baseline
        del output
        return {
            "milliseconds_per_batch": milliseconds,
            "peak_extra_mib": peak_extra / 2**20,
        }

    float_first = measure(old)
    uint8_first = measure(new)
    return {
        "batch_size": batch_size,
        "timesteps": 2,
        "cameras": len(keys),
        "native_hw": [180, 320],
        "crop_hw": [76, 76],
        "iterations": iterations,
        "float_first": float_first,
        "uint8_first": uint8_first,
        "speedup": (
            float_first["milliseconds_per_batch"]
            / uint8_first["milliseconds_per_batch"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("data", "preprocess", "all"),
        default="all",
    )
    parser.add_argument("--data-batch-size", type=int, default=256)
    parser.add_argument("--preprocess-batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else None
        ),
    }
    if args.benchmark in {"data", "all"}:
        result["data"] = benchmark_data(args.data_batch_size, args.iterations)
    if args.benchmark in {"preprocess", "all"}:
        result["preprocess"] = benchmark_preprocess(
            args.preprocess_batch_size,
            args.iterations,
        )
    encoded = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
