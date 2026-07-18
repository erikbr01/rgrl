"""HTTP inference server for rgrl policy checkpoints (FastAPI).

Loads a checkpoint and serves predict_action over HTTP. Run it on the GPU
machine; an env process (robomimic sim or a real robot) talks to it through
rgrl.inference.policy_client.HttpPolicy, which assembles obs windows
client-side using /meta — the env process never needs torch or rgrl.data,
only `requests` (see pyproject.toml's `eval` extra).

Endpoints
---------
GET  /health   {"status": "ok", "device": ..., "step": ...}
GET  /meta     {"obs_horizon", "horizon", "n_action_steps", "action_dim",
                "shape_meta" (the target shape_meta this checkpoint trained
                against — one uniform obs entry per key, image or state, no
                policy-type branch), "env_meta" (the robomimic simulator
                kwargs the store recorded, so an env process can build its
                sim without ever loading the store itself)}
POST /infer    body: wire.encode_batch bytes, one array per shape_meta obs
               key, each (B, obs_horizon, *key_shape) — uint8 for rgb keys,
               float32 for low_dim. May additionally carry
               wire.ACTION_PREFIX_KEY ((B, P, action_dim) float32 committed
               actions): the request is then served by predict_action_rtc —
               real-time chunking, see rgrl/inference/runner.py — and the
               returned action chunk is (B, P + n_action_steps, action_dim)
               with the prefix repeated at its head.
               response: wire.encode_batch bytes,
               {"action": (B, n_action_steps, action_dim) float32,
                "act:<layer>": ... for each of InferenceConfig.capture_layers
                — forward activations captured during this call, for
                policy_graphs-style inspection of the policy's internals}.

Usage:
    uv run python -m rgrl.inference.policy_server \
        --checkpoint runs/lift_image/latest.pt --device cuda:0 --port 5001
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response

from rgrl.inference.activations import ACTIVATION_KEY_PREFIX, ActivationCapture
from rgrl.inference.checkpoint import InferenceConfig, load_policy, shape_meta_json
from rgrl.inference.wire import ACTION_PREFIX_KEY, decode_batch, encode_batch

log = logging.getLogger(__name__)


def build_app(checkpoint: str, device: str = "cuda:0", inference: InferenceConfig | None = None):
    """Load the checkpoint and return the FastAPI app."""
    import torch

    inference = inference or InferenceConfig()
    if inference.cuda_graph and inference.capture_layers:
        raise ValueError(
            "cuda_graph + capture_layers is not supported: forward hooks only fire "
            "during the one-time graph capture, not on later graph replays, so "
            "captured activations would silently go stale after the first call."
        )
    device_t = torch.device(device)
    log.info("loading checkpoint: %s", checkpoint)
    policy, store, step = load_policy(Path(checkpoint), inference, device_t)
    log.info("ready on %s (step %d)", device, step)
    capture = ActivationCapture(policy, inference.capture_layers) if inference.capture_layers else None

    predict = policy
    if inference.cuda_graph:
        if device_t.type != "cuda":
            raise ValueError("cuda_graph requires device='cuda:*'")
        from rgrl.inference.cuda_graph import GraphedPolicy

        predict = GraphedPolicy(policy)

    meta = {
        "obs_horizon": policy.cfg.obs_horizon,
        "horizon": policy.cfg.horizon,
        "n_action_steps": policy.cfg.n_action_steps,
        "action_dim": int(store.meta.keys["action"].shape[0]),
        "shape_meta": shape_meta_json(store),
        "env_meta": store.meta.env_meta,
        "capture_layers": list(inference.capture_layers),
    }
    autocast = torch.autocast(device_type=device_t.type, dtype=torch.bfloat16, enabled=inference.bf16)

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device, "step": step}

    @app.get("/meta")
    def get_meta():
        return meta

    @app.post("/infer")
    async def infer(request: Request):
        arrays = decode_batch(await request.body())
        prefix = arrays.pop(ACTION_PREFIX_KEY, None)
        batch = {k: torch.as_tensor(v, device=device_t) for k, v in arrays.items()}
        with torch.no_grad(), autocast:
            if prefix is None:
                action = predict.predict_action(batch)
            else:
                action = predict.predict_action_rtc(batch, torch.as_tensor(prefix, device=device_t))
        response = {"action": action.float().cpu().numpy()}
        if capture is not None:
            for layer, act in capture.pop().items():
                response[f"{ACTIVATION_KEY_PREFIX}{layer}"] = act
        return Response(content=encode_batch(response), media_type="application/octet-stream")

    return app


def serve(
    checkpoint: str,
    device: str = "cuda:0",
    host: str = "127.0.0.1",
    port: int = 5001,
    inference: InferenceConfig | None = None,
) -> None:
    import uvicorn

    app = build_app(checkpoint, device, inference)
    uvicorn.run(app, host=host, port=port)


@dataclasses.dataclass
class _Args:
    checkpoint: str
    device: str = "cuda:0"
    host: str = "127.0.0.1"
    port: int = 5001
    sampler: str | None = None  # ddim | ddpm; override the checkpoint's trained schedule
    inference_steps: int | None = None
    compile: bool = True  # torch.compile the UNet; warms up at load time (see load_policy)
    warmup_batch_size: int = 1  # batch size to compile/warm up for; match your clients' actual B
    cuda_graph: bool = False  # capture the whole predict_action call -- see rgrl/inference/cuda_graph.py
    bf16: bool = False
    use_ema: bool = True
    capture_layers: tuple[str, ...] = ()  # dotted submodule paths -- see rgrl/inference/activations.py


def main() -> None:
    import tyro

    args = tyro.cli(_Args)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    inference = InferenceConfig(
        use_ema=args.use_ema,
        bf16=args.bf16,
        compile=args.compile,
        warmup_batch_size=args.warmup_batch_size,
        cuda_graph=args.cuda_graph,
        sampler=args.sampler,
        inference_steps=args.inference_steps,
        capture_layers=list(args.capture_layers),
    )
    serve(args.checkpoint, args.device, args.host, args.port, inference)


if __name__ == "__main__":
    main()
