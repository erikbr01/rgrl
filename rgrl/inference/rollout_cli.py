"""Shared construction helpers for rgrl's three rollout entrypoints --
eval.py (batched, non-realtime statistical eval), run_realtime.py
(real-time-paced rollouts with sim auto-success labels), and
collect_rollouts.py (real-time-paced rollouts with human keyboard labeling +
recording). All three build a policy the same way (checkpoint vs. remote
server, and which kind of server) and need the same facts about a loaded
policy (obs_horizon, shape_meta, env_meta, ...) before they can build an env
to run it in -- this module is the one place that logic lives, instead of
being copied per script.
"""

from __future__ import annotations


def build_policy(
    ckpt: str | None,
    server: str | None,
    inference,
    device: str = "cuda:0",
    external_camera: str = "ext1",
    wrist_key: str = "wrist_image_left",
    ext_key: str | None = None,
):
    """ws(s):// -> an openpi server (WebSocketPolicy); http(s):// -> an rgrl
    policy_server (HttpPolicy); a checkpoint path with no server -> in-process
    (LocalPolicy); neither given -> default to a same-machine policy_server.
    `inference` (an InferenceConfig) is only consulted on the LocalPolicy
    path; `external_camera`/`wrist_key`/`ext_key` only on the WebSocketPolicy
    path (openpi's obs-key mapping)."""
    if server is not None:
        if server.startswith(("ws://", "wss://")):
            from rgrl.inference.policy_client import WebSocketPolicy

            return WebSocketPolicy(
                server, external_camera=external_camera, wrist_key=wrist_key, ext_key=ext_key
            )
        from rgrl.inference.policy_client import HttpPolicy

        return HttpPolicy(server)
    if ckpt is None:
        from rgrl.inference.policy_client import HttpPolicy

        return HttpPolicy("http://127.0.0.1:5001")
    from pathlib import Path

    from rgrl.inference.policy_client import LocalPolicy

    return LocalPolicy(str(Path(ckpt)), device=device, inference=inference)


def meta_from_policy(policy) -> dict:
    """obs_horizon/action_dim/shape_meta/env_meta/n_action_steps -- whatever
    an env needs to construct itself from a loaded policy, learned uniformly
    regardless of backend: HttpPolicy's /meta, or LocalPolicy's in-process
    store/cfg. WebSocketPolicy (openpi) has no equivalent -- it's droid-only,
    and DroidEnv's build() doesn't consume this in the first place."""
    from rgrl.inference.policy_client import HttpPolicy, LocalPolicy

    if isinstance(policy, HttpPolicy):
        return policy.ensure_meta()
    if isinstance(policy, LocalPolicy):
        # deferred: checkpoint.py imports torch, only needed on this in-process path --
        # HttpPolicy clients (e.g. a torch-free robot workstation) never hit it
        from rgrl.inference.checkpoint import shape_meta_json

        store = policy.store
        return {
            "obs_horizon": policy.policy.cfg.obs_horizon,
            "action_dim": int(store.meta.keys["action"].shape[0]),
            "shape_meta": shape_meta_json(store),
            "env_meta": store.meta.env_meta,
            "n_action_steps": policy.policy.cfg.n_action_steps,
        }
    raise TypeError(
        f"{type(policy).__name__} has no /meta equivalent -- pass a robomimic/droid "
        "backend only with --ckpt or a http(s):// server"
    )
