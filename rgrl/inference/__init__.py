"""Inference-time pieces: policy serving (HTTP) and env/policy rollout
contracts (Robosuite sim with Robomimic-compatible metadata, DROID real-world).

Kept deliberately separate from rgrl's training code so a policy-serving
process never needs to import an env's dependencies (robosuite/mujoco, or
the real-robot `droid` package) and vice versa — see
rgrl/inference/checkpoint.py's docstring for why this split exists.
"""
