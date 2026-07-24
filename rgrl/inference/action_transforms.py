"""Inverse of diffusion_policy's abs-action encoding, for stepping a
robosuite env with actions recorded/trained in that space.

Policies trained against an `abs_action: true` store (see rgrl/data.py)
output per-arm `pos(3) + rotation_6d(6) + gripper(1)` — diffusion_policy's
`RobomimicLowdimRunner.undo_transform_action` convention (Zhou et al. 2019
continuity representation for the rotation component, chosen there because a
6D representation has no discontinuity a network has to learn around, unlike
axis-angle or quaternions). robosuite's OSC_POSE controller, however, only
understands `pos(3) + axis_angle(3) + gripper(1)` — this module converts one
to the other. No pytorch3d/diffusion_policy dependency: rotation_6d_to_matrix
and the matrix logarithm map (matrix -> axis-angle) are both closed-form and
short enough to keep local, in keeping with rgrl's no-heavy-deps design.

Absolute vs. delta actions is a separate axis from the rotation
representation: a store's `abs_action` flag also needs the env's controller
reconfigured to accept absolute poses (`control_delta: False`) — see
rgrl/inference/envs/robomimic.py, which does that reconfiguration and calls
`undo_abs_rotation_action` before every env.step().
"""

from __future__ import annotations

import numpy as np


def rotation_6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rotation_6d -> (..., 3) axis-angle, via Gram-Schmidt
    orthogonalization to a rotation matrix (Zhou et al. 2019, Section B) then
    the standard SO(3) logarithm map."""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    r = np.stack([b1, b2, b3], axis=-2)  # rows b1,b2,b3 -> (..., 3, 3)

    cos_theta = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    axis = np.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]],
        axis=-1,
    )
    sin_theta = np.sin(theta)
    # Generic branch: (R - R^T) is 2 sin(theta) * skew(axis), so theta/(2 sin)
    # recovers the axis-angle vector. small covers both theta ~= 0 (-> 0) and
    # the theta ~= pi singularity, handled separately below.
    small = np.abs(sin_theta) < 1e-6
    scale = np.where(small, 0.0, theta / np.where(small, 1.0, 2 * sin_theta))
    out = axis * scale[..., None]

    # theta ~= pi: sin(theta) -> 0 makes the antisymmetric part vanish, so the
    # generic branch collapses to ~0 instead of pi*axis. Recover the axis from
    # the symmetric part R + I = 2 n n^T (its largest-norm column is parallel to
    # the axis), signed from the tiny antisymmetric part.
    near_pi = small & (cos_theta < 0)
    if np.any(near_pi):
        rp = r + np.eye(3)
        col = np.argmax(np.linalg.norm(rp, axis=-2), axis=-1)
        axis_pi = np.take_along_axis(rp, col[..., None, None], axis=-1)[..., 0]
        axis_pi = axis_pi / (np.linalg.norm(axis_pi, axis=-1, keepdims=True) + 1e-12)
        sign = np.sign((axis * axis_pi).sum(-1, keepdims=True))
        sign = np.where(sign == 0, 1.0, sign)
        out = np.where(near_pi[..., None], np.pi * axis_pi * sign, out)
    return out


def undo_abs_rotation_action(action: np.ndarray) -> np.ndarray:
    """(..., 10*n_arms) pos+rot6d+gripper per arm -> (..., 7*n_arms)
    pos+axis_angle+gripper per arm, e.g. 20-dim dual-arm -> 14-dim."""
    raw_shape = action.shape
    if raw_shape[-1] % 10 != 0:
        raise ValueError(
            f"expected action dim a multiple of 10 (pos3+rotation_6d6+gripper1 per arm), "
            f"got {raw_shape[-1]}"
        )
    n_arms = raw_shape[-1] // 10
    a = action.reshape(*raw_shape[:-1], n_arms, 10)
    pos, rot6d, gripper = a[..., :3], a[..., 3:9], a[..., 9:]
    axis_angle = rotation_6d_to_axis_angle(rot6d)
    out = np.concatenate([pos, axis_angle, gripper], axis=-1)
    return out.reshape(*raw_shape[:-1], n_arms * 7)
