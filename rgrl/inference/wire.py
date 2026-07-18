"""The HTTP wire format, defined exactly once and shared by
policy_server.py and policy_client.py: a dict of named arrays, encoded as
np.savez bytes. No pickling (allow_pickle=False) — every payload is either
uint8 image frames or float32 vectors.
"""

from __future__ import annotations

import io

import numpy as np

# Reserved request key for real-time chunking: (B, P, action_dim) float32
# committed actions sent alongside the obs window — the server routes such
# requests to predict_action_rtc instead of predict_action. The ":" makes a
# collision with a real obs key impossible (same convention as
# rgrl/inference/activations.py's "act:" response prefix).
ACTION_PREFIX_KEY = "rtc:action_prefix"


def encode_batch(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def decode_batch(data: bytes) -> dict[str, np.ndarray]:
    npz = np.load(io.BytesIO(data), allow_pickle=False)
    return {k: npz[k] for k in npz.files}
