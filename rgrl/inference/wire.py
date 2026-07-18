"""The HTTP wire format, defined exactly once and shared by
policy_server.py and policy_client.py: a dict of named arrays, encoded as
np.savez bytes. No pickling (allow_pickle=False) — every payload is either
uint8 image frames or float32 vectors.
"""

from __future__ import annotations

import io

import numpy as np


def encode_batch(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def decode_batch(data: bytes) -> dict[str, np.ndarray]:
    npz = np.load(io.BytesIO(data), allow_pickle=False)
    return {k: npz[k] for k in npz.files}
