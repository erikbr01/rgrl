"""Tee stdout to a log file for the duration of a `with` block.

Every experiment gets one self-contained directory: checkpoints, configs,
training log, and (per eval invocation) a results subfolder — see
rgrl/train_loop.py and rgrl/eval_robomimic.py.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_stdout(log_path: str | Path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", buffering=1)  # line-buffered: survives a crash
    orig = sys.stdout
    sys.stdout = _Tee(orig, f)
    try:
        yield
    finally:
        sys.stdout = orig
        f.close()
