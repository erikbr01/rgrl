"""Logging setup shared by the training/eval loops.

Every module gets its own logger via `logging.getLogger(__name__)`
("rgrl.train_loop", "rgrl.eval_robomimic", ...), all children of the "rgrl"
logger configured here with one console handler — the standard way to get
per-module verbosity control later without touching call sites.

Per-run file capture (log_to_file) attaches to the ROOT logger, not to
"rgrl"'s console handler, so it also picks up records from libraries that
log independently (e.g. robosuite's warnings) without printing them twice:
robosuite attaches its own console handler directly to its own logger
(propagate=True by default), so a record it emits already prints once via
that handler and propagates up to root — attaching a *console* handler at
root would print it again, but a *file* handler at root just gives it a
second, silent destination.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

_FORMATTER = logging.Formatter("%(message)s")

_pkg_logger = logging.getLogger("rgrl")
_pkg_logger.setLevel(logging.INFO)
if not _pkg_logger.handlers:  # idempotent across re-imports / multiple entrypoints
    _console = logging.StreamHandler()
    _console.setFormatter(_FORMATTER)
    _pkg_logger.addHandler(_console)


@contextmanager
def log_to_file(log_path: str | Path):
    """Also write every log record — rgrl's own plus any propagating library
    record — to log_path for the duration of the block."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(_FORMATTER)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
