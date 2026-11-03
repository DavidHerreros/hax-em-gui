"""Timing + traceback logging for the viewer backends.

HAX_VIEWER_TIMING=1 prints a per-frame breakdown of where a morph frame's time
actually goes: network vs copy-home vs marching cubes + smoothing.
"""

from __future__ import annotations

import os
import sys
import traceback

_TIMING = os.environ.get("HAX_VIEWER_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


def timing(stage: str, **parts: float) -> None:
    """Print one timing line. `parts` are seconds; shown as ms in declaration order."""
    if not _TIMING:
        return
    body = " | ".join(f"{k} {v * 1e3:8.1f} ms" for k, v in parts.items())
    print(f"[viewer:timing] {stage:12s} {body}", flush=True)


def log_exc(context: str) -> None:
    """Print the current exception's traceback to this backend's log.

    Request handlers convert exceptions into ``HTTPException`` so the browser
    gets a readable ``detail``, which discards the traceback — and a one-line
    ``detail`` is rarely enough to explain a decode failure. The manager that
    spawned the backend captures stdout+stderr per session
    (hax_gui/viewer_logs.py), so writing it here is what makes the full stack
    recoverable afterwards.
    """
    print(f"[viewer] {context}:", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.stderr.flush()
