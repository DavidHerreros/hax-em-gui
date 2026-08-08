"""Per-session log files for the viewer subprocesses.

The three viewer managers (:mod:`hax_gui.viewer`, :mod:`hax_gui.mrc_viewer`,
:mod:`hax_gui.md_viewer`) each spawn a backend as a subprocess. They used to send
its stdout *and* stderr to ``/dev/null``, which meant a backend that raised on a
request answered 500 with a one-line ``detail`` and left the traceback nowhere —
and a backend that died during startup produced only "failed to start".

Each session now writes to its own file here. Two things read it back: a failed
launch quotes the :func:`tail` into the error the browser sees, and the session's
log path is published in the manager's ``list()`` payload.

The parent keeps no handle. :func:`spawn_log` hands the file to ``Popen`` and
closes the parent's copy immediately — the child holds its own descriptor for as
long as it lives, so there is no lifecycle to unwind in ``stop()``/``shutdown()``.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import IO, Iterator

from .config import settings

# Logs of dead sessions are the ones worth having, so they are kept rather than
# deleted on stop. This bounds the pile: on each spawn, all but the newest N are
# removed. They are a few kB each, so N is generous.
RETAIN = 100

# How much of a log a failed launch quotes back to the caller. Enough for a JAX
# traceback plus the CUDA-plugin noise that usually precedes it.
TAIL_LINES = 40


def log_path(kind: str, vid: str) -> Path:
    """Path of the log for one session. ``kind`` is the viewer flavour
    (``viewer``/``mrcviewer``/``mdviewer``), used only to make the name readable."""
    return settings.viewer_logs_dir / f"{kind}-{vid}.log"


@contextlib.contextmanager
def spawn_log(path: Path) -> Iterator[IO[bytes]]:
    """Open ``path`` for a subprocess to inherit, then close the parent's copy.

    Falls back to ``os.devnull`` if the log directory is unwritable, so a
    read-only or full data dir degrades to the old behaviour instead of stopping
    the user from opening a viewer.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("wb")
    except OSError:
        handle = open(os.devnull, "wb")
    try:
        yield handle
    finally:
        handle.close()


def prune(kind: str) -> None:
    """Delete all but the :data:`RETAIN` newest logs of this flavour."""
    try:
        logs = sorted(settings.viewer_logs_dir.glob(f"{kind}-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for old in logs[RETAIN:]:
        with contextlib.suppress(OSError):
            old.unlink()


def tail(path: Path, lines: int = TAIL_LINES) -> str:
    """The last ``lines`` lines of a log, or ``""`` if it can't be read.

    Decoded leniently: the backends print tracebacks and progress lines, and a
    partially-written UTF-8 sequence at the tail must not mask the error we are
    trying to report.
    """
    try:
        text = path.read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:]).strip()


def bind_failure(path: Path) -> bool:
    """Did this session die because its port was taken?

    The managers probe for a free port and then let the child bind it — a window
    in which anything else on the machine can grab it. That loss is visible only
    as a startup exit whose log names the bind, which is what a caller checks
    for before retrying on a fresh port.
    """
    text = tail(path, 200).lower()
    return ("address already in use" in text
            or "error while attempting to bind" in text)


def startup_error(message: str, path: Path) -> str:
    """``message``, with the tail of the log appended when there is one.

    This is what turns "viewer backend exited during startup" into something
    actionable (a missing checkpoint, a CUDA OOM, an import error) without
    needing shell access to the machine running the GUI.
    """
    excerpt = tail(path)
    if not excerpt:
        return f"{message} (no output in {path})"
    return f"{message}\n\n--- {path.name} (last {TAIL_LINES} lines) ---\n{excerpt}"
