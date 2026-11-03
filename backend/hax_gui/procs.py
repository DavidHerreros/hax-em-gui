"""Process identity helpers shared by everything that signals a pid it stored.

A pid on its own is *not* safe to signal across time — the kernel recycles them,
so a pid recorded earlier may name something else entirely now. Pairing the pid
with its start time (`proc_key`) makes reuse detectable: a recycled pid is
necessarily younger than the one recorded.
"""

from __future__ import annotations

import os


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just not ours to signal
    return True


def proc_key(pid: int) -> str | None:
    """A cheap identity for a pid: its start time, in clock ticks since boot."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    # Field 2 (comm) is parenthesised and may itself contain spaces and ')', so split
    # only what follows the *last* ')'. starttime is field 22 overall, i.e. index 19
    # of the remainder.
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return rest[19].decode()
    except (ValueError, IndexError):
        return None
