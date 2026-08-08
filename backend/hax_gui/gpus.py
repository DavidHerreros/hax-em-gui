"""GPU discovery via nvidia-smi (best-effort; empty on machines without GPUs)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any

_QUERY = "index,name,memory.used,memory.total,utilization.gpu"

# The launcher (and every open tab's poll) asks for the GPU list; a fork+exec
# of nvidia-smi per request adds up. Memory/utilisation numbers a couple of
# seconds old are equally useful, so serve a short-lived cache.
_CACHE_TTL = 2.0
_cache: tuple[float, list[dict[str, Any]]] | None = None
_cache_lock = threading.Lock()


def list_gpus() -> list[dict[str, Any]]:
    global _cache
    with _cache_lock:
        if _cache is not None and time.monotonic() - _cache[0] < _CACHE_TTL:
            return _cache[1]
    gpus = _query_gpus()
    with _cache_lock:
        _cache = (time.monotonic(), gpus)
    return gpus


def _query_gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, mem_used, mem_total, util = parts[:5]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_used_mb": _int(mem_used),
                "memory_total_mb": _int(mem_total),
                "utilization_pct": _int(util),
            }
        )
    return gpus


def _int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
