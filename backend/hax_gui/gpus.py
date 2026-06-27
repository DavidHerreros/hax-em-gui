"""GPU discovery via nvidia-smi (best-effort; empty on machines without GPUs)."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

_QUERY = "index,name,memory.used,memory.total,utilization.gpu"


def list_gpus() -> list[dict[str, Any]]:
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
