"""Watch mode: deterministic health checks on a running run's training metrics.

"Tell me if this run goes wrong" — a watched job's event files are re-read once
a minute and three cheap heuristics run over its primary loss curve:

  diverged  — the latest value is NaN/Inf (the run is producing garbage NOW);
  rising    — the recent mean sits well above the previous window's (sustained
              climb, not a batch-to-batch wiggle);
  plateau   — the relative change over the last stretch is below noise level
              (informational: further epochs may be wasted compute).

Each alert kind fires once per watched run, through the same /ws/events bus the
job notifications use, so it lands as a desktop notification and a toast. No
LLM anywhere — these are measured numbers; the assistant can be asked about the
run afterwards, with the facts already attached.

Watches live in memory: they are a monitoring preference of the person at the
screen, not run state — a server restart simply forgets them.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from . import events
from . import metrics as metrics_reader
from .jobs import manager

INTERVAL_S = 60.0
_watched: dict[str, dict[str, Any]] = {}   # job_id -> {"fired": set[str]}
_task: asyncio.Task | None = None


def watch(job_id: str) -> dict[str, Any]:
    _watched.setdefault(job_id, {"fired": set()})
    _ensure_task()
    return {"watching": sorted(_watched)}


def unwatch(job_id: str) -> dict[str, Any]:
    _watched.pop(job_id, None)
    return {"watching": sorted(_watched)}


def watching() -> list[str]:
    return sorted(_watched)


def _ensure_task() -> None:
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_loop())


async def _loop() -> None:
    while _watched:
        await asyncio.sleep(INTERVAL_S)
        for job_id in list(_watched):
            try:
                await _check(job_id)
            except Exception:  # noqa: BLE001 - a broken check must not kill the loop
                pass


async def _check(job_id: str) -> None:
    job = manager.get(job_id)
    if job is None or job["status"] not in ("running", "pending", "queued"):
        _watched.pop(job_id, None)             # finished: the normal notification covers it
        return
    workdir = job.get("workdir")
    if not workdir or job["status"] != "running":
        return
    series = await asyncio.to_thread(_loss_series, workdir)
    if not series or len(series) < 3:
        return
    state = _watched.get(job_id)
    if state is None:
        return
    fired: set[str] = state["fired"]
    label = job.get("name") or job["program"]

    def alert(kind: str, level: str, text: str) -> None:
        if kind in fired:
            return
        fired.add(kind)
        events.publish({"kind": "watch", "job_id": job_id, "alert": kind, "level": level,
                        "title": text, "name": label, "project_id": job.get("project_id")})

    last = series[-1]
    if not math.isfinite(last):
        alert("diverged", "warn", f"{label}: loss is NaN/Inf — the run has diverged")
        return
    finite = [v for v in series if math.isfinite(v)]
    n = len(finite)
    if n >= 20:
        recent = finite[-5:]
        prior = finite[-10:-5]
        if prior and sum(recent) / 5 > 1.10 * (sum(prior) / 5) > 0:
            alert("rising", "warn",
                  f"{label}: loss is climbing ({sum(prior) / 5:.4g} → {sum(recent) / 5:.4g})")
    if n >= 30:
        span = finite[-max(10, n // 3):]
        lo, hi = min(span), max(span)
        mid = (abs(lo) + abs(hi)) / 2 or 1.0
        if (hi - lo) / mid < 0.005:
            alert("plateau", "info",
                  f"{label}: loss has plateaued at {span[-1]:.4g} — further epochs may add little")


def _loss_series(workdir: str) -> list[float]:
    """The primary loss curve: the first scalar tag containing 'loss' (any run),
    else the first scalar at all. Values in step order."""
    data = metrics_reader.scalars(workdir, max_points=500).get("tags", {})
    if not data:
        return []
    tag = next((t for t in data if "loss" in t.lower()), next(iter(data)))
    runs = data[tag]
    points = next(iter(runs.values())) if runs else []
    # The reader JSON-sanitises NaN/Inf to None; restore them as NaN so the
    # divergence check sees what actually happened.
    return [float("nan") if p[2] is None else float(p[2]) for p in points]
