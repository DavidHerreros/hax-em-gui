"""Keep a project honest when a run is re-made: invalidate everything derived from it.

Re-running a step does not just replace that step's outputs. Its run folder is emptied
and rebuilt in place (``JobManager.create`` with ``replace_job_id``), and every run
downstream was fed *from that folder*: ``adapters.produce`` either hands the consumer a
path inside it (checkpoints, particle metadata, volumes) or copies a file out of it
(latents). So the moment an upstream run is re-made, its dependents are reading a path
whose contents were thrown away — absent until the new run rewrites it, and then a
different result than the one they were computed from — or holding a copy of data that no
longer exists, all while still displaying as ``succeeded``. Nothing used to say so.

(The folder keeps its *name* across a rerun, which is why this is about content and not
about broken paths. Both are equally invalidating downstream; only one of them used to
also break the stored command outright.)

This module is the one place that knows what to do about it:

* :func:`dependents_of` — the closure of runs a set of jobs feeds, from their bindings.
* :func:`invalidate` — mark them stale (and, only when the user has said so, cancel the
  ones still running, since they are consuming the folder about to be deleted).
* :func:`queue_jobs` — hand them to the project's workflow scheduler, which is already
  the thing that knows how to wait for an upstream and re-run a project job in place.

Marking is deliberately not deletion: see :meth:`JobManager.mark_stale`. The caller is
expected to have warned the user first — every entry point goes through a dialog.
"""

from __future__ import annotations

from typing import Any

from .jobs import manager as jobs
from .workflows import manager as workflows

# What the caller asked to happen to the invalidated runs.
CASCADE_MODES = ("stale", "queue")


class DependentsRunning(Exception):
    """Downstream runs are still executing and the caller did not authorise stopping them.

    Not merely a courtesy: they are reading the run folder that is about to be emptied
    out from under them, so letting them continue would produce a result from half an
    input. The caller re-submits with ``cancel_running=True`` once the user has agreed.
    """

    def __init__(self, running: list[dict[str, Any]]):
        self.running = running
        names = ", ".join(brief(j)["label"] for j in running)
        super().__init__(f"downstream runs are still active: {names}")


def brief(job: dict[str, Any]) -> dict[str, Any]:
    """The little a dialog needs to name a run and say what state it is in."""
    return {
        "id": job["id"],
        "label": job.get("name") or job["program"],
        "program": job["program"],
        "status": job["status"],
        "active": bool(job.get("active")),
        "stale": bool(job.get("stale")),
    }


def dependents_of(job_ids: list[str]) -> list[dict[str, Any]]:
    """Every run downstream of any of `job_ids`, deduplicated, none of them in the set."""
    seen: dict[str, dict[str, Any]] = {}
    for jid in job_ids:
        for dep in jobs.dependents(jid):
            if dep["id"] not in job_ids:
                seen[dep["id"]] = dep
    return list(seen.values())


async def invalidate(dependents: list[dict[str, Any]], cancel_running: bool = False) -> dict[str, Any]:
    """Mark `dependents` stale, having first stopped any that are still running.

    Raises :class:`DependentsRunning` unless the caller has explicitly authorised the
    cancellation — nothing here kills a training the user has not been told about.
    """
    running = [d for d in dependents if d.get("active")]
    if running and not cancel_running:
        raise DependentsRunning([dict(r) for r in running])
    for job in running:
        await jobs.cancel(job["id"])
    marked = jobs.mark_stale([d["id"] for d in dependents])
    return {"stale": marked, "cancelled": [r["id"] for r in running]}


async def queue_jobs(project_id: str | None, job_ids: list[str]) -> dict[str, Any] | None:
    """Queue these runs to re-run themselves, in dependency order, once their inputs land.

    Delegated wholesale to the project's live workflow: its scheduler already waits for
    an upstream to succeed before starting a step, re-runs a project job *in place*
    rather than duplicating it, and respects the per-device limit. Nothing here needs to
    know about any of that — it only has to name the steps.

    The step whose rerun triggered this is deliberately *not* queued: it has already been
    launched by the caller (as a plain rerun, or a continue-from-checkpoint, which the
    scheduler has no way to express). The scheduler adopts it — it sees a node whose job
    is running and waits it out — so the dependents still start only once it succeeds.
    """
    if not project_id or not job_ids:
        return None
    wf = workflows.open_project(project_id)
    nodes = workflows.nodes_for_jobs(wf["id"], job_ids)
    if not nodes:
        return None
    await workflows.run(wf["id"], only=nodes)
    return {"workflow_id": wf["id"], "nodes": nodes}
