"""Watch mode: deterministic health checks on a running run's training metrics.

"Tell me if this run goes wrong" — a watched job's event files are re-read once
a minute and three cheap heuristics run over its primary loss curve:

  diverged  — the latest value is NaN/Inf (the run is producing garbage NOW);
  rising    — the recent mean sits well above the previous window's (sustained
              climb, not a batch-to-batch wiggle);
  plateau   — the relative change over the last stretch is below noise level
              (informational: further epochs may be wasted compute).

When the assistant is installed its rule table replaces these three (more kinds,
each with its evidence), and an actionable finding carries the restart it would
propose. Whether that is only shown, offered with an Apply button, or done right
away follows the assistant's autonomy level. Acting always goes through the job
manager and never deletes anything: a restart from scratch is a NEW job in a new
folder (the old run is kept); a resume is left to the job's own Continue button.

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

from . import adapters, events, registry
from . import assistant as assistant_bridge
from . import metrics as metrics_reader
from .command import ValidationError, strip_obsolete
from .jobs import ACTIVE, ProjectNotFound, manager

INTERVAL_S = 60.0
_watched: dict[str, dict[str, Any]] = {}   # job_id -> {"fired": set[str]}
_origins: dict[str, str] = {}              # relaunched job_id -> folder of the run the chain began with
_applying: set[str] = set()
_task: asyncio.Task | None = None


class ApplyError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


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
    if assistant_bridge.available():
        try:
            report = await asyncio.to_thread(assistant_bridge.watch_report, workdir)
        except Exception:  # noqa: BLE001 - e.g. tbparse missing: use the inline heuristics
            report = None
        if report is not None:
            if job_id in _watched:
                await _assistant_alerts(job, _watched[job_id]["fired"], report)
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


# -- assistant rules: alerts, proposals, and acting on them -------------------
async def _assistant_alerts(job: dict[str, Any], fired: set[str], report: dict[str, Any]) -> None:
    label = job.get("name") or job["program"]
    proposal = report.get("proposal")
    for f in report["findings"]:
        # `finished`: the normal job notification covers it.
        if f["kind"] == "finished" or f["kind"] in fired:
            continue
        fired.add(f["kind"])
        event = {"kind": "watch", "job_id": job["id"], "alert": f["kind"],
                 "level": "info" if f["severity"] == "info" else "warn",
                 "title": f"{label}: {f['title']}", "name": label,
                 "project_id": job.get("project_id"), "evidence": f["evidence"],
                 "severity": f["severity"], "calibrated": f["calibrated"]}
        if proposal and proposal["kind"] == f["kind"]:
            event.update(await _offer(job, f, proposal))
        events.publish(event)


async def _offer(job: dict[str, Any], finding: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    """The proposal part of an alert, per the autonomy level; at `auto` it is carried out."""
    out: dict[str, Any] = {"proposal": {k: proposal[k] for k in
                                        ("restart", "changes", "reason", "details", "text")}}
    try:
        decision = await asyncio.to_thread(assistant_bridge.decide_relaunch, job["workdir"])
    except Exception as exc:  # noqa: BLE001
        decision, out["blocked"] = "propose", str(exc)
    # As the assistant's Watcher: auto never acts on a rule that is not calibrated.
    if decision == "proceed" and not finding["calibrated"]:
        decision = "confirm"
    out["decision"] = decision
    if decision == "propose" and "blocked" not in out:
        reason = await asyncio.to_thread(assistant_bridge.ensure_workspace)
        if reason:
            out["blocked"] = reason
    elif decision == "confirm":
        out["needs_confirmation"] = True
    elif decision == "proceed":
        try:
            done = await apply_proposal(job["id"], finding["kind"], by="auto")
            out["acted"] = {"applied": done["applied"], "restart": done["restart"],
                            "message": done["message"],
                            "new_job_id": (done.get("job") or {}).get("id")}
            out["title"] = f"{job.get('name') or job['program']}: {finding['title']} {done['message']}"
        except ApplyError as exc:
            out["blocked"] = exc.detail
    return out


async def apply_proposal(job_id: str, kind: str, by: str = "user") -> dict[str, Any]:
    """Carry out the assistant's remedy for finding `kind` on a job, through the job
    manager. The remedy is re-derived here from the run's own signals."""
    if job_id in _applying:                    # a double click must not relaunch twice
        raise ApplyError(409, "a restart of this job is already in progress")
    _applying.add(job_id)
    try:
        return await _apply(job_id, kind, by)
    finally:
        _applying.discard(job_id)


async def _apply(job_id: str, kind: str, by: str) -> dict[str, Any]:
    if not assistant_bridge.available():
        raise ApplyError(400, "assistant not installed")
    job = manager.get(job_id)
    if job is None:
        raise ApplyError(404, "job not found")
    workdir = job.get("workdir")
    if not workdir or not job.get("project_id"):
        raise ApplyError(422, "this job has no run folder in a project")
    active = job["status"] in ACTIVE
    try:
        proposal = await asyncio.to_thread(assistant_bridge.remedy_for, workdir, kind, active)
    except Exception as exc:  # noqa: BLE001
        raise ApplyError(409, f"could not read the run's signals: {exc}")
    if proposal is None:
        raise ApplyError(409, f"the run no longer shows '{kind}', or no restart applies to it")
    origin = _origins.get(job_id, workdir)
    try:
        decision = await asyncio.to_thread(assistant_bridge.decide_relaunch, workdir)
        if decision == "propose":
            raise assistant_bridge.Refused(
                "autonomy is advisory, so the assistant only proposes (switch with /autonomy ask)"
                if assistant_bridge.autonomy_level() == "advisory" else
                assistant_bridge.ensure_workspace() or f"{workdir} is outside the assistant's workspaces")
        await asyncio.to_thread(assistant_bridge.guard_relaunch, job["program"], workdir,
                                proposal["changes"])
    except assistant_bridge.Refused as exc:
        _audit(job, proposal, "blocked", by, origin)
        raise ApplyError(403, str(exc))
    if proposal["restart"] == "scratch" and assistant_bridge.applied_before(origin, job_id, kind):
        raise ApplyError(409, "this proposal was already applied: the relaunched job exists")
    used, allowed = assistant_bridge.relaunch_budget(origin)
    if used >= allowed:
        raise ApplyError(409, f"relaunch budget used up ({used} of {allowed}); not relaunching")
    if proposal["restart"] == "resume":
        # the manager's continue clears the run's metrics folder: left to the user's own Continue button
        return {"applied": False, "restart": "resume", "text": proposal["text"],
                "message": "not applied: use the job's Continue button to resume from its checkpoint"}
    new = await _scratch(job, proposal["changes"])
    _audit(job, proposal, "relaunched", by, origin, new)
    _origins[new["id"]] = origin
    _register(new, workdir)
    _watched[new["id"]] = {"fired": set()}     # keep watching the relaunched run, afresh
    _ensure_task()
    where = (f"relaunched from scratch as {new.get('name') or new['id']} in a new folder "
             "(the old run is kept)")
    return {"applied": True, "restart": proposal["restart"], "job": new,
            "text": proposal["text"], "message": f"— the assistant {where}."}


async def _scratch(job: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """A NEW job in the same project (own folder) with the run's arguments plus `changes`."""
    program = job["program"]
    args = dict(job.get("args") or {})
    for arg in (registry.output_arg(program), registry.ssd_arg(program)):
        if arg:
            args.pop(arg, None)
    args, dropped = strip_obsolete(program, args)
    bindings = job.get("bindings") or None
    for field in bindings or {}:
        args.pop(field, None)         # re-materialised from the source runs into the new folder
    args.update(changes)
    try:
        manager.precheck(program, args, job.get("gpu"), bindings)
    except ValidationError as exc:
        raise ApplyError(422, f"the relaunch would be refused: {exc.errors}")
    if job["status"] in ACTIVE:
        await manager.cancel(job["id"])
    try:
        return await manager.create(program, args, job.get("gpu"), job["project_id"],
                                    name=f"{(job.get('name') or program).removesuffix(' (relaunch)')} (relaunch)",
                                    bindings=bindings, launch=True, dropped_args=dropped)
    except ProjectNotFound:
        raise ApplyError(409, "the job's project no longer exists")
    except ValidationError as exc:
        raise ApplyError(422, f"the relaunch was refused: {exc.errors}")
    except adapters.AdapterError as exc:
        raise ApplyError(422, f"the relaunch was refused: {exc}")


def _audit(job: dict[str, Any], proposal: dict[str, Any], decision: str, by: str,
           origin: str, new: dict[str, Any] | None = None) -> None:
    try:
        assistant_bridge.audit(origin, {
            "run_dir": job["workdir"], "job_id": job["id"], "finding": proposal["finding"],
            "decision": decision, "decided_by": by, "autonomy": assistant_bridge.autonomy_level(),
            "restart": proposal["restart"], "changes": proposal["changes"],
            "new_job_id": new and new["id"], "new_run_dir": new and new.get("workdir"),
            "via": "hax-gui"})
    except Exception:  # noqa: BLE001 - the audit trail is best-effort
        pass


def _register(new: dict[str, Any], origin: str) -> None:
    # A job still queued for its GPU has no pid yet: it stays unowned, so the assistant asks first.
    job = manager.get(new["id"]) or new
    if not job.get("pid") or not job.get("workdir"):
        return
    try:
        assistant_bridge.register_job(job["workdir"], job["pid"], job.get("command") or [], origin)
    except Exception:  # noqa: BLE001 - unowned just means the assistant keeps asking
        pass
