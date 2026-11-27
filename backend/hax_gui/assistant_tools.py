"""The chat's acting tools, backed by the GUI (the assistant's free tier).

The assistant's own launch_workflow, stop_run and relaunch_run run plain processes in plain
folders the GUI does not know. In the GUI the tool loop gets these instead: a workflow is planned
and built as the workflow card does it (assistant_workflows) and run by the workflow engine as jobs
of a GUI project; stop and relaunch go through the job manager, a relaunch being a NEW job
(watch.relaunch), and take a job id, job name, run folder or project (find_jobs); list_jobs shows
the jobs. The tool loop applies the autonomy level before any of them acts. They run in the chat
turn's worker thread, act on the event loop, and do nothing once the turn is cancelled. Also here:
which GUI project a request means, and which job a "stop the run" means and the checked stop of the
chat's stop card (the same resolution in both tiers).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from . import assistant as assistant_bridge
from . import assistant_monitor, assistant_workflows, projects, watch
from .jobs import ACTIVE, manager

NEW_PROJECT = "a new project"
STOPPED = {"status": "not_executed", "reason": "the chat turn was stopped"}
STOPPABLE = ACTIVE - {"cancelling"}
ALL = "all of them"
LIST_MAX = 25
_stopping: set[str] = set()
_MINE_RE = re.compile(r"\b(?:my|our|existing)\s+(?:[\w.-]+\s+){0,2}project\b|\bproject\s+(?:called|named)\b", re.I)


def _listed() -> list[dict[str, Any]]:
    try:
        return projects.list_projects()
    except projects.ProjectsRootUnset:
        return []


def named_projects(text: str) -> tuple[list[dict[str, Any]], bool]:
    """The GUI projects ``text`` names (by name or id, not inside a path), and whether it asks
    for an existing project at all."""
    if not text or not re.search(r"\bprojects?\b", text, re.I):
        return [], False
    found = [p for p in _listed() if any(re.search(rf"(?<![\w./-]){re.escape(k)}(?![\w/-])", text, re.I)
                                         for k in {p["id"], p["name"]} if k)]
    return found, bool(found or _MINE_RE.search(text))


def resolve_project(text: str, interaction, default: str | None = None, name: str | None = None) -> str | None:
    """The id of the GUI project a request means (None: a new one): the project ``name`` or the text
    names, else ``default`` (the chat's). Asks when that is unclear; Skip raises ``Cancelled``."""
    from hax_assistant.interaction import Question
    if name:
        found = [p for p in _listed() if name.strip().lower() in (p["id"].lower(), p["name"].lower())]
        mine = True
    else:
        found, mine = named_projects(text)
    if len(found) == 1:
        return found[0]["id"]
    pool = found or _listed()
    if not mine or not pool or (not found and default and not name):
        return default
    labels = {(p["name"] if p["name"] == p["id"] else f"{p['name']} ({p['id']})"): p["id"] for p in pool}
    if found:
        what = f"Several of your projects match “{name}”" if name else "The request names several of your projects"
    else:
        what = f"There is no project “{name}”" if name else "The request mentions one of your projects"
    answer = interaction.ask(Question("project", f"{what}. Which project should this go into?",
                                      kind="choice", choices=[*labels, NEW_PROJECT]))
    return labels.get(answer)


def place(action: Any, text: str, interaction, default: str | None = None, route_job_id: str | None = None) -> Any:
    """A proposal with what the GUI resolves for it: the project a workflow or draft job goes into
    (``project_id``), or the jobs a stop request may mean (``jobs``; ``route_job_id`` is the page's job)."""
    if isinstance(action, dict) and action.get("kind") == "stop_job":
        return {**action, **stop_target(text, action.get("job_id") or route_job_id, default)}
    if not isinstance(action, dict) or action.get("kind") not in ("workflow", "create_job"):
        return action
    try:
        project_id = resolve_project(text, interaction, default)
    except Exception:  # noqa: BLE001 - skipped, or the turn was stopped: the chat's project
        project_id = default
    return {**action, "project_id": project_id}


class StopError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _name(job: dict[str, Any]) -> str:
    return job.get("name") or job["program"]


def _stop_decision(workdir: str, level: str | None = None) -> tuple[str, str | None]:
    """The autonomy decision for stopping the job in ``workdir``, and why it is only a proposal."""
    decision = assistant_bridge.decide_stop(workdir, level)
    if decision != "propose":
        return decision, None
    return decision, ("autonomy is advisory, so the assistant only proposes (switch with /autonomy ask)"
                      if (level or assistant_bridge.autonomy_level()) == "advisory" else
                      assistant_bridge.ensure_workspace() or f"{workdir} is outside the assistant's workspaces")


def find_jobs(ref: str, project_id: str | None = None,
              every: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], str]:
    """The GUI jobs ``ref`` names, and how: a job id or run folder ("job": that job); a job name ("name":
    the jobs so named in the project, or anywhere without one); a project (name, id or folder) or a
    workflow started here ("active": its waiting or running jobs, else "ended": its latest job, if any).
    ``([], "")`` when it names none of these."""
    ref = (ref or "").strip().rstrip("/")
    if not ref:
        return [], ""
    every = manager.list(limit=1_000_000) if every is None else every
    base = Path(ref).name if ref.startswith(("/", "~")) else ref
    hit = [j for j in every if j["id"] == ref or (j.get("workdir") and Path(j["workdir"]).name == base)]
    if hit:
        return hit[:1], "job"
    named = [j for j in every if _name(j).lower() == ref.lower() and (not project_id or j.get("project_id") == project_id)]
    if named:
        return named, "name"
    proj = next((p for p in _listed() if base.lower() in (p["id"].lower(), p["name"].lower())), None)
    if proj is not None:
        group = [j for j in every if j.get("project_id") == proj["id"]]
    else:
        ids = assistant_monitor.jobs_of_workflow(ref)
        group = [j for j in every if j["id"] in ids]
        if not group:
            return [], ""
    active = [j for j in group if j["status"] in STOPPABLE]
    return (active, "active") if active else (group[:1], "ended")


def stop_target(text: str, job_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """The jobs a stop request may mean, each with the autonomy decision for it: what the request names
    (see find_jobs), else ``job_id`` (the attached run's, else the page's), else the waiting or running
    jobs of the chat's project (of every project when it has none). A path that names nothing of the GUI
    comes back as ``unknown``; a project or workflow with nothing running as ``named``."""
    every = manager.list(limit=1_000_000)
    refs = [w.strip(".,;:!?¿¡'\"()") for w in (text or "").split()]
    refs += [p["id"] for p in _listed()
             if " " in p["name"] and re.search(rf"(?<![\w./-]){re.escape(p['name'])}(?![\w/-])", text or "", re.I)]
    jobs: list[dict[str, Any]] = []
    for ref in refs:
        found, how = find_jobs(ref, project_id, every)
        if how in ("active", "ended") and not found:
            return {"jobs": [], "named": ref}
        if found:
            jobs = found
            break
    if not jobs:
        paths = [r for r in refs if r.startswith(("/", "~"))]
        if paths:
            return {"jobs": [], "unknown": paths[0]}
        job = manager.get(job_id) if job_id else None
        jobs = [job] if job else [j for j in every if j["status"] in STOPPABLE
                                  and (not project_id or j.get("project_id") == project_id)]
    rows = []
    for j in jobs:
        row = {"id": j["id"], "name": _name(j), "program": j["program"], "status": j["status"]}
        if j["status"] in STOPPABLE and j.get("workdir"):
            row["decision"], row["reason"] = _stop_decision(j["workdir"])
        rows.append(row)
    return {"jobs": rows}


def _audit_stop(job: dict[str, Any], decision: str, by: str, level: str | None) -> None:
    try:
        assistant_bridge.audit(job["workdir"], {"run_dir": job["workdir"], "job_id": job["id"], "action": "stop",
                                                "decision": decision, "decided_by": by,
                                                "autonomy": level or assistant_bridge.autonomy_level(),
                                                "via": "hax-gui"})
    except Exception:  # noqa: BLE001 - the audit trail is best-effort
        pass


async def stop_job(job_id: str, *, auto: bool = False, by: str = "user", level: str | None = None) -> dict[str, Any]:
    """Stop a job for the chat through the job manager; its run folder and checkpoint are kept. Checked
    again here: the job still waits or runs, its folder is in a workspace, the autonomy level (``level``,
    default the setting) allows it, and ``auto`` (no click) only while the decision is proceed (the
    assistant's own job at auto)."""
    job = manager.get(job_id)
    if job is None:
        raise StopError(404, "job not found")
    if job_id in _stopping or job["status"] == "cancelling":
        raise StopError(409, "this job is already being stopped")
    if job["status"] not in STOPPABLE:
        raise StopError(409, f"the job already ended ({job['status']}): there is nothing to stop")
    decision, reason = await asyncio.to_thread(_stop_decision, job.get("workdir") or "", level)
    if decision == "propose" or (auto and decision != "proceed"):
        _audit_stop(job, "blocked", "auto" if auto else by, level)
        raise StopError(403, reason or "autonomy is no longer auto, or the assistant did not launch this job: "
                                       "stop it with a click")
    _stopping.add(job_id)
    try:
        await manager.cancel(job_id)
    finally:
        _stopping.discard(job_id)
    _audit_stop(job, "stopped", "auto" if auto else by, level)
    return {"job_id": job_id, "name": _name(job), "status": manager.get(job_id)["status"], "auto": auto}


async def _watch_line(job: dict[str, Any], chat_id: str, stop: bool) -> str:
    if stop:
        return assistant_monitor.unwatch_from_chat(job)
    if job["status"] not in STOPPABLE:
        return f"“{_name(job)}” already ended ({job['status']}): there is nothing to watch."
    return assistant_monitor.watch_from_chat(job, chat_id)


def watch_reply(action: dict[str, Any], text: str, route_job_id: str | None, project_id: str | None,
                chat_id: str | None, loop: asyncio.AbstractEventLoop) -> str:
    """The one line a guided "watch this run" (or "stop watching") gets: the run the request names, else
    the attached one, else the page's, linked to this chat and watched (or unlinked)."""
    every = manager.list(limit=1_000_000)
    named = next((found for found, _ in (find_jobs(w.strip(".,;:!?¿¡'\"()"), project_id, every)
                                         for w in text.split()) if found), [])
    jid = action.get("job_id") or route_job_id
    jobs = named or [j for j in [manager.get(jid) if jid else None] if j]
    if len(jobs) != 1:
        return "Which run? Attach it (the paperclip) or open its page, then ask again."
    if not chat_id:
        return "This chat could not be saved, so its findings could not be posted here."
    return asyncio.run_coroutine_threadsafe(_watch_line(jobs[0], chat_id, not action.get("on", True)), loop).result()


def chat_tools(loop: asyncio.AbstractEventLoop, interaction, cancel, project_id: str | None = None,
               chat_id: str | None = None, review: dict[str, Any] | None = None) -> list:
    """list_jobs, watch_run, plan_workflow, launch_workflow, stop_run and relaunch_run for one chat turn;
    ``project_id`` is the chat's project, ``chat_id`` the chat (what it launches or watches is linked to
    it). ``review``: the assistant reviewing a monitor finding (assistant_monitor): it acts only on that
    job, an OK it needs becomes an Apply in the chat, and it launches nothing."""
    from hax_assistant import safety
    from hax_assistant.agent import Tool
    from hax_assistant.interaction import Cancelled, Question
    from hax_assistant.llm.base import ToolSpec

    def on_loop(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def spec(name, description, props, required):
        return ToolSpec(name, description, {"type": "object", "properties": props, "required": required})

    targets: dict[tuple, str | None] = {}
    shown: dict[tuple, dict] = {}
    picks: dict[tuple, tuple[list[dict], str | None]] = {}

    def preview(a: dict) -> tuple[str | None, dict]:
        # resolved and planned once per turn: the confirmation shows the plan that is created
        where = (a.get("request"), a.get("project"))
        if where not in targets:
            try:
                targets[where] = resolve_project(a.get("request") or "", interaction, project_id, a.get("project"))
            except Cancelled:
                raise ValueError("no project was chosen, so nothing was planned") from None
        key = (*where, a.get("name"), a.get("gpu"))
        if key not in shown:
            shown[key] = assistant_workflows.preview(a["request"], targets[where], a.get("name"), a.get("gpu"))
        return targets[where], shown[key]

    def where_text(out: dict) -> str:
        p = out["project"]
        return f"a new project “{p['name']}”" if p["new"] else f"your project “{p['name']}”"

    def list_jobs(ctx, project=None, status=None):
        pid = project_id
        if project:
            p = next((p for p in _listed() if project.strip().lower() in (p["id"].lower(), p["name"].lower())), None)
            if p is None:
                raise ValueError(f"there is no project “{project}”")
            pid = p["id"]
        rows = manager.list_for_project(pid) if pid else manager.list(limit=1_000_000)
        if status:
            rows = [j for j in rows if j["status"] == status or (status == "active" and j["status"] in ACTIVE)]
        out = {"project": pid or "all projects", "total": len(rows), "jobs": [
            {"id": j["id"], "name": _name(j), "program": j["program"], "status": j["status"],
             "started": j.get("started_at"), "run_dir": j.get("workdir"),
             "launched_by_assistant": bool(j.get("workdir")) and safety.is_owned(j["workdir"]),
             **assistant_monitor.step_of(j["id"])} for j in rows[:LIST_MAX]]}
        if len(rows) > LIST_MAX:
            out["truncated"] = f"only the newest {LIST_MAX} of {len(rows)}; narrow it with status"
        return out

    def plan_workflow(ctx, request, project=None, name=None, gpu=None):
        _, out = preview({"request": request, "project": project, "name": name, "gpu": gpu})
        return {"template": out["plan"]["template"], "ok": out["ok"], "project": where_text(out),
                "blocking": [f"{c['name']}: {c['message']}" for c in out["plan"]["checks"] if c["status"] == "fail"]
                + out["errors"], "not_in_the_workflow": [f"{o['title']}: {o['reason']}" for o in out["omitted"]],
                "plan": out["plan"]["describe"]}

    def describe_launch(a: dict) -> str:
        try:
            _, out = preview(a)
        except Exception as e:  # noqa: BLE001
            return f"launch a workflow for: {a.get('request')} (it could not be planned: {e})"
        return (f"create and start this workflow in {where_text(out)} (its steps run as new jobs there; "
                f"nothing that exists is changed):\n{out['plan']['describe']}")

    def launch_workflow(ctx, request, project=None, name=None, gpu=None):
        if cancel.is_set():
            return STOPPED
        if review:
            return {"status": "not_executed", "reason": "a review of a finding launches no workflow"}
        target, out = preview({"request": request, "project": project, "name": name, "gpu": gpu})
        auto = assistant_bridge.decide_launch() == "proceed"
        try:
            made = on_loop(assistant_workflows.create(request, target, name, gpu, run=True, auto=auto,
                                                      fingerprint=out["fingerprint"], chat_id=chat_id))
        except assistant_workflows.CreateError as exc:
            return {"status": "not_created", "reason": exc.detail}
        return {"status": "started" if made["started"] else "saved_not_started", "run_error": made.get("run_error"),
                "project": {"id": made["project_id"], "name": (projects.get(made["project_id"]) or {}).get("name")},
                "workflow": {"id": made["workflow_id"], "name": made["name"]},
                "steps": on_loop(assistant_workflows.step_jobs(made["workflow_id"])),
                "note": "a step gets its job and run folder when it starts; list_jobs shows them, and stop_run or "
                        "relaunch_run take a job id, run folder, job name or the project"}

    def chosen(a: dict, verb: str) -> tuple[list[dict], str | None]:
        """The job(s) a stop or relaunch is about, resolved once per turn (asking when several match),
        and why there is nothing to act on."""
        ref = a.get("job") or a.get("run_dir")
        if not ref:
            raise ValueError("say which job: a job id, run folder, job name or project")
        if (ref, verb) not in picks:
            found, how = find_jobs(ref, project_id)
            if not how:
                raise ValueError(f"{ref} is not the folder of a job of the GUI, nor a job id, job name, project "
                                 "or workflow: list_jobs shows the jobs")
            if how == "ended":
                why = (f"nothing is waiting or running there; its latest job “{_name(found[0])}” is "
                       f"{found[0]['status']}") if found else "no job has run there yet"
                picks[ref, verb] = [], why
            elif len(found) == 1:
                picks[ref, verb] = found, None
            else:
                labels = {f"{_name(j)} · {j['status']} · {Path(j['workdir']).name}": j for j in found}
                try:
                    answer = interaction.ask(Question("job", f"Several jobs match “{ref}”. Which one should I {verb}?",
                                                      kind="choice", choices=[*labels, *([ALL] if verb == "stop" else [])]))
                    picks[ref, verb] = (list(labels.values()) if answer == ALL else [labels[answer]]), None
                except Cancelled:
                    picks[ref, verb] = [], "no job was chosen"
            if review and any(j["id"] != review["job_id"] for j in picks[ref, verb][0]):
                raise ValueError("a review acts only on the job of the finding it reviews")
        return picks[ref, verb]

    def describe_jobs(verb: str, a: dict) -> str:
        ref = a.get("job") or a.get("run_dir")
        try:
            found, why = chosen(a, verb)
        except Exception as e:  # noqa: BLE001
            return f"{verb} {ref} ({e})"
        if why:
            return f"{verb} {ref}: nothing to do ({why})"
        return f"{verb} " + "; ".join(f"the job “{_name(j)}” ({j['status']}) in {j['workdir']}" for j in found)

    def owned(verb: str):
        # under auto the assistant acts alone only on jobs it launched ("all of them": only when it launched
        # all); in a review, only on a calibrated finding, as the watcher
        return lambda a: (not review or review["calibrated"]) and bool(chosen(a, verb)[0]) and \
            all(safety.is_owned(j["workdir"]) for j in chosen(a, verb)[0])

    def awaiting(proposed: dict) -> dict | None:
        # in a review an OK becomes an Apply in the chat: nothing is done now
        confirming, interaction.confirming = getattr(interaction, "confirming", False), False
        if not (review and confirming):
            return None
        review["proposed"] = proposed
        return {"status": "awaiting_confirmation", "reason": "the user decides with Apply in the chat"}

    def watch_run(ctx, job=None, run_dir=None, stop=False):
        if not chat_id:
            raise ValueError("this chat is not saved, so findings cannot be posted to it")
        found, why = chosen({"job": job, "run_dir": run_dir}, "watch")
        if why:
            return {"status": "not_watched", "reason": why}
        return {"status": "unwatched" if stop else "watching", "message": on_loop(_watch_line(found[0], chat_id, stop))}

    def stop_run(ctx, job=None, run_dir=None):
        if cancel.is_set():
            return STOPPED
        found, why = chosen({"job": job, "run_dir": run_dir}, "stop")
        if why:
            return {"status": "not_running", "reason": why}
        wait = awaiting({"tool": "stop_run"})
        if wait:
            return wait
        done = []
        for j in found:
            try:
                r = on_loop(stop_job(j["id"], by="chat", level=ctx.autonomy_level))
                done.append({"job": r["name"], "job_status": r["status"]})
            except StopError as exc:
                done.append({"job": _name(j), "not_stopped": exc.detail})
        ok = sum("not_stopped" not in d for d in done)
        if review and ok:
            review["acted"] = {"applied": True, "message": f"the assistant stopped “{done[0]['job']}” (its run folder is kept)"}
        return {"status": "stopped" if ok == len(done) else "partly_stopped" if ok else "not_stopped", "jobs": done}

    def describe_relaunch(a: dict) -> str:
        changes = ", ".join(f"--{k} {v}" for k, v in (a.get("changes") or {}).items()) or "no parameter changes"
        how = "as a new job in a new folder (the old run is kept)" if a.get("restart") == "scratch" else \
            "by resuming (left to the job's own Continue button)"
        return f"{describe_jobs('relaunch', a)} {how}; {changes}"

    def relaunch_run(ctx, restart, changes, job=None, run_dir=None):
        if cancel.is_set():
            return STOPPED
        found, why = chosen({"job": job, "run_dir": run_dir}, "relaunch")
        if why:
            return {"status": "not_relaunched", "reason": why}
        if review and restart == "resume" and review["kind"] == "diverged":
            return {"status": "not_relaunched", "reason": "a diverged checkpoint is never resumed: restart from scratch"}
        wait = awaiting({"tool": "relaunch_run", "restart": restart, "changes": dict(changes or {})})
        if wait:
            return wait
        try:
            done = on_loop(watch.relaunch(found[0], restart, changes))
        except watch.ApplyError as exc:
            return {"status": "not_relaunched", "reason": exc.detail}
        if review and done["applied"]:
            review["acted"] = {"applied": True, "message": done["message"], "new_job_id": done["job"]["id"]}
        return {"status": "relaunched" if done["applied"] else "not_applied", "message": done["message"],
                "new_job": (done.get("job") or {}).get("name")}

    job = {"type": "string", "description": "the job: job id, job name, run folder, or project"}
    run_dir = {"type": "string", "description": "same as job (a run folder)"}
    wf_args = {"request": {"type": "string", "description": "what to do, with the data paths (e.g. particles, "
                                                            "volume, mask, pixel size)"},
               "project": {"type": "string", "description": "one of the user's GUI projects (name or id) to "
                                                            "add it to; omit for the chat's project, or a new one"},
               "name": {"type": "string", "description": "workflow name"},
               "gpu": {"type": "string", "description": "GPU index"}}
    return [
        Tool(spec("list_jobs", "The jobs of a GUI project (default: the chat's project; all projects when it has "
                               "none), newest first: id, name, program, status, run folder, workflow step, and "
                               "whether the assistant launched it. Read-only.",
                  {"project": {"type": "string", "description": "project name or id"},
                   "status": {"type": "string", "description": "only this status (running, queued, succeeded, "
                                                               "failed, cancelled), or active"}}, []), list_jobs),
        Tool(spec("watch_run", "Watch a job of the GUI from this chat: the run monitor's findings for it are "
                               "posted here (stop: true stops watching it). Changes it proposes still follow "
                               "the autonomy level; a job the user started is never changed without their OK.",
                  {"job": job, "run_dir": run_dir, "stop": {"type": "boolean", "description": "stop watching it"}},
                  []), watch_run),
        Tool(spec("plan_workflow", "Plan a multi-step workflow (continuous motion, improve resolution, "
                                   "reconstruction) in one of the user's GUI projects or a new one: steps, "
                                   "parameters with reasons and pre-flight checks. Creates nothing.",
                  wf_args, ["request"]), plan_workflow),
        Tool(spec("launch_workflow", "Create a workflow in one of the user's GUI projects (or a new one) and "
                                     "start it in the GUI's workflow engine; its steps run as new jobs of "
                                     "that project. Returns the project, the workflow and each step's job.",
                  wf_args, ["request"]),
             launch_workflow, side_effect="launch", describe=describe_launch, owned=lambda a: True),
        Tool(spec("stop_run", "Stop a job of the GUI (its run folder and checkpoint are kept). A project or "
                              "workflow means its running job; several are asked about.",
                  {"job": job, "run_dir": run_dir}, []),
             stop_run, side_effect="stop", describe=lambda a: describe_jobs("stop", a), owned=owned("stop")),
        Tool(spec("relaunch_run", "Relaunch a job of the GUI from scratch, as a new job in a new folder (the "
                                  "old run is kept); a resume is left to the job's own Continue button.",
                  {"job": job, "run_dir": run_dir, "restart": {"type": "string", "enum": ["scratch", "resume"]},
                   "changes": {"type": "object", "description": "flag name (no dashes) -> new value"}},
                  ["restart", "changes"]),
             relaunch_run, side_effect="relaunch", describe=describe_relaunch, owned=owned("relaunch")),
    ]
