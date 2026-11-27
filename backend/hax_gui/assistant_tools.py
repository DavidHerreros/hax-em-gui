"""The chat's acting tools, backed by the GUI (the assistant's free tier).

The assistant's own launch_workflow, stop_run and relaunch_run run plain processes in plain
folders the GUI does not know. In the GUI the tool loop gets these instead: a workflow is planned
and built as the workflow card does it (assistant_workflows) and run by the workflow engine as jobs
of a GUI project; stop and relaunch go through the job manager, a relaunch being a NEW job
(watch.relaunch). The tool loop applies the autonomy level before any of them runs. They run in the
chat turn's worker thread, do their GUI work on the event loop, and do nothing once the turn is
cancelled. Also here: which GUI project a request means.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from . import assistant as assistant_bridge
from . import assistant_workflows, projects, watch
from .jobs import ACTIVE, manager

NEW_PROJECT = "a new project"
STOPPED = {"status": "not_executed", "reason": "the chat turn was stopped"}
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


def place(action: Any, text: str, interaction, default: str | None = None) -> Any:
    """A workflow or draft-job proposal with the GUI project it goes into (``project_id``)."""
    if not isinstance(action, dict) or action.get("kind") not in ("workflow", "create_job"):
        return action
    try:
        project_id = resolve_project(text, interaction, default)
    except Exception:  # noqa: BLE001 - skipped, or the turn was stopped: the chat's project
        project_id = default
    return {**action, "project_id": project_id}


async def _job_at(run_dir: str) -> dict[str, Any] | None:
    """The newest GUI job whose run folder is ``run_dir``."""
    p = Path(run_dir).expanduser().resolve()
    return next((j for j in manager.list(limit=1_000_000)
                 if j.get("workdir") and Path(j["workdir"]).name == p.name and Path(j["workdir"]).resolve() == p),
                None)


def chat_tools(loop: asyncio.AbstractEventLoop, interaction, cancel, project_id: str | None = None) -> list:
    """launch_workflow, plan_workflow, stop_run and relaunch_run for one chat turn; ``project_id`` is
    the chat's project."""
    from hax_assistant import safety
    from hax_assistant.agent import Tool
    from hax_assistant.interaction import Cancelled
    from hax_assistant.llm.base import ToolSpec

    def on_loop(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def spec(name, description, props, required):
        return ToolSpec(name, description, {"type": "object", "properties": props, "required": required})

    targets: dict[tuple, str | None] = {}
    shown: dict[tuple, dict] = {}

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
        target, out = preview({"request": request, "project": project, "name": name, "gpu": gpu})
        auto = assistant_bridge.decide_launch() == "proceed"
        try:
            made = on_loop(assistant_workflows.create(request, target, name, gpu, run=True, auto=auto,
                                                      fingerprint=out["fingerprint"]))
        except assistant_workflows.CreateError as exc:
            return {"status": "not_created", "reason": exc.detail}
        return {"status": "started" if made["started"] else "saved_not_started", "workflow": made["name"],
                "project": where_text(out), "run_error": made.get("run_error"),
                "note": "it runs in the GUI's workflow engine; its jobs show in the project"}

    def gui_job(run_dir: str) -> dict:
        job = on_loop(_job_at(run_dir))
        if job is None:
            raise ValueError(f"{run_dir} is not the folder of a job of the GUI: in the GUI the assistant "
                             "stops and relaunches the GUI's own jobs only, through its job manager")
        return job

    def describe_job(verb: str, a: dict) -> str:
        try:
            job = gui_job(a.get("run_dir", ""))
        except Exception:  # noqa: BLE001
            return f"{verb} the run in {a.get('run_dir')} (not a job of the GUI: it will be refused)"
        return f"{verb} the job “{job.get('name') or job['program']}” ({job['status']}) in {job['workdir']}"

    def stop_run(ctx, run_dir):
        if cancel.is_set():
            return STOPPED
        job = gui_job(run_dir)
        safety.require_inside(job["workdir"], "run folder")
        if job["status"] not in ACTIVE:
            return {"status": "not_running", "job_status": job["status"]}
        on_loop(manager.cancel(job["id"]))
        return {"status": "stopped", "job": job.get("name") or job["program"]}

    def describe_relaunch(a: dict) -> str:
        changes = ", ".join(f"--{k} {v}" for k, v in (a.get("changes") or {}).items()) or "no parameter changes"
        how = "as a new job in a new folder (the old run is kept)" if a.get("restart") == "scratch" else \
            "by resuming (left to the job's own Continue button)"
        return f"{describe_job('relaunch', a)} {how}; {changes}"

    def relaunch_run(ctx, run_dir, restart, changes):
        if cancel.is_set():
            return STOPPED
        try:
            done = on_loop(watch.relaunch(gui_job(run_dir), restart, changes))
        except watch.ApplyError as exc:
            return {"status": "not_relaunched", "reason": exc.detail}
        return {"status": "relaunched" if done["applied"] else "not_applied", "message": done["message"],
                "new_job": (done.get("job") or {}).get("name")}

    run_dir = {"type": "string", "description": "the run's folder (a job of the GUI)"}
    wf_args = {"request": {"type": "string", "description": "what to do, with the data paths (e.g. particles, "
                                                            "volume, mask, pixel size)"},
               "project": {"type": "string", "description": "one of the user's GUI projects (name or id) to "
                                                            "add it to; omit for the chat's project, or a new one"},
               "name": {"type": "string", "description": "workflow name"},
               "gpu": {"type": "string", "description": "GPU index"}}
    owned = lambda a: safety.is_owned(a.get("run_dir", ""))   # noqa: E731
    return [
        Tool(spec("plan_workflow", "Plan a multi-step workflow (continuous motion, improve resolution, "
                                   "reconstruction) in one of the user's GUI projects or a new one: steps, "
                                   "parameters with reasons and pre-flight checks. Creates nothing.",
                  wf_args, ["request"]), plan_workflow),
        Tool(spec("launch_workflow", "Create a workflow in one of the user's GUI projects (or a new one) and "
                                     "start it in the GUI's workflow engine; its steps run as new jobs of "
                                     "that project.", wf_args, ["request"]),
             launch_workflow, side_effect="launch", describe=describe_launch, owned=lambda a: True),
        Tool(spec("stop_run", "Stop a running job of the GUI.", {"run_dir": run_dir}, ["run_dir"]),
             stop_run, side_effect="stop", describe=lambda a: describe_job("stop", a), owned=owned),
        Tool(spec("relaunch_run", "Relaunch a job of the GUI from scratch, as a new job in a new folder (the "
                                  "old run is kept); a resume is left to the job's own Continue button.",
                  {"run_dir": run_dir, "restart": {"type": "string", "enum": ["scratch", "resume"]},
                   "changes": {"type": "object", "description": "flag name (no dashes) -> new value"}},
                  ["run_dir", "restart", "changes"]),
             relaunch_run, side_effect="relaunch", describe=describe_relaunch, owned=owned),
    ]
