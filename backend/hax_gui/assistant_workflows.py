"""Assistant-planned workflows, built as GUI workflow graphs.

The assistant plans a request deterministically (server-side, from the request text alone)
and this module turns that plan into a graph for the GUI's own workflow engine, so every
step runs as a job of a project and shows up there. It only ever saves a new workflow,
validates it and starts it: nothing here stops, reruns or removes anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from . import adapters, projects, registry, settings_store, watch
from . import assistant as assistant_bridge
from .command import ValidationError
from .jobs import manager as jobs
from .workflows import WorkflowError, manager as wf_manager

INTERNAL_NOTES = {
    "select_states": "not added: the state selection runs inside the assistant, not as a hax program",
    "analysis": "not added: the analysis (report, 3D views, per-particle table) runs inside the assistant; "
                "write it after the run with the report button or `hax-assistant report`",
}
UNBOUND_NOTE = ("not added: it needs the few representative states the assistant selects, and the "
                "builder's latents input would hand it every particle's latent (one volume each). "
                "Decode states after the run, e.g. with the assistant's per-domain motion action.")
WATCH_FOR_S = 600.0            # how long to wait for a training step's job to appear
_creating: set[tuple] = set()
_tasks: set[asyncio.Task] = set()


class CreateError(Exception):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status, self.detail = status, detail


def _step_key(src: str) -> str:
    """The plan step a ``step:<key>[/sub]`` or ``out:<key>:<name>`` link points at."""
    kind, _, ref = src.partition(":")
    return ref.split("/", 1)[0] if kind == "step" else ref.split(":", 1)[0]


def build_graph(plan: dict[str, Any]) -> dict[str, Any]:
    """One node per hax step of ``plan``, wired by its step links. ``steps`` says what each node
    runs and why, ``omitted`` which plan steps the builder cannot run, ``errors`` what failed."""
    nodes: list[dict] = []
    edges: list[dict] = []
    steps, omitted, errors = [], [], []
    node_of: dict[str, dict] = {}
    title_of: dict[str, str] = {}
    gpu = "" if plan.get("gpu") in (None, "") else str(plan["gpu"])
    for s in plan["steps"]:
        links = {flag: _step_key(src) for flag, src in s["links"].items()}
        if s["internal"] or any(key not in node_of for key in links.values()):
            omitted.append({"index": s["index"], "title": s["title"],
                            "reason": INTERNAL_NOTES[s["program"]] if s["internal"] else UNBOUND_NOTE})
            continue
        program = s["program"]
        auto = {registry.output_arg(program), registry.ssd_arg(program)}
        args = {k: v for k, v in s["args"].items() if k not in auto}
        node = {"id": f"n{len(nodes) + 1}", "program": program, "args": args, "gpu": gpu,
                "name": s["title"], "labels": [], "x": 0, "y": 0}
        wired, inherited = {}, {}
        for flag, key in links.items():
            src = node_of[key]
            kind = adapters.bindable_fields(program).get(flag)
            if kind is None or kind not in adapters.job_produces(src["program"], src["args"]):
                errors.append(f"{s['title']}: --{flag} cannot be wired from {src['name']} in the builder")
                continue
            edges.append({"id": f"e{len(edges) + 1}", "from": src["id"], "to": node["id"],
                          "field": flag, "kind": kind})
            args.pop(flag, None)
            wired[flag] = title_of[key]
            # As the canvas does on wiring: these come from the source run, wired or literal.
            for name in adapters.inherited_fields(program).get(flag, []):
                args.pop(name, None)
                ups = [e for e in edges if e["to"] == src["id"] and e["field"] == name]
                for e in ups:
                    edges.append({**e, "id": f"e{len(edges) + 1}", "to": node["id"]})
                if not ups and name in src["args"]:
                    args[name] = src["args"][name]
                if ups or name in src["args"]:
                    inherited[name] = title_of[key]
        nodes.append(node)
        node_of[s["key"]], title_of[s["key"]] = node, f"{s['index']}. {s['title']}"
        steps.append({"node": node["id"], "index": s["index"], "title": s["title"], "note": s["note"],
                      "program": program, "args": args, "watch": s["watch"],
                      "reasons": {k: v for k, v in s["reasons"].items() if k in args and k not in inherited},
                      "wired": wired, "inherited": inherited})
    graph = {"nodes": nodes, "edges": edges, "orientation": "horizontal"}   # zero coords: tidied on open
    return {"graph": graph, "steps": steps, "omitted": omitted, "errors": errors}


def check_graph(graph: dict[str, Any]) -> list[str]:
    """The engine's graph validation, plus per step the validation a job submission gets."""
    errors = list(wf_manager.validate(graph))
    for n in graph["nodes"]:
        values = dict(n["args"])
        for e in graph["edges"]:
            if e["to"] == n["id"]:      # filled from the upstream job at run time
                values[e["field"]] = ["/placeholder"] if adapters.is_multi_bindable(n["program"], e["field"]) \
                    else "/placeholder"
        try:
            jobs.precheck(n["program"], values, n.get("gpu") or None)
        except ValidationError as exc:
            errors += [f"{n['name']}: --{k} {v}" for k, v in exc.errors.items()]
    return errors


def _target(project_id: str | None) -> tuple[Path, dict[str, Any]]:
    try:
        if project_id:
            folder = projects.dir_for(project_id)
            if folder is None:
                raise CreateError(404, "project not found")
            return folder, {"id": project_id, "name": (projects.get(project_id) or {}).get("name"), "new": False}
        root = settings_store.projects_root()
    except projects.ProjectsRootUnset as exc:
        raise CreateError(409, str(exc))
    if root is None:
        raise CreateError(409, "No projects folder configured. Set one in Settings.")
    # Planned against a stand-in: projects.create picks the real, new folder on creation.
    return root / "(new project)", {"id": None, "name": None, "new": True}


def preview(request: str, project_id: str | None = None, name: str | None = None,
            gpu: str | None = None) -> dict[str, Any]:
    """Re-plan ``request`` and build its graph. Read-only (blocking: run it off the event loop)."""
    folder, project = _target(project_id)
    try:
        plan = assistant_bridge.plan_workflow(request, str(folder), name, gpu)
    except assistant_bridge.Refused as exc:
        raise CreateError(422, str(exc))
    built = build_graph(plan)
    errors = built["errors"] + (check_graph(built["graph"]) if built["graph"]["nodes"] else
                                ["none of the planned steps can run in the workflow builder"])
    if project["new"]:
        project["name"] = plan["name"]
    level = assistant_bridge.autonomy_level()
    fingerprint = hashlib.sha256(json.dumps(built["graph"], sort_keys=True).encode()).hexdigest()[:16]
    return {"plan": plan, "graph": built["graph"], "steps": built["steps"], "omitted": built["omitted"],
            "errors": errors, "ok": plan["ok"] and not errors, "project": project, "fingerprint": fingerprint,
            "autonomy": {"level": level, "decision": assistant_bridge.decide_launch()}}


async def create(request: str, project_id: str | None = None, name: str | None = None,
                 gpu: str | None = None, *, run: bool = False, auto: bool = False,
                 fingerprint: str | None = None) -> dict[str, Any]:
    """Save the planned workflow (and with ``run``, start it when autonomy allows). ``auto``: the
    client acts on its own, which only the ``auto`` level permits. ``fingerprint``: the graph the
    user was shown; a re-plan that differs from it is refused."""
    key = (request, project_id, name, gpu)
    if key in _creating:                       # a double click must not create it twice
        raise CreateError(409, "this workflow is already being created")
    _creating.add(key)
    try:
        return await _create(request, project_id, name, gpu, run, auto, fingerprint)
    finally:
        _creating.discard(key)


async def _create(request, project_id, name, gpu, run, auto, fingerprint) -> dict[str, Any]:
    decision = await asyncio.to_thread(assistant_bridge.decide_launch)
    if run and decision == "propose":
        raise CreateError(403, "autonomy is advisory, so the assistant runs nothing: open the workflow "
                               "as a draft and run it yourself (or switch with /autonomy ask)")
    if auto and decision != "proceed":
        raise CreateError(403, "autonomy is no longer auto, so the assistant does not act on its own")
    out = await asyncio.to_thread(preview, request, project_id, name, gpu)
    failed = [c for c in out["plan"]["checks"] if c["status"] == "fail"]
    if failed or out["errors"]:
        raise CreateError(422, {"error": "the workflow was not created", "failed_checks": failed,
                                "errors": out["errors"]})
    if fingerprint and fingerprint != out["fingerprint"]:
        raise CreateError(409, "the plan changed since it was shown (e.g. the free GPU or memory moved); "
                               "nothing was created: ask again to review the new plan")
    plan = out["plan"]
    if not project_id:
        try:
            project_id = projects.create(plan["name"])["id"]     # always a new folder
        except (projects.ProjectsRootUnset, OSError) as exc:
            raise CreateError(409, f"could not create the project: {exc}")
    wf = wf_manager.save(project_id, plan["name"], out["graph"])
    result = {"workflow_id": wf["id"], "project_id": project_id, "name": wf["name"],
              "decision": decision, "started": False}
    if run:
        try:
            await wf_manager.run(wf["id"])
        except WorkflowError as exc:
            return {**result, "run_error": "; ".join(exc.errors)}
        result["started"] = True
        _follow_training(wf["id"], [s["node"] for s in out["steps"] if s["watch"]])
    return result


def _follow_training(wid: str, node_ids: list[str]) -> None:
    if node_ids:
        task = asyncio.get_running_loop().create_task(_watch_when_launched(wid, set(node_ids)))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)


async def _watch_when_launched(wid: str, todo: set[str], every: float = 2.0) -> None:
    """Turn on health monitoring for the training steps as their jobs appear (bounded wait)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + WATCH_FOR_S
    while todo and loop.time() < end:
        wf = wf_manager.get(wid)
        if wf is None:
            return
        state = wf.get("run_state") or {}
        for nid in list(todo):
            s = state.get(nid) or {}
            if s.get("job_id"):
                watch.watch(s["job_id"])
                todo.discard(nid)
            elif s.get("status") not in ("pending", "queued", "running"):
                todo.discard(nid)              # never launched (failed, blocked, cancelled)
        if todo:
            await asyncio.sleep(every)
