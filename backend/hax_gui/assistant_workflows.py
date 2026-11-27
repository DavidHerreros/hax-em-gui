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

from . import adapters, assistant_monitor, events, projects, registry, settings_store, watch
from . import assistant as assistant_bridge
from .command import ValidationError
from .jobs import ACTIVE, manager as jobs
from .workflows import WorkflowError, manager as wf_manager

INTERNAL_NOTES = {
    "select_states": "not added: the state selection runs inside the assistant, not as a hax program; after "
                     "the predict step, the assistant's select_states tool does it",
    "analysis": "not added: the analysis (report, 3D views, per-particle table) runs inside the assistant; "
                "write it after the run with the report button or `hax-assistant report`",
}
UNBOUND_NOTE = ("not added: it needs the few representative states the assistant selects, and the "
                "builder's latents input would hand it every particle's latent (one volume each). "
                "Decode states after the run: the assistant's select_states tool, then launch_job with the "
                "decode arguments it returns (or the per-domain motion action).")
NO_STATES_WIRE = ("decode_states_from_latents needs a few representative states, but a wired latents input "
                  "hands it every particle's latent (one volume each): end the workflow at the predict step, "
                  "then use select_states and launch_job")
WATCHED = ("hetsiren", "zernike3deep")      # their training steps are watched
REBIND_WAIT_S = 30.0          # how long a relaunch waits for the old step's workflow run to settle
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


def custom_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], gpu: str | None = None) -> dict[str, Any]:
    """A graph from steps the model wrote (``nodes`` [{id, program, args, name}], ``edges`` [{from, to,
    field}]), wired as a planned workflow is (inherited fields included). Saves nothing."""
    errors: list[str] = []
    by_id: dict[str, dict] = {}
    for n in nodes:
        nid = str(n.get("id") or "").strip()
        if not nid or nid in by_id or not n.get("program"):
            errors.append(f"every step needs its own id and a program ({nid or n})")
            continue
        by_id[nid] = n
    links: dict[str, dict[str, str]] = {nid: {} for nid in by_id}
    for e in edges:
        src, dst, field = (str(e.get(k) or "") for k in ("from", "to", "field"))
        if src not in by_id or dst not in by_id:
            errors.append(f"the edge {src} -> {dst} names a step that is not in the workflow")
            continue
        sp, dp = by_id[src]["program"], by_id[dst]["program"]
        inputs = adapters.bindable_fields(dp)
        made = adapters.job_produces(sp, by_id[src].get("args") or {})
        if field not in inputs:
            errors.append(f"{dst}: --{field} cannot take another step's output; {dp} can be wired on "
                          f"{', '.join('--' + f for f in inputs) or 'nothing'}")
        elif dp == "decode_states_from_latents" and field == "latents_file":
            errors.append(f"{dst}: {NO_STATES_WIRE}")
        elif inputs[field] not in made:
            errors.append(f"{dst}: --{field} takes {inputs[field]}, which {src} ({sp}) does not produce "
                          f"(it produces {', '.join(sorted(made)) or 'nothing another step can take'})")
        elif field in links[dst]:
            errors.append(f"{dst}: --{field} is wired twice (several sources: build it in the workflow editor)")
        else:
            links[dst][field] = src
    order, todo = [], dict(links)
    while todo:
        ready = [nid for nid, ins in todo.items() if all(s in order for s in ins.values())]
        if not ready:
            return {"graph": None, "steps": [], "errors": errors + [f"the steps {', '.join(todo)} wait on each other"]}
        order += ready
        for nid in ready:
            del todo[nid]
    steps = []
    for i, nid in enumerate(order, 1):
        n, args = by_id[nid], dict(by_id[nid].get("args") or {})
        title = n.get("name") or (f"{n['program']} {args['mode']}" if args.get("mode") else n["program"])
        steps.append({"index": i, "key": nid, "title": title, "program": n["program"], "args": args,
                      "links": {f: f"step:{s}" for f, s in links[nid].items()}, "internal": False, "note": "",
                      "watch": n["program"] in WATCHED and args.get("mode") == "train", "reasons": {}})
    built = build_graph({"steps": steps, "gpu": gpu})
    return {"graph": built["graph"], "steps": built["steps"], "errors": errors + built["errors"]}


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
                 fingerprint: str | None = None, chat_id: str | None = None) -> dict[str, Any]:
    """Save the planned workflow (and with ``run``, start it when autonomy allows). ``auto``: the
    client acts on its own, which only the ``auto`` level permits. ``fingerprint``: the graph the
    user was shown; a re-plan that differs from it is refused. ``chat_id``: the chat it was asked
    from, where the monitor's findings for its jobs are posted."""
    key = (request, project_id, name, gpu)
    if key in _creating:                       # a double click must not create it twice
        raise CreateError(409, "this workflow is already being created")
    _creating.add(key)
    try:
        return await _create(request, project_id, name, gpu, run, auto, fingerprint, chat_id)
    finally:
        _creating.discard(key)


async def _create(request, project_id, name, gpu, run, auto, fingerprint, chat_id) -> dict[str, Any]:
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
    return await _save_and_run(project_id, plan["name"], out["graph"], out["steps"], out["project"]["new"],
                               decision, run, chat_id)


async def create_custom(graph: dict[str, Any], steps: list[dict[str, Any]], project_id: str | None, name: str,
                        *, auto: bool, chat_id: str | None) -> dict[str, Any]:
    """Save and start a workflow the model designed (custom_graph), with the guards create() applies."""
    decision = await asyncio.to_thread(assistant_bridge.decide_launch)
    if decision == "propose":
        raise CreateError(403, "autonomy is advisory, so the assistant runs nothing (switch with /autonomy ask)")
    if auto and decision != "proceed":
        raise CreateError(403, "autonomy is no longer auto, so the assistant does not act on its own")
    new = not project_id
    if new:
        try:
            project_id = projects.create(name)["id"]
        except (projects.ProjectsRootUnset, OSError) as exc:
            raise CreateError(409, f"could not create the project: {exc}")
    return await _save_and_run(project_id, name, graph, steps, new, decision, True, chat_id)


async def _save_and_run(project_id, name, graph, steps, new_project, decision, run, chat_id) -> dict[str, Any]:
    wf = wf_manager.save(project_id, name, graph)
    # Open views refresh their project and workflow lists (the jobs announce themselves).
    events.publish({"kind": "workflow", "workflow_id": wf["id"], "project_id": project_id,
                    "new_project": new_project})
    result = {"workflow_id": wf["id"], "project_id": project_id, "name": wf["name"],
              "decision": decision, "started": False}
    if run:
        try:
            await wf_manager.run(wf["id"])
        except WorkflowError as exc:
            return {**result, "run_error": "; ".join(exc.errors)}
        result["started"] = True
        origin = str(projects.dir_for(project_id))
        assistant_monitor.remember_workflow(wf["id"], chat_id, origin, steps)   # followed again after a restart
        follow(wf["id"], steps, origin, chat_id)
    return result


def follow(wid: str, steps: list[dict[str, Any]], origin: str, chat_id: str | None) -> None:
    task = asyncio.get_running_loop().create_task(_follow(wid, steps, origin, chat_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _follow(wid: str, steps: list[dict[str, Any]], origin: str, chat_id: str | None,
                  every: float = 2.0) -> None:
    """Follow a workflow started here to its end: each step's job is linked to the chat (its findings go
    there) and, once it starts, registered as launched by the assistant (under auto it may then stop or
    relaunch it); a training step is watched. Parked after a restart, it is followed until resumed."""
    todo, linked = {s["node"]: s for s in steps}, set()
    while todo:
        wf = wf_manager.get(wid)
        if wf is None:
            break
        state = wf.get("run_state") or {}
        for nid, step in list(todo.items()):
            job = jobs.get((state.get(nid) or {}).get("job_id") or "")
            if job is None:
                continue
            if job["id"] not in linked:
                linked.add(job["id"])
                assistant_monitor.link(job["id"], chat_id, workflow_id=wid, workflow=wf["name"], step=step["title"])
            if job.get("pid"):
                if not assistant_bridge.owned(job["workdir"]):
                    watch.register(job, origin)
                if step["watch"]:
                    watch.watch(job["id"])
                del todo[nid]
        if wf["status"] not in ("running", "interrupted"):
            break                              # finished, failed or stopped: no step starts any more
        await asyncio.sleep(every)
    assistant_monitor.forget_workflow(wid)


class _NoRebind(Exception):
    """Why a relaunched step cannot take its place in the workflow (the workflow is left as it was)."""


async def rebind_step(old: dict[str, Any], new: dict[str, Any], origin: str) -> dict[str, Any]:
    """Put the job a relaunch from scratch just created in the place of the step it replaces, in the
    workflow the assistant launched, and run the steps after it on it. The only caller of seed_state,
    and guarded: it changes that one step's entry and nothing else, only to this new job (same program
    and project, a step of no workflow yet), only once the old job has ended and the workflow's run has
    settled, and only while the entry still points at the old job. Anything else is refused and leaves
    the workflow as it was; every attempt is audited. Returns {ok, reason, workflow, step, later}."""
    step = assistant_monitor.step_of(old["id"])
    out = {"ok": False, "reason": None, "workflow": step.get("workflow"), "step": step.get("step"), "later": []}
    wid = step.get("workflow_id")
    if not wid:
        return out                                          # not a step of a workflow launched here
    try:
        # The old job has ended and the run has settled: the scheduler's last act writes a status other
        # than "running" (its _finish, or its stop/failure handler), so nothing writes run_state after.
        # Steps of other branches still running keep it "running": then this waits and is refused.
        loop = asyncio.get_running_loop()
        end = loop.time() + REBIND_WAIT_S
        while True:
            wf, ended = wf_manager.get(wid), (jobs.get(old["id"]) or {}).get("status") not in ACTIVE
            if wf is None:
                raise _NoRebind("the workflow no longer exists")
            if ended and wf["status"] != "running":
                break
            if loop.time() >= end:
                raise _NoRebind("the old run has not ended" if not ended else
                                "the workflow is still running other steps, so its record cannot be changed safely")
            await asyncio.sleep(0.25)
        # From here on nothing awaits until the write: no other coroutine can change the workflow meanwhile.
        before = dict(wf.get("run_state") or {})
        nodes = {n["id"]: n for n in wf["graph"]["nodes"]}
        nid = next((k for k, s in before.items() if (s or {}).get("job_id") == old["id"]), None)
        if nid is None:
            raise _NoRebind("the step no longer points at the old run (it was re-run or edited meanwhile)")
        fresh = jobs.get(new["id"])
        if fresh is None:
            raise _NoRebind("the new run does not exist")
        if fresh["program"] != nodes[nid]["program"]:
            raise _NoRebind("the new run is not the step's program")
        if fresh.get("project_id") != wf.get("project_id"):
            raise _NoRebind("the new run is in another project")
        if any((s or {}).get("job_id") == new["id"] for s in before.values()) or assistant_monitor.step_of(new["id"]):
            raise _NoRebind("the new run is already a step of a workflow")
        edges, seen, queue = wf["graph"]["edges"], {nid}, [nid]
        while queue:
            cur = queue.pop(0)
            for e in edges:
                if e["from"] == cur and e["to"] not in seen:
                    seen.add(e["to"])
                    queue.append(e["to"])
        later = [n for n in nodes if n in seen - {nid} and (before.get(n) or {}).get("status") != "succeeded"]
        if not later:
            raise _NoRebind("no later step is left to run")
        after = {**before, nid: {"job_id": new["id"], "status": "running", "error": None}}
        if set(after) != set(before) or any(after[k] != before[k] for k in before if k != nid):
            raise _NoRebind("only the relaunched step may change")

        def put(state: dict[str, Any]) -> None:
            wf_manager.seed_state(wid, state)
        put(after)
        try:
            await wf_manager.run(wid, only=later)          # the scheduler adopts the new job, then runs these
        except Exception:
            put(before)                                    # as it was: the new run goes on by itself
            raise
        out.update(ok=True, later=[nodes[n].get("name") or nodes[n]["program"] for n in later])
    except _NoRebind as exc:
        out["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - a relaunch never fails because of its workflow
        out["reason"] = f"the workflow could not be restarted ({exc})"
    _audit_rebind(origin, wid, old, new, out)
    if out["ok"]:
        chat_id, folder = assistant_monitor.chat_of(old["id"]), str(projects.dir_for(wf["project_id"]))
        assistant_monitor.link(new["id"], chat_id, workflow_id=wid, workflow=wf["name"], step=step["step"])
        steps = [{"node": n, "title": nodes[n].get("name") or nodes[n]["program"], "watch": False} for n in later]
        assistant_monitor.remember_workflow(wid, chat_id, folder, steps)
        follow(wid, steps, folder, chat_id)                # its later steps are the assistant's too
    return out


def _audit_rebind(origin: str, wid: str, old: dict[str, Any], new: dict[str, Any], out: dict[str, Any]) -> None:
    try:
        assistant_bridge.audit(origin, {
            "run_dir": old.get("workdir"), "job_id": old["id"], "decision": "rebound" if out["ok"] else "rebind_refused",
            "reason": out["reason"], "workflow_id": wid, "step": out["step"], "new_job_id": new["id"],
            "later_steps": out["later"], "via": "hax-gui"})
    except Exception:  # noqa: BLE001 - the audit trail is best-effort
        pass


async def step_jobs(wid: str, wait: float = 3.0) -> list[dict[str, Any]]:
    """Each step of a workflow with its job as far as it exists, after waiting up to ``wait`` seconds for
    the first one (the engine creates it on its next tick); a step not started yet has no job or folder."""
    loop = asyncio.get_running_loop()
    end = loop.time() + wait
    while True:
        wf = wf_manager.get(wid) or {}
        state = wf.get("run_state") or {}
        if any(s.get("job_id") for s in state.values()) or loop.time() >= end:
            break
        await asyncio.sleep(0.2)
    out = []
    for node in (wf.get("graph") or {}).get("nodes", []):
        s = state.get(node["id"]) or {}
        job = jobs.get(s.get("job_id") or "")
        out.append({"step": node.get("name"), "status": job["status"] if job else s.get("status", "pending"),
                    "job_id": job["id"] if job else None, "run_dir": job.get("workdir") if job else None})
    return out
