"""Workflow routes: the visual pipeline builder's catalog, CRUD, validation,
run/stop/resume, and project import/export."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import adapters, cascade, registry
from ..jobs import manager
from ..projects import ProjectsRootUnset
from ..workflows import StepsRunning, WorkflowError, manager as wf_manager
from .shared import auth, project_or_404 as _project_or_404

router = APIRouter()

# -- workflows (visual pipeline builder + DAG engine) ----------------------
@router.get("/api/workflows/catalog", dependencies=[Depends(auth)])
def workflow_catalog() -> list[dict[str, Any]]:
    """Program palette for the canvas: each entry's bindable input fields (input
    ports) and the artifact kinds it produces (output ports)."""
    out = []
    for p in registry.list_programs():
        out.append({
            **p,
            "bindable": adapters.bindable_fields(p["name"]),
            "bindable_multi": adapters.multi_bindable_fields(p["name"]),
            "inherits": adapters.inherited_fields(p["name"]),
            "produces": adapters.produces(p["name"]),
        })
    return out


class WorkflowBody(BaseModel):
    project_id: str | None = None
    name: str | None = None
    graph: dict[str, Any] = {"nodes": [], "edges": []}


class WorkflowUpdate(BaseModel):
    name: str | None = None
    graph: dict[str, Any] | None = None
    # How many steps may run at once on any one device (GPU index, or the CPU).
    # 1 = one training per GPU; raise it only when you know two fit on one card.
    max_per_device: int | None = None


class WorkflowRun(BaseModel):
    # Optional subset of node ids to run (a single node or a selection); when
    # omitted the whole workflow runs.
    nodes: list[str] | None = None
    # Running a subset re-makes those steps' outputs, invalidating the steps they feed
    # (see cascade.py). 'queue' runs those too, after their inputs land; 'stale' just
    # flags them. Ignored when the whole graph runs — nothing is left behind to flag.
    cascade: str = "stale"
    cancel_running: bool = False
    # The user's explicit yes to stopping steps in this request that are still executing,
    # so they can be relaunched with the settings as they now stand. Without it a busy
    # step answers 409 rather than quietly letting its current run continue.
    restart_running: bool = False


class WorkflowFromProject(BaseModel):
    name: str | None = None


class WorkflowCopy(BaseModel):
    project_id: str
    name: str | None = None


@router.get("/api/workflows", dependencies=[Depends(auth)])
def list_workflows(project_id: str | None = None) -> list[dict[str, Any]]:
    return wf_manager.list(project_id)


@router.post("/api/workflows", dependencies=[Depends(auth)])
def create_workflow(body: WorkflowBody) -> dict[str, Any]:
    return wf_manager.save(body.project_id, body.name, body.graph)


@router.get("/api/workflows/resumable", dependencies=[Depends(auth)])
def resumable_workflows() -> list[dict[str, Any]]:
    """Runs the server was restarted out from under, whose next step never started.

    Declared before /api/workflows/{wid} so it isn't swallowed by it. Only workflows
    where resuming would actually *launch* something land here: one whose step is still
    running was resumed silently at startup, and one with nothing left to do was settled.
    """
    return wf_manager.list_resumable()


@router.post("/api/workflows/{wid}/resume", dependencies=[Depends(auth)])
async def resume_workflow(wid: str) -> dict[str, Any]:
    """Carry on from where the run stopped. Completed steps keep their jobs and are not
    recomputed; the steps still queued are launched in order."""
    try:
        return await wf_manager.resume(wid)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@router.post("/api/workflows/{wid}/dismiss-resume", dependencies=[Depends(auth)])
def dismiss_workflow_resume(wid: str) -> dict[str, Any]:
    """Stop offering to resume this run (its queue is kept, so the canvas can still run it)."""
    wf = wf_manager.dismiss_resume(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf


@router.get("/api/workflows/{wid}", dependencies=[Depends(auth)])
def get_workflow(wid: str) -> dict[str, Any]:
    # Reconcile a project workflow with its jobs' live statuses on load (no-op for
    # standalone workflows / while running), so stale node states self-heal.
    wf = wf_manager.reconcile(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf


@router.put("/api/workflows/{wid}", dependencies=[Depends(auth)])
def update_workflow(wid: str, body: WorkflowUpdate) -> dict[str, Any]:
    try:
        return wf_manager.update(wid, name=body.name, graph=body.graph,
                                 max_per_device=body.max_per_device)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@router.delete("/api/workflows/{wid}", dependencies=[Depends(auth)])
def delete_workflow(wid: str) -> dict[str, str]:
    wf_manager.delete(wid)
    return {"status": "deleted"}


@router.post("/api/workflows/{wid}/validate", dependencies=[Depends(auth)])
def validate_workflow(wid: str) -> dict[str, Any]:
    wf = wf_manager.get(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return {"errors": wf_manager.validate(wf["graph"])}


@router.post("/api/workflows/{wid}/run", dependencies=[Depends(auth)])
async def run_workflow(wid: str, body: WorkflowRun = WorkflowRun()) -> dict[str, Any]:
    """Queue steps. When a run is already in flight the steps are appended to its
    queue and the live scheduler picks them up in turn.

    Running a *subset* re-makes those steps under everything they feed, so the steps
    downstream are invalidated the same way a rerun from the Jobs view invalidates them
    — and, with ``cascade='queue'``, simply join the same queue, which is where the
    dependency ordering they need already lives."""
    try:
        nodes = body.nodes
        if nodes:
            # From the graph, not the jobs' bindings: a step wired to a not-yet-run one
            # has no binding to find yet, and it is still downstream.
            down = wf_manager.downstream_nodes(wid, nodes)
            if down:
                deps = [j for j in (manager.get(jid) for jid in wf_manager.jobs_for_nodes(wid, down)) if j]
                await cascade.invalidate(deps, cancel_running=body.cancel_running)
                if body.cascade == "queue":
                    nodes = list(nodes) + down
        return await wf_manager.run(wid, nodes, restart_running=body.restart_running)
    except StepsRunning as exc:
        raise HTTPException(status_code=409, detail={
            "error": "those steps are still running", "steps": exc.steps})
    except cascade.DependentsRunning as exc:
        raise HTTPException(status_code=409, detail={
            "error": "downstream runs are still executing", "running": exc.running})
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")
    except WorkflowError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors})


@router.post("/api/workflows/{wid}/stop", dependencies=[Depends(auth)])
async def stop_workflow(wid: str) -> dict[str, Any]:
    """Halt a run: cancel the step executing now and dequeue everything still waiting."""
    try:
        return await wf_manager.stop(wid)
    except KeyError:
        raise HTTPException(status_code=404, detail="workflow not found")


@router.post("/api/workflows/from-project/{project_id}", dependencies=[Depends(auth)])
def workflow_from_project(project_id: str, body: WorkflowFromProject = WorkflowFromProject()) -> dict[str, Any]:
    """Open a project as a *live* workflow (get-or-create): nodes = the project's
    jobs, edges inferred from input bindings, kept in sync with the project. Editing
    / running / labeling this workflow modifies the actual project."""
    proj = _project_or_404(project_id)
    return wf_manager.open_project(project_id, body.name or proj["name"])


@router.post("/api/workflows/{wid}/copy", dependencies=[Depends(auth)])
def workflow_copy(wid: str, body: WorkflowCopy) -> dict[str, Any]:
    """Clone a workflow's graph into a new standalone workflow owned by another
    project. Self-contained: node/job links dropped, external data inputs cleared,
    internal dependencies preserved so it re-runs cleanly in the target project."""
    if wf_manager.get(wid) is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    _project_or_404(body.project_id)
    try:
        return wf_manager.copy_to_project(wid, body.project_id, body.name)
    except KeyError:
        raise HTTPException(status_code=404, detail="target project not found")


@router.post("/api/workflows/{wid}/save-as-project", dependencies=[Depends(auth)])
def workflow_save_as_project(wid: str, body: WorkflowFromProject = WorkflowFromProject()) -> dict[str, Any]:
    """Create a new project and bind this workflow to it as the project's live
    view, so further edits/runs of the workflow populate the new project."""
    wf = wf_manager.get(wid)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    name = (body.name or wf.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="project name required")
    try:
        return wf_manager.save_as_project(wid, name)
    except ProjectsRootUnset as exc:
        raise HTTPException(status_code=409, detail=str(exc))

