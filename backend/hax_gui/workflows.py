"""Workflow engine: persist a DAG of program steps and execute it.

A workflow is a node graph (built on the canvas in the SPA):
  nodes  = [{id, program, args, gpu, name, x, y}, ...]
  edges  = [{id, "from": <node id>, to: <node id>, field, kind}, ...]
An edge wires an upstream node's output (of artifact `kind`) into a downstream
node's bindable input `field`. Execution reuses the existing job manager + input
bindings: each node becomes a real job, and a downstream node is launched with a
binding pointing at its upstream node's job once that job has succeeded.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from . import adapters, projects, registry
from .config import settings
from .jobs import ACTIVE, manager as jobs
from .labels import manager as labels


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowError(Exception):
    """Validation/build error with one or more user-facing messages."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class WorkflowManager:
    def __init__(self) -> None:
        self._db: sqlite3.Connection | None = None
        self._running: dict[str, asyncio.Task] = {}

    def init(self) -> None:
        settings.ensure_dirs()
        self._db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                name TEXT,
                graph TEXT NOT NULL,
                run_state TEXT,
                status TEXT NOT NULL DEFAULT 'idle',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # `kind`: 'standalone' (a saved graph whose runs create new jobs) or
        # 'project' (a live, editable view of a project — runs rerun the project's
        # own jobs in place). Added idempotently for older databases.
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(workflows)").fetchall()}
        if "kind" not in cols:
            self._db.execute("ALTER TABLE workflows ADD COLUMN kind TEXT NOT NULL DEFAULT 'standalone'")
        # A workflow left 'running' from a previous server process is stale.
        self._db.execute("UPDATE workflows SET status='interrupted' WHERE status='running'")
        self._db.commit()

    # -- CRUD --------------------------------------------------------------
    def save(self, project_id: str | None, name: str | None, graph: dict[str, Any],
             kind: str = "standalone") -> dict[str, Any]:
        wid = uuid.uuid4().hex[:12]
        now = _now()
        self._db.execute(
            "INSERT INTO workflows (id, project_id, name, graph, run_state, status, created_at, updated_at, kind) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (wid, project_id, (name or "").strip() or "Untitled workflow", json.dumps(graph), None, "idle", now, now, kind),
        )
        self._db.commit()
        return self.get(wid)

    def update(self, wid: str, name: str | None = None, graph: dict[str, Any] | None = None) -> dict[str, Any]:
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        sets, params = [], []
        if name is not None:
            sets.append("name=?"); params.append(name.strip() or "Untitled workflow")
        if graph is not None:
            sets.append("graph=?"); params.append(json.dumps(graph))
        if not sets:
            return wf
        sets.append("updated_at=?"); params.append(_now())
        params.append(wid)
        self._db.execute(f"UPDATE workflows SET {', '.join(sets)} WHERE id=?", params)
        self._db.commit()
        return self.get(wid)

    def get(self, wid: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM workflows WHERE id=?", (wid,)).fetchone()
        return self._row(row) if row else None

    def list(self, project_id: str | None = None) -> list[dict[str, Any]]:
        if project_id:
            rows = self._db.execute(
                "SELECT * FROM workflows WHERE project_id=? ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT * FROM workflows ORDER BY updated_at DESC").fetchall()
        return [self._row(r) for r in rows]

    def delete(self, wid: str) -> None:
        task = self._running.pop(wid, None)
        if task and not task.done():
            task.cancel()
        self._db.execute("DELETE FROM workflows WHERE id=?", (wid,))
        self._db.commit()

    def seed_state(self, wid: str, state: dict) -> dict[str, Any]:
        """Attach a run_state to a workflow without marking it running. Used when
        importing a project so each node shows its existing job's status."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        self._db.execute(
            "UPDATE workflows SET run_state=?, updated_at=? WHERE id=?",
            (json.dumps(state), _now(), wid),
        )
        self._db.commit()
        return self.get(wid)

    # -- a project as a live workflow --------------------------------------
    @staticmethod
    def _clean_args(job: dict[str, Any]) -> dict[str, Any]:
        """A job's args minus auto-managed dirs and bound (edge-fed) fields."""
        args = dict(job.get("args") or {})
        for auto in (registry.output_arg(job["program"]), registry.ssd_arg(job["program"])):
            if auto:
                args.pop(auto, None)
        for field in (job.get("bindings") or {}):
            args.pop(field, None)
        return args

    @staticmethod
    def _labels_of(job_id: str) -> list[str]:
        try:
            return [lab["id"] for lab in labels.labels_for(job_id)]
        except Exception:  # noqa: BLE001
            return []

    def build_from_project(self, project_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Infer a fresh workflow graph from a project's jobs. Each job becomes a
        node (carrying its `job_id`); each job's input bindings become edges."""
        job_list = list(reversed(jobs.list_for_project(project_id)))  # oldest first
        id_map = {j["id"]: f"n{i + 1}" for i, j in enumerate(job_list)}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        state: dict[str, Any] = {}
        eseq = 1
        for j in job_list:
            nid = id_map[j["id"]]
            nodes.append({"id": nid, "program": j["program"], "args": self._clean_args(j),
                          "gpu": j.get("gpu") or "0", "name": j.get("name") or "",
                          "labels": self._labels_of(j["id"]), "x": 0, "y": 0, "job_id": j["id"]})
            state[nid] = {"job_id": j["id"], "status": j["status"], "error": None}
            for field, b in (j.get("bindings") or {}).items():
                for one in (b if isinstance(b, list) else [b]):   # multi-valued inputs -> many edges
                    if one.get("job_id") in id_map:
                        edges.append({"id": f"e{eseq}", "from": id_map[one["job_id"]], "to": nid,
                                      "field": field, "kind": one.get("kind")})
                        eseq += 1
        return {"nodes": nodes, "edges": edges, "orientation": "horizontal"}, state

    def open_project(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        """Get-or-create the project's *live* workflow. Re-opening returns the same
        workflow, reconciled with the project's current jobs (added/removed jobs,
        statuses, labels), so the canvas stays a true view of the project."""
        row = self._db.execute(
            "SELECT * FROM workflows WHERE project_id=? AND kind='project' ORDER BY created_at LIMIT 1",
            (project_id,),
        ).fetchone()
        if row is not None:
            wid = self._row(row)["id"]
            graph, state = self._synced_graph(self._row(row)["graph"], project_id)
            self._db.execute(
                "UPDATE workflows SET graph=?, run_state=?, updated_at=? WHERE id=?",
                (json.dumps(graph), json.dumps(state), _now(), wid),
            )
            self._db.commit()
            return self.get(wid)
        graph, state = self.build_from_project(project_id)
        wf = self.save(project_id, name or "Project workflow", graph, kind="project")
        return self.seed_state(wf["id"], state)

    def reconcile(self, wid: str) -> dict[str, Any] | None:
        """Refresh a *project* workflow from its jobs' current statuses, so the
        canvas is a true live view (a job that finished/cancelled outside a
        workflow run no longer shows a stale node status, and downstream nodes
        un-block once their inputs actually succeed). No-op for standalone
        workflows and while a run is in progress."""
        wf = self.get(wid)
        if wf is None:
            return None
        if wf.get("kind") != "project" or not wf.get("project_id"):
            return wf
        # Skip only while *this* process is genuinely running the workflow (its
        # _execute task owns run_state then). A stored status of 'running' with no
        # live task is stale — e.g. the run task ended abnormally, or the job was
        # driven/finished outside a workflow run — so we must still reconcile from
        # the jobs' real statuses instead of leaving nodes stuck on 'running'.
        if wid in self._running and not self._running[wid].done():
            return wf
        graph, state = self._synced_graph(wf["graph"], wf["project_id"])
        statuses = [s.get("status") for s in state.values()]
        overall = ("failed" if "failed" in statuses
                   else "running" if any(s in ("pending", "running", "cancelling") for s in statuses)
                   else "succeeded" if statuses and all(s == "succeeded" for s in statuses)
                   else "idle")
        self._db.execute(
            "UPDATE workflows SET graph=?, run_state=?, status=?, updated_at=? WHERE id=?",
            (json.dumps(graph), json.dumps(state), overall, _now(), wid),
        )
        self._db.commit()
        return self.get(wid)

    def _synced_graph(self, stored: dict[str, Any], project_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reconcile a stored project-workflow graph with the project's jobs:
        keep node positions / user-added (not-yet-run) nodes, drop nodes whose job
        was deleted, refresh linked nodes from their job, add nodes for new jobs,
        and rebuild edges from job bindings (plus user edges among un-run nodes)."""
        job_list = list(reversed(jobs.list_for_project(project_id)))
        by_job = {j["id"]: j for j in job_list}
        nodes = [dict(n) for n in stored.get("nodes", [])]
        # 1. drop nodes whose linked job vanished
        nodes = [n for n in nodes if not n.get("job_id") or n["job_id"] in by_job]
        node_for_job = {n["job_id"]: n for n in nodes if n.get("job_id")}
        next_n = max([int(re.sub(r"\D", "", n["id"]) or 0) for n in nodes], default=0)
        # 2. refresh linked nodes; add a node for each job not yet represented
        for j in job_list:
            n = node_for_job.get(j["id"])
            if n is None:
                next_n += 1
                n = {"id": f"n{next_n}", "program": j["program"], "x": 0, "y": 0, "job_id": j["id"]}
                nodes.append(n); node_for_job[j["id"]] = n
            n["program"] = j["program"]
            n["args"] = self._clean_args(j)
            n["gpu"] = j.get("gpu") or n.get("gpu") or "0"
            n["name"] = j.get("name") or ""
            n["labels"] = self._labels_of(j["id"])
        present = {n["id"] for n in nodes}
        # 3. edges: from job bindings (linked) + stored edges touching an un-run node
        binding_edges = []
        for j in job_list:
            nid = node_for_job[j["id"]]["id"]
            for field, b in (j.get("bindings") or {}).items():
                for one in (b if isinstance(b, list) else [b]):
                    src = node_for_job.get(one.get("job_id"))
                    if src:
                        binding_edges.append({"from": src["id"], "to": nid, "field": field, "kind": one.get("kind")})
        linked_ids = {n["id"] for n in nodes if n.get("job_id")}
        user_edges = [e for e in stored.get("edges", [])
                      if e.get("from") in present and e.get("to") in present
                      and (e["from"] not in linked_ids or e["to"] not in linked_ids)]
        seen, edges, eseq = set(), [], 0
        for e in binding_edges + user_edges:
            key = (e["from"], e["to"], e.get("field"))
            if key in seen:
                continue
            seen.add(key); eseq += 1
            edges.append({"id": f"e{eseq}", "from": e["from"], "to": e["to"], "field": e.get("field"), "kind": e.get("kind")})
        graph = {"nodes": nodes, "edges": edges, "orientation": stored.get("orientation", "horizontal"),
                 "savedLayout": stored.get("savedLayout")}
        state = {}
        for n in nodes:
            jid = n.get("job_id")
            if jid and jid in by_job:
                state[n["id"]] = {"job_id": jid, "status": by_job[jid]["status"], "error": None}
            else:
                state[n["id"]] = {"job_id": None, "status": "idle", "error": None}
        return graph, state

    def save_as_project(self, wid: str, name: str) -> dict[str, Any]:
        """Create a new project and bind this workflow to it as the project's live
        view, so running/editing the workflow populates the new project."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        proj = projects.create(name)
        self._db.execute(
            "UPDATE workflows SET project_id=?, kind='project', name=?, updated_at=? WHERE id=?",
            (proj["id"], (name or "").strip() or "Project workflow", _now(), wid),
        )
        self._db.commit()
        return self.get(wid)

    # -- validation --------------------------------------------------------
    @staticmethod
    def topo_order(nodes: list[dict], edges: list[dict]) -> list[str]:
        """Kahn topological sort; raises WorkflowError on a cycle."""
        ids = [n["id"] for n in nodes]
        indeg = {i: 0 for i in ids}
        adj: dict[str, list[str]] = {i: [] for i in ids}
        for e in edges:
            if e["from"] in adj and e["to"] in indeg:
                adj[e["from"]].append(e["to"])
                indeg[e["to"]] += 1
        queue = [i for i in ids if indeg[i] == 0]
        order: list[str] = []
        while queue:
            i = queue.pop(0)
            order.append(i)
            for j in adj[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
        if len(order) != len(ids):
            raise WorkflowError(["the workflow has a cycle — wiring must form a DAG"])
        return order

    def validate(self, graph: dict[str, Any]) -> list[str]:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        errors: list[str] = []
        if not nodes:
            return ["the workflow is empty — add at least one program"]
        by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            if not registry.get_schema(n.get("program", "")):
                errors.append(f"unknown program '{n.get('program')}'")
        for e in edges:
            src, dst = by_id.get(e.get("from")), by_id.get(e.get("to"))
            if src is None or dst is None:
                errors.append("an edge references a missing node"); continue
            field, kind = e.get("field"), e.get("kind")
            field_kind = adapters.bindable_fields(dst["program"]).get(field)
            if field_kind is None:
                errors.append(f"'{dst['program']}' has no bindable input '{field}'")
            elif kind != field_kind:
                errors.append(f"input '{dst['program']}.{field}' expects '{field_kind}', not '{kind}'")
            elif kind not in adapters.produces(src["program"]):
                errors.append(f"'{src['program']}' cannot provide '{kind}' for '{dst['program']}.{field}'")
        # one source per input field (a field can't be wired twice) — except
        # multi-valued inputs (e.g. FlexConsensus --input_space), which take many.
        seen: set[tuple] = set()
        for e in edges:
            dst = by_id.get(e.get("to"))
            if dst and adapters.is_multi_bindable(dst["program"], e.get("field")):
                continue
            key = (e.get("to"), e.get("field"))
            if key in seen:
                errors.append(f"input '{e.get('field')}' on a node is wired more than once")
            seen.add(key)
        try:
            self.topo_order(nodes, edges)
        except WorkflowError as exc:
            errors.extend(exc.errors)
        return errors

    # -- execution ---------------------------------------------------------
    async def run(self, wid: str, only: list[str] | None = None) -> dict[str, Any]:
        """Run the whole workflow, or just the nodes in `only` (a subset run). For
        a subset run, nodes outside the set keep their existing run_state, so an
        already-succeeded upstream node supplies its job to the nodes being run."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        if wf["status"] == "running" or (wid in self._running and not self._running[wid].done()):
            raise WorkflowError(["workflow is already running"])
        errors = self.validate(wf["graph"])
        if errors:
            raise WorkflowError(errors)
        nodes = wf["graph"]["nodes"]
        ids = {n["id"] for n in nodes}
        only_set = set(only) & ids if only is not None else None
        if only is not None and not only_set:
            raise WorkflowError(["none of the selected steps exist in this workflow"])
        prev = wf["run_state"] or {}
        # nodes being run reset to pending; others keep their prior state (job + status)
        state = {}
        for n in nodes:
            if only_set is None or n["id"] in only_set:
                state[n["id"]] = {"job_id": None, "status": "pending", "error": None}
            else:
                state[n["id"]] = prev.get(n["id"], {"job_id": None, "status": "idle", "error": None})
        self._set_run(wid, "running", state)
        self._running[wid] = asyncio.create_task(self._execute(wid, only_set))
        return self.get(wid)

    async def _execute(self, wid: str, only: set[str] | None = None) -> None:
        try:
            wf = self.get(wid)
            graph = wf["graph"]
            nodes = {n["id"]: n for n in graph["nodes"]}
            edges = graph["edges"]
            project_id = wf["project_id"]
            is_project = wf.get("kind") == "project"   # live view: rerun jobs in place
            graph_dirty = False
            order = self.topo_order(graph["nodes"], edges)
            state = wf["run_state"] or {i: {"job_id": None, "status": "pending", "error": None} for i in nodes}

            for nid in order:
                if only is not None and nid not in only:
                    continue  # subset run: leave untouched (its job feeds downstream deps)
                deps = [e for e in edges if e["to"] == nid]
                # block if any upstream did not succeed
                if any(state.get(e["from"], {}).get("status") != "succeeded" for e in deps):
                    state[nid] = {"job_id": None, "status": "blocked",
                                  "error": "an upstream step did not succeed"}
                    self._set_run(wid, "running", state)
                    continue

                node = nodes[nid]
                args = dict(node.get("args", {}))
                for auto in (registry.output_arg(node["program"]), registry.ssd_arg(node["program"])):
                    if auto:
                        args.pop(auto, None)
                bindings: dict[str, Any] = {}
                for e in deps:
                    one = {"job_id": state[e["from"]]["job_id"], "kind": e["kind"]}
                    if adapters.is_multi_bindable(node["program"], e["field"]):
                        bindings.setdefault(e["field"], []).append(one)   # multi-valued input
                    else:
                        bindings[e["field"]] = one
                for field in bindings:           # drop any stale literal so the binding wins
                    args.pop(field, None)
                # In a live project workflow, a node linked to an existing job
                # reruns *that* job in place (modifying the project), instead of
                # creating a duplicate. New nodes create a job and get linked.
                replace = node.get("job_id") if (is_project and node.get("job_id") and jobs.get(node["job_id"])) else None
                try:
                    job = await jobs.create(node["program"], args, node.get("gpu"), project_id,
                                            name=node.get("name"), bindings=bindings, replace_job_id=replace)
                except Exception as exc:  # noqa: BLE001 - surface build/validation failures per node
                    state[nid] = {"job_id": None, "status": "failed", "error": str(exc)}
                    self._set_run(wid, "running", state)
                    continue

                jid = job["id"]
                if is_project and node.get("job_id") != jid:
                    node["job_id"] = jid; graph_dirty = True   # link the new job to its node
                if node.get("labels"):           # carry the node's labels onto the job
                    try:
                        labels.set_job_labels(jid, node["labels"])
                    except Exception:  # noqa: BLE001 - labels are non-critical
                        pass
                state[nid] = {"job_id": jid, "status": "running", "error": None}
                self._set_run(wid, "running", state)
                final = await self._wait_job(jid)
                state[nid] = {"job_id": jid, "status": final,
                              "error": None if final == "succeeded" else "step failed (see the job log)"}
                self._set_run(wid, "running", state)

            if graph_dirty:           # persist new node<->job links for the live view
                self._db.execute("UPDATE workflows SET graph=? WHERE id=?", (json.dumps(graph), wid))
                self._db.commit()
            # judge overall status by the nodes actually run this pass
            judged = [s for nid, s in state.items() if only is None or nid in only]
            done = "succeeded" if all(s["status"] == "succeeded" for s in judged) else "failed"
            self._set_run(wid, done, state)
        except asyncio.CancelledError:
            self._set_run(wid, "cancelled", self.get(wid)["run_state"])
            raise
        except Exception as exc:  # noqa: BLE001
            wf = self.get(wid)
            self._set_run(wid, "failed", (wf or {}).get("run_state") or {})
        finally:
            self._running.pop(wid, None)

    async def _wait_job(self, job_id: str) -> str:
        """Poll until the job reaches a terminal status; return it. ``cancelling``
        is transitional (the process is still being torn down), so keep waiting
        through it — otherwise the step would be recorded as 'cancelling' forever
        instead of its real terminal 'cancelled'."""
        while True:
            job = jobs.get(job_id)
            if job is None:
                return "failed"
            if job["status"] not in ACTIVE and job["status"] != "cancelling":
                return job["status"]
            await asyncio.sleep(2.0)

    # -- helpers -----------------------------------------------------------
    def _set_run(self, wid: str, status: str, state: dict) -> None:
        self._db.execute(
            "UPDATE workflows SET status=?, run_state=?, updated_at=? WHERE id=?",
            (status, json.dumps(state), _now(), wid),
        )
        self._db.commit()

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["graph"] = json.loads(data["graph"]) if data.get("graph") else {"nodes": [], "edges": []}
        data["run_state"] = json.loads(data["run_state"]) if data.get("run_state") else None
        return data


manager = WorkflowManager()
