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
import contextlib
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
        # `resumable`: parked mid-run, waiting for the user to say whether to carry on.
        if "resumable" not in cols:
            self._db.execute("ALTER TABLE workflows ADD COLUMN resumable INTEGER NOT NULL DEFAULT 0")
        self._db.commit()
        self._plan_resume()

    def _plan_resume(self) -> None:
        """Decide what becomes of the workflows the previous server process left running.

        A scheduler is not a process — it is an ``asyncio`` task living inside the server —
        so it always dies with the server, even though the *job* it was waiting on now
        survives (see jobs.py). Every 'running' workflow therefore has to be re-judged
        against the jobs table, which ``JobManager._reconcile`` has already made truthful:
        it recovered the exit status of anything that finished while we were down, and
        pid-reuse-checked anything that claims to still be alive.

        Three outcomes. The question that separates them is *would resuming start new
        compute right now, with nobody watching?* — not how long the server was down,
        which is only a proxy for it and a poor one (a training happily survives an
        overnight outage; one can equally be OOM-killed during a five-second restart):

        * **a step is still running** → resume immediately. Nothing new is launched; the
          scheduler simply re-adopts the job already in flight. This is the update-restart
          case, and prompting here would strand a GPU until someone clicked a dialog.
        * **nothing alive, but the next step is ready** → park it (``resumable``). Resuming
          *would* launch a training, so it waits for an explicit go-ahead.
        * **nothing left to run** → hand it to the scheduler anyway; with nothing ready it
          launches nothing and just settles the final status. A workflow that quietly
          finished while we were down should not nag to be resumed.
        """
        assert self._db is not None
        self._auto: list[str] = []
        for row in self._db.execute("SELECT id FROM workflows WHERE status='running'").fetchall():
            wid = row["id"]
            try:
                wf = self.get(wid)
                if wf is None:
                    continue
                state = self._adopt_job_states(wid)
                live = any(s.get("status") == "running" for s in state.values())
                ready, _ = self._partition(wf["graph"], state)
                if not live and ready:
                    self._db.execute(
                        "UPDATE workflows SET status='interrupted', resumable=1 WHERE id=?", (wid,))
                else:
                    self._auto.append(wid)
            except Exception:  # noqa: BLE001 - a broken graph must never block startup
                self._db.execute("UPDATE workflows SET status='interrupted' WHERE id=?", (wid,))
        self._db.commit()

    async def resume_all(self) -> None:
        """Restart the schedulers of the workflows that survived (see :meth:`_plan_resume`)."""
        pending, self._auto = getattr(self, "_auto", []), []
        for wid in pending:
            self._running[wid] = asyncio.create_task(self._execute(wid, resumed=True))

    def _adopt_job_states(self, wid: str) -> dict[str, Any]:
        """Rebuild run_state from the jobs' real statuses, and persist it.

        run_state is only the scheduler's *memory* of what it last saw. Across a restart
        that memory is stale — a step it remembers as 'running' may have finished hours ago
        — and the jobs table is the truth. (Project workflows self-healed this way already,
        via :meth:`reconcile`; standalone ones never did, which is why resuming one used to
        strand its finished steps and re-run them from scratch.)
        """
        state = self._fresh_state(wid)
        for nid, s in list(state.items()):
            jid = s.get("job_id")
            if not jid:
                continue                      # queued but never launched: nothing to adopt
            job = jobs.get(jid)
            if job is None:
                state[nid] = {"job_id": None, "status": "idle", "error": None}
                continue
            st = job["status"]
            if st in ACTIVE or st == "cancelling":
                st = "running"                # still in flight; the scheduler adopts it
            state[nid] = {
                "job_id": jid, "status": st,
                "error": None if st in ("running", "succeeded")
                else s.get("error") or "step did not finish (see the job log)",
            }
        self._db.execute("UPDATE workflows SET run_state=? WHERE id=?", (json.dumps(state), wid))
        self._db.commit()
        return state

    # -- resume (user-driven) ----------------------------------------------
    def list_resumable(self) -> list[dict[str, Any]]:
        """The workflows parked by :meth:`_plan_resume`, with enough detail to decide."""
        rows = self._db.execute(
            "SELECT id, name, project_id, run_state, updated_at FROM workflows "
            "WHERE resumable=1 ORDER BY updated_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            state = json.loads(r["run_state"]) if r["run_state"] else {}
            counts: dict[str, int] = {}
            for s in state.values():
                counts[s.get("status")] = counts.get(s.get("status"), 0) + 1
            out.append({
                "id": r["id"], "name": r["name"], "project_id": r["project_id"],
                "updated_at": r["updated_at"],
                "done": counts.get("succeeded", 0),
                "remaining": counts.get("pending", 0),
                "total": sum(n for st, n in counts.items() if st != "idle"),
            })
        return out

    async def resume(self, wid: str) -> dict[str, Any]:
        """Carry on a parked workflow. Only the steps that never ran are launched: the
        ones that already succeeded keep their jobs, and the steps downstream bind to
        those jobs, so nothing that already completed is recomputed."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        if self._is_live(wid):
            return wf
        state = self._adopt_job_states(wid)
        self._set_run(wid, "running", state)          # also clears `resumable`
        self._running[wid] = asyncio.create_task(self._execute(wid, resumed=True))
        return self.get(wid)

    def dismiss_resume(self, wid: str) -> dict[str, Any] | None:
        """Stop offering to resume this one. The queue is left in run_state, so it can
        still be picked up by hand from the canvas later."""
        self._db.execute("UPDATE workflows SET resumable=0 WHERE id=?", (wid,))
        self._db.commit()
        return self.get(wid)

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
            # Carry over node->job links the client doesn't know about. Saving during a
            # live run (which is how you queue a new step onto it) posts the graph as the
            # canvas last saw it, so a link the run wrote since then would be erased.
            prior = {n["id"]: n for n in (wf["graph"] or {}).get("nodes", [])}
            for n in graph.get("nodes", []):
                if not n.get("job_id") and prior.get(n["id"], {}).get("job_id"):
                    n["job_id"] = prior[n["id"]]["job_id"]
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
            wf = self._row(row)
            wid = wf["id"]
            # Parked awaiting a resume decision: re-syncing would rebuild run_state from the
            # jobs alone, flattening the steps that never launched (no job to sync from) to
            # 'idle' — quietly emptying the queue we are offering to resume. Leave it be.
            if wf.get("resumable"):
                return wf
            graph, state = self._synced_graph(wf["graph"], project_id)
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
        # Parked awaiting a resume decision: its run_state holds the queue that resuming
        # will pick up, and _synced_graph would flatten those never-launched steps (no
        # job_id) to 'idle' — quietly emptying the queue we are asking about.
        if wf.get("resumable"):
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
    # The runner is a re-entrant scheduler, not a one-shot walk over a snapshot taken at
    # launch. Each tick it re-reads the graph and run_state from the DB, so steps queued
    # onto a live run — and edits saved while it runs — are picked up. Steps still
    # execute strictly one at a time (two trainings at once would fight for the GPU).
    WAITING = ("pending", "running")

    async def run(self, wid: str, only: list[str] | None = None) -> dict[str, Any]:
        """Queue nodes for execution: `only` selects a subset, omit it for the whole
        graph. If a run is already in flight the nodes are *appended to its queue* and
        the live scheduler picks them up when their turn comes; otherwise a run starts.
        Nodes outside the set keep their existing state, so an already-succeeded
        upstream node still supplies its job to the nodes being queued."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        errors = self.validate(wf["graph"])
        if errors:
            raise WorkflowError(errors)
        nodes = wf["graph"]["nodes"]
        ids = {n["id"] for n in nodes}
        only_set = set(only) & ids if only is not None else None
        if only is not None and not only_set:
            raise WorkflowError(["none of the selected steps exist in this workflow"])

        live = self._is_live(wid)
        prev = wf["run_state"] or {}
        state: dict[str, Any] = {}
        for n in nodes:
            nid = n["id"]
            if only_set is None or nid in only_set:
                if live and prev.get(nid, {}).get("status") in self.WAITING:
                    state[nid] = prev[nid]        # already queued or running: leave it be
                else:
                    state[nid] = {"job_id": None, "status": "pending", "error": None}
            else:
                s = prev.get(nid, {"job_id": None, "status": "idle", "error": None})
                # A pending/running left over from a run with no scheduler behind it is a
                # memory, not a fact — believe the job instead. Leaving it as-is would
                # stall its dependents forever; blanket-idling it (what this used to do)
                # threw away a step that had really finished, forcing a recompute.
                if not live and s.get("status") in self.WAITING:
                    job = jobs.get(s.get("job_id") or "")
                    if job is None:
                        s = {**s, "status": "idle"}            # never launched: a stale queue entry
                    elif job["status"] in ACTIVE or job["status"] == "cancelling":
                        s = {**s, "status": "running"}         # still training: adopt, don't relaunch
                    else:
                        s = {**s, "status": job["status"]}     # it did finish — keep its result
                state[nid] = s
        self._set_run(wid, "running", state)
        if not live:
            self._running[wid] = asyncio.create_task(self._execute(wid))
        return self.get(wid)

    async def stop(self, wid: str) -> dict[str, Any]:
        """Stop a live run: halt the scheduler, cancel the step currently executing, and
        drop everything still queued back to idle. The scheduler only ever *polled* jobs,
        so the running subprocess has to be cancelled explicitly here."""
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        task = self._running.pop(wid, None)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        state = self._fresh_state(wid)
        for nid, s in list(state.items()):
            st = s.get("status")
            if st == "running" and s.get("job_id"):
                await jobs.cancel(s["job_id"])
                state[nid] = {**s, "status": "cancelled", "error": None}
            elif st == "pending":
                state[nid] = {"job_id": None, "status": "idle", "error": None}   # dequeued
        self._set_run(wid, "cancelled", state)
        return self.get(wid)

    def _is_live(self, wid: str) -> bool:
        task = self._running.get(wid)
        return task is not None and not task.done()

    def _partition(self, graph: dict[str, Any], state: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Split the queued steps into (ready, blocked).

        A queued step is ready once every upstream has succeeded; it stays queued while an
        upstream is still queued/running, and is blocked once an upstream can no longer
        succeed (failed, or never queued at all).
        """
        edges = graph["edges"]
        ready: list[str] = []
        blocked: list[str] = []
        for nid in self.topo_order(graph["nodes"], edges):
            if state.get(nid, {}).get("status") != "pending":
                continue
            dep_st = [state.get(e["from"], {}).get("status") for e in edges if e["to"] == nid]
            if any(s != "succeeded" and s not in self.WAITING for s in dep_st):
                blocked.append(nid)
            elif any(s in self.WAITING for s in dep_st):
                continue
            else:
                ready.append(nid)
        return ready, blocked

    async def _execute(self, wid: str, resumed: bool = False) -> None:
        # `resumed`: this scheduler is picking up a run started by a previous server
        # process. Seed the judged set from what that run already got through, or a resume
        # with nothing left to do would settle on 'idle' instead of its real verdict.
        executed: set[str] = set()
        if resumed:
            executed = {nid for nid, s in self._fresh_state(wid).items()
                        if s.get("status") not in (None, "idle")}
        try:
            while True:
                wf = self.get(wid)
                if wf is None:
                    return
                graph = wf["graph"]
                nodes = {n["id"]: n for n in graph["nodes"]}
                edges = graph["edges"]
                state = self._sync_state(wid, nodes)

                # A step left 'running' by a previous server process has nothing awaiting
                # it: the _run_node that was watching it died with that server, while the
                # job itself kept on training. Adopt it — wait it out exactly as _run_node
                # would have. Never relaunch it; it is still on the GPU.
                orphan = next((nid for nid, s in state.items()
                               if s.get("status") == "running" and s.get("job_id")), None)
                if orphan:
                    executed.add(orphan)
                    await self._await_node(wid, orphan, state[orphan]["job_id"])
                    continue                       # it is terminal now: re-evaluate

                ready, blocked = self._partition(graph, state)

                if blocked:
                    state = self._fresh_state(wid)
                    for nid in blocked:
                        state[nid] = {"job_id": None, "status": "blocked",
                                      "error": "an upstream step did not succeed"}
                        executed.add(nid)
                    self._set_run(wid, "running", state)
                    continue                       # blocking cascades: re-evaluate

                if not ready:
                    break                          # queue drained (a running step is an
                                                   # orphan by now, handled above)

                nid = ready[0]                     # strictly sequential: one step at a time
                executed.add(nid)
                deps = [e for e in edges if e["to"] == nid]
                await self._run_node(wid, nid, nodes[nid], deps, wf["project_id"],
                                     wf.get("kind") == "project")
                # loop round: re-reads the graph, so steps queued while that job ran run next

            self._finish(wid, executed)
        except asyncio.CancelledError:
            wf = self.get(wid)
            self._set_run(wid, "cancelled", (wf or {}).get("run_state") or {})
            raise
        except Exception:  # noqa: BLE001
            wf = self.get(wid)
            self._set_run(wid, "failed", (wf or {}).get("run_state") or {})
        finally:
            self._running.pop(wid, None)

    async def _run_node(self, wid: str, nid: str, node: dict[str, Any], deps: list[dict[str, Any]],
                        project_id: str | None, is_project: bool) -> None:
        """Launch one step and wait for it out. run_state is re-read from the DB after
        every await: a step may have been queued onto this live run while we waited, and
        writing back a stale copy would silently drop it."""
        state = self._fresh_state(wid)
        # Never launch a step whose job is already running — adopt it and wait it out.
        # Without this, a workflow resumed while its step was still training would start a
        # *second* training on the same GPU, and for a project workflow `replace` below
        # would delete the run folder the first one is still writing into.
        alive = state.get(nid, {}).get("job_id") or node.get("job_id")
        if alive:
            existing = jobs.get(alive)
            if existing and existing["status"] in ACTIVE:
                state[nid] = {"job_id": alive, "status": "running", "error": None}
                self._set_run(wid, "running", state)
                await self._await_node(wid, nid, alive)
                return
        args = dict(node.get("args", {}))
        for auto in (registry.output_arg(node["program"]), registry.ssd_arg(node["program"])):
            if auto:
                args.pop(auto, None)
        bindings: dict[str, Any] = {}
        for e in deps:
            one = {"job_id": state.get(e["from"], {}).get("job_id"), "kind": e["kind"]}
            if adapters.is_multi_bindable(node["program"], e["field"]):
                bindings.setdefault(e["field"], []).append(one)   # multi-valued input
            else:
                bindings[e["field"]] = one
        for field in bindings:               # drop any stale literal so the binding wins
            args.pop(field, None)
        # In a live project workflow, a node linked to an existing job reruns *that* job
        # in place (modifying the project) instead of creating a duplicate.
        replace = node.get("job_id") if (is_project and node.get("job_id") and jobs.get(node["job_id"])) else None
        try:
            job = await jobs.create(node["program"], args, node.get("gpu"), project_id,
                                    name=node.get("name"), bindings=bindings, replace_job_id=replace)
        except Exception as exc:  # noqa: BLE001 - surface build/validation failures per node
            state = self._fresh_state(wid)
            state[nid] = {"job_id": None, "status": "failed", "error": str(exc)}
            self._set_run(wid, "running", state)
            return

        jid = job["id"]
        if is_project:
            self._link_job(wid, nid, jid)
        if node.get("labels"):                # carry the node's labels onto the job
            try:
                labels.set_job_labels(jid, node["labels"])
            except Exception:  # noqa: BLE001 - labels are non-critical
                pass

        state = self._fresh_state(wid)
        state[nid] = {"job_id": jid, "status": "running", "error": None}
        self._set_run(wid, "running", state)

        await self._await_node(wid, nid, jid)

    async def _await_node(self, wid: str, nid: str, jid: str) -> None:
        """Wait a step's job out and record how it ended. Split out of :meth:`_run_node` so
        a scheduler resumed after a restart can adopt a job it never launched."""
        final = await self._wait_job(jid)
        state = self._fresh_state(wid)
        state[nid] = {"job_id": jid, "status": final,
                      "error": None if final == "succeeded" else "step failed (see the job log)"}
        self._set_run(wid, "running", state)

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

    def _fresh_state(self, wid: str) -> dict[str, Any]:
        """run_state straight from the DB — never a copy held across an await."""
        wf = self.get(wid)
        return dict((wf or {}).get("run_state") or {})

    def _sync_state(self, wid: str, nodes: dict[str, Any]) -> dict[str, Any]:
        """Reconcile run_state with the graph as it stands now: forget deleted nodes and
        give nodes added mid-run an `idle` entry, so they show up on the canvas and can
        be queued — they do not start on their own just for being saved."""
        state = self._fresh_state(wid)
        synced = {nid: s for nid, s in state.items() if nid in nodes}
        for nid in nodes:
            synced.setdefault(nid, {"job_id": None, "status": "idle", "error": None})
        if synced != state:
            self._set_run(wid, "running", synced)
        return synced

    def _link_job(self, wid: str, nid: str, jid: str) -> None:
        """Point a project-workflow node at the job it just launched, merged into the
        graph *as it stands now* — the old runner wrote its launch-time snapshot back
        wholesale, silently destroying nodes and edges added while the run was going."""
        wf = self.get(wid)
        if wf is None:
            return
        graph = wf["graph"]
        for n in graph.get("nodes", []):
            if n["id"] == nid:
                if n.get("job_id") == jid:
                    return
                n["job_id"] = jid
                break
        else:
            return
        self._db.execute("UPDATE workflows SET graph=? WHERE id=?", (json.dumps(graph), wid))
        self._db.commit()

    def _finish(self, wid: str, executed: set[str]) -> None:
        state = self._fresh_state(wid)
        judged = [state[nid]["status"] for nid in executed if nid in state]
        if not judged:
            done = "idle"
        else:
            done = "succeeded" if all(s == "succeeded" for s in judged) else "failed"
        self._set_run(wid, done, state)

    # -- helpers -----------------------------------------------------------
    def _set_run(self, wid: str, status: str, state: dict) -> None:
        # A scheduler is driving this workflow again, so the parked "resume?" offer is
        # spent — whether it was taken up, superseded by a fresh Run, or stopped.
        self._db.execute(
            "UPDATE workflows SET status=?, run_state=?, resumable=0, updated_at=? WHERE id=?",
            (status, json.dumps(state), _now(), wid),
        )
        self._db.commit()

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["graph"] = json.loads(data["graph"]) if data.get("graph") else {"nodes": [], "edges": []}
        data["run_state"] = json.loads(data["run_state"]) if data.get("run_state") else None
        data["resumable"] = bool(data.get("resumable"))
        return data


manager = WorkflowManager()
