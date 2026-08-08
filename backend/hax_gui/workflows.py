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

from . import adapters, db, projects, registry
from .command import strip_obsolete
from .config import settings
from .jobs import ACTIVE, manager as jobs
from .labels import manager as labels


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_name(name: str, taken: set[str]) -> str:
    """``"<base> copy"``, then ``"<base> copy 2"``, ``"<base> copy 3"``, …

    A copy of a copy re-uses the base rather than stacking suffixes ("Name copy copy"),
    and the number is the first one free in `taken`, so copies of any sibling land in
    the same series. Mirrors `copyName` in the workflow canvas (views/workflows.js).
    """
    trimmed = name.strip()
    base = re.sub(r"\s+copy(\s+\d+)?$", "", trimmed, flags=re.I).strip() or trimmed
    candidate = f"{base} copy"
    i = 2
    while candidate in taken:
        candidate = f"{base} copy {i}"
        i += 1
    return candidate


class WorkflowError(Exception):
    """Validation/build error with one or more user-facing messages."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class StepsRunning(Exception):
    """Steps in the request are still executing, and the caller has not said to stop them.

    Relaunching one means stopping it first: its process holds the GPU, and its run folder
    is about to be rebuilt underneath it. Watching the existing run instead is *not* a
    substitute — the whole point of the request is usually that the settings changed, and
    quietly letting the old run continue makes an edited parameter look like it was
    applied when the step goes on to finish, successfully, on the previous ones.
    """

    def __init__(self, steps: list[dict[str, Any]]):
        self.steps = steps
        super().__init__("; ".join(s["label"] for s in steps))


class WorkflowManager:
    def __init__(self) -> None:
        self._db: db.LockedConnection | None = None
        self._running: dict[str, asyncio.Task] = {}

    def init(self) -> None:
        settings.ensure_dirs()
        self._db = db.connect(settings.db_path)
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
        # `max_per_device`: how many steps may run at once on any one device (see
        # :meth:`_device_of`). 1 — one training per GPU — is the safe default.
        if "max_per_device" not in cols:
            self._db.execute("ALTER TABLE workflows ADD COLUMN max_per_device INTEGER NOT NULL DEFAULT 1")
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
            if st in ACTIVE:                  # incl. 'cancelling' — not final yet
                st = "running"                # still in flight; the scheduler adopts it
            state[nid] = {
                "job_id": jid, "status": st, "stale": bool(job.get("stale")),
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

    def update(self, wid: str, name: str | None = None, graph: dict[str, Any] | None = None,
               max_per_device: int | None = None) -> dict[str, Any]:
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        sets, params = [], []
        if name is not None:
            sets.append("name=?"); params.append(name.strip() or "Untitled workflow")
        if max_per_device is not None:
            # Takes effect on the next scheduler tick (see TICK_INTERVAL), so raising it
            # mid-run starts filling the extra slots within a couple of seconds, while the
            # running steps carry on — no need to stop and re-run.
            sets.append("max_per_device=?"); params.append(max(1, min(8, int(max_per_device))))
        if graph is not None:
            # Carry over node->job links the client doesn't know about. Saving during a
            # live run (which is how you queue a new step onto it) posts the graph as the
            # canvas last saw it, so a link the run wrote since then would be erased.
            prior = {n["id"]: n for n in (wf["graph"] or {}).get("nodes", [])}
            for n in graph.get("nodes", []):
                if not n.get("job_id") and prior.get(n["id"], {}).get("job_id"):
                    n["job_id"] = prior[n["id"]]["job_id"]
            self._push_renames(wf, graph, prior)
            sets.append("graph=?"); params.append(json.dumps(graph))
        if not sets:
            return wf
        sets.append("updated_at=?"); params.append(_now())
        params.append(wid)
        self._db.execute(f"UPDATE workflows SET {', '.join(sets)} WHERE id=?", params)
        self._db.commit()
        return self.get(wid)

    @staticmethod
    def _push_renames(wf: dict[str, Any], graph: dict[str, Any], prior: dict[str, Any]) -> None:
        """Carry a renamed step's name onto the job it has already run.

        A node's name only ever reached its job at launch (see :meth:`_run_node`), while a
        project canvas refreshes node names *from* the job on every load — so renaming a
        step that has already run, or is running right now, was quietly undone unless the
        step was launched again.

        Only names this save actually *changed* are pushed, and only while the job still
        carries the name this canvas was built from — a compare-and-swap. The client holds
        its own copy of the graph for the length of a run, so without that check a late
        save could resurrect a stale name over one set meanwhile from the Jobs view. The
        job is taken from the server's own link — node.job_id, filled in by the caller, or
        the run's mapping for a standalone workflow — never from what the client believed.
        """
        state = wf.get("run_state") or {}
        for n in graph.get("nodes", []):
            was = prior.get(n.get("id"))
            if was is None or (was.get("name") or "") == (n.get("name") or ""):
                continue
            jid = n.get("job_id") or (state.get(n["id"]) or {}).get("job_id")
            job = jobs.get(jid) if jid else None
            if job is None or (job.get("name") or "") != (was.get("name") or ""):
                continue                      # no job yet, or renamed elsewhere since
            jobs.rename(jid, n.get("name") or "")

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

    def delete_for_project(self, project_id: str) -> None:
        """Drop a deleted project's workflows (rows *and* any live scheduler task);
        orphaned rows used to accumulate forever and their schedulers kept running."""
        for wf in self.list(project_id):
            self.delete(wf["id"])

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
        """A job's args minus auto-managed dirs and bound (edge-fed) fields.

        Options the program has since dropped are deliberately *kept* here and removed at
        launch instead (:meth:`_run_node`), so the removal is something the run can report
        rather than a silent edit made while the canvas syncs. The form ignores them —
        it renders from the schema — so they are invisible either way.
        """
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
                          # Empty means CPU — the job ran without one. Defaulting to "0"
                          # here put every CPU step on card 0, where it then contends with
                          # the trainings for a device slot (see _device_of).
                          "gpu": j.get("gpu") or "", "name": j.get("name") or "",
                          "labels": self._labels_of(j["id"]), "x": 0, "y": 0, "job_id": j["id"]})
            state[nid] = {"job_id": j["id"], "status": j["status"], "stale": bool(j.get("stale")), "error": None}
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
            graph, state = self._synced_graph(wf["graph"], project_id, wf.get("run_state"))
            self._db.execute(
                "UPDATE workflows SET graph=?, run_state=?, updated_at=? WHERE id=?",
                (json.dumps(graph), json.dumps(state), _now(), wid),
            )
            self._db.commit()
            return self.get(wid)
        graph, state = self.build_from_project(project_id)
        wf = self.save(project_id, name or "Project workflow", graph, kind="project")
        return self.seed_state(wf["id"], state)

    def forget_job(self, job_id: str) -> None:
        """Drop every reference to a job that has just been deleted, from all workflows.

        Called by the delete-job endpoint. Without it, removing a step from the canvas
        only *looked* permanent: the job went, but the node stayed in the stored graph
        and came back on the next visit — the full reconcile that would have dropped it
        never runs for a standalone workflow, and is skipped while a run is live.
        """
        for wf in self.list():
            self._prune_deleted_jobs(wf["id"], {job_id})

    def _prune_deleted_jobs(self, wid: str, only: set[str] | None = None) -> bool:
        """Forget nodes/run_state entries pointing at jobs that no longer exist.

        Runs for *every* workflow on load, including standalone ones and while a run is
        in progress — unlike :meth:`_synced_graph` it only ever removes, never re-derives
        the graph from the jobs, so it is safe on a live run (the scheduler re-reads the
        graph each tick and ``_sync_state`` follows) and on a parked resumable one (steps
        that never launched have no job id and are left alone).

        A *project* canvas is a view of its jobs, so a node whose job is gone goes with
        it. A *standalone* workflow's node is a step definition that outlives any one
        run, so it is kept and only its dangling run link is cleared. ``only`` limits the
        purge to specific job ids (used when a single job is deleted); by default any
        dangling reference is cleaned. Returns True if anything changed.
        """
        wf = self.get(wid)
        if wf is None:
            return False
        graph = wf["graph"] or {"nodes": [], "edges": []}
        state = dict(wf.get("run_state") or {})
        nodes = graph.get("nodes", [])

        def gone(jid: str | None) -> bool:
            if not jid or (only is not None and jid not in only):
                return False
            return jobs.get(jid) is None

        is_project = wf.get("kind") == "project"
        drop = {n["id"] for n in nodes if is_project and gone(n.get("job_id"))}
        if drop:
            graph["nodes"] = [n for n in nodes if n["id"] not in drop]
            graph["edges"] = [e for e in graph.get("edges", [])
                              if e.get("from") not in drop and e.get("to") not in drop]
        # Stale links on the nodes that stay: a standalone node keeps its place but goes
        # back to un-run, so it does not advertise a status for a job that is gone.
        cleared = False
        for nid in list(state):
            if nid in drop:
                del state[nid]; cleared = True
            elif gone(state[nid].get("job_id")):
                state[nid] = {"job_id": None, "status": "idle", "error": None}; cleared = True
        for n in graph.get("nodes", []):
            if gone(n.get("job_id")):
                n.pop("job_id", None); cleared = True
        if not (drop or cleared):
            return False
        self._db.execute(
            "UPDATE workflows SET graph=?, run_state=?, updated_at=? WHERE id=?",
            (json.dumps(graph), json.dumps(state), _now(), wid),
        )
        self._db.commit()
        return True

    def reconcile(self, wid: str) -> dict[str, Any] | None:
        """Refresh a *project* workflow from its jobs' current statuses, so the
        canvas is a true live view (a job that finished/cancelled outside a
        workflow run no longer shows a stale node status, and downstream nodes
        un-block once their inputs actually succeed). No-op for standalone
        workflows and while a run is in progress."""
        # Dropping deleted jobs is *not* part of that no-op: it has to happen on every
        # load, or a step removed from a standalone canvas — or from a project canvas
        # while a run is live — reappears the next time the page is opened.
        self._prune_deleted_jobs(wid)
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
        graph, state = self._synced_graph(wf["graph"], wf["project_id"], wf.get("run_state"))
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

    def _synced_graph(self, stored: dict[str, Any], project_id: str,
                      prior_state: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
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
            # The job is the authority here: this node *has* one, so an empty gpu means
            # "runs on the CPU", not "unknown, guess 0". Falling back to "0" silently
            # reassigned every CPU step to GPU 0 on the next reconcile — and since the
            # scheduler contends on device, that step then queued behind whatever was
            # training on card 0 instead of starting straight away on the CPU.
            n["gpu"] = j.get("gpu") or ""
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
            # A step that failed *before* it could produce a run has no job to be derived
            # from — and its node is still linked to whatever ran last time, which did
            # succeed. Rebuilding blindly from the jobs therefore overwrote the verdict
            # with that older success and dropped the reason with it, so a step that could
            # not launch (a stored argument the program no longer accepts, a source run
            # that never produced its output) flashed 'running' on the canvas and then read
            # 'succeeded' seconds later, having done nothing. The failure is carried over
            # instead; the next launch that does produce a job replaces it.
            prior = (prior_state or {}).get(n["id"]) or {}
            if prior.get("status") == "failed" and not prior.get("job_id"):
                state[n["id"]] = {"job_id": None, "status": "failed", "error": prior.get("error")}
            elif jid and jid in by_job:
                state[n["id"]] = {"job_id": jid, "status": by_job[jid]["status"],
                                  "stale": bool(by_job[jid].get("stale")), "error": None}
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

    def copy_to_project(self, wid: str, target_project_id: str, name: str | None = None) -> dict[str, Any]:
        """Clone a workflow's graph into a new standalone workflow owned by another
        project, so an entire pipeline can be reused elsewhere.

        The clone is self-contained and unlinked from the source: every node's `job_id`
        is dropped (so nothing points at the source project's jobs) and the run-state
        starts fresh. Dependencies *within* the copied graph travel as edges and rebind
        automatically at run time; data inputs coming from *outside* the graph (a node's
        bindable field with no incoming edge -- raw metadata, an un-copied job's output)
        are cleared, to be set in the target project before the copy runs. Auto-managed
        output/SSD dirs are dropped too, as they are per run."""
        src = self.get(wid)
        if src is None:
            raise KeyError(wid)
        if projects.dir_for(target_project_id) is None:
            raise KeyError(target_project_id)
        graph = json.loads(json.dumps(src.get("graph") or {"nodes": [], "edges": [], "orientation": "horizontal"}))
        for node in graph.get("nodes", []) or []:
            node.pop("job_id", None)               # unlink from the source project's jobs
            program = node.get("program")
            args = dict(node.get("args") or {})
            for auto in (registry.output_arg(program), registry.ssd_arg(program)):
                if auto:
                    args.pop(auto, None)
            # Data inputs are never carried as literals in a portable copy: an input fed
            # by another step travels as an edge (rebinds at run time), and one from
            # outside the graph is cleared to be set in the target project. Either way
            # the field is dropped here — so no source-project path can leak through.
            for field in adapters.bindable_fields(program):
                args.pop(field, None)
            node["args"] = args
        new_name = (name or "").strip() or _copy_name(
            src.get("name") or "Workflow",
            {(w.get("name") or "").strip() for w in self.list(target_project_id)},
        )
        return self.save(target_project_id, new_name, graph, kind="standalone")

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

    # -- what a step feeds -------------------------------------------------
    def downstream_nodes(self, wid: str, node_ids: list[str]) -> list[str]:
        """The steps `node_ids` feed, transitively, excluding `node_ids` themselves.

        Taken from the graph rather than from the jobs' bindings, because a canvas edge
        exists before the run that would turn it into a binding does: a step wired to a
        not-yet-run one has no binding to find, and a standalone workflow's steps may
        have no jobs at all. Returned in graph order so the UI lists them as they sit on
        the canvas.
        """
        wf = self.get(wid)
        if wf is None:
            raise KeyError(wid)
        graph = wf["graph"] or {}
        edges = graph.get("edges", [])
        seen, queue = set(node_ids), list(node_ids)
        while queue:
            cur = queue.pop(0)
            for e in edges:
                if e.get("from") == cur and e.get("to") not in seen:
                    seen.add(e["to"])
                    queue.append(e["to"])
        start = set(node_ids)
        return [n["id"] for n in graph.get("nodes", []) if n["id"] in seen and n["id"] not in start]

    def jobs_for_nodes(self, wid: str, node_ids: list[str]) -> list[str]:
        """The job each of these steps last ran as, skipping the ones that never ran.

        A project node carries its link on the node itself; a standalone one only in
        run_state (its steps make a new job per run), so both are consulted.
        """
        wf = self.get(wid)
        if wf is None:
            return []
        state = wf.get("run_state") or {}
        by_node = {n["id"]: n for n in (wf["graph"] or {}).get("nodes", [])}
        out = []
        for nid in node_ids:
            jid = by_node.get(nid, {}).get("job_id") or (state.get(nid) or {}).get("job_id")
            if jid and jid not in out:
                out.append(jid)
        return out

    def nodes_for_jobs(self, wid: str, job_ids: list[str]) -> list[str]:
        """The reverse of :meth:`jobs_for_nodes`: the steps these jobs are linked to."""
        wf = self.get(wid)
        if wf is None:
            return []
        state = wf.get("run_state") or {}
        wanted = set(job_ids)
        out = []
        for n in (wf["graph"] or {}).get("nodes", []):
            jid = n.get("job_id") or (state.get(n["id"]) or {}).get("job_id")
            if jid in wanted and n["id"] not in out:
                out.append(n["id"])
        return out

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
    # launch. Each tick it re-reads the graph, run_state and max_per_device from the DB,
    # so steps queued onto a live run — and edits saved while it runs — are picked up. A
    # tick happens whenever a step finishes *and* every TICK_INTERVAL regardless, so a
    # change that frees a slot is acted on while the other steps are still training.
    #
    # Steps run concurrently, limited by the resource they actually contend for: the
    # device. Every ready step whose device has a free slot starts; the rest wait. With
    # the default `max_per_device` of 1 that means one training per GPU, so two branches
    # pinned to GPU 0 and GPU 1 run side by side while two steps both on GPU 0 still
    # queue. Raising it to 2 is the deliberate "I know these two fit on one card"
    # setting — nothing here checks free VRAM.
    #
    # Concurrency is safe here only because of a discipline the sequential runner already
    # kept: run_state is re-read from the DB immediately before every write, with no
    # `await` in between. That makes each read-modify-write atomic with respect to the
    # event loop, so two steps finishing at once cannot clobber each other's status.
    WAITING = ("pending", "running")
    CPU_DEVICE = "cpu"
    # How long the scheduler may sit on a stale reading of the graph / device caps.
    TICK_INTERVAL = 2.0

    @staticmethod
    def _device_of(node: dict[str, Any]) -> str:
        """The contended resource a step occupies: a GPU index, or the CPU.

        Anything that isn't a GPU index shares one notional CPU device, so two heavy
        CPU steps (``preprocess_volumes``) don't both start just because neither
        claimed a card.
        """
        gpu = node.get("gpu")
        gpu = "" if gpu is None else str(gpu).strip()
        return gpu if gpu and gpu.lower() != "cpu" else WorkflowManager.CPU_DEVICE

    async def run(self, wid: str, only: list[str] | None = None,
                  restart_running: bool = False) -> dict[str, Any]:
        """Queue nodes for execution: `only` selects a subset, omit it for the whole
        graph. If a run is already in flight the nodes are *appended to its queue* and the
        live scheduler starts each as soon as its dependencies are met and its device has
        a free slot; otherwise a run starts.
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

        # Steps in the request that are still executing have to be stopped before they can
        # be relaunched, and that is the user's call, not ours — so it is asked for
        # explicitly rather than assumed either way.
        by_id = {n["id"]: n for n in nodes}
        busy: list[dict[str, Any]] = []
        for nid in sorted(only_set or ()):
            jid = (prev.get(nid) or {}).get("job_id") or by_id[nid].get("job_id")
            job = jobs.get(jid) if jid else None
            if job is not None and job["status"] in ACTIVE:
                busy.append({"node": nid, "job_id": jid,
                             "label": by_id[nid].get("name") or job["program"],
                             "program": job["program"], "status": job["status"]})
        if busy and not restart_running:
            raise StepsRunning(busy)
        for b in busy:
            await jobs.cancel(b["job_id"])
            # Wait for it to actually leave the active states, or _run_node's adoption
            # guard would see a job still 'cancelling' and settle for watching it.
            for _ in range(20):
                job = jobs.get(b["job_id"])
                if job is None or job["status"] not in ACTIVE:
                    break
                await asyncio.sleep(0.25)
            # ...and for the watcher that was following it to record its verdict. That
            # watcher re-reads run_state and overwrites its own node entry, so queueing
            # the step before its 'cancelled' lands would simply be clobbered — the
            # relaunch would go missing and the step would sit there cancelled.
            for _ in range(24):
                s = self._fresh_state(wid).get(b["node"]) or {}
                if s.get("status") not in self.WAITING:
                    break
                await asyncio.sleep(0.25)
        busy_nodes = {b["node"] for b in busy}
        if busy:
            prev = self._fresh_state(wid)     # re-read: the waits above moved it on

        state: dict[str, Any] = {}
        for n in nodes:
            nid = n["id"]
            if only_set is None or nid in only_set:
                # A step the user has explicitly asked to run is only left alone when it is
                # *genuinely* still in flight. run_state is the scheduler's memory, and
                # while a run is live nothing refreshes it from the jobs table (reconcile()
                # bails out on a live workflow) — so a step whose job finished outside this
                # run, or whose watcher died with a previous server, can sit at 'running'
                # indefinitely. Trusting that memory dropped the request on the floor: the
                # step was never queued, the scheduler adopted it on the next tick, and
                # _wait_job returned the already-final status at once — so it flashed
                # 'running' and settled 'succeeded' without having re-run anything.
                if live and nid not in busy_nodes and self._in_flight(prev.get(nid)):
                    state[nid] = prev[nid]        # already queued or really running
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
                    elif job["status"] in ACTIVE:
                        s = {**s, "status": "running"}         # still training: adopt, don't relaunch
                    else:
                        s = {**s, "status": job["status"]}     # it did finish — keep its result
                state[nid] = s
        self._set_run(wid, "running", state)
        # Re-checked rather than reusing `live` from the top: stopping a busy step above
        # can take seconds, and the scheduler that was watching it may have run out of
        # work and exited in the meantime. Spawning off the stale reading left the queue
        # we have just written with nobody to execute it.
        if not self._is_live(wid):
            self._running[wid] = asyncio.create_task(self._execute(wid))
        return self.get(wid)

    async def stop(self, wid: str) -> dict[str, Any]:
        """Stop a live run: halt the scheduler, cancel every step currently executing, and
        drop everything still queued back to idle. The scheduler only ever *polled* jobs,
        so the running subprocesses have to be cancelled explicitly here."""
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

    @staticmethod
    def _in_flight(s: dict[str, Any] | None) -> bool:
        """Is this run_state entry describing something that is *actually* still going?

        The status alone cannot say: it is what the scheduler last wrote, not what is
        true now. Only the job settles it — except for a step queued but never launched,
        which has no job yet and is exactly what 'pending' with no id means.
        """
        if not s or s.get("status") not in WorkflowManager.WAITING:
            return False
        jid = s.get("job_id")
        if not jid:
            return s.get("status") == "pending"     # queued, awaiting its turn
        job = jobs.get(jid)
        return job is not None and job["status"] in ACTIVE

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
        # Steps currently in flight, each with the task watching it and the device it
        # occupies. A step is only ever in here once.
        inflight: dict[str, asyncio.Task] = {}
        device: dict[str, str] = {}
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
                # would have. Never relaunch it; it is still on the GPU. It also occupies
                # a device slot, so nothing new is scheduled onto the card it is using.
                for nid, s in state.items():
                    if s.get("status") == "running" and s.get("job_id") and nid not in inflight:
                        executed.add(nid)
                        device[nid] = self._device_of(nodes.get(nid, {}))
                        inflight[nid] = asyncio.create_task(
                            self._guarded(wid, nid, self._await_node(wid, nid, s["job_id"])))

                ready, blocked = self._partition(graph, state)

                if blocked:
                    state = self._fresh_state(wid)
                    for nid in blocked:
                        state[nid] = {"job_id": None, "status": "blocked",
                                      "error": "an upstream step did not succeed"}
                        executed.add(nid)
                    self._set_run(wid, "running", state)
                    continue                       # blocking cascades: re-evaluate

                # Start every ready step whose device still has a free slot. `ready` is in
                # topological order, so when slots are scarce the earlier steps win.
                cap = max(1, int(wf.get("max_per_device") or 1))
                used: dict[str, int] = {}
                for nid in inflight:
                    used[device[nid]] = used.get(device[nid], 0) + 1
                for nid in ready:
                    # A step stays 'pending' until its _run_node gets past jobs.create —
                    # which suspends, since it launches a subprocess. A sibling finishing
                    # inside that window ticks the scheduler, which would otherwise see
                    # this step as still queued and launch it a *second* time. The `used`
                    # accounting below hides this at a cap of 1 (the step already holds the
                    # only slot on its device); at 2+ it does not, and the result is two
                    # trainings of the same step on one GPU.
                    if nid in inflight:
                        continue
                    dev = self._device_of(nodes[nid])
                    if used.get(dev, 0) >= cap:
                        continue                   # that device is busy; it waits its turn
                    used[dev] = used.get(dev, 0) + 1
                    executed.add(nid)
                    device[nid] = dev
                    deps = [e for e in edges if e["to"] == nid]
                    inflight[nid] = asyncio.create_task(self._guarded(
                        wid, nid,
                        self._run_node(wid, nid, nodes[nid], deps, wf["project_id"],
                                       wf.get("kind") == "project")))

                if not inflight:
                    # Nothing running and nothing startable — but a request can land in
                    # the window between the partition above and this check (run() writes
                    # run_state from the request handler, not from here), and exiting on
                    # the older reading would strand it with no scheduler behind it.
                    if any(s.get("status") == "pending" for s in self._fresh_state(wid).values()):
                        continue
                    break                          # the queue really is drained (or every
                                                   # ready step's device is held by a step
                                                   # this scheduler does not own)

                # Wait for the first step to finish, then loop: re-reads the graph, so
                # steps queued while these ran are picked up, and the slot just freed is
                # handed to whatever is ready now.
                #
                # The timeout is what makes "re-reads each tick" true while a long step is
                # in flight. Without it the only wake-up is a step *finishing*, so anything
                # that should let a queued step start right now — raising max_per_device,
                # or pointing a step at an idle device (another GPU, or the CPU) — did
                # nothing until the running training ended, sometimes hours later. The
                # step just sat at 'pending'. Polling matches _await_node, which watches
                # its job on the same cadence.
                done, _ = await asyncio.wait(inflight.values(), timeout=self.TICK_INTERVAL,
                                             return_when=asyncio.FIRST_COMPLETED)
                for nid, task in list(inflight.items()):
                    if task in done:
                        del inflight[nid]
                        device.pop(nid, None)

            self._finish(wid, executed)
        except asyncio.CancelledError:
            # Stop the watchers too, or they would outlive the scheduler and keep writing
            # into run_state after stop() has settled it.
            for task in inflight.values():
                task.cancel()
            wf = self.get(wid)
            self._set_run(wid, "cancelled", (wf or {}).get("run_state") or {})
            raise
        except Exception:  # noqa: BLE001
            for task in inflight.values():
                task.cancel()
            wf = self.get(wid)
            self._set_run(wid, "failed", (wf or {}).get("run_state") or {})
        finally:
            self._running.pop(wid, None)

    async def _guarded(self, wid: str, nid: str, coro) -> None:
        """Run one step's coroutine so an unexpected failure sinks that step, not the run.

        With steps sequential an exception here could only ever end the workflow that was
        already stuck on it; concurrently it would also abandon the sibling steps still
        training, which is much worse than recording one failed node.
        """
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            state = self._fresh_state(wid)
            state[nid] = {"job_id": state.get(nid, {}).get("job_id"), "status": "failed",
                          "error": str(exc) or "the step's scheduler failed"}
            self._set_run(wid, "running", state)

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
        # Second line of defence for a program whose CLI moved on: _clean_args strips
        # obsolete options when the graph is synced from the jobs, but the client posts the
        # graph from its own copy, which may predate that. Without this the step dies in
        # build_command before it can create a job.
        args, dropped = strip_obsolete(node["program"], args)
        # In a live project workflow, a node linked to an existing job reruns *that* job
        # in place (modifying the project) instead of creating a duplicate.
        replace = node.get("job_id") if (is_project and node.get("job_id") and jobs.get(node["job_id"])) else None
        try:
            job = await jobs.create(node["program"], args, node.get("gpu"), project_id,
                                    name=node.get("name"), bindings=bindings, replace_job_id=replace,
                                    dropped_args=dropped)
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
            if job["status"] not in ACTIVE:   # ACTIVE covers the transitional 'cancelling'
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
        data["max_per_device"] = int(data.get("max_per_device") or 1)
        return data


manager = WorkflowManager()
