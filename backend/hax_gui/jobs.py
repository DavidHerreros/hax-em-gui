"""Job manager: launch hax programs as subprocesses, persist metadata, stream logs.

Single-server, plain-subprocess execution (PLAN.md §10). Jobs run detached from
any one client so long trainings survive browser disconnects; the UI reattaches
to live logs on reconnect. Job metadata lives in SQLite; full logs are files.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
import shutil
import signal
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapters, mdpaths, projects, registry, settings_store
from .command import ValidationError, build_command
from .config import settings

# In-memory tail (characters) kept per job for fast WS replay on reconnect. Raw
# text including ANSI codes and carriage returns, so the client terminal can
# reconstruct progress lines exactly.
LIVE_BUFFER_CHARS = 256_000

ACTIVE = {"pending", "running"}

# The subprocess' output streams, kept separate so the UI can show one tab each.
STREAMS = ("out", "err")


def _collapse_progress(text: str) -> str:
    """Collapse in-place progress frames for a log-replay snapshot.

    A ``\\r`` that isn't followed by a newline rewrites the line from column 0
    (how tqdm draws a progress bar), so the terminal only ever shows the *last*
    frame. For a replay we keep just that last frame of each newline-delimited
    segment — dropping the thousands of redundant progress writes that would
    otherwise dominate the size cap and evict the earliest real lines (e.g.
    "Creating Array Record…"). Fully-committed lines are untouched.
    """
    if "\r" not in text:
        return text
    out = []
    for seg in text.split("\n"):
        if seg.endswith("\r"):        # CRLF line ending: drop the lone trailing CR
            seg = seg[:-1]
        if "\r" in seg:               # in-place progress frames: keep the last
            seg = seg.rsplit("\r", 1)[-1]
        out.append(seg)
    return "\n".join(out)


class ProjectNotFound(Exception):
    def __init__(self, project_id: str):
        super().__init__(f"unknown project '{project_id}'")
        self.project_id = project_id


class JobNotFound(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"unknown job '{job_id}'")
        self.job_id = job_id


class JobActive(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"job '{job_id}' is still running")
        self.job_id = job_id


class JobNotSaved(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"job '{job_id}' is not in the saved state")
        self.job_id = job_id


class JobNotResumable(Exception):
    def __init__(self, job_id: str):
        super().__init__(f"job '{job_id}' has no loadable checkpoint to continue from")
        self.job_id = job_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self) -> None:
        self._db: sqlite3.Connection | None = None
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        # Recent raw text tail per job, split by stream: job_id -> {stream: deque}.
        self._buffers: dict[str, dict[str, deque[str]]] = {}
        self._buflens: dict[str, dict[str, int]] = {}

    # -- lifecycle ---------------------------------------------------------
    def init(self) -> None:
        settings.ensure_dirs()
        self._db = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                program TEXT NOT NULL,
                name TEXT,
                args TEXT NOT NULL,
                gpu TEXT,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                pid INTEGER,
                return_code INTEGER,
                log_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                project_id TEXT,
                workdir TEXT,
                cwd TEXT,
                bindings TEXT
            )
            """
        )
        # Add columns to pre-existing databases (idempotent).
        for col in ("project_id TEXT", "workdir TEXT", "cwd TEXT", "name TEXT", "bindings TEXT"):
            try:
                self._db.execute(f"ALTER TABLE jobs ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        # Any job still marked active from a previous run lost its process.
        self._db.execute(
            "UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('pending','running')",
            (_now(),),
        )
        # A job left mid-cancel ('cancelling') also lost its process — finalise it
        # as cancelled so it never sticks at 'cancelling' across a restart.
        self._db.execute(
            "UPDATE jobs SET status='cancelled', finished_at=? WHERE status='cancelling'",
            (_now(),),
        )
        self._db.commit()

    # -- creation / launch -------------------------------------------------
    async def create(
        self,
        program: str,
        values: dict[str, Any],
        gpu: str | None,
        project_id: str,
        name: str | None = None,
        bindings: dict[str, dict[str, Any]] | None = None,
        launch: bool = True,
        replace_job_id: str | None = None,
        recreate_ssd: bool = False,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Create a run. With ``launch=False`` the job is fully prepared (run
        folder, materialised inputs, validated command) but left in the ``saved``
        state to execute later via :meth:`start`.

        With ``replace_job_id`` set, the existing job row is updated in place
        (same id / list position) and its previous run folder is removed, instead
        of inserting a new job — this backs the "rerun replaces the current run"
        behaviour."""
        project_dir = projects.dir_for(project_id)
        if project_dir is None:
            raise ProjectNotFound(project_id)

        # One uniquely-named folder per run holds its outputs + log (CryoSPARC-style).
        # An in-place continue (resume) instead reuses the interrupted run's folder, so
        # the program finds the checkpoint it left there and picks up where it stopped.
        workdir = None
        if resume and replace_job_id:
            prev = self.get(replace_job_id)
            if prev and prev.get("workdir") and Path(prev["workdir"]).is_dir():
                workdir = Path(prev["workdir"])
        if workdir is None:
            resume = False   # nothing to resume from -> fall back to a fresh run
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            workdir = project_dir / f"{program}_{stamp}_{uuid.uuid4().hex[:4]}"

        # Auto-fill the program's output-dir argument with the (future) run folder.
        values = dict(values)
        out_arg = registry.output_arg(program)
        if out_arg:
            values[out_arg] = str(workdir)

        # Auto-fill the SSD scratch dir (shared per project+program) when a
        # scratch root is configured, so users never pass an SSD path.
        ssd_dir = None
        ssd_arg = registry.ssd_arg(program)
        ssd_root = settings_store.ssd_scratch_root()
        if ssd_arg and ssd_root is not None:
            ssd_dir = ssd_root / project_id / program
            values[ssd_arg] = str(ssd_dir)

        # The run folder is created up-front so bound inputs (derived from a
        # previous job's output) can be materialised into it. Any failure below
        # cleans the folder back up so a bad submission leaves nothing behind.
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            values = self._resolve_bindings(values, bindings, workdir)
            command = build_command(program, values, gpu)  # raises ValidationError
        except (ValidationError, adapters.AdapterError):
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        if ssd_dir is not None:
            # The SSD scratch is shared per project+program and reused across runs
            # (the generator skips rebuilding when cached shards exist). Optionally
            # discard it so it's rebuilt from the current inputs — only on an actual
            # launch, so a merely-saved job can't clobber a concurrent run's scratch.
            if recreate_ssd and launch and ssd_dir.exists():
                shutil.rmtree(ssd_dir, ignore_errors=True)
            ssd_dir.mkdir(parents=True, exist_ok=True)

        # Launch from the metadata file's folder so relative image references in
        # .star/.xmd resolve; fall back to the run folder.
        cwd = self._launch_cwd(program, values, workdir)

        job_id = replace_job_id or uuid.uuid4().hex[:12]
        log_path = workdir / "run.log"
        status = "pending" if launch else "saved"
        name_val = (name or "").strip() or None
        # Persist the input bindings so a later rerun can re-materialise them into
        # a fresh run folder (the produced files live in this job's workdir).
        bindings_val = json.dumps(bindings) if bindings else None
        if replace_job_id:
            # Replace in place: keep the same id (and created_at, so the row stays
            # put in lists) and discard the previous run folder.
            old = self.get(replace_job_id)
            self._exec(
                "UPDATE jobs SET program=?, name=?, args=?, gpu=?, command=?, status=?, log_path=?, "
                "workdir=?, cwd=?, bindings=?, return_code=NULL, finished_at=NULL WHERE id=?",
                (program, name_val, json.dumps(values), gpu or None, json.dumps(command),
                 status, str(log_path), str(workdir), cwd, bindings_val, replace_job_id),
            )
            if old and old.get("workdir"):
                old_wd = Path(old["workdir"])
                if old_wd != workdir and old_wd.exists():
                    shutil.rmtree(old_wd, ignore_errors=True)
        else:
            self._exec(
                "INSERT INTO jobs (id, program, name, args, gpu, command, status, log_path, created_at, project_id, workdir, cwd, bindings) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, program, name_val, json.dumps(values), gpu or None,
                 json.dumps(command), status, str(log_path), _now(), project_id, str(workdir), cwd, bindings_val),
            )
        self._init_buffers(job_id)
        if launch:
            if resume:
                # Reused folder: clear metrics and mark the resume point in the kept log.
                self._prepare_resume(program, str(workdir), str(log_path))
            await self._launch(job_id, command, str(log_path), cwd, append_out=resume)
        return self.get(job_id)

    async def start(self, job_id: str) -> dict[str, Any]:
        """Launch a previously saved job using its stored command."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job["status"] in ACTIVE:
            raise JobActive(job_id)
        if job["status"] != "saved":
            raise JobNotSaved(job_id)
        self._init_buffers(job_id)
        self._exec("UPDATE jobs SET status='pending', return_code=NULL, finished_at=NULL WHERE id=?", (job_id,))
        await self._launch(job_id, job["command"], job["log_path"], job.get("cwd"))
        return self.get(job_id)

    def _resolve_bindings(
        self, values: dict[str, Any], bindings: dict[str, dict[str, Any]] | None, workdir: Path
    ) -> dict[str, Any]:
        """Turn each {field: {job_id, kind}} binding into a real input file.

        The produced file path replaces the field's value, so the program reads
        a job-derived input. Failures surface as field validation errors.
        """
        if not bindings:
            return values
        for field, binding in bindings.items():
            # A list binding feeds a multi-valued input (e.g. FlexConsensus
            # --input_space): materialise every source into its own file.
            if isinstance(binding, list):
                paths = []
                for i, one in enumerate(binding):
                    source = self.get(one.get("job_id", ""))
                    if source is None:
                        raise ValidationError({field: "a source job no longer exists"})
                    try:
                        paths.append(adapters.produce(one.get("kind", ""), source, workdir, f"{field}_{i}"))
                    except adapters.AdapterError as exc:
                        raise ValidationError({field: str(exc)})
                values[field] = paths
                continue
            source = self.get(binding.get("job_id", ""))
            if source is None:
                raise ValidationError({field: "source job no longer exists"})
            try:
                values[field] = adapters.produce(binding.get("kind", ""), source, workdir, field)
            except adapters.AdapterError as exc:
                raise ValidationError({field: str(exc)})
        return values

    async def rerun(self, job_id: str) -> dict[str, Any]:
        """Re-launch a job in place: the same job is replaced with a fresh run
        (new run folder, the old one removed) rather than creating a new job."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if not job.get("project_id"):
            raise ProjectNotFound("")
        if job["status"] in ACTIVE:
            raise JobActive(job_id)
        program = job["program"]
        args = dict(job["args"])
        # Drop the auto-managed dirs so create() regenerates them for this run.
        for arg in (registry.output_arg(program), registry.ssd_arg(program)):
            if arg:
                args.pop(arg, None)
        # Re-pass the original bindings so create() re-materialises job-derived
        # inputs into the fresh run folder (the previous folder is removed). Drop
        # the stale produced paths first so a missing binding can't slip through.
        bindings = job.get("bindings")
        if bindings:
            for field in bindings:
                args.pop(field, None)
        return await self.create(program, args, job["gpu"], job["project_id"],
                                 name=job.get("name"), bindings=bindings, replace_job_id=job_id)

    async def continue_job(self, job_id: str) -> dict[str, Any]:
        """Resume a finished run **in place**: relaunch its exact command so the program
        picks up the intermediate checkpoint it wrote inside its own output folder
        (restoring weights + optimizer + epoch — no ``--reload``, that is the network's
        own auto-resume).

        The job row and run folder are kept: only the metrics are cleared (so the curves
        restart cleanly) and stdout is appended to (its history is preserved); stderr
        starts fresh. Contrast :meth:`rerun`, which wipes the folder for a clean run."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job["status"] in ACTIVE:
            raise JobActive(job_id)
        if not adapters.is_resumable(job["program"], job.get("workdir")):
            raise JobNotResumable(job_id)
        self._prepare_resume(job["program"], job.get("workdir"), job["log_path"])
        self._init_buffers(job_id)
        self._exec("UPDATE jobs SET status='pending', return_code=NULL, finished_at=NULL WHERE id=?", (job_id,))
        await self._launch(job_id, job["command"], job["log_path"], job.get("cwd"), append_out=True)
        return self.get(job_id)

    def _prepare_resume(self, program: str, workdir: str | None, log_path: str) -> None:
        """Ready a run folder for an in-place continue: drop its metrics so they don't
        overlap the resumed run, and mark the resume point in the preserved stdout log."""
        metrics = adapters.resume_metrics_dir(program, workdir)
        if metrics is not None and metrics.exists():
            shutil.rmtree(metrics, ignore_errors=True)
        with contextlib.suppress(OSError):
            with open(log_path, "ab") as f:
                f.write(f"\n\n──────── resumed {_now()} ────────\n".encode())

    def rename(self, job_id: str, name: str) -> dict[str, Any]:
        """Set (or clear) a job's custom display name."""
        if self.get(job_id) is None:
            raise JobNotFound(job_id)
        self._exec("UPDATE jobs SET name=? WHERE id=?", ((name or "").strip() or None, job_id))
        return self.get(job_id)

    @staticmethod
    def _launch_cwd(program: str, values: dict[str, Any], workdir: Path) -> str:
        wd_arg = registry.working_dir_arg(program)
        if wd_arg and values.get(wd_arg):
            md = Path(str(values[wd_arg])).expanduser()
            # Where the metadata's image refs actually resolve from — usually its
            # own folder, but Scipion writes them relative to the project root.
            base = mdpaths.image_base_dir(md)
            if base is not None:
                return str(base)
            if md.parent.is_dir():
                return str(md.parent)
        return str(workdir)

    async def _launch(self, job_id: str, command: list[str], log_path: str, cwd: str | None = None,
                      append_out: bool = False) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                # Keep stderr on its own pipe so the UI can show it in a separate
                # tab (its own log file, buffer and stream-tagged chunks).
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                # hax colorizes output only on a TTY (honoring NO_COLOR/FORCE_COLOR).
                # We capture stdout through a pipe (not a TTY), but the log viewer
                # renders ANSI, so request colors explicitly with FORCE_COLOR.
                env={**os.environ, "FORCE_COLOR": "1"},
                # Own session/process group: a hax program spawns its real workers
                # via `forkserver` (JAX/CUDA), and those grandchildren inherit the
                # stdout pipe. Cancelling must signal the whole group, or they keep
                # the pipe open, the reader never sees EOF, and the job is stuck
                # "cancelling" forever. start_new_session => pgid == proc.pid.
                start_new_session=True,
            )
        except OSError as exc:
            self._exec(
                "UPDATE jobs SET status='failed', return_code=-1, finished_at=? WHERE id=?",
                (_now(), job_id),
            )
            self._publish(job_id, {"type": "chunk", "stream": "err", "data": f"[launch error] {exc}\n"})
            self._publish(job_id, {"type": "status", "status": "failed", "return_code": -1})
            return

        self._procs[job_id] = proc
        self._exec(
            "UPDATE jobs SET status='running', pid=?, started_at=? WHERE id=?",
            (proc.pid, _now(), job_id),
        )
        self._publish(job_id, {"type": "status", "status": "running", "pid": proc.pid})
        asyncio.create_task(self._reader(job_id, proc, log_path, append_out))

    async def _reader(self, job_id: str, proc: asyncio.subprocess.Process, log_path: str,
                      append_out: bool = False) -> None:
        # Drain stdout and stderr concurrently into their own log files, then wait
        # for exit. Both pipes must reach EOF before proc.wait() returns — the hax
        # program's forkserver workers inherit them, so a subset closing isn't done.
        # On an in-place continue stdout is appended (its history is kept); stderr
        # always starts fresh.
        await asyncio.gather(
            self._pump(job_id, proc.stdout, log_path, "out", append=append_out),
            self._pump(job_id, proc.stderr, self._err_path(log_path), "err"),
        )
        return_code = await proc.wait()
        self._procs.pop(job_id, None)
        current = self.get(job_id)
        status = "cancelled" if current and current["status"] == "cancelling" else (
            "succeeded" if return_code == 0 else "failed"
        )
        self._exec(
            "UPDATE jobs SET status=?, return_code=?, finished_at=? WHERE id=?",
            (status, return_code, _now(), job_id),
        )
        self._publish(job_id, {"type": "status", "status": status, "return_code": return_code})

    async def _pump(self, job_id: str, stream: asyncio.StreamReader | None, path: str, kind: str,
                    append: bool = False) -> None:
        """Drain one output stream: raw bytes to its log file (ANSI preserved),
        decoded text to the in-memory tail and to WS subscribers tagged by stream.
        ``append`` keeps the existing file (an in-place continue) instead of truncating."""
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with open(path, "ab" if append else "wb") as log:
            while True:
                data = await stream.read(4096)
                if not data:
                    break
                log.write(data)
                log.flush()
                text = decoder.decode(data)
                if text:
                    self._buffer_append(job_id, text, kind)
                    self._publish(job_id, {"type": "chunk", "stream": kind, "data": text})

    @staticmethod
    def _err_path(log_path: str) -> str:
        """The stderr log file beside the stdout one: run.log -> run.err.log."""
        p = Path(log_path)
        return str(p.with_name(p.stem + ".err" + p.suffix))

    def _init_buffers(self, job_id: str) -> None:
        self._buffers[job_id] = {s: deque() for s in STREAMS}
        self._buflens[job_id] = {s: 0 for s in STREAMS}

    async def cancel(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is None:
            return
        self._exec("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
        # Terminate the whole process group (workers + the launcher), not just the
        # direct child — otherwise grandchildren keep the stdout pipe open and the
        # reader never reaches the terminal-status update. SIGTERM first for a
        # clean shutdown, then SIGKILL if it doesn't exit promptly.
        self._signal_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._signal_group(proc, signal.SIGKILL)

    @staticmethod
    def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        # The process may exit on its own between our check and the signal; a group
        # owned by another user (shouldn't happen here) raises PermissionError.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), sig)

    def delete(self, job_id: str) -> None:
        """Remove a finished job: its DB record and its run folder (outputs+log)."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job["status"] in ACTIVE:
            raise JobActive(job_id)
        workdir = job.get("workdir")
        if workdir:
            wd = Path(workdir)
            # Only remove a per-run folder (named <program>_<stamp>_<id>), never
            # a project root or arbitrary path.
            if wd.is_dir() and wd.name.startswith(job["program"] + "_"):
                shutil.rmtree(wd, ignore_errors=True)
        self._buffers.pop(job_id, None)
        self._buflens.pop(job_id, None)
        self._subscribers.pop(job_id, None)
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))

    def delete_for_project(self, project_id: str) -> None:
        """Drop all job records for a project (used when the project is deleted).

        Refuses if any of its jobs is still active. The run folders themselves
        live inside the project folder, which the caller removes wholesale.
        """
        jobs = self.list_for_project(project_id)
        if any(j["status"] in ACTIVE for j in jobs):
            raise JobActive(project_id)
        for j in jobs:
            jid = j["id"]
            self._buffers.pop(jid, None)
            self._buflens.pop(jid, None)
            self._subscribers.pop(jid, None)
        self._exec("DELETE FROM jobs WHERE project_id=?", (project_id,))

    # -- pub/sub for live logs --------------------------------------------
    def _buffer_append(self, job_id: str, text: str, stream: str) -> None:
        bufs = self._buffers.setdefault(job_id, {s: deque() for s in STREAMS})
        lens = self._buflens.setdefault(job_id, {s: 0 for s in STREAMS})
        buf = bufs.setdefault(stream, deque())
        buf.append(text)
        lens[stream] = lens.get(stream, 0) + len(text)
        while lens[stream] > LIVE_BUFFER_CHARS and len(buf) > 1:
            lens[stream] -= len(buf.popleft())

    def subscribe(self, job_id: str) -> tuple[asyncio.Queue, str]:
        """Register a live subscriber and atomically snapshot the buffered tail.

        Done without an await in between so no chunk is duplicated or dropped.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(queue)
        snapshot = "".join(self._buffers.get(job_id, {}).get("out", ()))
        return queue, snapshot

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id)
        if subs:
            subs.discard(queue)

    def read_log(self, job_id: str, stream: str = "out", cap: int = 4_000_000, raw_cap: int = 64_000_000) -> str:
        """Full persisted log for one stream ("out"/"err"), tqdm frames collapsed.

        The in-memory buffer is only a recent tail, so reopening a job (running or
        finished) replays from the file to keep the earliest lines. We read up to
        ``raw_cap`` bytes from the end (enough for any realistic run in full),
        collapse repeated ``\\r`` progress frames to their final state, then cap the
        result to ``cap`` characters — so early lines survive even when a run has
        written megabytes of progress output.
        """
        job = self.get(job_id)
        if not job:
            return ""
        log_path = job["log_path"] if stream == "out" else self._err_path(job["log_path"])
        try:
            p = Path(log_path)
            size = p.stat().st_size
            with open(p, "rb") as f:
                if size > raw_cap:
                    f.seek(size - raw_cap)
                data = f.read()
        except OSError:
            return "".join(self._buffers.get(job_id, {}).get(stream, ()))
        return _collapse_progress(data.decode("utf-8", errors="replace"))[-cap:]

    def _publish(self, job_id: str, message: dict[str, Any]) -> None:
        for queue in self._subscribers.get(job_id, set()):
            queue.put_nowait(message)

    # -- reads -------------------------------------------------------------
    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self._exec("SELECT * FROM jobs WHERE id=?", (job_id,), fetch="one")
        return self._row(row) if row else None

    def list(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._exec(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit), fetch="all",
            )
        else:
            rows = self._exec("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,), fetch="all")
        return [self._row(r) for r in rows]

    def count_active(self) -> int:
        """Number of jobs currently pending or running (used to gate updates)."""
        row = self._exec(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('pending','running')",
            fetch="one",
        )
        return int(row["n"]) if row else 0

    def list_for_project(self, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit), fetch="all",
        )
        return [self._row(r) for r in rows]

    def tail(self, job_id: str, max_lines: int | None = None, stream: str = "out") -> list[str]:
        job = self.get(job_id)
        if not job:
            return []
        log_path = job["log_path"] if stream == "out" else self._err_path(job["log_path"])
        try:
            with open(log_path, encoding="utf-8") as log:
                lines = log.read().splitlines()
        except FileNotFoundError:
            lines = list(self._buffers.get(job_id, {}).get(stream, ()))
        return lines[-max_lines:] if max_lines else lines

    # -- helpers -----------------------------------------------------------
    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["args"] = json.loads(data["args"])
        data["command"] = json.loads(data["command"])
        data["bindings"] = json.loads(data["bindings"]) if data.get("bindings") else None
        data["active"] = data["status"] in ACTIVE
        return data

    def _exec(self, sql: str, params: tuple = (), fetch: str | None = None):
        assert self._db is not None, "JobManager.init() not called"
        cur = self._db.execute(sql, params)
        result = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        self._db.commit()
        return result


manager = JobManager()
