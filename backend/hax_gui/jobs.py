"""Job manager: launch hax programs as subprocesses, persist metadata, stream logs.

Single-server, plain-subprocess execution (PLAN.md §10). Jobs run detached from
any one client so long trainings survive browser disconnects; the UI reattaches
to live logs on reconnect. Job metadata lives in SQLite; full logs are files.

Jobs also outlive the *server*. Stopping, restarting or updating the GUI leaves a
running training alone, and the next server process reattaches to it. Three things
make that work, and all three are load-bearing:

* **The job owns its log file descriptors** (`_launch`). They used to be pipes whose
  only read end lived in this process, so the moment the server exited the job's next
  write got EPIPE and the whole tree — the program and every grain worker that
  inherited the fd — died. `start_new_session=True` never protected against that: the
  pipe, not a signal, was the leash. Writing straight to the file changes nothing about
  what is written (a pipe and a file are both non-TTYs, so the program emits the same
  ANSI and the same `\\r` progress frames), it just stops the server's exit from
  killing the job.
* **The job records its own exit status** (`_wrap`). After a restart we are no longer
  the job's parent, so `waitpid` is gone and a reattached job could only ever be seen
  to vanish, never to have succeeded or failed.
* **Identity, not just a pid** (`_proc_key`). Pids get recycled, so a stored pid alone
  is not safe to signal after a restart — it could name an innocent process by then.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import os
import shlex
import shutil
import signal
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapters, mdpaths, projects, registry, settings_store, usage
from .command import ValidationError, build_command, strip_obsolete
from .config import settings

# In-memory tail (characters) kept per job for fast WS replay on reconnect. Raw
# text including ANSI codes and carriage returns, so the client terminal can
# reconstruct progress lines exactly.
LIVE_BUFFER_CHARS = 256_000

# Statuses a job can still move out of on its own. 'cancelling' belongs here: the
# process group has been signalled but nothing is final until the supervisor reaps it
# and writes 'cancelled'. Leaving it out made the log WS treat a cancelling job as
# finished and hang up, so the client never saw that final status.
ACTIVE = {"pending", "running", "cancelling"}

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
    def __init__(self, job_id: str, reason: str | None = None):
        super().__init__(reason or f"job '{job_id}' has no loadable checkpoint to continue from")
        self.job_id = job_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# How often a reattached job's log file is polled for new bytes. The old pipe reader
# woke on every write; this is a poll, so it trades a little latency for the job being
# able to outlive us. The terminal collapses `\r` frames anyway, so coalescing a few
# progress updates into one chunk looks no different on screen.
TAIL_INTERVAL = 0.2


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just not ours to signal
    return True


def _proc_key(pid: int) -> str | None:
    """A cheap identity for a pid: its start time, in clock ticks since boot.

    A pid on its own is *not* safe to signal across a restart — the kernel recycles
    them, so a pid stored yesterday may name something else entirely today, and
    `killpg` on it would take out an innocent process group. Pairing the pid with its
    start time makes reuse detectable: a recycled pid is necessarily younger than the
    one we recorded.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    # Field 2 (comm) is parenthesised and may itself contain spaces and ')', so split
    # only what follows the *last* ')'. starttime is field 22 overall, i.e. index 19
    # of the remainder.
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return rest[19].decode()
    except (ValueError, IndexError):
        return None


def _group_alive(pgid: int, key: str | None) -> bool:
    """Is the job's process-group leader still the very process we launched?"""
    if not pgid or pgid <= 0 or not _pid_alive(pgid):
        return False
    if not key:
        return True          # pre-migration row: no identity recorded, trust the pid
    now = _proc_key(pgid)
    return now is not None and now == key


def _group_members(pgid: int) -> list[int]:
    """Every live pid in a process group."""
    out: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for d in entries:
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            if os.getpgid(pid) == pgid:
                out.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    return out


def _exit_path(log_path: str) -> str:
    """Where the job records its own exit status: run.log -> run.exit."""
    p = Path(log_path)
    return str(p.with_name(p.stem + ".exit"))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _read_exit(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _wrap(command: list[str], exit_path: str) -> list[str]:
    """Run `command`, then leave its exit status somewhere a future server can read it.

    `exit $rc` keeps the wrapper's own exit code equal to the program's, so the common
    case — we are still the parent and simply wait on it — behaves exactly as before.
    """
    return ["/bin/sh", "-c",
            f'{shlex.join(command)}; rc=$?; printf %s "$rc" > {shlex.quote(exit_path)}; exit $rc']


class JobManager:
    def __init__(self) -> None:
        self._db: sqlite3.Connection | None = None
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        # Recent raw text tail per job, split by stream: job_id -> {stream: deque}.
        self._buffers: dict[str, dict[str, deque[str]]] = {}
        self._buflens: dict[str, dict[str, int]] = {}
        # Byte offset each stream's tailer has *published* up to, per job. This is
        # the cut point a log-WebSocket snapshot must stop at: everything past it
        # arrives through the subscriber queue, so reading the file further would
        # duplicate those chunks (see tail_positions / read_log's `upto`).
        self._tail_pos: dict[str, dict[str, int]] = {}

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
        for col in ("project_id TEXT", "workdir TEXT", "cwd TEXT", "name TEXT", "bindings TEXT",
                    "pgid INTEGER", "proc_key TEXT", "stale INTEGER NOT NULL DEFAULT 0"):
            try:
                self._db.execute(f"ALTER TABLE jobs ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        self._reconcile()

    def _reconcile(self) -> None:
        """Work out what actually happened to the jobs of the previous server process.

        This used to be a blanket "anything that was running is now interrupted", which
        was true only because the log pipe killed every job when the server exited. Jobs
        now survive that, so we have to look: a job whose process group is still alive is
        still *running* and gets picked up by :meth:`reattach`; one that finished while we
        were down left its exit status behind (see `_wrap`) and is finalised from it.
        """
        assert self._db is not None
        rows = self._db.execute(
            "SELECT id, pgid, proc_key, status, log_path FROM jobs "
            "WHERE status IN ('pending','running','cancelling')"
        ).fetchall()
        self._reattach: list[dict[str, Any]] = []
        for r in rows:
            pgid = int(r["pgid"] or 0)
            if _group_alive(pgid, r["proc_key"]):
                self._reattach.append(dict(r))     # outlived us; reattach() resumes it
                continue
            # Gone. Did it get to say why? ('pending' never launched, so it never can.)
            rc = _read_exit(_exit_path(r["log_path"]))
            if r["status"] == "cancelling":
                final = "cancelled"
            elif rc is None:
                final = "interrupted"              # killed from outside, or the box went down
            else:
                final = "succeeded" if rc == 0 else "failed"
            self._db.execute(
                "UPDATE jobs SET status=?, return_code=?, finished_at=? WHERE id=?",
                (final, rc, _now(), r["id"]),
            )
        self._db.commit()

    async def reattach(self) -> None:
        """Resume supervision of the jobs that outlived the previous server process."""
        pending, self._reattach = getattr(self, "_reattach", []), []
        for r in pending:
            out_path = r["log_path"]
            err_path = self._err_path(out_path)
            # Tail from the current end: the UI replays the history from the file itself
            # (read_log), so re-publishing it over the socket would only duplicate it.
            asyncio.create_task(self._supervise_detached(
                r["id"], int(r["pgid"] or 0), r["proc_key"], out_path, err_path,
                _size(out_path), _size(err_path),
            ))

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
        dropped_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a run. With ``launch=False`` the job is fully prepared (run
        folder, materialised inputs, validated command) but left in the ``saved``
        state to execute later via :meth:`start`.

        With ``replace_job_id`` set, the existing job row is updated in place
        (same id / list position) and its previous run folder is *reused* — cleared
        and rebuilt, rather than replaced by a differently-named one — instead of
        inserting a new job. This backs the "rerun replaces the current run"
        behaviour."""
        project_dir = projects.dir_for(project_id)
        if project_dir is None:
            raise ProjectNotFound(project_id)

        # One folder per job holds its outputs + log, named after the job itself:
        # `<program>_<job_id>`. The name is *stable*, so re-running lands back in the
        # same folder instead of minting a timestamped one and deleting the old.
        #
        # That stability is the point. Most job-derived inputs are referenced in place
        # (adapters.produce hands the consumer a path *inside* the producer's run folder
        # for checkpoints, particle metadata and volumes), so a rerun that moved the
        # folder invalidated every downstream run's stored command by path alone — on top
        # of the content going stale, which is the honest part and is still flagged (see
        # mark_stale / cascade.py). Nothing is preserved by a fresh name either: a rerun
        # already deleted the previous folder outright.
        #
        # An in-place continue (resume) reuses the interrupted run's folder *as it is*,
        # so the program finds the checkpoint it left there and picks up where it stopped
        # — including a legacy timestamped folder from before this scheme.
        job_id = replace_job_id or uuid.uuid4().hex[:12]
        prev = self.get(replace_job_id) if replace_job_id else None
        prev_wd = Path(prev["workdir"]) if prev and prev.get("workdir") else None
        workdir = project_dir / f"{program}_{job_id}"
        if resume:
            if prev_wd is not None and prev_wd.is_dir():
                workdir = prev_wd
            else:
                resume = False   # nothing to resume from -> fall back to a fresh run

        # Auto-fill the program's output-dir argument with the (future) run folder.
        values = dict(values)
        out_arg = registry.output_arg(program)
        if out_arg:
            values[out_arg] = str(workdir)

        # Auto-fill the SSD scratch dir (shared per project) when a scratch root is
        # configured, so users never pass an SSD path.
        #
        # Shared by every program in the project, deliberately. hax caches the particle
        # images inside this folder as an array-record copy, and that copy is expensive:
        # building it means reading the whole stack once (410 GB for 1M particles at box
        # 320 -- an hour off a spinning disk). It is also program-agnostic -- HetSIREN,
        # MoDART and reconstruct_volume all build and read exactly the same shards -- so a
        # per-program folder made each of them pay that price separately. One folder per
        # project means whoever runs first pays, and everyone after reads from the SSD.
        #
        # This is safe because hax names each cache directory after the *dataset* it holds
        # (MetaDataGenerator._cache_dir: a digest of the metadata path, particle count, box
        # and precision). Two datasets in one project therefore get two directories rather
        # than silently sharing one -- which is what previously made a shared folder
        # dangerous, and why this used to be split per program.
        ssd_dir = None
        ssd_arg = registry.ssd_arg(program)
        ssd_root = settings_store.ssd_scratch_root()
        if ssd_arg and ssd_root is not None:
            ssd_dir = ssd_root / project_id
            values[ssd_arg] = str(ssd_dir)

        # The run folder is created up-front so bound inputs (derived from a
        # previous job's output) can be materialised into it. A failure below cleans up
        # a folder we just created, so a bad submission leaves nothing behind — but a
        # *reused* folder is left alone: it still holds a real previous run, and a
        # rejected submission must not be what destroys it. (Its inputs are what they
        # were: the only files touched so far are this job's own materialised ones,
        # which carry the field's name and are rewritten identically every time.)
        existed = workdir.is_dir()
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            # Off the event loop: materialising a binding can parse a million-row
            # metadata file into an .npy (tens of seconds). Blocking here froze the
            # entire server — every request, every log WebSocket, the scheduler —
            # for the duration, indistinguishable from a hang.
            values = await asyncio.to_thread(self._resolve_bindings, values, bindings, workdir)
            command = build_command(program, values, gpu)  # raises ValidationError
        except (ValidationError, adapters.AdapterError):
            if not existed:
                shutil.rmtree(workdir, ignore_errors=True)
            raise
        if not resume:
            # Command accepted: now clear whatever the previous run left here, keeping the
            # inputs just materialised above. Not optional — the networks auto-resume from
            # an intermediate checkpoint found in their own output folder (adapters.
            # _RESUMABLE), so a leftover one would silently turn this fresh run into a
            # continue. A brand-new folder makes this a no-op.
            #
            # Off the loop: this rmtree walks a whole previous run (checkpoints,
            # decoded volume series) — minutes on a big one.
            await asyncio.to_thread(
                self._clear_run_folder, workdir, self._materialised(values, workdir))
        if ssd_dir is not None:
            # The SSD scratch is shared per project and reused across runs (the generator
            # skips rebuilding when cached shards exist). Optionally discard it so it's
            # rebuilt from the current inputs — only on an actual launch, so a merely-saved
            # job can't clobber a concurrent run's scratch.
            #
            # Only THIS job's dataset is discarded, never the whole folder: it now holds the
            # caches of every dataset in the project, and wiping it would throw away hours of
            # rebuilding for datasets this run has nothing to do with.
            if recreate_ssd and launch and ssd_dir.exists():
                # Off the loop: the array-record cache is enormous (the module
                # comment above sizes it at 410 GB for 1M particles) — deleting
                # it inline blocked the whole server for up to an hour.
                await asyncio.to_thread(
                    self._purge_dataset_cache, ssd_dir,
                    values.get(registry.working_dir_arg(program)))
            ssd_dir.mkdir(parents=True, exist_ok=True)

        # Launch from the metadata file's folder so relative image references in
        # .star/.xmd resolve; fall back to the run folder.
        cwd = self._launch_cwd(program, values, workdir)

        log_path = workdir / "run.log"
        status = "pending" if launch else "saved"
        # `None` means "not specified", and is not the same as "". Replacing a job keeps
        # the name it already carries, so a caller that never had it — a workflow node
        # posted from a graph built before the step was named, an API client that omits
        # the field — cannot silently erase a name the user set from the Jobs view. An
        # explicit empty string is what clears it.
        name_val = None if name is None else (name.strip() or None)
        if name is None and prev is not None:
            name_val = prev.get("name")
        # Persist the input bindings so a later rerun can re-materialise them into
        # the rebuilt run folder (the produced files live in this job's workdir).
        bindings_val = json.dumps(bindings) if bindings else None
        if replace_job_id:
            # Replace in place: keep the same id (and created_at, so the row stays put in
            # lists). The run folder is normally the same one, already cleared above.
            old = prev
            # `stale=0`: this run has just been re-made from its current inputs, so
            # whatever invalidated it (see :meth:`mark_stale`) no longer applies.
            #
            # The clock is reset with it. `created_at` is deliberately kept (that is what
            # holds the row's place in the lists), so it describes the *slot*, not the run
            # sitting in it — started_at/finished_at are what describe the run, and leaving
            # the previous run's times on a replaced row makes the UI quote a run that no
            # longer exists. `_launch` fills started_at back in immediately when launching;
            # a merely-saved replacement correctly ends up with neither.
            self._exec(
                "UPDATE jobs SET program=?, name=?, args=?, gpu=?, command=?, status=?, log_path=?, "
                "workdir=?, cwd=?, bindings=?, stale=0, return_code=NULL, started_at=NULL, "
                "finished_at=NULL WHERE id=?",
                (program, name_val, json.dumps(values), gpu or None, json.dumps(command),
                 status, str(log_path), str(workdir), cwd, bindings_val, replace_job_id),
            )
            # A different folder only happens when this job's *name* changed under it: a
            # run made before folders were named after the job (legacy `<program>_<stamp>_
            # <hex>`), or one whose program was swapped in an edit. Either way the old
            # folder is now orphaned, so re-making the run is also what migrates it.
            if old and old.get("workdir"):
                old_wd = Path(old["workdir"])
                if old_wd != workdir and old_wd.exists():
                    # A whole orphaned run folder — delete it off the loop.
                    await asyncio.to_thread(shutil.rmtree, old_wd, ignore_errors=True)
                    usage.forget(str(old_wd))
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
                await asyncio.to_thread(self._prepare_resume, program, str(workdir), str(log_path))
            # Say so in the run's own log rather than only in the reply: this is the
            # record someone reads months later when asking why a repeated run differs
            # from the original.
            notice = None
            if dropped_args:
                notice = (f"[hax-gui] {program} no longer accepts "
                          f"{', '.join('--' + a for a in dropped_args)}; re-running without "
                          f"{'it' if len(dropped_args) == 1 else 'them'}.\n\n")
            await self._launch(job_id, command, str(log_path), cwd, append_out=resume, notice=notice)
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
        # A saved run that was invalidated (:meth:`mark_stale`) holds inputs materialised
        # from a source run that has been re-made since: copies of data that no longer
        # exists, and references into a folder whose contents were thrown away. Launching
        # its stored command would read whatever is there now (or nothing), so rebuild the
        # run from its bindings instead of launching what we have.
        if job.get("stale"):
            return await self.rerun(job_id)
        self._init_buffers(job_id)
        self._exec("UPDATE jobs SET status='pending', stale=0, return_code=NULL, finished_at=NULL WHERE id=?", (job_id,))
        await self._launch(job_id, job["command"], job["log_path"], job.get("cwd"))
        return self.get(job_id)

    def relink_project(self, project_id: str) -> list[dict[str, Any]]:
        """Recover input bindings that were flattened to literal paths.

        Editing a run and re-launching it (the launcher form) submits a *literal* value
        for every field, including one that used to be fed by another run's output: the
        job's stored args already held that run's resolved output path, so the edit
        froze it in and dropped the binding. The workflow edge, which is derived from the
        binding, then vanished and the input read as hardcoded.

        This walks the project's jobs and, for each bindable field still holding a plain
        path, looks for the one other job whose output *is* that exact path (see
        :func:`adapters.reference_path`). A unique match restores the binding; the
        workflow edge reappears on the next reconcile. Ambiguous matches (two runs at the
        same path) and copied inputs (latents, whose path points into the consumer's own
        folder) are left alone rather than guessed. Idempotent: an already-bound field is
        skipped, so it is safe to run repeatedly.
        """
        job_list = self.list_for_project(project_id)
        # Each producer's in-place output path per kind, computed once.
        produced: list[tuple[str, str, str]] = []       # (abspath, kind, producer_id)
        kinds = {k for j in job_list for k in adapters.bindable_fields(j["program"]).values()}
        for producer in job_list:
            for kind in kinds:
                ref = adapters.reference_path(kind, producer)
                if ref:
                    produced.append((os.path.abspath(ref), kind, producer["id"]))

        restored: list[dict[str, Any]] = []
        for consumer in job_list:
            fields = adapters.bindable_fields(consumer["program"])
            if not fields:
                continue
            bindings = dict(consumer.get("bindings") or {})
            args = consumer.get("args") or {}
            changed = False
            for field, kind in fields.items():
                if field in bindings:
                    continue                             # already fed by a run
                val = args.get(field)
                if not isinstance(val, str) or not val:
                    continue                             # unset, or a multi-valued list
                target = os.path.abspath(val)
                srcs = {pid for path, k, pid in produced
                        if k == kind and path == target and pid != consumer["id"]}
                if len(srcs) == 1:
                    bindings[field] = {"job_id": next(iter(srcs)), "kind": kind}
                    changed = True
                    restored.append({"job": consumer["id"], "field": field,
                                     "kind": kind, "source": bindings[field]["job_id"]})
            if changed:
                self._exec("UPDATE jobs SET bindings=? WHERE id=?",
                           (json.dumps(bindings), consumer["id"]))
        return restored

    @staticmethod
    def _materialised(values: dict[str, Any], workdir: Path) -> set[Path]:
        """This run's own inputs that live *inside* ``workdir`` — what the clear-out must
        not take with it.

        Most job-derived inputs are referenced where they already are (a checkpoint or a
        metadata file inside the producer's folder), so they are not in here; the ones
        that are, are the copied/extracted artifacts ``adapters.produce`` writes as
        ``<workdir>/<field>.npy``, which cost a file copy to re-make.

        Every value is scanned rather than only the bound fields, because a binding that
        was flattened to a literal path by an edit (see :meth:`relink_project`) still
        points at that same copied file — and clearing the folder from under it would
        leave the run reading an input that isn't there.
        """
        keep: set[Path] = set()
        root = workdir.resolve()
        for val in values.values():
            for one in (val if isinstance(val, list) else [val]):
                if not isinstance(one, str) or not one:
                    continue
                with contextlib.suppress(OSError):
                    path = Path(one).resolve()
                    if path.parent == root:
                        keep.add(path)
        return keep

    @staticmethod
    def _clear_run_folder(workdir: Path, keep: set[Path] | None = None) -> None:
        """Empty a run folder in place, keeping `keep`.

        The folder itself survives (it is this job's identity now, and the paths pointing
        at it must stay valid). Anything that cannot be removed is left rather than raised
        on: a rerun failing outright because one file is held open would be worse than a
        folder with a leftover in it, and the run is about to overwrite its own outputs
        anyway. Deleting files a *running* reader still holds open is safe on POSIX — it
        keeps reading the data until it closes — and the cascade dialog is what stops
        downstream runs from being left in that state (see cascade.py).
        """
        keep = {p.resolve() for p in (keep or set())}
        with contextlib.suppress(OSError):
            for entry in workdir.iterdir():
                if entry.resolve() in keep:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    with contextlib.suppress(OSError):
                        entry.unlink()
        # The measured size on record is the previous run's; drop it so the project page
        # doesn't quote it for up to a TTL after the folder was emptied.
        usage.forget(str(workdir))

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
        """Re-launch a job in place: the same job is replaced with a fresh run — its own
        run folder cleared and rebuilt, keeping the path — rather than creating a new job."""
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
        # ...and anything the program has since stopped accepting, or this run could never
        # be repeated once its CLI moved on (see command.strip_obsolete).
        args, dropped = strip_obsolete(program, args)
        # Re-pass the original bindings so create() re-materialises job-derived
        # inputs into the fresh run folder (the previous folder is removed). Drop
        # the stale produced paths first so a missing binding can't slip through.
        bindings = job.get("bindings")
        if bindings:
            for field in bindings:
                args.pop(field, None)
        return await self.create(program, args, job["gpu"], job["project_id"],
                                 name=job.get("name"), bindings=bindings, replace_job_id=job_id,
                                 dropped_args=dropped)

    # -- what a run feeds, and what re-making it invalidates ---------------
    def dependents(self, job_id: str) -> list[dict[str, Any]]:
        """Every run downstream of `job_id`, transitively.

        A job's ``bindings`` name the runs it consumes, so the reverse map over the
        project *is* the dependency graph — the same one the canvas draws, whose edges
        are derived from these bindings (see ``workflows._synced_graph``). Anything
        reachable here was computed from this run's outputs, and stops being valid the
        moment those outputs are re-made.

        Returned oldest first, which for a pipeline built front to back reads in the
        order the runs were chained. Execution order is not decided here: queueing hands
        the set to the workflow scheduler, which topologically sorts it itself.
        """
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        project_id = job.get("project_id")
        if not project_id:
            return []
        siblings = list(reversed(self.list_for_project(project_id)))    # oldest first
        by_id = {j["id"]: j for j in siblings}
        consumers: dict[str, list[str]] = {}
        for consumer in siblings:
            for binding in (consumer.get("bindings") or {}).values():
                # A multi-valued input (e.g. FlexConsensus --input_space) holds a list of
                # sources, so every one of them feeds this consumer.
                for one in (binding if isinstance(binding, list) else [binding]):
                    src = (one or {}).get("job_id")
                    if src and src != consumer["id"]:
                        consumers.setdefault(src, []).append(consumer["id"])
        seen, queue = {job_id}, [job_id]
        while queue:
            for cid in consumers.get(queue.pop(0), []):
                if cid not in seen and cid in by_id:
                    seen.add(cid)
                    queue.append(cid)
        return [j for j in siblings if j["id"] in seen and j["id"] != job_id]

    def mark_stale(self, job_ids: list[str]) -> list[str]:
        """Flag runs whose inputs were re-made under them; returns the ids actually marked.

        Deliberately *not* destructive. A stale run keeps its outputs and its log — they
        are still a real result, just one computed from a version of its input that no
        longer exists — so it can be inspected and compared until the user chooses to
        re-run it, at which point the folder is rebuilt like any other rerun. What the
        flag does buy is honesty: the UI marks it, it cannot be continued in place, and
        it is called out when offered as the input of a new run.
        """
        marked = []
        for jid in job_ids:
            job = self.get(jid)
            if job is None or job.get("stale"):
                continue
            self._exec("UPDATE jobs SET stale=1 WHERE id=?", (jid,))
            marked.append(jid)
        return marked

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
        # Continuing reuses this folder *and* this command, so it would read the inputs
        # materialised from a source run that has since been re-made — paths that are now
        # gone. An invalidated run can only be re-run, which rebuilds them.
        if job.get("stale"):
            raise JobNotResumable(
                job_id, "this run's inputs were re-made after it finished — re-run it instead of continuing")
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

    def copy_prefill(self, job_id: str) -> dict[str, Any]:
        """The recipe for re-creating a job in *another* project, as a launcher prefill.

        We copy the program and its parameters, but not anything tied to the source
        project or its run folder: the auto-managed output/SSD dirs are dropped (the new
        project regenerates them), and every job-derived input field is cleared -- both
        real bindings (which point at the source project's jobs) and literal data paths
        (raw metadata etc.) -- so they are chosen fresh in the target project. The result
        seeds the New Run form; nothing is created until the user saves or launches."""
        job = self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        program = job["program"]
        values = dict(job.get("args") or {})
        for auto in (registry.output_arg(program), registry.ssd_arg(program)):
            if auto:
                values.pop(auto, None)
        for field in adapters.bindable_fields(program):
            values.pop(field, None)
        return {"program": program, "gpu": job.get("gpu") or "", "values": values, "name": job.get("name") or ""}

    @staticmethod
    def _purge_dataset_cache(ssd_dir: Path, md_path: str | None) -> None:
        """Delete the cached images of one dataset from the project's shared scratch folder.

        hax writes each cache as ``<scratch>/images_mmap*_<digest>/`` with a ``manifest.json``
        naming the dataset it holds, so we match on that rather than recomputing the digest
        here — the GUI must not have to know how hax hashes a dataset. A cache with no
        manifest predates that scheme (or isn't ours) and is left alone.

        With no metadata path to match (a program whose scratch isn't keyed to an ``--md``),
        nothing is deleted: silently wiping every dataset in the project would be far worse
        than ignoring the checkbox.

        Deleting shards from under a job that is *currently* streaming them is safe on POSIX:
        the reader holds the files open, so the data survives until it closes. The next run
        rebuilds them.
        """
        if not md_path:
            return
        target = os.path.abspath(str(md_path))
        for manifest in ssd_dir.glob("images_mmap*/manifest.json"):
            try:
                held = json.loads(manifest.read_text()).get("md")
            except (OSError, json.JSONDecodeError):
                continue
            if held and os.path.abspath(str(held)) == target:
                shutil.rmtree(manifest.parent, ignore_errors=True)

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
                      append_out: bool = False, notice: str | None = None) -> None:
        out_path, err_path = log_path, self._err_path(log_path)
        exit_path = _exit_path(log_path)
        with contextlib.suppress(OSError):
            os.remove(exit_path)   # a stale status from a previous run must not be read back
        # On an in-place continue stdout is appended (its history is kept) and we stream
        # from where it already ended; stderr always starts fresh.
        out_off = _size(out_path) if append_out else 0

        try:
            # The job owns these fds — this is what lets it outlive the server. They used
            # to be pipes whose only read end was ours, so our exit gave the job (and every
            # grain worker that inherited the fd) EPIPE on its next write and killed the
            # whole tree. What gets *written* is unchanged: a pipe and a regular file are
            # both non-TTYs, so the program emits the same ANSI and the same `\r` progress
            # frames, and the terminal in the UI still overwrites them in place.
            out_f = open(out_path, "ab" if append_out else "wb")
            err_f = open(err_path, "wb")
            if notice:
                # Written before the child inherits the fd, so its own output follows on
                # from here (parent and child share one file description, hence one offset).
                out_f.write(notice.encode())
                out_f.flush()
        except OSError as exc:
            self._fail_launch(job_id, exc)
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                *_wrap(command, exit_path),
                stdout=out_f,
                stderr=err_f,
                cwd=cwd,
                # hax colorizes output only on a TTY (honoring NO_COLOR/FORCE_COLOR). The
                # log viewer renders ANSI, so ask for colors explicitly.
                env={**os.environ, "FORCE_COLOR": "1"},
                # Own session/process group: a hax program spawns its real workers via
                # `forkserver` (JAX/CUDA). Signalling must reach the whole group, or the
                # workers survive the leader and keep holding the GPU.
                # start_new_session => pgid == proc.pid.
                start_new_session=True,
            )
        except OSError as exc:
            self._fail_launch(job_id, exc)
            return
        finally:
            out_f.close()   # the child has its own dup; we must not keep these open
            err_f.close()

        key = _proc_key(proc.pid)
        self._procs[job_id] = proc
        self._exec(
            "UPDATE jobs SET status='running', pid=?, pgid=?, proc_key=?, started_at=? WHERE id=?",
            (proc.pid, proc.pid, key, _now(), job_id),
        )
        self._publish(job_id, {"type": "status", "status": "running", "pid": proc.pid})
        asyncio.create_task(self._supervise(job_id, proc, out_path, err_path, out_off, 0))

    def _fail_launch(self, job_id: str, exc: OSError) -> None:
        self._exec(
            "UPDATE jobs SET status='failed', return_code=-1, finished_at=? WHERE id=?",
            (_now(), job_id),
        )
        self._publish(job_id, {"type": "chunk", "stream": "err", "data": f"[launch error] {exc}\n"})
        self._publish(job_id, {"type": "status", "status": "failed", "return_code": -1})

    async def _supervise(self, job_id: str, proc: asyncio.subprocess.Process,
                         out_path: str, err_path: str, out_off: int, err_off: int) -> None:
        """Watch a job we launched: stream its logs, then record how it ended."""
        stop = asyncio.Event()
        tails = [
            asyncio.create_task(self._tail(job_id, out_path, "out", out_off, stop)),
            asyncio.create_task(self._tail(job_id, err_path, "err", err_off, stop)),
        ]
        try:
            # `_wrap` re-raises the program's own code, so this is still the real one.
            return_code = await proc.wait()
        finally:
            self._procs.pop(job_id, None)
        stop.set()
        await asyncio.gather(*tails, return_exceptions=True)
        await self._reap_group(job_id, proc.pid)
        self._finalize(job_id, return_code)

    async def _supervise_detached(self, job_id: str, pgid: int, key: str | None,
                                  out_path: str, err_path: str, out_off: int, err_off: int) -> None:
        """Watch a job that outlived the previous server process.

        We are not its parent any more — it was reparented to init when the old server
        died — so there is no `wait()` to await and no exit code to collect from the
        kernel. Poll the process group instead, and take the status the job left behind.
        """
        self._publish(job_id, {"type": "status", "status": "running", "pid": pgid})
        stop = asyncio.Event()
        tails = [
            asyncio.create_task(self._tail(job_id, out_path, "out", out_off, stop)),
            asyncio.create_task(self._tail(job_id, err_path, "err", err_off, stop)),
        ]
        while _group_alive(pgid, key):
            await asyncio.sleep(1.0)
        stop.set()
        await asyncio.gather(*tails, return_exceptions=True)
        await self._reap_group(job_id, pgid)
        self._finalize(job_id, _read_exit(_exit_path(out_path)))

    async def _tail(self, job_id: str, path: str, kind: str, offset: int,
                    stop: asyncio.Event) -> None:
        """Stream one of a job's log files to the live subscribers.

        Replaces reading the subprocess' pipe. The job writes the file itself now, so we
        forward the bytes exactly as it wrote them — ANSI intact, `\\r` progress frames
        intact — which is what keeps the terminal overwriting progress lines in place
        rather than scrolling them.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pos = offset
        self._tail_pos.setdefault(job_id, {})[kind] = pos
        while True:
            data = b""
            try:
                with open(path, "rb") as f:
                    f.seek(pos)
                    data = f.read()
            except OSError:
                pass
            if data:
                pos += len(data)
                text = decoder.decode(data)
                if text:
                    self._buffer_append(job_id, text, kind)
                    self._publish(job_id, {"type": "chunk", "stream": kind, "data": text})
                    self._tail_pos.setdefault(job_id, {})[kind] = pos
                continue                      # keep draining while there is more
            if stop.is_set():
                return                        # the job is gone and the file is drained
            await asyncio.sleep(TAIL_INTERVAL)

    def _finalize(self, job_id: str, return_code: int | None) -> None:
        current = self.get(job_id)
        if current is None:
            return
        if current["status"] == "cancelling":
            status = "cancelled"
        elif return_code is None:
            # The process is gone but never recorded a status: killed from outside, or the
            # machine went down under it. We cannot invent a return code for it.
            status = "interrupted"
        else:
            status = "succeeded" if return_code == 0 else "failed"
        self._exec(
            "UPDATE jobs SET status=?, return_code=?, finished_at=? WHERE id=?",
            (status, return_code, _now(), job_id),
        )
        self._publish(job_id, {"type": "status", "status": status, "return_code": return_code})

    async def _reap_group(self, job_id: str, pgid: int) -> None:
        """Kill whatever is left in a finished job's process group.

        hax runs its data loaders as grain/forkserver children. If the program dies
        abnormally they are orphaned and keep running — holding GPU memory — and nothing
        has ever cleaned them up: with the old pipe the server just blocked forever
        waiting for an EOF they were still holding open. Once the leader has exited,
        anything still in its group is exactly that debris.
        """
        left = [p for p in _group_members(pgid) if p != pgid]
        if not left:
            return
        self._publish(job_id, {"type": "chunk", "stream": "err",
                               "data": f"\n[hax-gui] cleaning up {len(left)} orphaned "
                                       f"worker process(es) left by this job\n"})
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        for _ in range(20):                                   # 5s to go quietly
            await asyncio.sleep(0.25)
            if not [p for p in _group_members(pgid) if p != pgid]:
                return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

    @staticmethod
    def _err_path(log_path: str) -> str:
        """The stderr log file beside the stdout one: run.log -> run.err.log."""
        p = Path(log_path)
        return str(p.with_name(p.stem + ".err" + p.suffix))

    def _init_buffers(self, job_id: str) -> None:
        self._buffers[job_id] = {s: deque() for s in STREAMS}
        self._buflens[job_id] = {s: 0 for s in STREAMS}

    async def cancel(self, job_id: str) -> None:
        """Stop a running job — including one inherited from a previous server process.

        Driven off the persisted pgid rather than the in-memory handle, which is empty
        after a restart (that is why cancelling a reattached job used to be a silent
        no-op). The identity check matters: pids are recycled, so signalling a stored pid
        without confirming it is still *our* process could kill a stranger.
        """
        job = self.get(job_id)
        if job is None or job["status"] not in ACTIVE:
            return
        pgid, key = int(job.get("pgid") or 0), job.get("proc_key")
        if not _group_alive(pgid, key):
            return                                   # already gone; the supervisor will finalise it
        self._exec("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
        self._publish(job_id, {"type": "status", "status": "cancelling"})
        # Signal the whole group, not just the leader: hax spawns its real workers via
        # forkserver, and killing only the leader leaves them running on the GPU.
        # SIGTERM for a clean shutdown, then SIGKILL if it does not go promptly.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        for _ in range(20):                          # 5s grace
            await asyncio.sleep(0.25)
            if not _group_alive(pgid, key):
                return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)

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
            # Only remove a per-run folder (named <program>_<job_id>, or the legacy
            # <program>_<stamp>_<id>), never a project root or arbitrary path.
            if wd.is_dir() and wd.name.startswith(job["program"] + "_"):
                shutil.rmtree(wd, ignore_errors=True)
                usage.forget(str(wd))
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

    def tail_positions(self, job_id: str) -> dict[str, int]:
        """Byte offset each stream's tailer has published up to (may be empty).

        Captured in the same event-loop tick as :meth:`subscribe`, this is the
        exact cut for a replay snapshot: bytes below it are read from the file,
        bytes at/after it arrive as live chunks — nothing duplicated, nothing
        lost, even when the file read itself happens later in a worker thread.
        """
        return dict(self._tail_pos.get(job_id, {}))

    def read_log(self, job_id: str, stream: str = "out", cap: int = 4_000_000,
                 raw_cap: int = 16_000_000, upto: int | None = None) -> str:
        """Persisted log for one stream ("out"/"err"), tqdm frames collapsed.

        The in-memory buffer is only a recent tail, so reopening a job (running or
        finished) replays from the file to keep the earliest lines. We read up to
        ``raw_cap`` bytes from the end (a truncation notice marks anything beyond),
        collapse repeated ``\\r`` progress frames to their final state, then cap the
        result to ``cap`` characters — so early lines survive even when a run has
        written megabytes of progress output.

        ``upto`` stops the read at that byte offset (the tailer's published
        position): a live subscriber gets everything past it as queued chunks, so
        reading further here would replay those bytes twice. This is a blocking
        read of up to ``raw_cap`` bytes — call it from a worker thread, never on
        the event loop.
        """
        job = self.get(job_id)
        if not job:
            return ""
        log_path = job["log_path"] if stream == "out" else self._err_path(job["log_path"])
        truncated = False
        try:
            p = Path(log_path)
            end = p.stat().st_size
            if upto is not None:
                end = min(end, upto)
            start = max(0, end - raw_cap)
            truncated = start > 0
            with open(p, "rb") as f:
                f.seek(start)
                data = f.read(end - start)
        except OSError:
            return "".join(self._buffers.get(job_id, {}).get(stream, ()))
        text = _collapse_progress(data.decode("utf-8", errors="replace"))[-cap:]
        if truncated:
            text = f"[hax-gui] log truncated: showing the last {raw_cap // 1_000_000} MB\n" + text
        return text

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
        """Number of jobs still active — pending, running or being cancelled
        (used to gate updates)."""
        row = self._exec(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ({})".format(
                ",".join("?" * len(ACTIVE))),
            tuple(sorted(ACTIVE)),
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
        """The last ``max_lines`` of a job's log, reading only the end of the file.

        The old ``read().splitlines()`` slurped the whole log — a 500 MB
        progress-bar log was a >1 GB allocation per request, and forty threadpool
        slots of that is an OOM. Seek to a bounded window from the end instead
        (generous 4 KB/line, 8 MB floor/ceiling) and drop the first, possibly
        partial, line when the window didn't reach the start.
        """
        job = self.get(job_id)
        if not job:
            return []
        log_path = job["log_path"] if stream == "out" else self._err_path(job["log_path"])
        window = min(32_000_000, max(8_000_000, (max_lines or 0) * 4096))
        try:
            with open(log_path, "rb") as log:
                size = os.fstat(log.fileno()).st_size
                start = max(0, size - window)
                log.seek(start)
                text = log.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            lines = list(self._buffers.get(job_id, {}).get(stream, ()))
            return lines[-max_lines:] if max_lines else lines
        lines = text.splitlines()
        if start > 0 and lines:
            lines = lines[1:]                       # first line is likely cut mid-way
        return lines[-max_lines:] if max_lines else lines

    # -- helpers -----------------------------------------------------------
    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["args"] = json.loads(data["args"])
        data["command"] = json.loads(data["command"])
        data["bindings"] = json.loads(data["bindings"]) if data.get("bindings") else None
        data["active"] = data["status"] in ACTIVE
        # Its inputs were re-made after it ran, so its outputs no longer correspond to
        # anything that exists (see :meth:`mark_stale`).
        data["stale"] = bool(data.get("stale"))
        # The kinds this run can feed downstream, narrowed by its mode (a train run
        # yields its checkpoint but no latents/particles, a predict run the reverse),
        # so the chaining UI won't offer a train run where a predict run is required.
        data["produces"] = adapters.job_produces(data.get("program", ""), data["args"])
        return data

    def _exec(self, sql: str, params: tuple = (), fetch: str | None = None):
        assert self._db is not None, "JobManager.init() not called"
        cur = self._db.execute(sql, params)
        result = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        self._db.commit()
        return result


manager = JobManager()
