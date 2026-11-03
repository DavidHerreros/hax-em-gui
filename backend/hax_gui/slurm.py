"""Slurm submission backend for :mod:`hax_gui.jobs`.

The local backend already made jobs *external* to the server: each job writes
its own stdout/stderr files (the server merely tails them) and records its exit
code to a file beside the log (see ``jobs._wrap``). Slurm slots straight into
that contract — ``sbatch`` points ``-o``/``-e`` at the very same log paths and
the batch script ends with the same exit-code write — so log streaming, the
WebSocket replay, exit-code recovery and server restarts all work unchanged.
What replaces the process-group machinery is: submit with ``sbatch``, poll
``squeue`` for the state, cancel with ``scancel``.

Everything here is best-effort against the cluster's CLI: no Slurm REST API,
no persistent daemon connection — the same binaries a user would type. The
binary names are overridable via ``HAX_GUI_SBATCH`` etc. (which is also how the
test suite substitutes a fake scheduler).
"""

from __future__ import annotations

import asyncio
import os
import shlex

from . import settings_store

SBATCH = os.environ.get("HAX_GUI_SBATCH", "sbatch")
SQUEUE = os.environ.get("HAX_GUI_SQUEUE", "squeue")
SACCT = os.environ.get("HAX_GUI_SACCT", "sacct")
SCANCEL = os.environ.get("HAX_GUI_SCANCEL", "scancel")

# squeue states that mean "not running yet, still waiting for resources".
PENDING_STATES = {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD"}


class SlurmError(Exception):
    pass


def enabled() -> bool:
    return settings_store.load().get("execution_backend") == "slurm"


def build_script(command: list[str], exit_path: str, out_path: str, err_path: str,
                 name: str, cwd: str | None = None) -> str:
    """The sbatch script for one hax run.

    ``--open-mode=append`` because the caller prepares (truncates or keeps) the
    log files itself — exactly as the local backend does — so an in-place
    continue keeps its stdout history here too. The trailing exit-code write is
    ``jobs._wrap``'s contract, which is what lets a restarted server finalise a
    job that finished while it was down.
    """
    s = settings_store.load()
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={_sanitize(name)}",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
        "#SBATCH --open-mode=append",
    ]
    if cwd:
        lines.append(f"#SBATCH --chdir={cwd}")
    for key, flag in (("slurm_partition", "--partition"), ("slurm_gres", "--gres"),
                      ("slurm_time", "--time"), ("slurm_account", "--account")):
        val = str(s.get(key) or "").strip()
        if val:
            lines.append(f"#SBATCH {flag}={val}")
    # Free-form extras, one sbatch option per line (e.g. "--mem=64G").
    for extra in str(s.get("slurm_extra") or "").splitlines():
        extra = extra.strip()
        if extra:
            lines.append(f"#SBATCH {extra}")
    lines.append("")
    # Environment prelude: whatever the compute nodes need before hax exists —
    # `module load cuda`, `source activate hax`, … Verbatim shell lines.
    prelude = [ln for ln in str(s.get("slurm_prelude") or "").splitlines() if ln.strip()]
    if prelude:
        lines += prelude + [""]
    lines += [
        "export FORCE_COLOR=1",   # the GUI's terminal renders ANSI; hax colorizes on request
        f'{shlex.join(command)}; rc=$?; printf %s "$rc" > {shlex.quote(exit_path)}; exit $rc',
        "",
    ]
    return "\n".join(lines)


async def _run(argv: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except OSError as exc:  # binary missing: the one error every call site must surface well
        raise SlurmError(f"{argv[0]}: {exc}") from exc
    out, err = await proc.communicate(stdin.encode() if stdin is not None else None)
    return proc.returncode or 0, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


async def submit(script: str) -> str:
    """Submit a batch script (via stdin, no temp file) and return the Slurm job id."""
    rc, out, err = await _run([SBATCH, "--parsable"], stdin=script)
    if rc != 0 or not out:
        raise SlurmError(err or out or f"sbatch exited {rc}")
    # --parsable prints "jobid[;cluster]"
    return out.split(";", 1)[0].strip()


async def state(slurm_id: str) -> str | None:
    """The job's current squeue state (e.g. PENDING/RUNNING), or None once it has
    left the queue (finished, failed, cancelled — squeue forgets completed jobs)."""
    rc, out, _err = await _run([SQUEUE, "-h", "-j", str(slurm_id), "-o", "%T"])
    # A finished job yields empty output; an unknown id makes squeue exit non-zero.
    if rc != 0 or not out:
        return None
    return out.splitlines()[0].strip().upper() or None


async def final_state(slurm_id: str) -> str | None:
    """Best-effort terminal state from accounting (CANCELLED/TIMEOUT/…), for jobs
    that left no exit file. None when sacct is unavailable or has no record."""
    try:
        rc, out, _err = await _run(
            [SACCT, "-n", "-P", "-X", "-j", str(slurm_id), "-o", "State"])
    except SlurmError:
        return None
    if rc != 0 or not out:
        return None
    return out.splitlines()[0].strip().upper().split(" ")[0] or None


async def cancel(slurm_id: str) -> None:
    try:
        await _run([SCANCEL, str(slurm_id)])
    except SlurmError:
        pass   # scancel gone missing: the poll loop will still see whatever happens


def _sanitize(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)
    return (keep or "hax")[:64]
