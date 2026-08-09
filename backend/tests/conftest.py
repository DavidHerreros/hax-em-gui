"""Shared fixtures: an isolated data dir, a stub hax CLI, and the app under ASGI.

Environment is pinned *before* hax_gui is imported (its Settings singleton reads
the environment at import time): every test run gets its own data dir and a stub
`hax_project_manager`, so no test can touch a real ~/.hax_gui or launch a real
JAX program. Jobs launched by tests run the stub, which exits immediately —
unless HAXSTUB_GATE names an existing file, in which case it waits for the file
to disappear (how the cancel test holds a job in `running`).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="hax_gui_test_"))
GATE = TMP / "gate"

os.environ["HAX_GUI_DATA_DIR"] = str(TMP / "data")
os.environ["HAX_GUI_MANAGE_OLLAMA"] = "0"
os.environ["HAX_GUI_ASSISTANT_WARMUP"] = "0"
os.environ["HAXSTUB_GATE"] = str(GATE)

_stub = TMP / "hax_stub.sh"
_stub.write_text(
    "#!/bin/bash\n"
    'echo "stub: $@"\n'
    'while [[ -n "$HAXSTUB_GATE" && -e "$HAXSTUB_GATE" ]]; do sleep 0.2; done\n'
    "exit 0\n"
)
_stub.chmod(0o755)
os.environ["HAX_PROJECT_MANAGER"] = str(_stub)

# ── fake Slurm scheduler ─────────────────────────────────────────────────────
# Three tiny scripts emulating the sbatch/squeue/scancel contract hax_gui.slurm
# relies on: sbatch reads the batch script on stdin, honours its #SBATCH
# --output/--error lines, runs it in its own session and prints a job id;
# squeue -j reports RUNNING while that session leader is alive; scancel kills
# the group (and leaves a marker sacct reports as CANCELLED). The binary names
# are injected via HAX_GUI_SBATCH etc., read by hax_gui.slurm at import.
FAKE_SLURM = TMP / "fakeslurm"
FAKE_SLURM.mkdir(exist_ok=True)
os.environ["FAKE_SLURM_DIR"] = str(FAKE_SLURM)

_sbatch = TMP / "fake_sbatch.sh"
_sbatch.write_text(
    "#!/bin/bash\n"
    'd="$FAKE_SLURM_DIR"\n'
    'n=$(( $(ls "$d" 2>/dev/null | grep -c "\\.sh$") + 1 ))\n'
    's="$d/job_$n.sh"\n'
    "cat > \"$s\"\n"
    "out=$(sed -n 's/^#SBATCH --output=//p' \"$s\" | head -1)\n"
    "err=$(sed -n 's/^#SBATCH --error=//p' \"$s\" | head -1)\n"
    'setsid bash "$s" >> "${out:-/dev/null}" 2>> "${err:-/dev/null}" &\n'
    'echo $! > "$d/job_$n.pid"\n'
    'echo "$n"\n'
)
_squeue = TMP / "fake_squeue.sh"
_squeue.write_text(
    "#!/bin/bash\n"
    'while [ $# -gt 0 ]; do case "$1" in -j) id=$2; shift 2;; *) shift;; esac; done\n'
    'pid=$(cat "$FAKE_SLURM_DIR/job_$id.pid" 2>/dev/null)\n'
    '[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo RUNNING || exit 1\n'
)
_scancel = TMP / "fake_scancel.sh"
_scancel.write_text(
    "#!/bin/bash\n"
    'id="$1"\n'
    'touch "$FAKE_SLURM_DIR/job_$id.cancelled"\n'
    'pid=$(cat "$FAKE_SLURM_DIR/job_$id.pid" 2>/dev/null)\n'
    '[ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null\n'
    "exit 0\n"
)
_sacct = TMP / "fake_sacct.sh"
_sacct.write_text(
    "#!/bin/bash\n"
    'while [ $# -gt 0 ]; do case "$1" in -j) id=$2; shift 2;; *) shift;; esac; done\n'
    '[ -e "$FAKE_SLURM_DIR/job_$id.cancelled" ] && echo CANCELLED || echo COMPLETED\n'
)
for f, var in ((_sbatch, "HAX_GUI_SBATCH"), (_squeue, "HAX_GUI_SQUEUE"),
               (_scancel, "HAX_GUI_SCANCEL"), (_sacct, "HAX_GUI_SACCT")):
    f.chmod(0o755)
    os.environ[var] = str(f)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

import httpx  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def run(loop):
    return loop.run_until_complete


@pytest.fixture(scope="session")
def client(run):
    from hax_gui.app import app

    cm = app.router.lifespan_context(app)
    run(cm.__aenter__())
    transport = httpx.ASGITransport(app=app)
    c = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")
    yield c
    run(c.aclose())
    run(cm.__aexit__(None, None, None))


@pytest.fixture(scope="session")
def project(client, run):
    """A projects_root + one project, configured through the real API."""
    root = TMP / "projects"
    root.mkdir(exist_ok=True)
    r = run(client.put("/api/settings", json={"projects_root": str(root)}))
    assert r.status_code == 200, r.text
    r = run(client.post("/api/projects", json={"name": "testproj"}))
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture()
def md_file():
    """A (fake) metadata file: jobs launch from its folder, so it must exist."""
    p = TMP / "particles.xmd"
    p.write_text("data_\n")
    return p


def wait_for_status(client, run, job_id: str, statuses: set[str], timeout: float = 30.0) -> dict:
    """Poll a job through the API until it reaches one of `statuses`."""
    async def _wait():
        for _ in range(int(timeout / 0.2)):
            r = await client.get(f"/api/jobs/{job_id}")
            assert r.status_code == 200, r.text
            job = r.json()
            if job["status"] in statuses:
                return job
            await asyncio.sleep(0.2)
        raise AssertionError(f"job {job_id} never reached {statuses}: {job['status']}")
    return run(_wait())
