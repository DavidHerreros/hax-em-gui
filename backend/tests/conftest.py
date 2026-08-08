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
