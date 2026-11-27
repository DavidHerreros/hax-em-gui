"""Jobs (local and sbatch) never inherit HAX_ASSISTANT_API_KEY from the GUI server's environment."""

from __future__ import annotations

from conftest import TMP, wait_for_status

from hax_gui import slurm
from hax_gui.config import settings

SECRET = "sk-test-0123456789abcdef"


def test_jobs_and_sbatch_do_not_inherit_the_key(client, run, project, md_file, monkeypatch):
    monkeypatch.setenv("HAX_ASSISTANT_API_KEY", SECRET)
    stub = TMP / "env_stub.sh"
    stub.write_text("#!/bin/bash\nenv\n")
    stub.chmod(0o755)
    monkeypatch.setattr(settings, "hax_cli", str(stub))
    r = run(client.post(f"/api/projects/{project['id']}/jobs",
                        json={"program": "filter_latents", "args": {"latents": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    wait_for_status(client, run, r.json()["id"], {"succeeded", "failed", "interrupted"})
    log = run(client.get(f"/api/jobs/{r.json()['id']}/log/full", params={"stream": "out"})).text
    assert "FORCE_COLOR=1" in log and SECRET not in log
    rc, out, _ = run(slurm._run(["env"]))
    assert rc == 0 and "PATH=" in out and SECRET not in out
