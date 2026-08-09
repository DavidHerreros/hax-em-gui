"""Slurm backend, end to end against the fake scheduler from conftest: submit,
log streaming through the shared files, exit-code finalisation, and scancel.
Everything flows through the real API and the real JobManager — only the four
scheduler binaries are fakes (HAX_GUI_SBATCH etc.).
"""

from __future__ import annotations

import pytest

from conftest import FAKE_SLURM, GATE, wait_for_status


@pytest.fixture()
def slurm_backend(client, run):
    r = run(client.put("/api/settings", json={"execution_backend": "slurm",
                                              "slurm_partition": "gpu", "slurm_gres": "gpu:1"}))
    assert r.status_code == 200, r.text
    yield
    r = run(client.put("/api/settings", json={"execution_backend": "local"}))
    assert r.status_code == 200, r.text


def _create(client, run, project, md_file):
    body = {"program": "estimate_latent_covariances", "args": {"md": str(md_file)}, "launch": True}
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json=body))
    assert r.status_code == 200, r.text
    return r.json()


def test_slurm_submit_run_and_finish(client, run, project, md_file, slurm_backend):
    job = _create(client, run, project, md_file)
    done = wait_for_status(client, run, job["id"], {"succeeded", "failed", "interrupted"})
    assert done["status"] == "succeeded", done
    assert done["slurm_id"], "the Slurm job id must be recorded on the row"
    # The batch script honoured the settings and carried the exit-code contract.
    script = (FAKE_SLURM / f"job_{done['slurm_id']}.sh").read_text()
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "exit.status" in script or "printf" in script
    # Logs streamed through the same files as a local run.
    r = run(client.get(f"/api/jobs/{job['id']}/log"))
    lines = r.json()["lines"]
    assert any("submitted to Slurm as job" in ln for ln in lines)
    assert any("stub:" in ln for ln in lines)


def test_slurm_cancel(client, run, project, md_file, slurm_backend):
    GATE.write_text("")            # the stub blocks until the gate file is removed
    try:
        job = _create(client, run, project, md_file)
        running = wait_for_status(client, run, job["id"], {"running"})
        assert running["slurm_id"]
        r = run(client.post(f"/api/jobs/{job['id']}/cancel"))
        assert r.status_code == 200
        done = wait_for_status(client, run, job["id"], {"cancelled", "failed", "interrupted"})
        assert done["status"] == "cancelled", done
    finally:
        GATE.unlink(missing_ok=True)


def test_local_backend_untouched_after_slurm(client, run, project, md_file):
    """After switching back to local, launches go through the subprocess path
    (no slurm_id) — the fixture teardown really restored the backend."""
    job = _create(client, run, project, md_file)
    done = wait_for_status(client, run, job["id"], {"succeeded", "failed", "interrupted"})
    assert done["status"] == "succeeded"
    assert not done.get("slurm_id")
