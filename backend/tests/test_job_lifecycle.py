"""Job lifecycle through the real API: create/launch, logs, cancel, rerun,
and reconcile-after-restart. Jobs run the stub CLI from conftest, so the whole
path — run folder, wrapped command, exit file, supervision, finalize — is real,
only the program itself is fake.
"""

from __future__ import annotations

import json

from conftest import GATE, wait_for_status


def _create(client, run, project, md_file, extra=None, launch=True):
    body = {"program": "estimate_latent_covariances",
            "args": {"md": str(md_file), **(extra or {})}, "launch": launch}
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json=body))
    assert r.status_code == 200, r.text
    return r.json()


def test_create_launch_and_finish(client, run, project, md_file):
    job = _create(client, run, project, md_file)
    done = wait_for_status(client, run, job["id"], {"succeeded", "failed", "interrupted"})
    assert done["status"] == "succeeded"
    assert done["workdir"] and done["workdir"].endswith(f"estimate_latent_covariances_{job['id']}")
    r = run(client.get(f"/api/jobs/{job['id']}/log"))
    assert r.status_code == 200
    assert any("stub:" in line for line in r.json()["lines"])


def test_missing_required_arg_is_422(client, run, project):
    r = run(client.post(f"/api/projects/{project['id']}/jobs",
                        json={"program": "estimate_latent_covariances", "args": {}, "launch": False}))
    assert r.status_code == 422
    assert "md" in r.json()["detail"]["errors"]


def test_cancel_running_job(client, run, project, md_file):
    GATE.write_text("")                       # the stub now blocks until this is removed
    try:
        job = _create(client, run, project, md_file)
        running = wait_for_status(client, run, job["id"], {"running"})
        assert running["status"] == "running"
        r = run(client.post(f"/api/jobs/{job['id']}/cancel"))
        assert r.status_code == 200
        done = wait_for_status(client, run, job["id"], {"cancelled", "failed", "interrupted"})
        assert done["status"] == "cancelled"
    finally:
        GATE.unlink(missing_ok=True)


def test_rerun_reuses_folder(client, run, project, md_file):
    job = _create(client, run, project, md_file)
    first = wait_for_status(client, run, job["id"], {"succeeded"})
    r = run(client.post(f"/api/jobs/{job['id']}/rerun", json={}))
    assert r.status_code == 200, r.text
    second = wait_for_status(client, run, job["id"], {"succeeded"})
    assert second["workdir"] == first["workdir"]   # stable folder is the contract
    assert second["id"] == first["id"]


def test_saved_then_started(client, run, project, md_file):
    job = _create(client, run, project, md_file, launch=False)
    assert job["status"] == "saved"
    r = run(client.post(f"/api/jobs/{job['id']}/start"))
    assert r.status_code == 200, r.text
    done = wait_for_status(client, run, job["id"], {"succeeded"})
    assert done["status"] == "succeeded"


def test_reconcile_after_restart(client, run, project, md_file):
    """A fresh manager over the same DB must finalize what happened while 'down':
    a job left `running` whose process is gone and whose exit file says 0 is
    reconciled to `succeeded`, not stuck running forever."""
    from hax_gui import jobs as jobs_mod

    job = _create(client, run, project, md_file)
    done = wait_for_status(client, run, job["id"], {"succeeded"})

    mgr = jobs_mod.manager
    # Rewind the finished job to how a crash would have left it: status=running,
    # a dead pgid, and the exit file (written by _wrap) still on disk.
    mgr._exec(
        "UPDATE jobs SET status='running', return_code=NULL, finished_at=NULL, "
        "pgid=999999, proc_key='0' WHERE id=?", (job["id"],))

    fresh = jobs_mod.JobManager()
    fresh.init()          # runs _reconcile()
    row = fresh.get(job["id"])
    assert row["status"] == "succeeded", row["status"]
    assert row["return_code"] == 0
    # the shared row is visible through the API too
    r = run(client.get(f"/api/jobs/{job['id']}"))
    assert r.json()["status"] == "succeeded"


def test_delete_job_cleans_up(client, run, project, md_file):
    job = _create(client, run, project, md_file)
    wait_for_status(client, run, job["id"], {"succeeded"})
    # attach a label so the delete-cleanup path has something to remove
    r = run(client.post("/api/labels", json={"name": "tmp", "color": "#123456"}))
    lid = r.json()["id"]
    run(client.put(f"/api/jobs/{job['id']}/labels", json={"label_ids": [lid]}))
    r = run(client.delete(f"/api/jobs/{job['id']}"))
    assert r.status_code == 200
    assert run(client.get(f"/api/jobs/{job['id']}")).status_code == 404
    from hax_gui.labels import manager as label_manager
    assert label_manager.labels_for(job["id"]) == []
    run(client.delete(f"/api/labels/{lid}"))
