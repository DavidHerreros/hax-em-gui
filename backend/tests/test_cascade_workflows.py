"""Cascade invalidation over the binding graph, and workflow graph logic
(validation, cycle detection, downstream closure)."""

from __future__ import annotations

import json

from conftest import wait_for_status


def _create(client, run, project, md_file):
    r = run(client.post(f"/api/projects/{project['id']}/jobs", json={
        "program": "estimate_latent_covariances", "args": {"md": str(md_file)}, "launch": True}))
    assert r.status_code == 200, r.text
    return r.json()


def test_cascade_marks_dependents_stale(client, run, project, md_file):
    from hax_gui import cascade
    from hax_gui.jobs import manager

    a = _create(client, run, project, md_file)
    b = _create(client, run, project, md_file)
    c = _create(client, run, project, md_file)
    for j in (a, b, c):
        wait_for_status(client, run, j["id"], {"succeeded"})
    # Wire B to consume A and C to consume B (bindings are what the graph is
    # derived from; written directly so no adapter materialisation is needed).
    manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                  (json.dumps({"md": {"job_id": a["id"], "kind": "particles"}}), b["id"]))
    manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                  (json.dumps({"md": {"job_id": b["id"], "kind": "particles"}}), c["id"]))

    deps = cascade.dependents_of([a["id"]])
    ids = [d["id"] for d in deps]
    assert ids == [b["id"], c["id"]]          # transitive, oldest first

    run(cascade.invalidate(deps))
    for j in (b, c):
        r = run(client.get(f"/api/jobs/{j['id']}"))
        assert r.json()["stale"] is True
    r = run(client.get(f"/api/jobs/{a['id']}"))
    assert r.json()["stale"] is False          # the source itself is not stale

    # rerunning a stale job clears the flag again (bindings dropped first: the
    # fabricated kind has no real artifact behind it for produce() to materialise)
    manager._exec("UPDATE jobs SET bindings=NULL WHERE id=?", (b["id"],))
    r = run(client.post(f"/api/jobs/{b['id']}/rerun", json={}))
    assert r.status_code == 200, r.text
    assert wait_for_status(client, run, b["id"], {"succeeded"})["stale"] is False


def test_rerun_refused_while_dependent_runs(client, run, project, md_file):
    from conftest import GATE
    from hax_gui.jobs import manager

    a = _create(client, run, project, md_file)
    wait_for_status(client, run, a["id"], {"succeeded"})
    GATE.write_text("")
    try:
        b = _create(client, run, project, md_file)
        wait_for_status(client, run, b["id"], {"running"})
        manager._exec("UPDATE jobs SET bindings=? WHERE id=?",
                      (json.dumps({"md": {"job_id": a["id"], "kind": "particles"}}), b["id"]))
        # A rerun of A must refuse (409) while B is reading A's folder …
        r = run(client.post(f"/api/jobs/{a['id']}/rerun", json={}))
        assert r.status_code == 409
        assert any(d["id"] == b["id"] for d in r.json()["detail"]["running"])
        # … and go through once the client re-submits with cancel_running.
        r = run(client.post(f"/api/jobs/{a['id']}/rerun", json={"cancel_running": True}))
        assert r.status_code == 200, r.text
        wait_for_status(client, run, b["id"], {"cancelled"})
        GATE.unlink(missing_ok=True)   # A's relaunched stub blocks on the gate too
        wait_for_status(client, run, a["id"], {"succeeded"})
    finally:
        GATE.unlink(missing_ok=True)


# ---- workflows -----------------------------------------------------------
def test_workflow_validate_and_topo(client, run, project):
    from hax_gui.workflows import manager as wf

    assert wf.validate({"nodes": [], "edges": []})            # empty graph is an error

    nodes = [{"id": "n1", "program": "estimate_latent_covariances"},
             {"id": "n2", "program": "estimate_latent_covariances"},
             {"id": "n3", "program": "no_such_program"}]
    errors = wf.validate({"nodes": nodes, "edges": []})
    assert any("no_such_program" in e for e in errors)

    # a cycle must be reported, whatever else is wrong with the edges
    cyc = {"nodes": nodes[:2],
           "edges": [{"from": "n1", "to": "n2", "field": "md", "kind": "particles"},
                     {"from": "n2", "to": "n1", "field": "md", "kind": "particles"}]}
    assert wf.validate(cyc)

    saved = wf.save(project["id"], "t", {
        "nodes": [{"id": "a", "program": "x"}, {"id": "b", "program": "x"}, {"id": "c", "program": "x"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
    })
    assert wf.downstream_nodes(saved["id"], ["a"]) == ["b", "c"]   # transitive closure
    wf.delete(saved["id"])


def test_workflow_crud_roundtrip(client, run, project):
    graph = {"nodes": [{"id": "a", "program": "estimate_latent_covariances"}], "edges": []}
    r = run(client.post("/api/workflows", json={"project_id": project["id"], "name": "wf1", "graph": graph}))
    assert r.status_code == 200, r.text
    wid = r.json()["id"]
    r = run(client.get(f"/api/workflows/{wid}"))
    assert r.json()["graph"]["nodes"][0]["id"] == "a"
    r = run(client.delete(f"/api/workflows/{wid}"))
    assert r.status_code == 200
    assert run(client.get(f"/api/workflows/{wid}")).status_code == 404
