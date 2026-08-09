"""One request per route module: the split app must answer everywhere.

Guards the routes/ package against a module that fails to import or a router
that falls out of app.py's include list — either would 404 a whole area.
"""

from __future__ import annotations

CASES = [
    # system
    ("GET", "/api/health", 200),
    ("GET", "/api/gpus", 200),
    ("GET", "/api/programs", 200),
    ("GET", "/api/updates", 200),
    ("GET", "/api/settings", 200),
    ("GET", "/api/branches", 200),
    ("GET", "/api/labels", 200),
    ("GET", "/api/files", 200),
    # assistant (+ chats: plain data, present without the assistant)
    ("GET", "/api/assistant/status", 200),
    ("GET", "/api/ollama/status", 200),
    ("GET", "/api/chats", 200),
    # projects
    ("GET", "/api/projects", 200),
    ("GET", "/api/projects/nope", 404),
    # jobs
    ("GET", "/api/jobs", 200),
    ("GET", "/api/jobs/nope", 404),
    # provenance versions (reproduce-run drift warning)
    ("GET", "/api/system/versions", 200),
    # metrics
    ("GET", "/api/tensorboard", 200),
    ("GET", "/api/metrics/jobs", 200),
    # viewers
    ("GET", "/api/viewer", 200),
    ("GET", "/api/mrcviewer", 200),
    ("GET", "/api/mdviewer", 200),
    ("GET", "/viewer/nope/", 404),
    ("GET", "/mrcviewer/nope/", 404),
    ("GET", "/mdviewer/nope/", 404),
    # workflows
    ("GET", "/api/workflows", 200),
    ("GET", "/api/workflows/catalog", 200),
    # SPA
    ("GET", "/", 200),
]


def test_every_route_module_answers(client, run):
    for method, path, want in CASES:
        r = run(client.request(method, path))
        assert r.status_code == want, f"{method} {path}: {r.status_code} != {want}: {r.text[:200]}"


def test_unknown_host_rejected(client, run):
    r = run(client.get("/api/health", headers={"host": "evil.example.com"}))
    assert r.status_code == 421   # DNS-rebinding guard (loopback bind)
