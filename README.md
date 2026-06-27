# Hax Web GUI

A modern web GUI to run every `hax-em` program (and, later, chained workflows)
against a GPU server from any browser. See [PLAN.md](PLAN.md) for the full design,
roadmap, and the remote-ChimeraX evaluation.

## Status

| Area | State |
|---|---|
| **argparse → form-schema introspector** | ✅ done & verified (all 13 programs, 0 errors) |
| **Backend: catalog / jobs / live logs / files / GPUs** | ✅ done & verified end-to-end |
| **Frontend: home, catalog, auto-forms, job console** | ✅ built (pure vanilla JS, served by backend) — pending in-browser check |
| **Home page** (logo, docs cards, GUI guide) + **dark/light theme** | ✅ built |
| **display_metrics (TensorBoard) embed** | ✅ done & verified (launched + reverse-proxied end-to-end) |
| **Project manager** (settings root, projects, per-run output folders) | ✅ done & verified (backend) / built (frontend) |
| **Collapsible left menu + right program launcher** | ✅ built |
| **Project tabs + multiple TensorBoard tabs** | ✅ built |
| annotate_space + interactive 3D | ⬜ Phase 2 |
| Workflow builder | ⬜ Phase 3 |
| Remote ChimeraX (WebRTC) | ⬜ Phase 0 spike pending |

## No build tools — by design

The frontend is **pure ES modules + hand-written CSS**: no npm, no Node, no
`node_modules`, no bundler, no CDN. The FastAPI backend serves the static files
directly, so the browser only ever talks to your server. This eliminates the
npm supply-chain install-time risk entirely — the only frontend code is the
small, readable set of files in `frontend/`.

## Layout

```
backend/
  hax_gui/
    introspect.py   # argparse -> JSON form-schema (keystone; no hax source changes)
    registry.py     # loads cached schemas, groups programs by category
    command.py      # form values -> hax_project_manager argv (+ validation)
    jobs.py         # async subprocess manager, SQLite, live log pub/sub
    tensorboard.py  # per-logdir TensorBoard launcher (for display_metrics)
    files.py        # sandboxed server file browser
    gpus.py         # nvidia-smi discovery
    app.py          # FastAPI: REST + WebSocket + TensorBoard proxy + static SPA
  scripts/generate_schemas.py
  schemas_cache/program_schemas.json   # generated; served by the backend
frontend/                  # pure ES modules + CSS, served statically by the backend
  index.html
  styles.css               # hand-written design system (no Tailwind)
  js/
    h.js                   # ~30-line hyperscript/DOM helper
    api.js                 # REST + WebSocket client
    router.js              # hash router (no server fallback needed)
    icons.js               # inline SVG icons
    app.js                 # app shell + routing + docs link + theme toggle
    views/                 # home, catalog, programForm, metrics, jobs, jobDetail, filePicker, ui
```

## Install

Just the backend — it serves the frontend too. In the conda env that has
`hax-em` installed:

```bash
pip install hax-em-gui            # from PyPI (installs deps + the `hax-gui` command)
# or from a clone, for development:
pip install -e .                  # editable install from the repo root
```

The SPA, the WebGL viewer and a seed schema cache are bundled inside the package,
so plain, editable, and `pip install git+<url>` installs all work. After
installing, regenerate the schema cache whenever a hax program's CLI changes
(it imports JAX, so it's not done automatically):

```bash
hax-gui schemas       # or, from a clone: python backend/scripts/generate_schemas.py
```

## Run it

```bash
hax-gui start                 # background, on http://127.0.0.1:8000 by default
hax-gui status                # is it running? where? pid + health
hax-gui restart               # relaunch (keeps the current host/port)
hax-gui stop                  # stop it
```

Pick a different address with `hax-gui start --host 0.0.0.0 --port 8051`, run it
attached to the terminal with `--foreground`, or auto-reload on edits with
`--reload`. Every command reports whether the server is running. (The classic
`uvicorn hax_gui.app:app --host 0.0.0.0 --port 8000` still works too.)

Then open **http://&lt;server&gt;:&lt;port&gt;** in any browser, on any machine. No
Node, no npm, no build step, no second server. Because the page and the API share
an origin, there is no CORS or proxy to configure.

## Configuration (backend env vars)

| Var | Default | Purpose |
|---|---|---|
| `HAX_GUI_DATA_DIR` | `~/.hax_gui` | SQLite DB + per-job log files |
| `HAX_GUI_SCHEMA_CACHE` | `backend/schemas_cache/program_schemas.json` | cached form-schemas |
| `HAX_PROJECT_MANAGER` | `hax_project_manager` | launcher (supports multi-token, e.g. `python -m hax.cli`) |
| `HAX_GUI_FILE_ROOTS` | `$HOME` | colon-separated roots the file browser may expose |
| `HAX_GUI_TOKEN` | _(empty)_ | shared bearer token; empty = open (dev only) |
| `HAX_GUI_CORS_ORIGINS` | `localhost:5173` | dev SPA origins |

## Remote use & security

Single-tenant (one shared token behind TLS — see PLAN.md §10). Put a reverse
proxy (Caddy/nginx) in front to terminate TLS and route `/api`, `/ws`, and the
SPA. The backend runs on the GPU server; the browser can be anywhere.
