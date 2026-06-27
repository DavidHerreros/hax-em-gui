# Hax Web GUI — Development Plan

A web-based GUI to run every `hax-em` program and chain them into workflows, usable
from a remote browser against a GPU server. This document defines scope, architecture,
the technical evaluation of remote ChimeraX rendering, and a phased roadmap.

---

## 1. Goals & requirements (recap)

1. **Modern, good-looking, intuitive** app-like UI with animations (menus, progress lines, transitions).
2. Link out to the existing **documentation** site (https://davidherreros.github.io/hax-em-docs/).
3. **Server/client split**: heavy compute runs on the GPU server; the GUI is used from any browser on another machine.
4. **All programs usable from the GUI**, including the two non-batch ones:
   - `display_metrics` (TensorBoard) and `annotate_space` (interactive PyQt5 + napari + ChimeraX).
5. **Fast, responsive 3D** rendering/interaction in the browser.
6. **Remote ChimeraX**: ChimeraX installed and running *on the server*, rendered *in the client browser*, as close to local performance as possible. (Feasibility evaluated in §4.)

---

## 2. Current-state findings (what the GUI must wrap)

| Program | Module | Shape | GUI implication |
|---|---|---|---|
| `hetsiren`, `zernike3deep`, `reconsiren`, `flexconsensus`, `latent_space_deconvolution`, `image_gray_scale_adjustment`, `volume_gray_scale_adjustment` | `hax/networks/` | argparse `main()`, 12–20 args, `--mode {train,predict,send_to_pickle}` | **Auto-generated form** + job runner + live log/metric streaming |
| `estimate_latent_covariances`, `decode_states_from_latents`, `filter_latents`, `modart` | `hax/programs/` | argparse `main()` | Auto-generated form + job runner |
| `display_metrics` | `hax/metrics/writer.py` | launches **TensorBoard** | TensorBoard is already a web server → **proxy/embed** |
| `annotate_space` | `hax/viewers/annotate_space/` | **PyQt5 + napari** desktop app; spawns **ChimeraX** via `Popen(chimerax "script.cxc")`; clustering/UMAP/KDTree/PCA on QThreads; maps exchanged over a raw TCP socket (`viewer_socket/`) | **Re-architect for web** (the only real porting effort). See §5. |

**Critical observations**
- Every program is launched the same way (`hax_project_manager <program> --args`) and exposes a parser whose `_actions` can be introspected → a single mechanism drives *all* program forms and stays in sync with the CLI.
- `--gpu` must be set **before any JAX import** (sets `CUDA_VISIBLE_DEVICES`). The job runner must pass GPU selection as the dispatcher already expects.
- The numerics (clustering, dimred, map decoding) in `annotate_space` are pure Python/JAX and **server-appropriate**; only the *rendering and interaction layer* (napari/Qt/ChimeraX) is desktop-bound and needs a web equivalent.

---

## 3. Target architecture

```
┌──────────────────────────── CLIENT (browser, any machine) ─────────────────────────────┐
│  React + TS SPA                                                                          │
│  • Program catalog & auto-generated forms      • Workflow (pipeline) builder            │
│  • Job dashboard (live logs, progress lines)   • Latent-space viewer (WebGL)            │
│  • Metrics (TensorBoard embed)                 • ChimeraX video pane (WebRTC)           │
└───────────────▲───────────────▲───────────────▲───────────────▲────────────────────────┘
        REST/JSON │   WebSocket   │     WebRTC      │   reverse-proxied HTTP
                  │  (logs/state) │ (video+input)   │  (TensorBoard)
┌─────────────────┴───────────────┴─────────────────┴─────────────────── SERVER (GPU) ────┐
│  FastAPI backend (same conda env as hax-em)                                              │
│  ├─ Program registry  ← introspects argparse from MODULES_DICT                           │
│  ├─ Job manager       → launches `hax_project_manager <prog>` subprocesses, tracks       │
│  │                       PID/GPU/status, streams stdout/stderr, parses progress          │
│  ├─ Workflow engine   → DAG of jobs, output→input wiring, dependency scheduling          │
│  ├─ File/Project API  → browse server paths, validate inputs, manage output dirs         │
│  ├─ Annotate service  → clustering/dimred/map-decode endpoints (reuse hax functions)     │
│  ├─ TensorBoard proxy → spawn + reverse-proxy display_metrics                            │
│  └─ ChimeraX session  → headless ChimeraX (EGL/Xvfb) + NVENC → WebRTC pixel stream       │
│  SQLite (job/workflow metadata) · Caddy/nginx (TLS, auth, static, reverse proxy)         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Recommended stack
- **Backend**: Python **FastAPI** + Uvicorn (lives in the hax conda env, so it can `import hax` directly for introspection and reuse map/cluster code). WebSockets for live logs/progress. SQLite for metadata (no external DB needed for single-server scientific use).
- **Job execution**: lightweight async subprocess manager reusing the existing CLI dispatcher. Pluggable backend so it can later target **Slurm/cluster** without changing the API.
- **Frontend**: **React + TypeScript + Vite**, **Tailwind + shadcn/ui** (or Mantine) for a modern component system, **Framer Motion** for animations (menu transitions, animated progress lines), **TanStack Query** (server state) + **Zustand** (UI state).
- **3D / latent space**: **deck.gl** or **regl/three.js** for the latent scatter (hundreds of thousands of points, GPU-accelerated, lasso selection); a WebGL volume/isosurface viewer (**Mol\***, **NGL**, or **niivue**) for maps rendered client-side.
- **Reverse proxy / security**: **Caddy** (automatic TLS) terminating one origin, routing `/api`, `/ws`, `/tensorboard`, `/webrtc`, and the static SPA. Token or OAuth auth at this layer.

### Why FastAPI in the hax env (not a separate service)
The backend can `import` each program's argparse parser to emit form schemas, and directly call the same clustering/dimred/decode functions `annotate_space` uses — no duplication, always in sync with the package.

---

## 4. Evaluation: remote ChimeraX (server-side render → browser) — **feasible**

**Short answer: yes, this is achievable with near-local responsiveness on a LAN, using GPU pixel-streaming over WebRTC. ChimeraX does not ship this, so we integrate proven building blocks rather than invent them.**

ChimeraX's own remote feature (`remotecontrol rest`) is **command/control only** — it does not stream rendered pixels. To render the server GPU's output in a browser there are two viable strategies:

### Strategy A — GPU pixel streaming (recommended for true ChimeraX fidelity)
Run ChimeraX **headless** on the server (offscreen **EGL**, or a virtual `Xvfb`/`Xorg` display bound to the NVIDIA GPU), capture its framebuffer, **hardware-encode with NVENC** (H.264/HEVC/AV1), and ship frames to the browser over **WebRTC**; the browser forwards mouse/keyboard/touch events back over a WebRTC data channel to drive ChimeraX.

- This is exactly how cloud-gaming and remote-workstation products work (NICE DCV, KasmVNC, Parsec). The open-source path is **Selkies-GStreamer** (WebRTC + NVENC HTML5 remote desktop) or a custom GStreamer `nvh264enc → webrtcbin` pipeline.
- **Latency on a LAN**: typically **~15–50 ms** glass-to-glass with NVENC — perceptually close to local. Over WAN it degrades gracefully (WebRTC adapts bitrate/resolution).
- **Pros**: pixel-perfect ChimeraX (its exact lighting, volume rendering, morph animations); zero client GPU/install beyond a browser; reuses the *existing* `chimerax_connection` morph-script logic almost unchanged.
- **Cons**: consumes a server GPU encode session per active viewer; needs headless GL setup; one ChimeraX session per user (fine for this audience).

### Strategy B — Geometry/isosurface streaming (recommended for the common case)
Decode the map on the server, extract the **isosurface mesh** (or stream the volume), send geometry to the browser, and render natively with **Mol\*/three.js**. Interaction (rotate/zoom) is then **100% local in the browser** — zero round-trip latency.

- **Pros**: maximum responsiveness for rotate/zoom; cheap on the server; scales to many users.
- **Cons**: not ChimeraX's renderer; ChimeraX-specific morph animation must be reproduced client-side (feasible: stream the per-frame morphed meshes).

### Recommendation: **hybrid**
- Use **Strategy B** for the everyday latent-space → map exploration (the bulk of `annotate_space` interaction) so rotate/zoom feels instant.
- Use **Strategy A** (WebRTC pixel stream of headless ChimeraX) for the **ChimeraX-specific workflows** (volume morph, salesman/random-walk trajectories) where the user explicitly wants ChimeraX's renderer. The existing `FlexMorphChimeraX` script generation is reused; we only swap the local `Popen` window for a headless session whose framebuffer is streamed.

This gives "fast and responsive 3D" for the common path **and** real ChimeraX where it matters. A **spike in Phase 0** will measure end-to-end latency on the actual server GPU to confirm the LAN target before committing.

---

## 5. Re-architecting `annotate_space` for the web

The desktop app splits cleanly into a **compute layer** (keep, move server-side) and a **view layer** (rebuild for web):

| Desktop piece | Web replacement |
|---|---|
| napari 2D/3D scatter of latent space | deck.gl/regl scatter with lasso/box selection, animated transitions |
| QThreads: clustering, UMAP, PCA, KDTree | FastAPI endpoints calling the *same* functions; results streamed over WebSocket |
| Map generation over raw TCP socket (`viewer_socket/`) | Replace socket with HTTP/WebSocket endpoints; decode map server-side, return mesh (Strategy B) |
| ChimeraX `Popen` window | Headless ChimeraX + WebRTC pixel pane (Strategy A) |
| magicgui control widgets | React control panel (sliders, combo boxes) with Framer Motion |

The numerics and the ChimeraX command scripts are reusable; the effort is the new web view + the two streaming services.

---

## 6. Auto-generated program forms

A single introspection layer keeps the GUI in lockstep with the CLI:
1. For each entry in `MODULES_DICT`, import the module and build its argparse parser.
2. Walk `parser._actions` → emit a **JSON schema** per program (name, type, `required`, `choices`, `default`, help text, `store_true` flags → toggles, `type=float/int` → numeric inputs, file-path args → file pickers).
3. Frontend renders the schema into a validated, sectioned form with inline help and a server-side file browser for path arguments.
4. Path-typed args get a **server file/folder picker** (the File/Project API), since paths are on the server.

This means **new or changed programs need zero GUI code** — the form regenerates from the parser.

---

## 7. Workflows (program combinations)

- A **visual pipeline builder** (node graph): drag programs onto a canvas, wire one program's output (e.g. a checkpoint dir, a `.star`/latent file) into another's input argument.
- The **workflow engine** schedules the DAG, passing resolved output paths as inputs, respecting dependencies and GPU availability.
- Ship a set of **curated workflow templates** mirroring the documented pipelines (e.g. train → decode states → annotate; consensus across latent spaces).
- Phase this *after* single-program execution is solid.

---

## 8. Phased roadmap

### Phase 0 — Foundations & de-risking spikes (1–2 wks)
- Scaffold repo: FastAPI backend (in hax env) + React/Vite/Tailwind frontend + Caddy dev proxy.
- **Spike 1 — argparse → JSON schema** introspection for all programs.
- **Spike 2 — headless ChimeraX + NVENC → WebRTC** latency test on the real server GPU. *Go/no-go and A-vs-B confirmation for §4.*
- Basic auth + TLS reverse proxy; "Hello, GPU server" smoke test from a remote browser.

### Phase 1 — Run any program (3–4 wks)
- Program catalog UI with the modern animated shell (sidebar, transitions, doc link).
- Auto-generated forms + server file picker.
- Job manager: launch subprocess with `--gpu`, track status, **stream logs + animated progress lines** over WebSocket, cancel/kill.
- Job dashboard & history (SQLite).
- **display_metrics**: spawn TensorBoard + reverse-proxy/embed.

### Phase 2 — Interactive 3D & annotate_space (4–6 wks)
- Latent-space viewer (deck.gl scatter, lasso selection, clustering/dimred via API).
- Client-side map rendering (Strategy B): server decodes → mesh → WebGL viewer.
- ChimeraX WebRTC pane (Strategy A) for morph/trajectory workflows.
- Port the `annotate_space` control panel and annotation/save flows to web.

### Phase 3 — Workflows & polish (3–4 wks)
- Visual pipeline builder + workflow engine + curated templates.
- Multi-GPU scheduling, queueing; optional Slurm backend.
- Theming (light/dark), accessibility, onboarding tour, full animation pass.
- Packaging: one-command server launch (`docker compose` or a `hax_gui` console entry point), docs.

### Cross-cutting
- Security (auth, path-traversal sandboxing of the file API, per-user job isolation).
- Tests (backend job manager, schema introspection, WebRTC signaling); CI.

---

## 9. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Remote ChimeraX latency worse than hoped | Phase 0 spike measures it first; hybrid falls back to client-side mesh (Strategy B) for the common case |
| Headless GL on the server (drivers/EGL/Xvfb) | Validate in Phase 0; document exact NVIDIA/EGL setup; containerize |
| `annotate_space` feature parity is large | Port incrementally; keep desktop app available during transition |
| GPU contention (training jobs vs ChimeraX encode vs interactive decode) | GPU-aware scheduler; reserve/queue; surface GPU usage in the dashboard |
| Long-running training jobs surviving disconnects | Jobs run as detached server processes; UI reattaches to live logs on reconnect |
| Argparse edge cases (custom `type=`, list args) | Schema layer handles common cases; per-program overrides for the few specials (e.g. `list_of_floats`) |

---

## 10. Locked decisions

These are settled and the plan above assumes them:

1. **Users — single user / small lab.** Auth is a single shared token behind TLS (Caddy). No per-user accounts, no per-user isolation. *Simplification:* the file/project API and job manager are single-tenant; we still sandbox file-API paths against traversal, but skip per-user output partitioning.
2. **Scheduler — plain subprocesses now.** Jobs launch `hax_project_manager <prog> --gpu N` directly on the one server with a GPU-aware queue. The job-manager interface is kept **pluggable** so a Slurm/cluster backend can be added later without touching the API or UI.
3. **Frontend — pure vanilla JavaScript (ES modules) + hand-written CSS.** No npm, no Node, no `node_modules`, no bundler, no CDN; the FastAPI backend serves the static files directly (same origin → no CORS/proxy). Chosen to eliminate the npm supply-chain install-time attack surface entirely — the only frontend code is the small, auditable set of files under `frontend/`. *(Supersedes the earlier React + Tailwind choice.)*
4. **ChimeraX — hybrid.** Client-side mesh (Strategy B) for the common rotate/zoom path; WebRTC pixel stream of headless ChimeraX (Strategy A) for morph/trajectory workflows. Phase 0 spike confirms the LAN latency target before committing to A.

### Immediate next step
Phase 0: scaffold the FastAPI + React/Vite/Caddy repo and run the two de-risking spikes (argparse→JSON schema; headless ChimeraX + NVENC→WebRTC latency on the real server GPU). Say the word and I'll start the scaffold.
