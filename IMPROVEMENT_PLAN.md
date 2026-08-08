# Hax GUI — Improvement Plan

Derived from a full four-part audit (backend core, frontend SPA, viewer subsystem,
assistant/integrations) performed 2026-08-08. Every item below cites the file (and
where useful the line) the finding anchors to.

## Branching strategy (read first)

**All work in this plan lands on a dedicated integration branch, not `main`:**

```bash
git checkout -b gui-improvements main
```

- Each phase (or large feature) gets its own short-lived branch off
  `gui-improvements` (e.g. `gui-improvements/handedness-fix`,
  `gui-improvements/job-queue`) and merges back into `gui-improvements` when its
  acceptance checks pass.
- `main` receives **nothing** from this plan until a phase has been exercised
  against real data on the GPU server from the integration branch. This keeps the
  released GUI stable while the new features are tested, and makes any regression
  attributable to a single phase.
- Behaviour-changing defaults are introduced **one at a time**, each in its own
  commit with a note in the commit message describing the old vs new behaviour
  (a prior regression became un-attributable because four defaults changed at
  once — do not repeat that).

---

## Phase 0 — Scientific correctness & security hotfixes

Small diffs, highest stakes. Do these before anything else; several invalidate
current output.

### 0.1 Mirrored 3D maps (handedness) — CRITICAL
- `viewer_app/server.py` `mesh_at` (~line 853–865): marching-cubes vertices are in
  array order `(z, y, x)` but rendered as `(x, y, z)` — every isosurface and
  ray-marched volume is mirror-imaged (α-helices render left-handed). The slice
  view (`slices.js`) is correct, so 2D and 3D views disagree.
- Fix: `verts = verts[:, ::-1]` before normalisation; swap the axis order in
  `grid_box` (`server.py:737-757`); drop the `.zyx` swizzle in `volume.js:68/77`.
  `slices.js` needs no change.
- **Add a chirality regression test**: synthesize a right-handed helix volume,
  mesh it, assert handedness of the output triangles.

### 0.2 `.map` files offered but unreadable
- `mrc_viewer.py:67` and `app.py` (`_MAP_SUFFIXES`) accept `.map`, but
  `ImageHandler` never dispatches it → cryptic "not a 3D volume" error. `.map` is
  the extension every EMDB deposition ships under.
- Fix: route `.map` (and ideally `.map.gz` / `.ccp4`) through the mrcfile path.

### 0.3 XSS via TensorBoard text summaries — CRITICAL
- `frontend/js/views/metricsPanel.js` (~line 1438): `h("div", { class, html })`
  injects run `add_text` content as raw `innerHTML`. A crafted tfevents file (from
  a shared cluster, downloaded dataset…) executes script in the GUI origin, where
  the bearer token sits in `localStorage`. This is the only unsanitised
  `innerHTML` in the app.
- Fix: render through `renderMarkdown()` (already node-safe) or strip to text.
  Add a CSP header server-side as defence in depth.

### 0.4 Authentication defaults
- `config.py:46` + `app.py:154-159`: empty token disables auth entirely, silently.
  The API is effectively RCE (arbitrary `hax_project_manager` argv, `pip install`
  via the updater, `/api/server/restart`). DNS rebinding defeats CORS even on a
  loopback bind, and body-less POSTs (`/rerun`, `/cancel`, `/start`) are
  CORS-simple, hence forgeable cross-site.
- Fix: generate and persist a token on first run; refuse to bind a non-loopback
  address without one; add a `Host`-header allowlist; log the auth state at
  startup.

### 0.5 Assistant grounds on a stale schema
- `hax-em-assistant` `config.py:111-128` reads the packaged seed
  `schemas_cache/program_schemas.json`; the GUI itself prefers the regenerated
  `~/.hax_gui/program_schemas.json` (`registry.py:41-45`). After a hax upgrade the
  assistant confidently recommends outdated flags.
- Fix: resolve the same cache path the registry uses.

---

## Phase 1 — Server responsiveness (event-loop hygiene)

The GUI freezes wholesale at every job transition. All of these are
`async def` endpoints doing heavy synchronous work directly on the loop.

### 1.1 `jobs.create()` blocking work → thread
- `jobs.py:313-497`: inline `shutil.rmtree` of the previous run folder and the SSD
  dataset cache (can be **hundreds of GB — minutes**), plus per-particle latents
  extraction (`adapters.py:480-497`). Fires from launch, rerun, continue, start,
  and every workflow step (`workflows.py:1143`).
- Fix: split into a prepare phase wrapped in `asyncio.to_thread(...)` and a small
  async launch phase.

### 1.2 Log reading
- `app.py:1159-1171` (WS open): two 64 MB decodes + `_collapse_progress` on the
  loop. Move to `run_in_threadpool`; drop `raw_cap` to ~2 MB with a
  "log truncated" marker.
- `jobs.py:1196-1206` `tail()`: reads the whole file then slices. Seek from the
  end instead.

### 1.3 Viewer-launch project scan
- `app.py:1541-1571` `_find_latent_metadata`: `rglob` of the entire project tree +
  parse of every metadata file, on the loop. Thread-hop it; cache by
  (project, mtime); stop at first match instead of `sorted()` materialising all.

### 1.4 SQLite
- Four connections to one DB, no WAL, no `busy_timeout`, shared across threads
  (`jobs.py:231`, `labels.py:34`, `workflows.py:81`, `chats.py:30`).
- Fix: `PRAGMA journal_mode=WAL` + `busy_timeout`; one connection per thread or a
  lock; stop committing on reads (`jobs.py:1228`).

### 1.5 Proxies & payloads
- `app.py:1710/2008/1447`: proxies fully buffer responses (a 512³ map ≈ 500 MB
  staged in the parent). Use streaming (`httpx` `stream=True` →
  `StreamingResponse`); add `GZipMiddleware`.
- `server.py:1342-1352` `/api/mrc`: temp-file write + `read_bytes()`. Use
  `FileResponse` with a cleanup `BackgroundTask`.

### 1.6 Misc loop blockers
- Sequential 5 s cancel grace per dependent (`cascade.py:84-85`) → parallelise.
- `nvidia-smi` fork per `/api/gpus` poll (`gpus.py:12-24`) → cache ~2 s.
- Projects-list N+1 (`app.py:727-731`) → `SELECT COUNT(*) GROUP BY project_id`.

---

## Phase 2 — Frontend robustness (kill the silent-data-loss class)

### 2.1 Shared `poller()` helper — structural fix
One helper with: in-flight guard, retry-on-error (never stop permanently),
signature comparison, `document.hidden` skip. Migrate all nine `setInterval`
sites. This structurally fixes:
- error-then-frozen pages (`jobs.js:161-174`, `projects.js:129-144`,
  `projectManager.js:64-74` — signature not reset in `catch`);
- workflow poll dying forever on one blip (`workflows.js:1535`);
- poll-amplification when a response is slow.

### 2.2 Shared resilient WebSocket helper
`onclose`/`onerror` + exponential-backoff reconnect + guarded `JSON.parse`.
Migrate `subscribeWS`, `assistantChat`, `assistantPrepareMotion`
(`api.js:279-328`). Today a server restart (which the GUI itself triggers on
update) silently freezes log views and permanently locks the assistant composer.

### 2.3 Poll-clobbers-edit bugs
- Workflow canvas: `adoptGraph` swaps the graph while the inspector holds the old
  node → edits silently lost on Save (`workflows.js:1526-1531`). Guard the swap on
  `dirty`, not just the re-render.
- Project rename input wiped by the 5 s poll — inverted `renamingId` condition
  (`projectManager.js:64-74`).
- Job-outputs poll destroys transient UI every 4 s — add a signature guard
  (`jobDetail.js:221, 365`).
- Apply `afterLocalRename` hook to `projects.js` rows (fix exists in `jobs.js:119`
  one file away).

### 2.4 Assistant turn plumbing
- Stop-then-send swallows the next question permanently (`assistant.js:93-100` +
  backend `app.py:469-477` treating any inbound frame as a stop). Tag frames
  (`{type:"stop"}` vs `{type:"user"}`); ignore stale stops.
- Empty reply duplicates the previous answer into the new bubble
  (`assistant.js:83-91` vs `347-357`).
- Transport errors are persisted into history and re-sent as context
  (`assistant.js:61-65`) — render errors outside the transcript.
- Serialise chat persistence (`assistant.js:106-116` — duplicate-chat race).
- Backend: session construction outside `try` can hang the socket forever
  (`app.py:436-454`); guard, and always enqueue the done sentinel.
- Streaming markdown is O(n²) per frame (`assistant.js:339-346`) — append-only
  rendering of the stable prefix.

### 2.5 Leaks & churn
- Launcher form subscription leak on fast program switching (`launcher.js:24-30,
  219`).
- `UpdateBanner` listener per Home visit (`updates.js:111`); `ResizeObserver`
  never disconnected (`metricsPanel.js:718`); frame/latent caches unbounded
  (`metricsPanel.js:796, 1156`).
- Workflow node drag re-renders the whole graph per pointermove
  (`workflows.js:976-985`) — move only the affected wires, throttle to rAF.
- Terminal snapshot parses char-by-char in one synchronous burst
  (`terminal.js:82-107`) — chunk across frames, cap snapshot size.
- Minimap rAF repaints at 60 fps for the whole run (`workflows.js:1362-1373`) —
  repaint on change only.

---

## Phase 3 — Process lifecycle & state integrity

### 3.1 Subprocess reaping and cross-restart cleanup
- TensorBoard/viewer children: `terminate()` with no `wait()` → zombies; not
  persisted → true orphans holding ports and **GPU memory** after a server crash
  (`tensorboard.py:94-103`, `viewer.py:132-148`). Persist `{pid, starttime, port}`
  to the data dir; reap with wait+kill escalation on stop and on next startup.
- `_reap_group` must re-verify process identity (pass `key`) before `killpg`
  (`jobs.py:1011-1033`) — currently the one code path that signals a group is the
  one that skips the PID-reuse check. Same trust-on-aliveness flaw in
  `ollama.py:79-83` (verify the process is actually Ollama before killing).

### 3.2 Viewer manager races
- `ensure()` is check-then-spawn with no lock in all three managers
  (`viewer.py:72-111`, `mrc_viewer.py`, `md_viewer.py`) — double-click spawns a
  duplicate JAX/CUDA subprocess and corrupts the reuse map. Per-key
  `asyncio.Lock`.
- `_free_port()` TOCTOU (`viewer.py:30-33`): retry on child bind failure instead
  of waiting out the 40 s health timeout.
- Add an idle timeout so abandoned viewer sessions release GPU memory.

### 3.3 State integrity
- **200-job truncation** (`jobs.py:1189-1194` `limit=200`) silently breaks cascade
  invalidation and workflow sync past 200 runs per project. Remove the limit for
  graph-deriving callers (paginate only the UI list).
- Cascade applied before validation: a 422 rerun has already SIGKILLed and staled
  dependents (`app.py:1044-1065`). Validate first.
- Delete cleanup: remove `job_labels` rows on job delete (`app.py:1125-1136`);
  remove workflows + chats on project delete (`app.py:794-802`).
- Atomic writes (temp-file + rename) + a lock for `settings.json`
  (`settings_store.py:35-47`), `hax_project.json`, `server.json`; recover, don't
  swallow, corrupt JSON (a corrupt settings file currently makes every project
  vanish silently).
- Corrupt schema cache should rebuild, not 500 forever (`registry.py:40-53`);
  lock `_load` so two threads can't co-write the cache file.
- Fix latent `store_false` inversion in form generation (`introspect.py:136-138`
  + `command.py:78-81`).

---

## Phase 4 — Architecture & testing

- **Split `app.py`** (2,323 lines) into the empty `routes/` package: jobs,
  projects, workflows, metrics, viewers, assistant, updates. Extract the ~300
  lines of artifact-resolution logic and the four copy-pasted proxies into
  modules.
- **Extract shared viewer frontend modules** (`wire.js` for the binary parsers,
  `viewerbase.js` for `makeLive`/token guards/snapshot): `app.js` and `mrc.js`
  have already diverged — `mrc.js` lacks the request-token guard (stale responses
  win), the boot try/catch (WebGL2 failure = infinite "Loading…"), and the frame
  cache budget (unbounded morph cache → tab OOM).
- Add a version/magic byte to the six hand-packed binary wire formats so a
  mismatch fails cleanly instead of parsing garbage geometry.
- Split `viewer_app/server.py` (1,434 lines): `meshing.py` (so `mrc_server` stops
  `sys.path`-hacking into a JAX-adjacent module), `landscape.py`, `exporter.py`.
- Replace deprecated `@app.on_event` with lifespan; make startup non-blocking
  (currently a sync handler doing index build + up-to-20 s Ollama poll before the
  server accepts connections — also fixes the assistant's dead-embeddings-for-
  process-lifetime bug by starting Ollama **before** building the index).
- **Tests** (currently zero): job lifecycle (create/rerun/cancel/reconcile after
  restart), cascade invalidation, workflow topo-scheduling, schema round-trip,
  chirality (0.1), and an HTTP smoke test per route module. Run them in CI on the
  integration branch.

---

## Phase 5 — Competent-GUI features (table stakes)

### 5.1 Global job queue + GPU admission control
- Today jobs start unconditionally (`jobs.py:484-496`); two launches on GPU 0
  both start and OOM. `max_per_device` exists only inside workflows
  (`workflows.py:1014-1031`).
- Server-wide scheduler: per-device slots, a `queued` state, optional free-VRAM
  check via the already-existing `gpus.list_gpus()` before dispatch.

### 5.2 Notifications
- On job finish/failure: browser `Notification` + `document.title`/favicon badge
  as the baseline; optional webhook/email in settings. Currently a user must keep
  a tab open and poll by eye.

### 5.3 Server-push job status
- One SSE or WS channel for job/workflow state transitions replaces the 3–5 s
  polling fan-out (and shrinks the DB write contention from Phase 1.4).

### 5.4 Provenance
- Stamp every job with hax version/commit + environment; record what
  `strip_obsolete` (`command.py:29-48`) dropped instead of dropping silently.
- Project export/import (tar a project with metadata) for moving between
  machines.

### 5.5 UI polish & accessibility
- One `modalShell()` (role="dialog", aria-modal, focus trap/restore) for all ten
  modals; `:focus-visible` styles; `prefers-reduced-motion`; `aria-live` for
  status/errors.
- Unsaved-changes guard: `beforeunload` + navigation guard on the workflow
  canvas `dirty` flag and half-filled launcher forms.
- Keyboard: Cmd/Ctrl-K palette or `g`-prefix navigation, Escape closes panels,
  Enter launches, undo (Ctrl-Z) on the workflow canvas.
- Jobs list: virtualization past ~100 rows; bulk select (multi-delete /
  multi-label); filters reflected in the URL; drop the per-row `animationDelay`
  stagger above ~30 rows.
- Boot resilience: loading/error state instead of a blank page (remove top-level
  `await` in `app.js:38-41`); a proper token entry view (not `window.prompt`);
  offline/backend-gone indicator.
- Theme: respect `prefers-color-scheme` for the initial default; add
  `<meta name="color-scheme">`.

---

## Phase 6 — Unique differentiators

The ingredients already exist in the repos; they mostly need wiring.

### 6.1 Assistant ⇄ job logs ("Diagnose this failure")
- `hax-em-assistant` already ships a failure-diagnosis rule table (GPU OOM, shape
  mismatch, NaN divergence, missing metadata labels, checkpoint errors) with fix
  hints (`context/logs.py`, `skills/diagnose_logs.py`) — **completely unwired**.
- Add a "Diagnose" button on failed jobs in `jobDetail.js` feeding the tail of
  `/api/jobs/{id}/log` into the diagnosis skill. Highest value-to-effort item in
  the entire plan.

### 6.2 Data-driven parameter recommendations
- `inspect_path` already measures box size, pixel size, particle count, CTF
  presence (`assistant_actions.py:41-100`); `list_gpus` knows free VRAM — none of
  it reaches the model. Inject these facts into the prompt so the assistant can
  recommend batch size / downsampling / latent dim from the actual data.

### 6.3 Cross-run reasoning
- Allow attaching multiple runs (or the project job list + dependency graph) to a
  chat so "did raising the latent dim help?" is answerable. Add a live-run watch
  mode ("is this converging, should I kill it?") off the existing metrics stream.

### 6.4 Viewer: from pretty to quantitative
- **Voxel size + origin** (currently discarded, `mrc_server.py:54-65`): unlocks Å
  scale bar, physically-correct multi-map alignment, and correct sampling rate on
  exported `.mrc` (today exports open in ChimeraX at 1 Å/px).
- **Measurement tools**: distance/angle/voxel-value probe — the camera-space
  depth readback (`gl.js:507`) already provides the machinery.
- **Map arithmetic**: subtract/difference maps (the standard way to look at
  conformational change).
- Clipping planes / slab mode; per-map transfer-function editor; surface mesh
  export (STL/OBJ — the mesh is already computed server-side); saved views /
  session restore.
- Longer term: PDB/mmCIF overlay (the main remaining reason users leave for
  ChimeraX); FSC / local-resolution colouring.
- Viewer performance backlog: mmap volumes instead of triple-copy eager load
  (`mrc_server.py:54-65, 289`); box-mean downsample instead of aliasing stride
  (`server.py:728-734`); cache the per-map histogram/stats
  (`mrc_server.py:154-168`); upload morph textures once per leg, not per frame
  (`mrc.js:443-445`); WebGL context-loss handling; shared secret on viewer child
  ports (unauthenticated on loopback today).

### 6.5 Remote execution (strategic, largest effort)
- A Slurm submission backend (jobs already survive server restarts and identify
  by pid+starttime, so the abstraction point is `JobManager` launch/supervise).
  This is the ceiling between "lab tool" and "facility tool" — schedule it after
  Phases 0–3 have made the single-host story solid.

---

## Acceptance checks per phase

- **P0**: chirality test passes; `.map` opens; crafted tfevents renders inert;
  server refuses `0.0.0.0` without a token.
- **P1**: rerun of a job with a 100 GB SSD cache keeps `/api/health` < 100 ms;
  opening a 300 MB log freezes nothing; no `database is locked` under a 10-tab
  poll storm.
- **P2**: kill/restart the server with Jobs + a chat open → both recover without
  reload; editing a workflow node while a run finishes preserves the edit.
- **P3**: `hax-gui stop` leaves zero orphaned TB/viewer processes (check
  `nvidia-smi` + `ps`); a 250-run project cascades staleness correctly.
- **P5**: two simultaneous launches on one GPU → second queues; closing the tab
  and finishing a job produces a desktop notification.
- **P6**: a failed OOM run shows an actionable diagnosis; exported `.mrc` opens
  in ChimeraX at the correct Å/px.
