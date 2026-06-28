
---

## Implementation status (added by Claude)

Backend verified end-to-end; frontend built but pending an in-browser check (no
browser in the dev environment).

### Big-ticket roadmap status

- **annotate_space + interactive 3D** — ✅ DONE via the browser-native WebGL
  viewer (`viewer_app/`): isosurface + SSAO, point cloud, volume ray-marching,
  morph/export, launchable from annotate_space and predict runs. *Keep in mind:*
  the native PyQt5/napari path still exists for users who want it.
- **Remote ChimeraX (WebRTC)** — ✅ Effectively superseded: the WebGL viewer
  renders the conformational landscape and decoded maps client-side (ChimeraX-
  like shading), so the planned headless-ChimeraX + NVENC→WebRTC pixel stream is
  no longer needed for the common path. *Keep in mind:* revisit only if true
  ChimeraX-renderer fidelity (its exact volume/render features) is ever required.
- **Workflow builder** — ✅ MVP done (PLAN §7): a **node-graph canvas** (`views/
  workflows.js`) where program nodes are wired output→input, backed by a DAG
  engine (`workflows.py`) that topologically schedules each node as a real job,
  passing an upstream node's output into the downstream input via the existing
  binding system; blocks downstream steps on failure. Backend + engine verified
  (validation, topo/cycle detection, binding wiring, failure propagation). *Keep
  in mind / next:* curated templates, parallel branch execution, canvas pan/zoom,
  live per-node log peek. The canvas UI itself needs an in-browser pass (no
  browser in the dev env).

#### Workflow builder — backlog (requested) — ✅ all implemented

- [x] **Layout modes** — a Horizontal/Vertical toggle in the topbar
  (`graph.orientation`, persisted). Port geometry is orientation-aware (inputs on
  the left edge + output on the right for horizontal; inputs along the top edge +
  output on the bottom for vertical) and wires bend along the active axis.
- [x] **File-field picker** — node-inspector parameters whose schema `widget` is
  `path` now render a 📂 browse button (`openFilePicker`) beside the text input,
  same server file browser as the launcher form.
- [x] **Inherit params on predict↔train wiring** — wiring a moded program's model
  output into another node of the *same* program (`*_model` kind, e.g. a HetSIREN
  train → HetSIREN predict) copies the source's parameters into the target
  (`inheritParams`), filling only unset/unwired fields, and defaults
  source→`mode=train`, target→`mode=predict`.
- [x] **Custom box names → job names** — a node's "name" field becomes the
  launched job's name.
- [x] **Open a project as a workflow** — "From project" imports the open project's
  jobs as an editable graph: nodes = jobs, edges inferred from each job's stored
  input bindings, seeded with every job's real status. Backend
  `WorkflowManager.build_from_project` + `POST /api/workflows/from-project/{pid}`;
  the canvas auto-tidies the imported graph.
- [x] **Run single nodes / a selection** — a ▶ button on each node runs just that
  step; shift/ctrl-click multi-selects and "Run selected" runs the subset; the
  backend `run(wid, only=…)` keeps untouched nodes' state so an already-succeeded
  upstream feeds the steps being run. Per-node status (saved / running /
  succeeded / failed / blocked) colors each node + its outgoing wires, with a
  legend under the palette.
- [x] **Drag-and-drop add + drop-to-connect** — palette programs are draggable
  onto the canvas (in addition to click-to-add); dropping a program on top of an
  existing node auto-connects the first compatible output→input port pair
  (`autoConnect`).
- [x] **Auto-layout button** — "Tidy" arranges the graph into longest-path
  dependency tiers along the active axis (`autoLayout`); also runs automatically
  on orientation change and on project import.

- [x] **1** — Left menu collapses to icons-only via a header toggle.
- [x] **2** — Project manager: Settings page sets a common projects root (gear icon
  at the bottom of the left menu); projects are folders under it; each run gets a
  unique `<program>_<timestamp>_<id>` subfolder with all outputs + `run.log`; the
  output-dir argument is auto-injected, so users never set an output path.
  CryoSPARC-style project view lists the project's executions.
- [x] **3** — Jobs list and each project's executions show the owning project.
- [x] **4** — Multiple projects open as tabs (Projects section).
- [x] **5** — Multiple TensorBoards as tabs (Metrics section); iframes persist
  across tab switches.
- [x] **6** — Programs moved into a collapsible **right** menu; forms render and
  launch from there into the active project.
- [x] **7** — SSD scratch folder in Settings: when set, each run that supports it
  gets `<ssd_root>/<project>/<program>`, auto-injected into `ssd_scratch_folder`.
  The SSD path is hidden from forms (like the output path). Leave empty to disable.
- [x] **8** — Relative image refs in metadata: each program is launched with its
  working directory set to the folder of its `--md` file (fallback: the run
  folder), so relative `.star`/`.xmd` image paths resolve. The cwd is shown in the
  job view. (Covers metadata-relative paths; project-root-relative layouts would
  need an explicit data-root override — not yet added.)
- [x] **9** — Rerun jobs (failed or passed): a Rerun button on the job view
  re-launches the same program with the same user args into the same project, in a
  fresh run folder (output/SSD dirs regenerated). `POST /api/jobs/{id}/rerun`.
- [x] **10** — Delete jobs: trash buttons on the jobs list, project executions, and
  the job view remove the job record and its run folder. `DELETE /api/jobs/{id}`
  (409 if still running — cancel first).
- [x] **11** — Terminal-style logs: backend streams raw chunks (so carriage-return
  progress bars update live) and stores raw bytes; a hand-written vanilla terminal
  renderer (`terminal.js`) interprets ANSI colors, `\r`/`\b`/tab, and erase-line.
- [x] **12** — Fixed a 500 (`ProcessLookupError`) when cancelling a job that had
  already exited.
- [x] **14** — First log line lost on reopen: the live replay buffer is a 256 KB
  tail, so long logs dropped early lines. Finished jobs now replay from the full
  log file instead.
- [x] **15** — Ctrl+C hang ("waiting for background tasks"): finished-job log
  WebSockets now close after replay instead of idling, and running-job sockets
  break on a server shutdown event.
- [x] **16** — Hax logo as the browser tab favicon.
- [x] **17** — Metrics auto-lists runs that have TensorBoard events (`/api/metrics/jobs`);
  click a run to open its metrics. Manual log-dir entry still available.
- [x] **18** — Form device selector: default is now the first GPU (GPU 0); a "CPU"
  option runs with `CUDA_VISIBLE_DEVICES=""` (passes `--gpu ""`). Job lists show
  "CPU" / "GPU N".
- [x] **19** — Polling no longer redraws the Jobs/Project lists every 3 s (dirty-check
  on id+status), fixing the periodic flicker/jump.
- [x] **13** — "Edit & rerun" on a finished job: opens the right-menu form preloaded
  with that job's inputs (GPU included; auto-managed output/SSD dirs dropped so they
  regenerate) targeting its project, so you can tweak parameters and launch a new
  run — also serves as "duplicate this execution". Instant "Rerun" stays for
  unchanged re-execution.
- [x] **21** — Custom run names: the launcher form has an optional **Run name**
  field, and any run can be renamed afterwards (⋮ menu on the Jobs list and a
  project's executions, or the pencil on the job view). Names show as the run's
  title with the program name kept as a subtitle. Backend: a `name` column on jobs
  and `PUT /api/jobs/{id}`; rerun carries the name over.
- [x] **22** — Job-output → next-program input chaining: fields that consume a
  previous run's output are marked **bindable** and offer "Use a job's output"
  (picker of compatible runs in the project) **or drag-and-drop** of an execution
  row onto the field. The first wired artifact is **latents**: a HetSIREN /
  Zernike3Deep / reconSIREN *predict* run (latents stored in the `latent_space`
  metadata column) or a `filter_latents` run (`filtered_latents.npy`) can feed
  `annotate_space` (`--z_space` / `--z_space_reduced`), `filter_latents`
  (`--latents`) and `decode_states_from_latents` (`--latents_file`). At launch the
  backend (`adapters.py`) materialises the `.npy` into the new run's folder — for
  metadata sources it parses the comma-separated `latent_space` column into an
  `(N, latent_dim)` array; for ready arrays it copies the file. `GET
  /api/projects/{id}/sources?kind=latents` lists compatible runs; the program schema
  exposes `bindable` (field→kind) and `producers` (kind→programs). Adding more
  artifact kinds/consumers is a table edit in `adapters.py`.
- [x] **23** — `--reload` from a job: every program's `--reload` is now bindable to a
  previous run's saved network, picked from the project (or dragged in) instead of
  typing a checkpoint path. Each network reloads its own type (hetsiren/HetSIREN,
  zernike3deep/Zernike3Deep, reconsiren/ReconSIREN, flexconsensus/FlexConsensus,
  latent_space_deconvolution/deconvolver, image/volume_gray_scale_adjustment), while
  `decode_states_from_latents` reloads any decoder network (hetsiren/zernike3deep/
  reconsiren). The adapter ("dir" style) references the source run's checkpoint folder
  in place (no copy). The picker only ever shows runs you can actually reload: a source
  qualifies only when it has a **complete, loadable checkpoint** — exactly what
  `NeuralNetworkCheckpointer.load` reads (`<dir>/checkpoints/config` + `.../state`).
  That precisely excludes predict-only runs (no checkpoint), still-training runs (final
  checkpoint not committed yet), and crashed runs that created the folder but never
  finished writing it.
- [x] **26** — Jobs tab grouped by project + filters: the Jobs page now groups runs
  under their owning project (alphabetical, "(no project)" last, with a per-group count)
  and has a filter bar with **Project / Program / Status** dropdowns (options derived
  from the live data, selection preserved across polls) plus a Clear button. Filtering
  is client-side over the polled job list; the per-row project chip was dropped since
  the group header shows it.
- [x] **28** — WebGL conformational-landscape viewer (browser-native alternative to
  the native PyQt5/napari annotate_space). Lives entirely in `hax-em-gui/viewer/`
  (does **not** touch hax-em; imports hax only as a library, like
  decode_states_from_latents). Standalone backend (`viewer/server.py`, hax env): loads
  network + latents, PCA-embeds the latent space (NumPy SVD, no UMAP dep), and on
  selection decodes a volume → `skimage` marching cubes → a binary triangle mesh
  (decoded volumes cached so the iso slider doesn't re-run the network). Frontend
  (`viewer/static/`, pure vanilla + **hand-written WebGL2**, zero deps): 2D latent
  scatter (Canvas2D) to pick a state + an orbit-camera isosurface renderer with an iso
  slider. Integrated into the GUI like TensorBoard: `backend/hax_gui/viewer.py` spawns a
  per-session subprocess, the app reverse-proxies it at `/viewer/<id>/`, and a new "3D
  viewer" tab embeds it in an iframe. Launchable from any run carrying latents + a
  network (`z_space`/`latents`/`latents_file` + `pickled_nn`/`reload`) via the job
  view's "3D viewer" button or the viewer tab's run list — also a fallback when the
  native viewer can't open (e.g. the WSL Qt/xcb issue). v1 = mesh isosurface; roadmap
  (in `viewer/PLAN.md`): WebGL2 volume ray-marching, 3D scatter, latent interpolation,
  .mrc export.
- [x] **29** — viewer fixes + landscape features: (a) fixed "No surface at any iso" —
  JAX arrays are read-only and skimage marching cubes needs a writable buffer, so the
  decoded volume is now copied (`np.array`); (b) contour is now in **sigma units**
  (`mean + σ·std`, default 2.5) instead of a linear fraction of [min,max], which was far
  too high for sparse cryo-EM maps; (c) the scatter is now a landscape explorer:
  **PCA axis selection** (choose which PCs for X/Y, with explained-variance labels),
  **UMAP** embedding (lazy), **k-means clustering** coloring with a k slider, and a
  **density heatmap** background. New endpoints: `/api/landscape`, `/api/embedding`,
  `/api/cluster`; `/api/decode` takes `sigma`. Verified end-to-end on a real HetSIREN
  checkpoint (500 states, dim 8).
- [x] **30** — viewer: ChimeraX-replacement features + morph/export. Backend
  (`viewer/server.py`, all verified on a real HetSIREN run): `/api/decode` now supports
  **latent interpolation** (`index2`,`t`) and **mass-constant contouring** (`mass`,
  solved by bisection so a morph holds enclosed mass constant); `/api/histogram` (volume
  value histogram); `/api/mrc` (export the decoded map via ImageHandler). Frontend:
  **ChimeraX-style histogram contour** widget (draggable threshold, `histogram.js`);
  **mass-constant morph** between two states (shift-click a target) with ping-pong
  playback and **video recording** (`MediaRecorder` → .webm); **lighting presets**
  (standard/soft/sharp); **background color**; **surface color**; **Save PNG** for both
  the landscape and volume windows with **DPI scale + transparent** options
  (`snapshot()`); **.mrc download**. Remaining native-GUI parity (3D landscape, lasso →
  layers, true volume ray-marching/MIP, series-over-layer, ray-traced lighting) is
  specced in `viewer/PLAN.md` ("Matching the native annotate_space GUI").
- [x] **31** — viewer: **3D landscape view** (`viewer/static/cloud.js`, new). A
  hand-written WebGL2 point-cloud renderer plots the latent landscape over any 3 PCs
  with an orbit camera, CPU-side click picking (click selects, shift-click sets the
  morph target), and selection/target highlight rings. Coloring mirrors the 2D
  scatter: by **cluster** (palette), by **local density** (O(N²) kNN count → ramp), or
  none. The landscape toolbar gains a **dims (2D/3D)** toggle and a **Z-axis** selector;
  the 2D and 3D views share one selection (`syncSelection`) and both feed the same
  decode/morph pipeline. UMAP is 2D-only, so switching to 3D forces PCA. **No backend
  change** — `/api/landscape` already returns up to 8 PCs. Save-PNG snapshots whichever
  landscape view is active (DPI scale + transparent honoured). Verified end-to-end on a
  real HetSIREN run (500 states, pca_k=8; static + API serve 200).
- [x] **32** — viewer refinements (UMAP-3D, multi-state morph, gray default, density σ):
  (a) **UMAP in 3D** — `Landscape.umap()` now computes 3 components (clamped, PC
  fallback); the 3D toggle no longer forces PCA, so UMAP works in both 2D (first 2 cols)
  and 3D (3 cols). Verified `/api/embedding?kind=umap` → 500×3. (b) **Multi-state morph**
  — the single morph target is replaced by an ordered **path**: shift-click appends any
  number of waypoints (pink rings in both 2D scatter and 3D cloud), `runMorph` walks
  every consecutive `[current,…path]` segment at constant mass and ping-pongs for a
  clean loop; a **Clear** button resets it; "frames/seg" controls per-segment sampling.
  Verified a 0→250 segment frame holds mass 16100. (c) **ChimeraX-gray default** — the
  surface default is now `#c8c8c8` (gl.js + the color input). (d) **Density σ slider** —
  density renders in 2D (Canvas2D heatmap, blur passes = σ) and 3D (per-point kNN ramp,
  radius = bbox-diag fraction), updates under PCA *and* UMAP, with a σ slider shown when
  color = density that rewidens it live. New file: none (extends `cloud.js`,
  `scatter.js`, `app.js`, `server.py`).
- [x] **33** — viewer: **layout redesign + lasso/layer system**. (a) **Layout** — the
  single-row pane toolbars were wrapping and overlapping the titles/canvas (buttons
  clipped, per the user screenshot). Rebuilt the shell in `index.html`/`style.css`:
  stacked `.pane-head` (title line + a `flex-wrap` `.toolbar` that grows its own height),
  a `.pane-body` with a collapsible **Layers rail** + a `position:relative`
  `.canvas-wrap` whose canvases are `absolute; inset:0` (2D/3D/lasso stack cleanly), a
  wrapping topbar with an ellipsised status, and labelled `surface`/`bg` swatches.
  (b) **Lasso** — an app-owned transparent overlay (`#lassoCanvas`, armed by the **▱
  Lasso** button) captures a drag-loop and selects enclosed points via a ray-cast
  polygon test against the active view's `projectedPoints()` (new on both `scatter.js`
  and `cloud.js`, so it works in 2D *and* 3D); selected points get a white ring
  (`setSelection`). (c) **Layers** — a left rail turns a selection into a named,
  recolorable, show/hide-able layer; **color = layers** paints points by layer (2D
  `setPointColors`, 3D `computeColors`); each layer's **▶** plays a constant-mass
  **volume series** (its states ordered along the current axis → the multi-state morph
  path). No backend change. Verified all static assets + API serve 200 on a real run;
  JS balance + DOM-id checks clean. Remaining in `viewer/PLAN.md`: volume ray-marcher,
  series polish, menu bar.
- [x] **34** — viewer: **volume ray-marcher + Stop + 3D-color fix**. (a) **Ray-marcher**
  (`viewer/static/volume.js`, new) — a WebGL2 3D-texture renderer: backend `/api/grid`
  streams the decoded map stride-downsampled to a uint8 payload (header: nx,ny,nz +
  vmin/vmax/mean/std; verified 128³→64³, body size matches, interp frames work). The
  shader ray-marches a [-1,1] cube on a full-screen quad (ray-box intersection via a new
  `mat4.invert`) with **MIP** and **emission/absorption** modes, **γ** + **brightness**
  transfer function, and a threshold driven live by the contour histogram. A **render**
  select (Surface / Volume·MIP / Volume·glow) toggles a second canvas `#glvol`; surface/
  bg color + Reset are shared; morphs force Surface. (b) **Stop** — a ■ Stop button (and
  `morphAbort` flag checked each frame) interrupts a running morph / volume series.
  (c) **3D color fix** — `cloud.js` now persists per-point colors and seeds gray on
  `setPoints`, so switching 2D→3D never shows uninitialised black points before
  `applyColor` runs. All static + API serve 200; JS balance + DOM-id checks clean.
- [x] **35** — viewer: **scrubbable morph/series timeline + play/pause**. The morph is now
  a timeline of `[a,b,t]` frames (forward + back so it loops cleanly), built cheaply
  (no network) whenever the path changes. **▶ Play / ⏸ Pause** toggles continuous
  playback; **■ Stop** halts and rewinds to frame 0; a **scrub slider** jumps to any
  frame (debounced), with a `framelbl` showing `i/N`. `renderFrame(i)` decodes the frame
  in the *current* render mode — surface (`/api/decode`, mass-constant) or volume
  (`/api/grid`, interpolated) — so the scrubber/play works for both the mesh morph and
  the volume series; **Record** captures a full pass on the active canvas. Editing the
  path or selecting a state pauses playback; a layer's ▶ builds the path and auto-plays.
  Replaces the old one-shot `runMorph`. JS balance + DOM-id checks clean; static + API
  serve 200.
- [x] **36** — viewer: **3D density overlay**. The 3D landscape's "density" color mode now
  renders an additive **glow cloud** (the analogue of the 2D heatmap) instead of tinting
  points. `cloud.js` gains a second shader program (`VERT_SPLAT`/`FRAG_SPLAT`): large
  soft gaussian point-sprites on a 4-stop heat ramp, drawn additively (`SRC_ALPHA, ONE`)
  behind the points with depth-writes off, sized/brightened by each point's local
  density (`setDensityField` + `showDensity`). Points themselves stay neutral gray under
  the glow (mirrors 2D's gray dots over the heatmap). The σ slider recomputes the field
  live (`density3` fed to `setDensityField`). Frontend-only; JS balance + DOM-id checks
  clean.
- [x] **37** — viewer: **glow slider + ChimeraX trackball + resizable/wider panels**.
  (a) **Glow intensity** — the 3D density overlay was blown out to white; added a `glow`
  slider (`densint`, default 0.35, was effectively 1.0) and shrank the splat size (60→48),
  so dense regions read without clipping. (b) **Trackball rotation** — new shared
  `orbit.js`: the model carries an accumulated rotation matrix (`mat4.axisAngle`/
  `identity`) and the camera sits fixed on +Z looking at the origin, so rotation is
  continuous in every axis (no gimbal lock / pole flip) and centered — dragging from any
  point feels right, like ChimeraX. Wired into `cloud.js` (bakes R into the view; picking
  + projection use it), `gl.js` (adds a `uModel`, rotates normals so the headlight stays
  fixed while the mesh turns), and `volume.js` (ray-marches in model space via
  `invert(proj·view·R)` + eye = `R⁻¹·eye`). All three `resetView` → `orbit.reset`. (c)
  **Layout** — the GUI's `.container` max-width (1100px) is lifted for `.viewer-view`
  (full-width); inside the viewer the `.split` is now flex with a draggable **divider**
  between the two panes and a **rail-resize** handle on the layers panel (`setupResizers`,
  dispatching a resize event so the canvases re-fit). JS balance + DOM-id + HTML-tag
  checks clean; all static (incl. new `orbit.js`) + API serve 200.
- [x] **38** — viewer: **glow tune + 2D-density fix + lasso add/remove + KMeans layers**.
  (a) 3D glow was still too bright: tighter gaussian, lower per-splat alpha, smaller size
  (34), default intensity 0.22, finer slider. (b) Fixed density not showing in 2D after a
  3D→2D switch (the dims toggle never re-applied color; now calls `applyColor`).
  (c) **Lasso modes** — Shift = add to selection, Alt/Ctrl = remove, plain = replace; the
  lasso is tinted teal/green/red by mode and the set is unioned/differenced accordingly.
  (d) **KMeans clustering** — a `Cluster` button + k input in the layers rail calls a new
  backend `/api/kmeans?k` (KMeans on the latents → labels + `cluster_centers_` registered
  as **decodable virtual indices** `n..n+k-1`, with their PCA coords). It creates one
  **layer per cluster** (colored from the palette) plus a **Centroids** layer. Centroids
  decode/histogram/grid/morph like real states (`Decoder.extra` + `_latent`, index checks
  use `decoder.total`). Decoded centroid maps are **auto-colored by their cluster**, and a
  centroid **series/morph smoothly lerps the surface/volume color** between clusters
  (`indexColor` map + `lerpColor` in `decode`/`loadVolume`/`renderFrame`). Backend verified
  end-to-end (kmeans, centroid decode/histogram/grid, reindex on recompute, 404 OOB); JS
  balance + DOM-id + server AST checks clean.
- [x] **39** — viewer: **export layers + playback speed + continuous contour/scrub**.
  (a) **Export** — the viewer is now spawned with `--metadata` + `--project-dir`; the GUI
  (`viewer.py`, `app.py`) auto-resolves the row-aligned particle metadata by scanning the
  project for a `_latent_space` file whose first latent matches the loaded latents
  (`_find_latent_metadata`). A new `Exporter` + `POST /api/save_layer` writes
  `<project>/<layer_name>/`: a subset `.xmd`/`.star`/`.cs` (poses/shifts/CTF/latent, via
  `md.table.iloc[real_indices]` — the `XmippMetaData(rows=…)` ctor drops labels, so we
  slice the DataFrame) for the layer's real particles, plus `<name>_average.mrc` decoded
  from the layer's **average latent** (centroid layers get just the volume). Frontend: a
  per-layer ⤓ button + "Save all" + format select, gated on `/api/export_info` (disabled
  standalone). Verified end-to-end: cluster → 5-row .xmd + 8 MB mrc; centroids → mrc only;
  resolver finds the right metadata. (Note: `.cs` depends on `XmippMetaData.write` support;
  `.xmd`/`.star` confirmed.) (b) **Speed** — a `speed` slider (fps) sets the play/record
  frame delay (`frameDelay`). (c) **Continuous** — contour (surface) + morph scrubber use
  a coalescing single-flight runner (`makeLive`) instead of a debounce, so they update
  continuously like the glow/σ sliders. JS balance + DOM-id + py-AST checks clean.
- [x] **40** — viewer: **ChimeraX-like surfaces + point opacity + collapsible path**.
  (a) **Shading** — rewrote `gl.js` to shade in view space with lights fixed to the
  camera (model rotates under them): Blinn-Phong **key + fill + ambient + specular**, a
  cheap derivative-based **ambient occlusion** (`fwidth(normal)` darkens crevices/edges),
  and **depth cueing** (far fragments darken). The `light` select is now
  **standard / smooth / ray traced**, each a parameter set (ambient/fill/spec/shine/ao/
  depthCue). (b) **Point opacity** — a `points` slider (0 = hide) on the landscape fades
  the 2D scatter (`globalAlpha`) and the 3D cloud (alpha uniform + blended points pass);
  highlight rings (selected/path/selection) stay visible so you can hide the dots to see
  the density glow. (c) **Collapsible path** — long morph paths (>6) collapse to a
  `N states · #a → … → #z` summary with a ▸/▾ toggle; expanded, `#morphto` scrolls
  (max-height) instead of flooding the bar. JS balance + DOM-id checks clean; all static
  serve 200.
- [x] **27** — annotate_space `--pickled_nn` bindable: surfaced `--pickled_nn` (a
  dynamic kwarg argparse can't see) as a real form field via `registry._EXTRA_FIELDS`,
  and made it bindable to a trained network just like `--reload` (kind `decoder_model`,
  producers hetsiren/zernike3deep/reconsiren). `load_model.py` calls
  `NeuralNetworkCheckpointer.load(pickled_nn)` directly, so the adapter hands over the
  checkpoint dir itself (dir form). `get_schema` now bakes extra fields + field-default
  overrides into one augmented (cached) schema used everywhere (form, `build_command`),
  replacing `schema_with_defaults`.
- [x] **25** — annotate_space `--server_functions_path` pre-filled: the form now
  defaults this field to hax's bundled `viewers/server_loading_functions/load_model.py`
  (resolved from the installed package via `importlib.util.find_spec`, so no JAX import
  and it follows the install location). Implemented as an editable field default
  (`registry.field_defaults` / `schema_with_defaults`), not a hidden/forced value, so
  the user can clear it for view-only mode or point it at a custom interface — and
  because enabling it also needs a `pickled_nn` network path. The cached schema is
  copied before overriding, never mutated.
- [x] **24** — "Saved" job state: the launcher form has a **Save job** button next to
  **Run**. Saving fully prepares the run (creates the run folder, materialises any
  job-bound inputs, validates and stores the command) but leaves it in a new `saved`
  state instead of launching — so a user can fill in a protocol now and execute it
  later. Saved jobs show a teal `saved` badge and offer **Run** (▶) from the ⋮ menu on
  the Jobs list / project executions and as a prominent button on the job view, which
  launches the stored command and streams its log live. Backend: `create(...,
  launch=False)` and `POST /api/jobs/{id}/start` (409 unless the job is `saved`).
  Saved jobs can be renamed/deleted like any other.
- [x] **20** — Project-manager flash cards: the Projects page now has a collapsible
  **lateral rail** of project cards next to the left menu. Each card shows the
  project name, run count, created date and path; clicking it opens the project as a
  workspace tab. A three-dot (⋮) menu on each card offers **Open / Rename / Delete**
  (rename is inline; delete removes the folder + all runs after confirm). A "New
  project" input sits at the top of the rail, and the rail collapses to a thin strip
  to reclaim space. Backend: `PUT /api/projects/{id}` (rename, keeps the folder id so
  existing runs still resolve), `DELETE /api/projects/{id}` (refuses if a job is still
  running; otherwise drops the job records and removes the project folder), and the
  project list now reports each project's run count.