# WebGL conformational-landscape viewer (`viewer/`)

A browser-native, **zero-dependency** alternative to the native PyQt5/napari
`annotate_space` viewer. Lives entirely under `hax-em-gui/viewer/` — it does
**not** modify `hax-em`. It imports hax only as a library (to load a trained
network and decode volumes), exactly like `decode_states_from_latents`.

## Why this exists
`annotate_space` is a native desktop app (OpenGL via Qt) — it can't render inside
a browser. This module re-implements the core loop (pick a point in latent space
→ decode its 3D map → inspect it) using a small backend that decodes volumes and
a hand-written WebGL2 frontend, so it embeds directly in the web GUI and works
for true server/client (remote browser) use.

## Architecture
```
browser ── /viewer/<id>/ ──> main FastAPI app ── reverse proxy ──> viewer subprocess (hax env)
  WebGL2 frontend                                                    loads network + latents
  (static/, no deps)                                                 decode → marching cubes → mesh
```

* **Backend** (`server.py`, standalone, runs in the hax conda env):
  - loads a network checkpoint (`NeuralNetworkCheckpointer.load`) and a latents
    `.npy`, exactly as `decode_states_from_latents` does;
  - computes a 2D embedding of the latent space (uses a provided reduced space,
    else PCA via NumPy SVD — no scikit-learn/UMAP dependency);
  - `GET /api/latents` → the 2D points;
  - `GET /api/decode?index=i&iso=f` → decode that latent → density grid →
    `skimage.measure.marching_cubes` at level `f` → a binary triangle mesh
    (positions + normals + indices). Decoded volumes are cached so moving the
    iso slider doesn't re-run the (expensive) network.
  - serves its own `static/` so it is fully runnable on its own.

* **Frontend** (`static/`, pure vanilla + raw WebGL2, no npm/CDN):
  - `scatter.js` — 2D latent scatter on a Canvas2D; click selects a state.
  - `gl.js` + `mat4.js` — a minimal WebGL2 isosurface renderer (orbit camera,
    headlight + wrap lighting). Mesh-based (server does marching cubes), which is
    far more robust to write than a volume ray-marcher.
  - `app.js` — wires scatter → decode fetch → renderer, plus the iso slider.

* **Integration** (`backend/hax_gui/viewer.py` + routes in `app.py`):
  - a session manager that spawns `server.py` per (latents, network) and a
    reverse-proxy at `/viewer/<id>/` (mirrors the TensorBoard proxy). The frontend
    opens it in an embedded iframe tab.

## Run standalone (for development / testing)
```bash
conda activate hax
python hax-em-gui/viewer/server.py \
    --latents /path/predicted_latents.npy \
    --reload  /path/run/HetSIREN \
    --port 8051
# open http://127.0.0.1:8051/
```

## Status / roadmap
- [x] mesh-based isosurface (server marching cubes), σ-contour slider, embedded via proxy.
- [x] latent scatter with **PCA axis selection** + **UMAP**, **k-means clustering** coloring,
      and a **density heatmap** background.
- [x] **3D landscape view** (`cloud.js`): WebGL2 point cloud over any 3 PCs, color by
      cluster (palette) / local density (kNN ramp) / none, orbit + CPU-side picking,
      selection synced with the 2D scatter, transparent/high-DPI snapshot. 2D/3D toggle
      in the landscape toolbar (`dims`), with a Z-axis selector. UMAP is 2D-only so 3D
      forces PCA. No backend change — `/api/landscape` already returns up to 8 PCs.
- [ ] live volume ray-marching (WebGL2 3D texture) for transfer-function rendering / "layers".
- [ ] lasso/box selection; latent-space interpolation between two states; trajectory morph.
- [ ] export the selected map as .mrc (backend writes via ImageHandler) and ChimeraX hand-off.

### Notes for future work
- JAX arrays are read-only -> always `np.array(..., dtype=np.float32)` (copy) before
  skimage, which needs a writable buffer. (Bit us once; see `Decoder.volume`.)
- Maps are sparse (median ~0, sharp positive tail), so contour in **sigma units**
  (`mean + sigma*std`), never a linear fraction of [min, max].

---

## Matching (and improving on) the native annotate_space GUI

The native app is napari-based: a **landscape** panel (2D/3D point cloud with a
density background + labeled cluster points), a **volume** panel (napari volume
rendering / MIP), a **layer list** (point layers per cluster, a volume layer), and
docked panels for *Landscape controls*, *Volume display controls*, *Landscape
clustering controls*, and *Saving controls* — plus a Morph-ChimeraX button.

### Proposed web layout (cleaner than the native dock soup)
```
┌──────────────────────────────────────────────────────────────────────┐
│ top bar: title · export(DPI, transparent) · status                    │
├───────────────┬───────────────────────────────┬──────────────────────┤
│  LAYERS rail  │        LANDSCAPE (2D/3D)        │   VOLUME (3D map)     │
│  (collapsible)│  density bg · clusters · labels │  histogram contour   │
│  • Cluster 1  │  axis selectors / PCA·UMAP      │  orbit · lighting     │
│  • KMeans     │  lasso/box select               │  morph · record       │
│  • Volume     │                                 │  save PNG · .mrc      │
│  + selection  │                                 │                      │
└───────────────┴───────────────────────────────┴──────────────────────┘
```
Replace the four native dock panels with **one left "Layers" rail** + **inline
per-panel toolbars** (already started: landscape head, contour strip, morph bar).
A small **menu bar** (View / Layers / Tools / Export) can host less-frequent
actions to keep toolbars clean.

### Feature status & approach
| Native feature | Web approach | Status |
|---|---|---|
| Landscape 2D + density | Canvas2D scatter + 2D-histogram heatmap; **3D = additive glow splats** | ✅ 2D + 3D |
| PCA/UMAP, axis (Dim 1/2/3) selectors | `/api/landscape` + `/api/embedding`; dropdowns | ✅ 2D + **3D** (cloud.js) |
| Clustering (KMeans, along PCA) | `/api/cluster?k` (coloring) + `/api/kmeans?k` (Cluster button → 1 layer/cluster + a decodable **centroid** layer); "cluster along PC" = bin a PC | ✅ KMeans + centroid layers; PC-binning pending |
| Cluster centroids morph + color | centroids are virtual decodable indices; centroid layer ▶ = series; decoded centroid maps auto-colored by cluster; morph lerps color | ✅ done |
| Lasso add/remove from selection | Shift = add, Alt/Ctrl = remove, drag = replace; lasso tinted by mode | ✅ done |
| Contour σ + **histogram** | `/api/histogram` + draggable widget (`histogram.js`) | ✅ done |
| Volume display (MIP, gamma, colormap) | **WebGL2 3D-texture ray-marcher** (`volume.js`): MIP + emission/absorption, γ + brightness, contour-driven threshold | ✅ done |
| Lighting standard/smooth/ray-traced | ChimeraX-style Blinn-Phong (key+fill+ambient+specular), derivative AO, depth cueing; presets standard/smooth/raytraced | ✅ done |
| Hide / fade landscape points | per-point opacity slider (0 = hide) on 2D + 3D; highlight rings stay visible | ✅ done |
| Collapsible morph path | long paths collapse to a `N states · #a → … → #z` summary with a ▸/▾ toggle (expanded scrolls) | ✅ done |
| Trackball rotation (all axes, centered) | shared `orbit.js`: accumulated rotation matrix, fixed camera, drag-from-anywhere | ✅ done (landscape + both volume modes) |
| Background color | `renderer.setBackground` | ✅ done |
| **Layers** (extract/add selection) | left rail; a layer = a named index set; lasso → layer | ✅ done |
| Lasso / box selection | overlay polygon hit-test → selection → layer (2D + 3D) | ✅ done |
| **Volume series / morph** (mass-constant) | `/api/decode`/`/api/grid` with `index2,t`; **scrubbable timeline** + play/pause + ■ stop; layer ▶ = series | ✅ multi-state morph + series-over-layer, surface & volume |
| Morph **video** | `MediaRecorder` on the active (surface or volume) canvas | ✅ done |
| Save PNG (DPI, transparent) — both windows | `snapshot({scale,transparent})` on scatter + renderer | ✅ done |
| Export decoded map (.mrc) | `/api/mrc` via ImageHandler | ✅ done |
| ChimeraX connection | **removed** — replaced by the above | ✅ avoided |

### Recently requested refinements (this batch)
- **UMAP in 3D** — `Landscape.umap()` now computes **3 components** (clamped to n-1,
  falls back to the first 3 PCs); 3D no longer forces PCA, so UMAP behaves like PCA in
  both 2D and 3D. ✅
- **Multi-state morph** — the single morph *target* is generalised to an ordered
  **path** of arbitrary length: shift-click appends waypoints, the morph interpolates
  consecutively through `[current, …path]` at constant mass (ping-pong for a clean
  loop), and the whole trajectory can be recorded. A **Clear** button resets the path.
  All waypoints get a pink ring in both the 2D scatter and 3D cloud. ✅
- **ChimeraX-gray default** — the surface default color is now ChimeraX's neutral light
  gray (`#c8c8c8`) instead of blue. ✅
- **Density σ slider** — density works in 2D (Canvas2D heatmap) and 3D (per-point kNN
  ramp), updates under both PCA and UMAP, and a **σ / bandwidth slider** (shown when
  color = density) widens/narrows it live: 2D = repeated box-blur passes, 3D = neighbour
  radius as a fraction of the bbox diagonal. ✅

### Layout redesign (this batch)
The single-row pane toolbars were wrapping and overlapping the titles/canvas (controls
clipped). Rebuilt the shell: **stacked `.pane-head`** (title line + a `flex-wrap`
`.toolbar` that grows its own height, never overlapping), a **`.pane-body`** holding a
collapsible **Layers rail** + a `position:relative` **`.canvas-wrap`** (canvases are
`absolute; inset:0`, so 2D/3D/lasso layers stack cleanly), a wrapping topbar with an
ellipsised status, and labelled `surface`/`bg` color swatches. Toolbars now reflow
instead of colliding. ✅

### Done (export + playback)
- **Export layers to disk** ✅ — each layer's ⤓ (and "Save all") writes
  `<project>/<layer_name>/`: a subset metadata `.xmd`/`.star`/`.cs` (poses/shifts/CTF/
  latent, via `md.table.iloc[indices]`) for the real particles, plus `<name>_average.mrc`
  decoded from the layer's **average latent**. The viewer is now launched with
  `--metadata` + `--project-dir`; the GUI auto-resolves the row-aligned metadata by
  scanning the project for a `_latent_space` file matching the latents
  (`_find_latent_metadata`). Export silently disables in standalone mode.
- **Playback speed** ✅ — a `speed` slider (frames/sec) drives the play/record delay.
- **Continuous contour + scrub** ✅ — the contour (surface mode) and the morph scrubber
  now use a coalescing single-flight runner (`makeLive`) instead of a debounce, so they
  update continuously (server-bounded) like the glow/σ sliders. Volume-mode contour was
  already instant (client uniform).

### Phased roadmap (remaining)
1. ~~**3D landscape**~~ — ✅ done (`cloud.js`): point cloud (3 PCs), cluster/density
   color, 3D picking, 2D/3D toggle.
2. ~~**Layer system + lasso/box select**~~ — ✅ done. App-owned **lasso** overlay
   (`#lassoCanvas`) works over the active 2D *or* 3D view (each exposes
   `projectedPoints()`; enclosure by ray-cast polygon test). Selected points get a white
   ring (`setSelection`). A **Layers rail** turns a selection into a named, recolorable,
   show/hide-able **layer**; "color = layers" paints by layer; each layer's **▶** plays a
   constant-mass **volume series** (states ordered along the current axis → morph path).
   No backend change — series reuses `/api/decode`.
3. ~~**Volume ray-marcher**~~ — ✅ done (`volume.js`). Backend `/api/grid` streams the
   decoded map stride-downsampled to a uint8 3D-texture payload (header: dims + vmin/
   vmax/mean/std). Frontend ray-marches a [-1,1] cube on a full-screen quad (ray-box
   intersection via `invert(proj·view)`): **MIP** and **emission/absorption** modes, a
   **γ** and **brightness** transfer function, and a threshold driven live by the
   contour histogram. A **render** select (Surface / Volume·MIP / Volume·glow) toggles a
   second canvas (`#glvol`); surface/bg color and Reset are shared. Morphs force Surface.
4. ~~**Series over a layer**~~ — ✅ done (a layer's ▶ orders its states along the current
   axis and plays the constant-mass morph; **■ Stop** interrupts it).
5. **Polish** — menu bar, PC-binning clusters, ray-traced lighting (SSAO/soft
   shadows approximation), per-layer color/opacity controls, optional volume-mode
   series (per-frame 3D-texture streaming).
