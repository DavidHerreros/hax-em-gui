// Orthogonal-slice viewer for a decoded/loaded volume. Renders the three
// canonical cuts (XY / XZ / YZ) from the same uint8 3D-texture payload the
// ray-marcher uses, with per-axis position sliders, a contrast window, a
// colormap and a crosshair. Clicking a panel re-centers the other two (standard
// ortho-viewer navigation). Canvas2D only — no WebGL, no dependencies.
//
// Grid layout (matches server `/api/grid`): voxel (ix,iy,iz) lives at
//   data[(iz*ny + iy)*nx + ix]   with x fastest.

// -- colormaps: control points interpolated at draw time -------------------
const CMAPS = {
  gray: [[0, 0, 0], [255, 255, 255]],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  inferno: [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]],
  hot: [[10, 0, 0], [230, 0, 0], [255, 210, 0], [255, 255, 255]],
};

function makeLUT(name) {
  const stops = CMAPS[name] || CMAPS.gray;
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const a = Math.min(stops.length - 1, Math.floor(t)), b = Math.min(stops.length - 1, a + 1);
    const f = t - a;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[a][c] * (1 - f) + stops[b][c] * f;
  }
  return lut;
}

export function createSlices(container) {
  let grid = null;                       // { nx, ny, nz, data }
  let lo = 0, hi = 1;                     // contrast window (fraction of 0..255)
  let lut = makeLUT("gray");
  let crosshair = true;
  let bg = "#04060f";
  const pos = { x: 0, y: 0, z: 0 };       // current slice indices

  // One panel per cut. `w`/`h` map a slice to its (col,row); `set` turns a click
  // at (col,row) into the two positions that panel controls; `axis` is the slice
  // index this panel's slider drives; `cx`/`cy` give the crosshair (col,row).
  const CUTS = [
    { key: "xy", label: "XY (axial · Z)", axis: "z",
      w: () => grid.nx, h: () => grid.ny,
      at: (c, r) => grid.data[(pos.z * grid.ny + r) * grid.nx + c],
      set: (c, r) => { pos.x = c; pos.y = r; },
      cx: () => pos.x, cy: () => pos.y },
    { key: "xz", label: "XZ (coronal · Y)", axis: "y",
      w: () => grid.nx, h: () => grid.nz,
      at: (c, r) => grid.data[(r * grid.ny + pos.y) * grid.nx + c],
      set: (c, r) => { pos.x = c; pos.z = r; },
      cx: () => pos.x, cy: () => pos.z },
    { key: "yz", label: "YZ (sagittal · X)", axis: "x",
      w: () => grid.ny, h: () => grid.nz,
      at: (c, r) => grid.data[(r * grid.ny + c) * grid.nx + pos.x],
      set: (c, r) => { pos.y = c; pos.z = r; },
      cx: () => pos.y, cy: () => pos.z },
  ];

  const panels = {};
  container.classList.add("slices");
  for (const cut of CUTS) {
    const canvas = document.createElement("canvas");
    canvas.className = "slice-canvas";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "0"; slider.value = "0";
    slider.className = "slice-slider";
    const idx = document.createElement("span");
    idx.className = "slice-idx muted small";
    const head = document.createElement("div");
    head.className = "slice-head";
    const lbl = document.createElement("span"); lbl.className = "slice-label"; lbl.textContent = cut.label;
    head.appendChild(lbl); head.appendChild(idx);
    const foot = document.createElement("div"); foot.className = "slice-foot";
    foot.appendChild(slider);
    const cell = document.createElement("div"); cell.className = "slice-cell";
    cell.appendChild(head); cell.appendChild(canvas); cell.appendChild(foot);
    container.appendChild(cell);

    const p = { cut, canvas, ctx: canvas.getContext("2d"), slider, idx, tf: null, view: { zoom: 1, panx: 0, pany: 0 } };
    panels[cut.key] = p;

    slider.addEventListener("input", () => { pos[cut.axis] = +slider.value; drawAll(); });
    canvas.addEventListener("wheel", (e) => onWheel(p, e), { passive: false });
    canvas.addEventListener("pointerdown", (e) => onDown(p, e));
    canvas.addEventListener("pointermove", (e) => onMove(p, e));
    canvas.addEventListener("pointerup", (e) => onUp(p, e));
  }
  // A fourth (empty) cell keeps the 2×2 grid balanced and shows a legend.
  const info = document.createElement("div");
  info.className = "slice-cell slice-info muted small";
  container.appendChild(info);

  // click = re-center the other two cuts · drag = pan · wheel = zoom (cursor-anchored)
  function devPos(p, e) {
    const rect = p.canvas.getBoundingClientRect();
    return { px: (e.clientX - rect.left) * (p.canvas.width / rect.width),
             py: (e.clientY - rect.top) * (p.canvas.height / rect.height) };
  }
  function voxelAt(p, px, py) {
    if (!p.tf) return null;
    return { c: (px - p.tf.ox) / p.tf.s, r: (py - p.tf.oy) / p.tf.s };
  }
  function recenter(p, px, py) {
    const v = voxelAt(p, px, py); if (!v) return;
    const c = Math.floor(v.c), r = Math.floor(v.r);
    if (c < 0 || r < 0 || c >= p.cut.w() || r >= p.cut.h()) return;
    p.cut.set(c, r); syncSliders(); drawAll();
  }

  let drag = null;
  function onDown(p, e) {
    if (!grid) return;
    drag = { p, ...devPos(p, e), moved: false, panx: p.view.panx, pany: p.view.pany };
    try { p.canvas.setPointerCapture(e.pointerId); } catch {}
  }
  function onMove(p, e) {
    if (!drag || drag.p !== p) return;
    const { px, py } = devPos(p, e);
    const dx = px - drag.px, dy = py - drag.py;
    if (!drag.moved && Math.hypot(dx, dy) > 3) drag.moved = true;
    if (drag.moved) { p.view.panx = drag.panx + dx; p.view.pany = drag.pany + dy; drawPanel(p); }
  }
  function onUp(p, e) {
    try { p.canvas.releasePointerCapture(e.pointerId); } catch {}
    if (drag && drag.p === p && !drag.moved) { const { px, py } = devPos(p, e); recenter(p, px, py); }
    drag = null;
  }
  function onWheel(p, e) {
    if (!grid || !p.tf) return;
    e.preventDefault();
    const { px, py } = devPos(p, e);
    const v = voxelAt(p, px, py);
    const nz = Math.max(1, Math.min(40, p.view.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    const w = p.cut.w(), h = p.cut.h();
    const s2 = Math.min(p.canvas.width / w, p.canvas.height / h) * nz;
    // keep the voxel under the cursor pinned: solve px = centerX + panx + v.c*s2
    p.view.zoom = nz;
    p.view.panx = px - (p.canvas.width - w * s2) / 2 - v.c * s2;
    p.view.pany = py - (p.canvas.height - h * s2) / 2 - v.r * s2;
    drawPanel(p);
  }

  function dpr() { return Math.min(window.devicePixelRatio || 1, 2); }

  function drawPanel(p) {
    const { canvas, ctx, cut } = p;
    const r = dpr();
    const cw = Math.max(1, Math.floor(canvas.clientWidth * r));
    const ch = Math.max(1, Math.floor(canvas.clientHeight * r));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    ctx.fillStyle = bg; ctx.fillRect(0, 0, cw, ch);
    if (!grid) { p.tf = null; return; }

    const w = cut.w(), h = cut.h();
    // rasterize the slice at native resolution into an ImageData
    const img = ctx.createImageData(w, h);
    const d = img.data;
    const span = Math.max(1e-6, hi - lo);
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const v = cut.at(col, row) / 255;                 // 0..1
        let t = (v - lo) / span; t = t < 0 ? 0 : t > 1 ? 1 : t;
        const li = (t * 255) | 0;
        const o = (row * w + col) * 4;
        d[o] = lut[li * 3]; d[o + 1] = lut[li * 3 + 1]; d[o + 2] = lut[li * 3 + 2]; d[o + 3] = 255;
      }
    }
    // blit letterboxed (preserve voxel aspect) with the panel's zoom + pan applied,
    // nearest-neighbour. `fit` is the base scale; `s` the zoomed scale.
    const fit = Math.min(cw / w, ch / h);
    const s = fit * p.view.zoom;
    const ox = (cw - w * s) / 2 + p.view.panx;
    const oy = (ch - h * s) / 2 + p.view.pany;
    const tmp = document.createElement("canvas"); tmp.width = w; tmp.height = h;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w, h, ox, oy, w * s, h * s);
    p.tf = { ox, oy, s };

    if (crosshair) {
      const cx = ox + (cut.cx() + 0.5) * s, cy = oy + (cut.cy() + 0.5) * s;
      ctx.strokeStyle = "rgba(45,212,191,0.7)"; ctx.lineWidth = Math.max(1, r);
      ctx.beginPath();
      ctx.moveTo(cx, oy); ctx.lineTo(cx, oy + h * s);
      ctx.moveTo(ox, cy); ctx.lineTo(ox + w * s, cy);
      ctx.stroke();
    }
    p.idx.textContent = `${cut.axis} = ${pos[cut.axis]} / ${(cut.axis === "z" ? grid.nz : cut.axis === "y" ? grid.ny : grid.nx) - 1}`;
  }

  function drawAll() { for (const k in panels) drawPanel(panels[k]); }

  function syncSliders() {
    for (const k in panels) {
      const p = panels[k];
      p.slider.value = String(pos[p.cut.axis]);
    }
  }

  window.addEventListener("resize", drawAll);

  return {
    el: container,
    // Swap in a new volume. When the grid keeps the same dimensions (e.g. successive
    // frames of a morph), the current cut positions, zoom and pan are preserved so the
    // view does not jump — you keep watching the same plane as the density moves through
    // it. A change of dimensions re-centers, as on a first load.
    setVolume(g) {
      const same = grid && grid.nx === g.nx && grid.ny === g.ny && grid.nz === g.nz;
      grid = g;
      if (same) {
        pos.x = Math.min(pos.x, g.nx - 1); pos.y = Math.min(pos.y, g.ny - 1); pos.z = Math.min(pos.z, g.nz - 1);
      } else {
        pos.x = g.nx >> 1; pos.y = g.ny >> 1; pos.z = g.nz >> 1;
      }
      for (const k in panels) {
        const p = panels[k];
        const n = p.cut.axis === "z" ? g.nz : p.cut.axis === "y" ? g.ny : g.nx;
        p.slider.max = String(n - 1);
      }
      syncSliders();
      info.textContent = `${g.nx} × ${g.ny} × ${g.nz} voxels · click = re-center · wheel = zoom · drag = pan · slider = scroll`;
      drawAll();
    },
    setContrast(l, h) { lo = Math.max(0, Math.min(1, l)); hi = Math.max(lo + 1e-3, Math.min(1, h)); drawAll(); },
    setColormap(name) { lut = makeLUT(name); drawAll(); },
    setCrosshair(on) { crosshair = !!on; drawAll(); },
    setBackground(hex) { bg = hex; drawAll(); },
    resetZoom() { for (const k in panels) panels[k].view = { zoom: 1, panx: 0, pany: 0 }; drawAll(); },
    render: drawAll,
    async snapshot({ scale = 2, transparent = false } = {}) {
      // Composite the 2×2 grid into one PNG at the requested scale.
      const rect = container.getBoundingClientRect();
      const W = Math.floor(rect.width * scale), H = Math.floor(rect.height * scale);
      const out = document.createElement("canvas"); out.width = W; out.height = H;
      const octx = out.getContext("2d");
      if (!transparent) { octx.fillStyle = bg; octx.fillRect(0, 0, W, H); }
      for (const k in panels) {
        const p = panels[k];
        const cr = p.canvas.getBoundingClientRect();
        octx.drawImage(p.canvas,
          (cr.left - rect.left) * scale, (cr.top - rect.top) * scale,
          cr.width * scale, cr.height * scale);
      }
      return await new Promise((res) => out.toBlob(res, "image/png"));
    },
  };
}
