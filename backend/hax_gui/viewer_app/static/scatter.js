// 2D latent-landscape scatter (Canvas2D): dynamic points, optional cluster
// coloring and density heatmap, click-to-select, shift-click to set a morph
// target, plus high-DPI / transparent snapshots. No dependencies.

const PALETTE = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f87171",
  "#22d3ee", "#fb923c", "#4ade80", "#e879f9", "#facc15", "#38bdf8",
  "#fca5a5", "#a3e635", "#c084fc", "#2dd4bf", "#fdba74", "#93c5fd",
];
export const clusterColor = (l) => PALETTE[((l % PALETTE.length) + PALETTE.length) % PALETTE.length];

export function createScatter(canvas, onSelect) {
  const ctx = canvas.getContext("2d");
  let points = [], labels = null, showDensity = false, pointColors = null;
  let selected = -1, hover = -1, pathSet = new Set(), selSet = new Set();
  let pathOrder = [], pathColor = "#f472b6";   // ordered morph path drawn as a polyline
  let freeCurves = [];   // [{ pts:[[x,y]...] data coords, color }] free-spline overlays
  let densBandwidth = 6, pointAlpha = 1.0, bgColor = null;   // bgColor set => opaque fill (dark density backdrop)
  let cmap = [[15, 23, 42], [30, 64, 120], [13, 148, 136], [250, 204, 21]];   // density colormap stops
  let bounds = { minX: -1, minY: -1, maxX: 1, maxY: 1 };   // data extent (fit)
  let view = { minX: -1, minY: -1, maxX: 1, maxY: 1 };     // visible window (zoom/pan); == bounds when fit
  const PAD = 26;

  function setBounds() {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (const [x, y] of points) { if (x < a) a = x; if (x > c) c = x; if (y < b) b = y; if (y > d) d = y; }
    if (!isFinite(a)) { a = -1; c = 1; b = -1; d = 1; }
    if (c - a < 1e-6) { a -= 1; c += 1; }
    if (d - b < 1e-6) { b -= 1; d += 1; }
    const mx = (c - a) * 0.05, my = (d - b) * 0.05;
    bounds = { minX: a - mx, minY: b - my, maxX: c + mx, maxY: d + my };
    view = { ...bounds };   // new data refits the view
  }
  // ── zoom / pan (wheel zooms toward the cursor; drag pans; dbl-click resets) ──
  function clampView() {
    const bw = bounds.maxX - bounds.minX, bh = bounds.maxY - bounds.minY;
    const vw = Math.min(bw, Math.max(bw / 40, view.maxX - view.minX));
    const vh = Math.min(bh, Math.max(bh / 40, view.maxY - view.minY));
    let cx = (view.minX + view.maxX) / 2, cy = (view.minY + view.maxY) / 2;
    cx = Math.min(bounds.maxX - vw / 2, Math.max(bounds.minX + vw / 2, cx));
    cy = Math.min(bounds.maxY - vh / 2, Math.max(bounds.minY + vh / 2, cy));
    view = { minX: cx - vw / 2, maxX: cx + vw / 2, minY: cy - vh / 2, maxY: cy + vh / 2 };
  }
  function zoomAt(cssX, cssY, factor) {
    const [bx, by] = unproject(cssX, cssY);
    const fx = (bx - view.minX) / (view.maxX - view.minX), fy = (by - view.minY) / (view.maxY - view.minY);
    const vw = (view.maxX - view.minX) / factor, vh = (view.maxY - view.minY) / factor;
    view = { minX: bx - fx * vw, maxX: bx + (1 - fx) * vw, minY: by - fy * vh, maxY: by + (1 - fy) * vh };
    clampView(); draw();
  }
  function panBy(dxCss, dyCss) {
    const r = dpr();
    const ddx = dxCss * r * (view.maxX - view.minX) / Math.max(1, canvas.width - 2 * PAD);
    const ddy = dyCss * r * (view.maxY - view.minY) / Math.max(1, canvas.height - 2 * PAD);
    view = { minX: view.minX - ddx, maxX: view.maxX - ddx, minY: view.minY + ddy, maxY: view.maxY + ddy };
    clampView(); draw();
  }
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  function resize() {
    const r = dpr(), w = Math.max(1, Math.floor(canvas.clientWidth * r)), h = Math.max(1, Math.floor(canvas.clientHeight * r));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function toScreen(x, y) {
    const w = canvas.width, h = canvas.height;
    return [PAD + (x - view.minX) / (view.maxX - view.minX) * (w - 2 * PAD),
      h - PAD - (y - view.minY) / (view.maxY - view.minY) * (h - 2 * PAD)];
  }
  // CSS-space screen px -> data coords (inverse of toScreen, in the current view)
  function unproject(sx, sy) {
    const r = dpr(), w = canvas.width, h = canvas.height, dx = sx * r, dy = sy * r;
    return [view.minX + (dx - PAD) / Math.max(1, w - 2 * PAD) * (view.maxX - view.minX),
      view.minY + (h - PAD - dy) / Math.max(1, h - 2 * PAD) * (view.maxY - view.minY)];
  }

  // Density color from the active colormap (alpha rises with t so faint→opaque). The
  // light theme needs a higher, more opaque range or the colormap washes out on the
  // pale background.
  function ramp(t, forceAlpha) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (cmap.length - 1), i = Math.min(cmap.length - 2, Math.floor(f)), u = f - i;
    const c = (a, b) => Math.round(a + (b - a) * u);
    // A dark backdrop (density mode) reads like the desktop app's inferno layer;
    // a light page background needs a higher floor or the colormap washes out.
    const light = !bgColor && document.documentElement.dataset.theme === "light";
    const alpha = forceAlpha != null ? forceAlpha : (light ? 0.5 + 0.48 * t : 0.18 + 0.72 * t);
    return `rgba(${c(cmap[i][0], cmap[i + 1][0])},${c(cmap[i][1], cmap[i + 1][1])},${c(cmap[i][2], cmap[i + 1][2])},${alpha})`;
  }

  // Smoothed point-count grid (shared by the heatmap fill and per-point tinting).
  function buildDensityGrid() {
    const G = 64, grid = new Float32Array(G * G);
    for (const [x, y] of points) {
      const gx = Math.min(G - 1, Math.max(0, Math.floor((x - bounds.minX) / (bounds.maxX - bounds.minX) * G)));
      const gy = Math.min(G - 1, Math.max(0, Math.floor((y - bounds.minY) / (bounds.maxY - bounds.minY) * G)));
      grid[gy * G + gx] += 1;
    }
    for (let p = 0; p < densBandwidth; p++) blur(grid, G);   // wider sigma = more passes
    let max = 0; for (let i = 0; i < grid.length; i++) max = Math.max(max, grid[i]);
    return { grid, G, max };
  }
  function densityAt(dg, x, y) {
    if (!dg || dg.max <= 0) return 0;
    const gx = Math.min(dg.G - 1, Math.max(0, Math.floor((x - bounds.minX) / (bounds.maxX - bounds.minX) * dg.G)));
    const gy = Math.min(dg.G - 1, Math.max(0, Math.floor((y - bounds.minY) / (bounds.maxY - bounds.minY) * dg.G)));
    return dg.grid[gy * dg.G + gx] / dg.max;
  }
  function drawDensity(dg) {
    if (!dg || dg.max <= 0) return;
    const { grid, G, max } = dg;
    const bw = bounds.maxX - bounds.minX, bh = bounds.maxY - bounds.minY;
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const v = grid[gy * G + gx] / max; if (v < 0.02) continue;
      // map the cell's data rectangle through toScreen so it tracks zoom / pan
      const [sx0, syTop] = toScreen(bounds.minX + gx / G * bw, bounds.minY + (gy + 1) / G * bh);
      const [sx1, syBot] = toScreen(bounds.minX + (gx + 1) / G * bw, bounds.minY + gy / G * bh);
      ctx.fillStyle = ramp(v);
      ctx.fillRect(sx0, syTop, sx1 - sx0 + 1, syBot - syTop + 1);
    }
  }

  function paint(bg) {
    const w = canvas.width, h = canvas.height;
    if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h); } else ctx.clearRect(0, 0, w, h);
    const dg = showDensity ? buildDensityGrid() : null;
    if (dg) drawDensity(dg);
    const r = 3 * dpr();
    for (let i = 0; i < points.length; i++) {
      const hl = i === selected || pathSet.has(i) || i === hover;
      if (!hl && pointAlpha <= 0) continue;                 // points hidden
      const [sx, sy] = toScreen(points[i][0], points[i][1]);
      ctx.globalAlpha = hl ? 1 : pointAlpha;
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      // In density mode tint each point by its local density (lifted off the dark
      // low end so sparse points stay visible) — the colored-sphere look.
      ctx.fillStyle = i === selected ? "#2dd4bf" : pathSet.has(i) ? "#f472b6" : i === hover ? "#e2e8f0"
        : pointColors ? pointColors[i]
        : dg ? ramp(0.32 + 0.68 * densityAt(dg, points[i][0], points[i][1]), 1)
        : labels ? clusterColor(labels[i]) : "rgba(148,163,184,0.6)";
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    drawFreeCurves();
    drawPath();
    // selection ring must contrast with the background -> dark only on a light
    // backdrop (the density mode forces a dark canvas regardless of page theme)
    const selCol = (!bgColor && document.documentElement.dataset.theme === "light") ? "rgba(15,23,42,0.9)" : "rgba(241,245,249,0.9)";
    for (const si of selSet) ring(si, selCol, 1.7, true);
    for (const pi of pathSet) ring(pi, "#f472b6");
    ring(selected, "#2dd4bf");
    function ring(i, col, scale = 2.4, halo = false) {
      if (i < 0 || !points[i]) return;
      const [sx, sy] = toScreen(points[i][0], points[i][1]);
      if (halo) {  // faint opposite-tone halo so the ring reads on any background
        ctx.beginPath(); ctx.arc(sx, sy, r * scale, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0,0,0,0.25)"; ctx.lineWidth = 3.2 * dpr(); ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(sx, sy, r * scale, 0, Math.PI * 2); ctx.strokeStyle = col; ctx.lineWidth = 1.6 * dpr(); ctx.stroke();
    }
  }
  // Connect the ordered morph waypoints with a polyline (dark halo + colored line
  // + small numbered ticks) so a path is visible on the landscape.
  function drawPath() {
    const pts = pathOrder.filter((i) => i >= 0 && points[i]).map((i) => toScreen(points[i][0], points[i][1]));
    if (pts.length < 2) return;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.strokeStyle = "rgba(15,23,42,0.55)"; ctx.lineWidth = 4.5 * dpr(); ctx.stroke();
    ctx.strokeStyle = pathColor; ctx.lineWidth = 2 * dpr(); ctx.stroke();
  }
  // Free-spline overlays: app passes an already-sampled dense polyline in data coords.
  function drawFreeCurves() {
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const fc of freeCurves) {
      const pts = (fc.pts || []).map(([x, y]) => toScreen(x, y));
      if (pts.length >= 2) {
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.strokeStyle = "rgba(15,23,42,0.55)"; ctx.lineWidth = 4.5 * dpr(); ctx.stroke();
        ctx.strokeStyle = fc.color || "#f472b6"; ctx.lineWidth = 2 * dpr(); ctx.stroke();
      }
      if (fc.handles) {                                  // editable control-point handles
        const rr = 5 * dpr();
        for (const [hx, hy] of fc.handles) {
          const [sx, sy] = toScreen(hx, hy);
          ctx.beginPath(); ctx.arc(sx, sy, rr, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(255,255,255,0.92)"; ctx.fill();
          ctx.strokeStyle = "rgba(15,23,42,0.65)"; ctx.lineWidth = 1.5 * dpr(); ctx.stroke();
        }
      }
    }
  }
  function draw() { resize(); paint(bgColor); if (onRenderCb) onRenderCb(); }
  let onRenderCb = null;

  function nearest(cx, cy) {
    const rect = canvas.getBoundingClientRect(), r = dpr(), px = (cx - rect.left) * r, py = (cy - rect.top) * r;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < points.length; i++) {
      const [sx, sy] = toScreen(points[i][0], points[i][1]);
      const dd = (sx - px) ** 2 + (sy - py) ** 2; if (dd < bestD) { bestD = dd; best = i; }
    }
    return best;
  }
  // click selects (suppressed if the pointer was a pan-drag); drag pans; wheel zooms
  // toward the cursor; double-click refits. (When the lasso/draw overlay is armed it
  // sits on top and intercepts these — the app forwards the wheel there instead.)
  let panning = false, panned = false, plx = 0, ply = 0;
  canvas.addEventListener("pointerdown", (e) => { if (e.button === 1) e.preventDefault(); panning = true; panned = false; plx = e.clientX; ply = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
  canvas.addEventListener("pointermove", (e) => {
    if (!panning) return;
    const dx = e.clientX - plx, dy = e.clientY - ply;
    if (Math.abs(dx) + Math.abs(dy) > 3) panned = true;
    if (panned) panBy(dx, dy);
    plx = e.clientX; ply = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => { panning = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("click", (e) => { if (panned) return; const i = nearest(e.clientX, e.clientY); if (i >= 0) onSelect(i, { shift: e.shiftKey }); });
  canvas.addEventListener("mousemove", (e) => { const i = nearest(e.clientX, e.clientY); if (i !== hover) { hover = i; draw(); } });
  canvas.addEventListener("mouseleave", () => { hover = -1; draw(); });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", () => { view = { ...bounds }; draw(); });
  window.addEventListener("resize", draw);

  return {
    setPoints(p) { points = p; setBounds(); draw(); },
    setLabels(l) { labels = l; draw(); },
    setPointColors(arr) { pointColors = arr; draw(); },   // per-point CSS color (layers mode); null clears
    setBackground(css) { bgColor = css || null; draw(); },   // opaque dark fill in density mode; null = transparent (theme)
    setDensity(on) { showDensity = on; draw(); },
    setDensityBandwidth(b) { densBandwidth = Math.max(1, b | 0); if (showDensity) draw(); },
    setColormap(stops) { if (stops && stops.length >= 2) cmap = stops; if (showDensity) draw(); },
    setPointAlpha(a) { pointAlpha = Math.max(0, Math.min(1, a)); draw(); },
    select(i) { selected = i; draw(); },
    setPath(arr) { pathSet = new Set(arr || []); draw(); },
    setPathOrder(arr, color) { pathOrder = arr || []; if (color) pathColor = color; draw(); },
    setFreeCurves(list) { freeCurves = list || []; draw(); },
    onRender(cb) { onRenderCb = cb; },   // called after every draw (overlay can re-sync to the new projection)
    // Project / unproject between data coords and CSS-space screen px (2D is invertible).
    projectData(pts) { const r = dpr(); return pts.map(([x, y]) => { const [sx, sy] = toScreen(x, y); return [sx / r, sy / r]; }); },
    unproject,
    zoomAt,   // app forwards the wheel here when the overlay is armed
    panBy,    // app forwards middle-mouse drags here when the overlay is armed (CSS px)
    resetView() { view = { ...bounds }; draw(); },
    setSelection(arr) { selSet = new Set(arr || []); draw(); },
    // CSS-space screen coords of every point (for app-driven lasso enclosure)
    projectedPoints() { const r = dpr(); return points.map(([x, y]) => { const [sx, sy] = toScreen(x, y); return [sx / r, sy / r]; }); },
    draw,
    get selected() { return selected; },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale); canvas.height = Math.floor(canvas.clientHeight * scale);
      paint(transparent ? null : (bgColor || "#020617"));
      const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
      canvas.width = ow; canvas.height = oh; draw();
      return blob;
    },
  };
}

function blur(grid, G) {
  const tmp = new Float32Array(grid.length), k = [0.25, 0.5, 0.25];
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { let s = 0; for (let d = -1; d <= 1; d++) { const xx = Math.min(G - 1, Math.max(0, x + d)); s += grid[y * G + xx] * k[d + 1]; } tmp[y * G + x] = s; }
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { let s = 0; for (let d = -1; d <= 1; d++) { const yy = Math.min(G - 1, Math.max(0, y + d)); s += tmp[yy * G + x] * k[d + 1]; } grid[y * G + x] = s; }
}
function ramp(t) {
  const stops = [[15, 23, 42], [30, 64, 120], [13, 148, 136], [250, 204, 21]];
  const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
  const c = (a, b) => Math.round(a + (b - a) * u);
  return `rgba(${c(stops[i][0], stops[i + 1][0])},${c(stops[i][1], stops[i + 1][1])},${c(stops[i][2], stops[i + 1][2])},${0.25 + 0.6 * t})`;
}
