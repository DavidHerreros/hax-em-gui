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
  let densBandwidth = 6, pointAlpha = 1.0;
  let cmap = [[15, 23, 42], [30, 64, 120], [13, 148, 136], [250, 204, 21]];   // density colormap stops
  let bounds = { minX: -1, minY: -1, maxX: 1, maxY: 1 };
  const PAD = 26;

  function setBounds() {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (const [x, y] of points) { if (x < a) a = x; if (x > c) c = x; if (y < b) b = y; if (y > d) d = y; }
    if (!isFinite(a)) { a = -1; c = 1; b = -1; d = 1; }
    if (c - a < 1e-6) { a -= 1; c += 1; }
    if (d - b < 1e-6) { b -= 1; d += 1; }
    const mx = (c - a) * 0.05, my = (d - b) * 0.05;
    bounds = { minX: a - mx, minY: b - my, maxX: c + mx, maxY: d + my };
  }
  const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
  function resize() {
    const r = dpr(), w = Math.max(1, Math.floor(canvas.clientWidth * r)), h = Math.max(1, Math.floor(canvas.clientHeight * r));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function toScreen(x, y) {
    const w = canvas.width, h = canvas.height;
    return [PAD + (x - bounds.minX) / (bounds.maxX - bounds.minX) * (w - 2 * PAD),
      h - PAD - (y - bounds.minY) / (bounds.maxY - bounds.minY) * (h - 2 * PAD)];
  }

  // Density color from the active colormap (alpha rises with t so faint→opaque).
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (cmap.length - 1), i = Math.min(cmap.length - 2, Math.floor(f)), u = f - i;
    const c = (a, b) => Math.round(a + (b - a) * u);
    return `rgba(${c(cmap[i][0], cmap[i + 1][0])},${c(cmap[i][1], cmap[i + 1][1])},${c(cmap[i][2], cmap[i + 1][2])},${0.25 + 0.6 * t})`;
  }

  function drawDensity() {
    const G = 64, grid = new Float32Array(G * G);
    for (const [x, y] of points) {
      const gx = Math.min(G - 1, Math.max(0, Math.floor((x - bounds.minX) / (bounds.maxX - bounds.minX) * G)));
      const gy = Math.min(G - 1, Math.max(0, Math.floor((y - bounds.minY) / (bounds.maxY - bounds.minY) * G)));
      grid[gy * G + gx] += 1;
    }
    for (let p = 0; p < densBandwidth; p++) blur(grid, G);   // wider sigma = more passes
    let max = 0; for (let i = 0; i < grid.length; i++) max = Math.max(max, grid[i]);
    if (max <= 0) return;
    const w = canvas.width, h = canvas.height, cw = (w - 2 * PAD) / G, ch = (h - 2 * PAD) / G;
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const v = grid[gy * G + gx] / max; if (v < 0.02) continue;
      ctx.fillStyle = ramp(v);
      ctx.fillRect(PAD + gx * cw, h - PAD - (gy + 1) * ch, cw + 1, ch + 1);
    }
  }

  function paint(bg) {
    const w = canvas.width, h = canvas.height;
    if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h); } else ctx.clearRect(0, 0, w, h);
    if (showDensity) drawDensity();
    const r = 3 * dpr();
    for (let i = 0; i < points.length; i++) {
      const hl = i === selected || pathSet.has(i) || i === hover;
      if (!hl && pointAlpha <= 0) continue;                 // points hidden
      const [sx, sy] = toScreen(points[i][0], points[i][1]);
      ctx.globalAlpha = hl ? 1 : pointAlpha;
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = i === selected ? "#2dd4bf" : pathSet.has(i) ? "#f472b6" : i === hover ? "#e2e8f0"
        : pointColors ? pointColors[i] : labels ? clusterColor(labels[i]) : "rgba(148,163,184,0.6)";
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    drawFreeCurves();
    drawPath();
    // selection ring must contrast with the background -> dark on the light theme
    const selCol = document.documentElement.dataset.theme === "light" ? "rgba(15,23,42,0.9)" : "rgba(241,245,249,0.9)";
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
  function draw() { resize(); paint(null); if (onRenderCb) onRenderCb(); }
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
  canvas.addEventListener("click", (e) => { const i = nearest(e.clientX, e.clientY); if (i >= 0) onSelect(i, { shift: e.shiftKey }); });
  canvas.addEventListener("mousemove", (e) => { const i = nearest(e.clientX, e.clientY); if (i !== hover) { hover = i; draw(); } });
  canvas.addEventListener("mouseleave", () => { hover = -1; draw(); });
  window.addEventListener("resize", draw);

  return {
    setPoints(p) { points = p; setBounds(); draw(); },
    setLabels(l) { labels = l; draw(); },
    setPointColors(arr) { pointColors = arr; draw(); },   // per-point CSS color (layers mode); null clears
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
    unproject(sx, sy) {
      const r = dpr(), w = canvas.width, h = canvas.height, dx = sx * r, dy = sy * r;
      return [bounds.minX + (dx - PAD) / Math.max(1, w - 2 * PAD) * (bounds.maxX - bounds.minX),
        bounds.minY + (h - PAD - dy) / Math.max(1, h - 2 * PAD) * (bounds.maxY - bounds.minY)];
    },
    setSelection(arr) { selSet = new Set(arr || []); draw(); },
    // CSS-space screen coords of every point (for app-driven lasso enclosure)
    projectedPoints() { const r = dpr(); return points.map(([x, y]) => { const [sx, sy] = toScreen(x, y); return [sx / r, sy / r]; }); },
    draw,
    get selected() { return selected; },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale); canvas.height = Math.floor(canvas.clientHeight * scale);
      paint(transparent ? null : "#020617");
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
