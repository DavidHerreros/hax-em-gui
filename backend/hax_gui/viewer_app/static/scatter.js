// 2D latent-landscape scatter (Canvas2D): dynamic points, optional cluster
// coloring and density heatmap, click-to-select, shift-click to set a morph
// target, plus high-DPI / transparent snapshots. No dependencies.
//
// Scales to ~1e6 points. Three things make that possible, and all three matter:
//   * points are rasterised into an ImageData above RASTER_MIN, instead of one
//     beginPath/arc/fill per point (a million path ops is seconds per frame);
//   * `nearest()` goes through a uniform grid index rather than scanning every point;
//   * hover and pan redraws are coalesced onto an animation frame.
// Below RASTER_MIN nothing changes — the pretty vector path still draws the discs.

const PALETTE = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f87171",
  "#22d3ee", "#fb923c", "#4ade80", "#e879f9", "#facc15", "#38bdf8",
  "#fca5a5", "#a3e635", "#c084fc", "#2dd4bf", "#fdba74", "#93c5fd",
];
export const clusterColor = (l) => PALETTE[((l % PALETTE.length) + PALETTE.length) % PALETTE.length];

// Above this many points, Canvas2D path ops stop being viable and we rasterise instead.
// At that density the 3px discs are almost pure overdraw anyway, so nothing is lost.
const RASTER_MIN = 20000;
const IDX_G = 256;           // nearest()-index grid resolution (256² cells)
const HOVER_TOL_CSS = 10;    // px; past this the cursor is over empty space, not a point
const CLICK_TOL_CSS = 48;    // px; a click further out than this falls back to a full scan

export function createScatter(canvas, onSelect) {
  const ctx = canvas.getContext("2d");
  let points = [], labels = null, showDensity = false, pointColors = null;
  let selected = -1, hover = -1, pathSet = new Set(), selSet = new Set();
  let pickFilter = null;   // when set, click/hover only consider points where pickFilter(i) is true
  let pathOrder = [], pathColor = "#f472b6";   // ordered morph path drawn as a polyline
  let freeCurves = [];   // [{ pts:[[x,y]...] data coords, color }] free-spline overlays
  let densBandwidth = 6, pointAlpha = 1.0, bgColor = null;   // bgColor set => opaque fill (dark density backdrop)
  let cmap = [[15, 23, 42], [30, 64, 120], [13, 148, 136], [250, 204, 21]];   // density colormap stops
  let bounds = { minX: -1, minY: -1, maxX: 1, maxY: 1 };   // data extent (fit)
  let view = { minX: -1, minY: -1, maxX: 1, maxY: 1 };     // visible window (zoom/pan); == bounds when fit
  const PAD = 26;

  // ── caches ────────────────────────────────────────────────────────────────
  // The density grid used to be rebuilt inside every paint(), and nearest() scanned
  // every point on every mousemove. Both are now computed once and invalidated only
  // when what they depend on actually changes.
  let densCache = null;    // { grid, G, max, bw } — depends on `points` + `bounds` + bandwidth
  let index = null;        // { G, start, items } uniform grid over `bounds`, for nearest()
  let layer = null;        // memoised offscreen points layer (raster path); see pointsLayer
  let drawPending = 0;
  // Bumped by anything that changes how the *base* points look but isn't captured by a
  // scalar in the layer key (the label array, the per-point colours, the colormap).
  let baseVer = 0;

  function invalidatePoints() { densCache = null; index = null; baseVer++; }
  function scheduleDraw() {
    if (drawPending) return;
    drawPending = requestAnimationFrame(() => { drawPending = 0; draw(); });
  }

  function setBounds() {
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity;
    for (const [x, y] of points) { if (x < a) a = x; if (x > c) c = x; if (y < b) b = y; if (y > d) d = y; }
    if (!isFinite(a)) { a = -1; c = 1; b = -1; d = 1; }
    if (c - a < 1e-6) { a -= 1; c += 1; }
    if (d - b < 1e-6) { b -= 1; d += 1; }
    const mx = (c - a) * 0.05, my = (d - b) * 0.05;
    bounds = { minX: a - mx, minY: b - my, maxX: c + mx, maxY: d + my };
    view = { ...bounds };   // new data refits the view
    invalidatePoints();
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
    clampView(); scheduleDraw();
  }
  function panBy(dxCss, dyCss) {
    const r = dpr();
    const ddx = dxCss * r * (view.maxX - view.minX) / Math.max(1, canvas.width - 2 * PAD);
    const ddy = dyCss * r * (view.maxY - view.minY) / Math.max(1, canvas.height - 2 * PAD);
    view = { minX: view.minX - ddx, maxX: view.maxX - ddx, minY: view.minY + ddy, maxY: view.maxY + ddy };
    clampView(); scheduleDraw();
  }
  // Mid-export the canvas is deliberately off its on-screen size, and everything drawn in
  // css px must scale with it — otherwise a 4x PNG comes out with screen-sized (i.e.
  // hairline) dots and text on a huge image.
  let snapping = false;
  const dpr = () => (snapping
    ? canvas.width / Math.max(1, canvas.clientWidth)
    : Math.min(window.devicePixelRatio || 1, 2));
  function resize() {
    if (snapping) return;
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

  // ── color ─────────────────────────────────────────────────────────────────
  // Numeric core of the density ramp. The raster path needs [r,g,b,a] and cannot afford
  // to build (and re-parse) a million `rgba(...)` strings per frame.
  function rampRGBA(t, forceAlpha) {
    t = Math.max(0, Math.min(1, t));
    const f = t * (cmap.length - 1), i = Math.min(cmap.length - 2, Math.floor(f)), u = f - i;
    const c = (a, b) => Math.round(a + (b - a) * u);
    // A dark backdrop (density mode) reads like the desktop app's inferno layer;
    // a light page background needs a higher floor or the colormap washes out.
    const light = !bgColor && document.documentElement.dataset.theme === "light";
    const alpha = forceAlpha != null ? forceAlpha : (light ? 0.5 + 0.48 * t : 0.18 + 0.72 * t);
    return [c(cmap[i][0], cmap[i + 1][0]), c(cmap[i][1], cmap[i + 1][1]), c(cmap[i][2], cmap[i + 1][2]), alpha];
  }
  function ramp(t, forceAlpha) {
    const [r, g, b, a] = rampRGBA(t, forceAlpha);
    return `rgba(${r},${g},${b},${a})`;
  }

  // CSS color -> [r,g,b,a], memoised: the layer/cluster palettes are a handful of
  // distinct strings shared across a million points, so parse each of them once.
  const cssCache = new Map();
  function parseCss(s) {
    let v = cssCache.get(s);
    if (v) return v;
    if (s[0] === "#") {
      const hex = s.length === 4
        ? [s[1] + s[1], s[2] + s[2], s[3] + s[3]]
        : [s.slice(1, 3), s.slice(3, 5), s.slice(5, 7)];
      v = [parseInt(hex[0], 16), parseInt(hex[1], 16), parseInt(hex[2], 16), 1];
    } else {
      const m = s.match(/[\d.]+/g) || [148, 163, 184, 1];
      v = [+m[0], +m[1], +m[2], m[3] != null ? +m[3] : 1];
    }
    cssCache.set(s, v);
    return v;
  }

  const SEL_RGBA = [45, 212, 191, 1], PATH_RGBA = [244, 114, 182, 1], HOVER_RGBA = [226, 232, 240, 1];
  const DEFAULT_RGBA = [148, 163, 184, 0.6];
  // i -> [r,g,b,a] for the *base* layer: the same fillStyle ladder the vector path uses,
  // minus the selected/hover/path cases. Those few points are drawn on top afterwards,
  // which is what lets the base layer survive a hover change (see pointsLayer).
  function baseColor(dg) {
    return (i) => {
      if (pointColors) return parseCss(pointColors[i]);
      if (dg) return rampRGBA(0.32 + 0.68 * densityAt(dg, points[i][0], points[i][1]), 1);
      if (labels) return parseCss(clusterColor(labels[i]));
      return DEFAULT_RGBA;
    };
  }

  // ── density grid ──────────────────────────────────────────────────────────
  // Smoothed point-count grid (shared by the heatmap fill and per-point tinting).
  function densityGrid() {
    if (densCache && densCache.bw === densBandwidth) return densCache;
    const G = 64, grid = new Float32Array(G * G);
    const bw = bounds.maxX - bounds.minX, bh = bounds.maxY - bounds.minY;
    for (const [x, y] of points) {
      const gx = Math.min(G - 1, Math.max(0, Math.floor((x - bounds.minX) / bw * G)));
      const gy = Math.min(G - 1, Math.max(0, Math.floor((y - bounds.minY) / bh * G)));
      grid[gy * G + gx] += 1;
    }
    for (let p = 0; p < densBandwidth; p++) blur(grid, G);   // wider sigma = more passes
    let max = 0; for (let i = 0; i < grid.length; i++) max = Math.max(max, grid[i]);
    densCache = { grid, G, max, bw: densBandwidth };
    return densCache;
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

  // ── points ────────────────────────────────────────────────────────────────
  // Pixel offsets of a filled disc of the given radius, computed once per radius.
  const discCache = new Map();
  function discOffsets(rad) {
    let o = discCache.get(rad);
    if (o) return o;
    const out = [];
    for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
      if (dx * dx + dy * dy <= rad * rad + 0.5) out.push(dx, dy);
    }
    o = Int8Array.from(out);
    discCache.set(rad, o);
    return o;
  }

  // The rasterised base layer, memoised. Splatting into an ImageData costs one write per
  // covered pixel and allocates nothing per point, where the vector path costs a path
  // object and a parsed fillStyle string per point.
  //
  // Crucially the layer depends only on the *base* appearance — points, view, colours,
  // alpha, density — and not on which point is hovered or selected. So moving the mouse
  // re-blits a cached bitmap instead of re-rasterising a million points, which is the
  // difference between a 60 ms hover and a free one. Anything that does change the base
  // appearance changes `key` and the layer is rebuilt.
  function pointsLayer(dg) {
    const w = canvas.width, h = canvas.height, n = points.length;
    const theme = (document.documentElement.dataset || {}).theme || "";
    const key = `${n}|${view.minX},${view.minY},${view.maxX},${view.maxY}|${w}x${h}|${pointAlpha}|`
      + `${showDensity ? densBandwidth : -1}|${baseVer}|${bgColor}|${theme}`;
    if (layer && layer.key === key) return layer.canvas;

    if (!layer || layer.canvas.width !== w || layer.canvas.height !== h) {
      const lc = document.createElement("canvas");
      lc.width = w; lc.height = h;
      layer = { canvas: lc, ctx: lc.getContext("2d"), img: null, key: null };
      layer.img = layer.ctx.createImageData(w, h);
    }
    const buf = layer.img.data;
    buf.fill(0);   // transparent; cheaper than reallocating the ImageData every frame

    // Shrink the disc as N grows: at a million points a 3px disc is nearly pure overdraw,
    // and the splat cost is quadratic in the radius.
    const rad = n > 500000 ? 1 : n > 100000 ? 2 : 3;
    const offs = discOffsets(rad);
    const colOf = baseColor(dg);
    const kx = (w - 2 * PAD) / (view.maxX - view.minX);
    const ky = (h - 2 * PAD) / (view.maxY - view.minY);

    if (pointAlpha > 0) {
      for (let i = 0; i < n; i++) {
        const p = points[i];
        const sx = (PAD + (p[0] - view.minX) * kx) | 0;
        const sy = (h - PAD - (p[1] - view.minY) * ky) | 0;
        if (sx + rad < 0 || sy + rad < 0 || sx - rad >= w || sy - rad >= h) continue;   // off-view
        const c = colOf(i);
        const sa = pointAlpha * c[3];
        if (sa <= 0) continue;
        for (let k = 0; k < offs.length; k += 2) {
          const x = sx + offs[k], y = sy + offs[k + 1];
          if (x < 0 || y < 0 || x >= w || y >= h) continue;
          const o = (y * w + x) << 2;
          // source-over onto the (initially transparent) layer, so overlapping points
          // accumulate the same way the vector path's successive fills would
          const da = buf[o + 3] / 255;
          const oa = sa + da * (1 - sa);
          if (oa <= 0) continue;
          const k1 = da * (1 - sa);
          buf[o]     = (c[0] * sa + buf[o]     * k1) / oa;
          buf[o + 1] = (c[1] * sa + buf[o + 1] * k1) / oa;
          buf[o + 2] = (c[2] * sa + buf[o + 2] * k1) / oa;
          buf[o + 3] = oa * 255;
        }
      }
    }
    layer.ctx.putImageData(layer.img, 0, 0);
    layer.key = key;
    return layer.canvas;
  }

  // The handful of points the base layer deliberately leaves out, drawn on top as proper
  // anti-aliased discs. Always a small set: the selection, the hover, and the morph
  // waypoints.
  function drawHighlights(r) {
    const spec = [];
    for (const i of pathSet) spec.push([i, "#f472b6"]);
    if (hover >= 0) spec.push([hover, "#e2e8f0"]);
    if (selected >= 0) spec.push([selected, "#2dd4bf"]);
    ctx.globalAlpha = 1;
    for (const [i, col] of spec) {
      if (i < 0 || !points[i]) continue;
      const [sx, sy] = toScreen(points[i][0], points[i][1]);
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();
    }
  }

  // The original vector path — kept verbatim for small N, where it looks better
  // (anti-aliased discs) and the cost is irrelevant.
  function vectorPoints(dg, r) {
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
  }

  function paint(bg) {
    const w = canvas.width, h = canvas.height;
    if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h); } else ctx.clearRect(0, 0, w, h);
    const dg = showDensity ? densityGrid() : null;
    if (dg) drawDensity(dg);
    const r = 3 * dpr();
    if (points.length >= RASTER_MIN) { ctx.drawImage(pointsLayer(dg), 0, 0); drawHighlights(r); }
    else vectorPoints(dg, r);
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

  // ── picking ───────────────────────────────────────────────────────────────
  // Uniform grid over `bounds`, in CSR form (start offsets + a point-id array), built
  // once per point set. Replaces the full O(N) scan that used to run on every mousemove.
  function buildIndex() {
    const G = IDX_G, n = points.length;
    const bw = bounds.maxX - bounds.minX, bh = bounds.maxY - bounds.minY;
    const cellOf = new Int32Array(n);
    const start = new Int32Array(G * G + 1);
    for (let i = 0; i < n; i++) {
      const gx = clampi(((points[i][0] - bounds.minX) / bw * G) | 0, G);
      const gy = clampi(((points[i][1] - bounds.minY) / bh * G) | 0, G);
      const c = gy * G + gx;
      cellOf[i] = c; start[c + 1]++;
    }
    for (let c = 0; c < G * G; c++) start[c + 1] += start[c];
    const items = new Int32Array(n);
    const cursor = start.slice(0, G * G);
    for (let i = 0; i < n; i++) items[cursor[cellOf[i]]++] = i;
    index = { G, start, items };
  }

  // Closest point to (cx, cy) within `tolCss` screen px, or -1. Only the grid cells the
  // tolerance disc actually touches get scanned.
  function nearest(cx, cy, tolCss) {
    const n = points.length;
    if (!n) return -1;
    if (!index) buildIndex();
    const rect = canvas.getBoundingClientRect(), r = dpr();
    const csx = cx - rect.left, csy = cy - rect.top;
    const px = csx * r, py = csy * r;
    const [qx, qy] = unproject(csx, csy);
    const tolPx = tolCss * r;
    // the same tolerance expressed in data units, so it tracks zoom
    const tolX = tolPx * (view.maxX - view.minX) / Math.max(1, canvas.width - 2 * PAD);
    const tolY = tolPx * (view.maxY - view.minY) / Math.max(1, canvas.height - 2 * PAD);
    const { G, start, items } = index;
    const bw = bounds.maxX - bounds.minX, bh = bounds.maxY - bounds.minY;
    const gx0 = clampi(((qx - tolX - bounds.minX) / bw * G) | 0, G), gx1 = clampi(((qx + tolX - bounds.minX) / bw * G) | 0, G);
    const gy0 = clampi(((qy - tolY - bounds.minY) / bh * G) | 0, G), gy1 = clampi(((qy + tolY - bounds.minY) / bh * G) | 0, G);
    let best = -1, bestD = tolPx * tolPx;
    for (let gy = gy0; gy <= gy1; gy++) {
      for (let gx = gx0; gx <= gx1; gx++) {
        const c = gy * G + gx;
        for (let k = start[c]; k < start[c + 1]; k++) {
          const i = items[k];
          if (pickFilter && !pickFilter(i)) continue;
          const [sx, sy] = toScreen(points[i][0], points[i][1]);
          const dd = (sx - px) ** 2 + (sy - py) ** 2;
          if (dd < bestD) { bestD = dd; best = i; }
        }
      }
    }
    return best;
  }
  // A click keeps the old "always selects the closest point" contract: try the indexed
  // neighbourhood first and only fall back to a full scan when the click landed in empty
  // space. That scan is now once per click, not once per mouse move.
  function nearestAnywhere(cx, cy) {
    const i = nearest(cx, cy, CLICK_TOL_CSS);
    if (i >= 0 || !points.length) return i;
    const rect = canvas.getBoundingClientRect(), r = dpr();
    const px = (cx - rect.left) * r, py = (cy - rect.top) * r;
    let best = -1, bestD = Infinity;
    for (let i2 = 0; i2 < points.length; i2++) {
      if (pickFilter && !pickFilter(i2)) continue;
      const [sx, sy] = toScreen(points[i2][0], points[i2][1]);
      const dd = (sx - px) ** 2 + (sy - py) ** 2; if (dd < bestD) { bestD = dd; best = i2; }
    }
    return best;
  }

  // click selects (suppressed if the pointer was a pan-drag); drag pans; wheel zooms
  // toward the cursor; double-click refits. (When the lasso/draw overlay is armed it
  // sits on top and intercepts these — the app forwards the wheel there instead.)
  let panning = false, panned = false, plx = 0, ply = 0;
  let hoverPending = 0, hx = 0, hy = 0;
  canvas.addEventListener("pointerdown", (e) => { if (e.button === 1) e.preventDefault(); panning = true; panned = false; plx = e.clientX; ply = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
  canvas.addEventListener("pointermove", (e) => {
    if (!panning) return;
    const dx = e.clientX - plx, dy = e.clientY - ply;
    if (Math.abs(dx) + Math.abs(dy) > 3) panned = true;
    if (panned) panBy(dx, dy);   // coalesced onto an animation frame
    plx = e.clientX; ply = e.clientY;
  });
  canvas.addEventListener("pointerup", (e) => { panning = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  // Always report the click (with coords) even when no base point is hit, so the app can
  // still resolve it to a centroid marker (which this renderer can't pick).
  canvas.addEventListener("click", (e) => { if (panned) return; const i = nearestAnywhere(e.clientX, e.clientY); onSelect(i, { shift: e.shiftKey, cx: e.clientX, cy: e.clientY }); });
  // Hover is coalesced onto an animation frame: mousemove fires far faster than we can
  // usefully repaint, and every repaint at 1e6 points is real work.
  canvas.addEventListener("mousemove", (e) => {
    if (panning) return;
    hx = e.clientX; hy = e.clientY;
    if (hoverPending) return;
    hoverPending = requestAnimationFrame(() => {
      hoverPending = 0;
      const i = nearest(hx, hy, HOVER_TOL_CSS);
      if (i !== hover) { hover = i; draw(); }
    });
  });
  canvas.addEventListener("mouseleave", () => { if (hover !== -1) { hover = -1; scheduleDraw(); } });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", () => { view = { ...bounds }; scheduleDraw(); });
  window.addEventListener("resize", scheduleDraw);
  // Same reason as in plot.js: a canvas built while its tab is hidden has no size, so
  // whatever we drew went nowhere. Redraw when the box we actually occupy changes.
  if (window.ResizeObserver) new ResizeObserver(() => scheduleDraw()).observe(canvas);

  return {
    setPoints(p) { points = p; setBounds(); draw(); },
    setPickFilter(fn) { pickFilter = typeof fn === "function" ? fn : null; },   // restrict click/hover to points where fn(i) is true
    setLabels(l) { labels = l; baseVer++; scheduleDraw(); },
    setPointColors(arr) { pointColors = arr; baseVer++; scheduleDraw(); },   // per-point CSS color (layers mode); null clears
    setBackground(css) { bgColor = css || null; scheduleDraw(); },   // opaque dark fill in density mode; null = transparent (theme)
    setDensity(on) { showDensity = on; baseVer++; scheduleDraw(); },
    setDensityBandwidth(b) { densBandwidth = Math.max(1, b | 0); if (showDensity) scheduleDraw(); },
    setColormap(stops) { if (stops && stops.length >= 2) cmap = stops; baseVer++; if (showDensity) scheduleDraw(); },
    setPointAlpha(a) { pointAlpha = Math.max(0, Math.min(1, a)); scheduleDraw(); },
    select(i) { selected = i; scheduleDraw(); },
    setPath(arr) { pathSet = new Set(arr || []); scheduleDraw(); },
    setPathOrder(arr, color) { pathOrder = arr || []; if (color) pathColor = color; scheduleDraw(); },
    setFreeCurves(list) { freeCurves = list || []; scheduleDraw(); },
    onRender(cb) { onRenderCb = cb; },   // called after every draw (overlay can re-sync to the new projection)
    // Project / unproject between data coords and CSS-space screen px (2D is invertible).
    projectData(pts) { const r = dpr(); return pts.map(([x, y]) => { const [sx, sy] = toScreen(x, y); return [sx / r, sy / r]; }); },
    unproject,
    zoomAt,   // app forwards the wheel here when the overlay is armed
    panBy,    // app forwards middle-mouse drags here when the overlay is armed (CSS px)
    resetView() { view = { ...bounds }; scheduleDraw(); },
    setSelection(arr) { selSet = new Set(arr || []); scheduleDraw(); },
    // CSS-space screen coords of every point (for app-driven lasso enclosure)
    projectedPoints() { const r = dpr(); return points.map(([x, y]) => { const [sx, sy] = toScreen(x, y); return [sx / r, sy / r]; }); },
    draw,
    get selected() { return selected; },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      snapping = true;
      canvas.width = Math.max(1, Math.floor(canvas.clientWidth * scale));
      canvas.height = Math.max(1, Math.floor(canvas.clientHeight * scale));
      paint(transparent ? null : (bgColor || "#020617"));
      const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
      snapping = false;
      canvas.width = ow; canvas.height = oh; draw();
      return blob;
    },
  };
}

const clampi = (v, G) => (v < 0 ? 0 : v >= G ? G - 1 : v);

function blur(grid, G) {
  const tmp = new Float32Array(grid.length), k = [0.25, 0.5, 0.25];
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { let s = 0; for (let d = -1; d <= 1; d++) { const xx = Math.min(G - 1, Math.max(0, x + d)); s += grid[y * G + xx] * k[d + 1]; } tmp[y * G + x] = s; }
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) { let s = 0; for (let d = -1; d <= 1; d++) { const yy = Math.min(G - 1, Math.max(0, y + d)); s += tmp[yy * G + x] * k[d + 1]; } grid[y * G + x] = s; }
}
