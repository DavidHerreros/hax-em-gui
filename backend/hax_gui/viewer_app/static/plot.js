// Minimal 2D plotter (line / scatter / histogram) on a canvas, with wheel-zoom,
// drag-pan, reset and a hover readout. No dependencies. Series carry either
// `points` ([[x,y],…]) for line/scatter or `bars` ([[x0,x1,h],…]) for histograms,
// plus a name+color.

const MARGIN = { l: 64, r: 16, t: 14, b: 34 };

// The hover box reports an actual datum, so it needs real precision — the tick
// formatters deliberately round to the grid spacing and would flatten a whole curve
// to the same string.
function fmtValue(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e7 || a < 1e-4) return v.toExponential(3);
  return String(+v.toPrecision(6));
}

export function createPlot(canvas) {
  const ctx = canvas.getContext("2d");
  let kind = "scatter";
  let series = [];             // [{ name, color, points?|bars? }]
  let bounds = null;           // data extent {x0,x1,y0,y1}
  let view = null;             // current (zoomed/panned) extent
  let labels = { x: "", y: "" };
  let cursor = null;           // {px, py} device px — drives the hover readout

  function dpr() { return Math.min(window.devicePixelRatio || 1, 2); }
  function resize() {
    const r = dpr();
    const w = Math.max(1, Math.floor(canvas.clientWidth * r)), h = Math.max(1, Math.floor(canvas.clientHeight * r));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  function area() {
    const r = dpr();
    return { l: MARGIN.l * r, t: MARGIN.t * r, w: canvas.width - (MARGIN.l + MARGIN.r) * r, h: canvas.height - (MARGIN.t + MARGIN.b) * r };
  }

  function computeBounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of series) {
      if (s.points) for (const [x, y] of s.points) {
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      if (s.bars) for (const [a, b, hgt] of s.bars) {
        if (a < x0) x0 = a; if (b > x1) x1 = b; if (0 < y0) y0 = 0; if (hgt > y1) y1 = hgt;
      }
    }
    if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }
    if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
    const px = (x1 - x0) * 0.03, py = (y1 - y0) * 0.05;
    return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
  }

  const sx = (x) => { const a = area(); return a.l + (x - view.x0) / (view.x1 - view.x0) * a.w; };
  const sy = (y) => { const a = area(); return a.t + (1 - (y - view.y0) / (view.y1 - view.y0)) * a.h; };
  const ix = (px) => { const a = area(); return view.x0 + (px - a.l) / a.w * (view.x1 - view.x0); };
  const iy = (py) => { const a = area(); return view.y0 + (1 - (py - a.t) / a.h) * (view.y1 - view.y0); };

  function niceTicks(lo, hi, n) {
    const span = hi - lo, step0 = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag, step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
    return { ticks, step };
  }

  // How many decimals the tick *spacing* needs. A fixed precision is what makes six
  // ticks across 0.9985–0.9995 all print as "0.999" — the labels have to resolve the
  // step, not the magnitude.
  function decimalsFor(step) {
    if (!(step > 0)) return 0;
    return Math.max(0, Math.min(10, Math.ceil(-Math.log10(step))));
  }

  // A formatter for one axis, given its tick step and the magnitude it has to cover.
  // Big step counts get SI suffixes (10k, 1.5M) instead of "1.0e+4" strings that are
  // both unreadable and wide enough to collide with their neighbours.
  function makeFmt(step, lo, hi) {
    const mag = Math.max(Math.abs(lo), Math.abs(hi));
    if (mag >= 1e4 && mag < 1e12) {
      const [div, suffix] = mag >= 1e9 ? [1e9, "G"] : mag >= 1e6 ? [1e6, "M"] : [1e3, "k"];
      const d = decimalsFor(step / div);
      return (v) => (v === 0 ? "0" : (v / div).toFixed(Math.min(d, 3)) + suffix);
    }
    const d = decimalsFor(step);
    if (d > 6 || (mag > 0 && mag < 1e-4)) return (v) => (v === 0 ? "0" : v.toExponential(1));
    return (v) => v.toFixed(d);
  }

  function draw() {
    resize();
    const r = dpr(), a = area();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!view) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--text").trim() || "#e5e7eb";
    const grid = css.getPropertyValue("--border").trim() || "#334155";
    const muted = css.getPropertyValue("--muted").trim() || "#94a3b8";

    // grid + ticks. Tick counts follow the plot's actual size, so a narrow card asks for
    // fewer labels instead of crowding them into each other.
    const nx = Math.max(2, Math.min(8, Math.round(a.w / (92 * r))));
    const ny = Math.max(2, Math.min(8, Math.round(a.h / (44 * r))));
    const yT = niceTicks(view.y0, view.y1, ny);
    const xT = niceTicks(view.x0, view.x1, nx);
    const fmtY = makeFmt(yT.step, view.y0, view.y1);
    const fmtX = makeFmt(xT.step, view.x0, view.x1);

    ctx.lineWidth = r; ctx.font = `${11 * r}px system-ui, sans-serif`; ctx.textBaseline = "middle";
    ctx.strokeStyle = grid; ctx.fillStyle = muted;
    ctx.textAlign = "right";
    for (const ty of yT.ticks) {
      const py = sy(ty);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(a.l, py); ctx.lineTo(a.l + a.w, py); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmtY(ty), a.l - 6 * r, py);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tx of xT.ticks) {
      const px = sx(tx);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(px, a.t); ctx.lineTo(px, a.t + a.h); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmtX(tx), px, a.t + a.h + 6 * r);
    }
    // frame
    ctx.strokeStyle = grid; ctx.globalAlpha = 1;
    ctx.strokeRect(a.l, a.t, a.w, a.h);

    // clip to the plot area for the data
    ctx.save();
    ctx.beginPath(); ctx.rect(a.l, a.t, a.w, a.h); ctx.clip();
    for (const s of series) {
      ctx.fillStyle = s.color; ctx.strokeStyle = s.color;
      if (kind === "histogram" && s.bars) {
        ctx.globalAlpha = series.length > 1 ? 0.55 : 0.85;
        for (const [x0, x1, hgt] of s.bars) {
          const px = sx(x0), pw = Math.max(1, sx(x1) - sx(x0)), py = sy(hgt), pb = sy(0);
          ctx.fillRect(px, py, pw, pb - py);
        }
        ctx.globalAlpha = 1;
      } else if (kind === "line" && s.points) {
        ctx.lineWidth = 1.5 * r; ctx.beginPath();
        let started = false;
        for (const [x, y] of s.points) {
          if (y == null || !isFinite(y)) { started = false; continue; }
          const px = sx(x), py = sy(y);
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      } else if (s.points) { // scatter
        const rad = 2.2 * r;
        for (const [x, y] of s.points) {
          if (y == null || !isFinite(y)) continue;
          ctx.beginPath(); ctx.arc(sx(x), sy(y), rad, 0, 6.2832); ctx.fill();
        }
      }
    }
    ctx.restore();

    // axis labels
    ctx.fillStyle = fg; ctx.globalAlpha = 1;
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.font = `${12 * r}px system-ui, sans-serif`;
    if (labels.x) ctx.fillText(labels.x, a.l + a.w / 2, canvas.height - 2 * r);
    if (labels.y) {
      ctx.save(); ctx.translate(12 * r, a.t + a.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "top"; ctx.fillText(labels.y, 0, 0); ctx.restore();
    }

    drawHover(a, r, fg, muted, grid);
  }

  // ---- hover readout -----------------------------------------------------
  // What every series is worth at the x under the cursor: a guide line, a dot on each
  // series, and a box listing them — the thing TensorBoard gives you and a bare canvas
  // does not.
  function hoverRows(dataX) {
    const rows = [];
    for (const s of series) {
      if (s.bars) {
        const b = s.bars.find(([x0, x1]) => dataX >= x0 && dataX <= x1);
        if (b) rows.push({ name: s.name, color: s.color, x: (b[0] + b[1]) / 2, y: b[2], bar: b });
        continue;
      }
      if (!s.points || !s.points.length) continue;
      let best = null, bestD = Infinity;
      for (const p of s.points) {
        if (p[1] == null || !isFinite(p[1])) continue;
        const d = Math.abs(p[0] - dataX);
        if (d < bestD) { bestD = d; best = p; }
      }
      if (best) rows.push({ name: s.name, color: s.color, x: best[0], y: best[1] });
    }
    return rows;
  }

  function drawHover(a, r, fg, muted, grid) {
    if (!cursor || drag || !series.length) return;
    if (cursor.px < a.l || cursor.px > a.l + a.w || cursor.py < a.t || cursor.py > a.t + a.h) return;

    const rows = hoverRows(ix(cursor.px));
    if (!rows.length) return;

    // Snap the guide to the series point nearest the cursor, not to the raw cursor x.
    const anchor = rows.reduce((m, q) => (Math.abs(sx(q.x) - cursor.px) < Math.abs(sx(m.x) - cursor.px) ? q : m), rows[0]);
    const gx = sx(anchor.x);

    ctx.save();
    ctx.beginPath(); ctx.rect(a.l, a.t, a.w, a.h); ctx.clip();
    ctx.strokeStyle = grid; ctx.globalAlpha = 0.9; ctx.lineWidth = r;
    ctx.setLineDash([3 * r, 3 * r]);
    ctx.beginPath(); ctx.moveTo(gx, a.t); ctx.lineTo(gx, a.t + a.h); ctx.stroke();
    ctx.setLineDash([]);
    for (const q of rows) {
      if (q.bar) continue;
      ctx.fillStyle = q.color; ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(sx(q.x), sy(q.y), 3.4 * r, 0, 6.2832); ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.lineWidth = r;
      ctx.stroke();
    }
    ctx.restore();

    // -- the box
    const pad = 7 * r, lh = 15 * r, sw = 8 * r;
    const named = rows.some((q) => q.name);
    ctx.font = `${11 * r}px system-ui, sans-serif`;
    const head = `${labels.x || "x"} ${fmtValue(anchor.x)}`;
    let w = ctx.measureText(head).width;
    const texts = rows.map((q) => {
      const t = `${q.name ? q.name + "  " : ""}${fmtValue(q.y)}`;
      w = Math.max(w, ctx.measureText(t).width + (named ? sw + 6 * r : 0));
      return t;
    });
    const bw = w + pad * 2, bh = pad * 2 + lh * (rows.length + 1);

    let bx = gx + 14 * r, by = cursor.py - bh / 2;
    if (bx + bw > a.l + a.w) bx = gx - 14 * r - bw;          // flip to the other side
    bx = Math.max(a.l + 2 * r, Math.min(bx, a.l + a.w - bw - 2 * r));
    by = Math.max(a.t + 2 * r, Math.min(by, a.t + a.h - bh - 2 * r));

    const css = getComputedStyle(document.documentElement);
    ctx.globalAlpha = 1;
    ctx.fillStyle = css.getPropertyValue("--panel").trim() || "#0f172a";
    ctx.strokeStyle = css.getPropertyValue("--border-2").trim() || grid;
    ctx.lineWidth = r;
    if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 8 * r); ctx.fill(); ctx.stroke(); }
    else { ctx.fillRect(bx, by, bw, bh); ctx.strokeRect(bx, by, bw, bh); }

    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillStyle = muted;
    ctx.fillText(head, bx + pad, by + pad + lh / 2);
    rows.forEach((q, i) => {
      const y = by + pad + lh * (i + 1) + lh / 2;
      let tx = bx + pad;
      if (named) {
        ctx.fillStyle = q.color;
        ctx.fillRect(tx, y - sw / 2, sw, sw);
        tx += sw + 6 * r;
      }
      ctx.fillStyle = fg;
      ctx.fillText(texts[i], tx, y);
    });
  }

  // ---- interaction ----
  canvas.addEventListener("wheel", (e) => {
    if (!view) return;
    e.preventDefault();
    const f = e.deltaY < 0 ? 0.85 : 1.176;   // in / out
    const mx = ix(e.offsetX * dpr()), my = iy(e.offsetY * dpr());
    view = {
      x0: mx + (view.x0 - mx) * f, x1: mx + (view.x1 - mx) * f,
      y0: my + (view.y0 - my) * f, y1: my + (view.y1 - my) * f,
    };
    draw();
  }, { passive: false });

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.offsetX, y: e.offsetY, v: { ...view } }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || !view) return;
    const a = area();
    const dx = (e.offsetX - drag.x) * dpr() / a.w * (drag.v.x1 - drag.v.x0);
    const dy = (e.offsetY - drag.y) * dpr() / a.h * (drag.v.y1 - drag.v.y0);
    view = { x0: drag.v.x0 - dx, x1: drag.v.x1 - dx, y0: drag.v.y0 + dy, y1: drag.v.y1 + dy };
    draw();
  });
  canvas.addEventListener("pointerup", (e) => { drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch {} });
  canvas.addEventListener("dblclick", () => resetZoom());
  window.addEventListener("resize", draw);

  // A canvas laid out inside a hidden container has clientWidth 0, so anything drawn
  // into it lands in a 1px bitmap and is effectively lost. That happens whenever a plot
  // is built on a tab that isn't on screen yet. Watch our own box instead of trusting
  // callers to redraw us at the right moment.
  if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(canvas);

  // Hover readout. Skipped while panning — you're moving the plot, not reading it.
  canvas.addEventListener("mousemove", (e) => {
    if (drag) return;
    cursor = { px: e.offsetX * dpr(), py: e.offsetY * dpr() };
    draw();
  });
  canvas.addEventListener("mouseleave", () => { if (cursor) { cursor = null; draw(); } });

  function resetZoom() { view = bounds ? { ...bounds } : null; draw(); }

  return {
    setSeries(k, ss, lbls) {
      kind = k; series = ss || []; labels = lbls || { x: "", y: "" };
      bounds = computeBounds(); view = { ...bounds };
      draw();
    },
    resetZoom,
    redraw: draw,
    async snapshot({ scale = 2 } = {}) {
      const ow = canvas.width, oh = canvas.height, oc = cursor;
      cursor = null;   // an exported chart must not carry whatever the mouse was hovering
      canvas.width = Math.floor(canvas.clientWidth * scale); canvas.height = Math.floor(canvas.clientHeight * scale);
      draw();
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      canvas.width = ow; canvas.height = oh; cursor = oc; draw();
      return blob;
    },
  };
}
