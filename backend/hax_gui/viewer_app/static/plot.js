// Minimal 2D plotter (line / scatter / histogram) on a canvas, with wheel-zoom,
// drag-pan and reset. No dependencies. Series carry either `points` ([[x,y],…])
// for line/scatter or `bars` ([[x0,x1,h],…]) for histograms, plus a name+color.

const MARGIN = { l: 58, r: 16, t: 14, b: 34 };

export function createPlot(canvas) {
  const ctx = canvas.getContext("2d");
  let kind = "scatter";
  let series = [];             // [{ name, color, points?|bars? }]
  let bounds = null;           // data extent {x0,x1,y0,y1}
  let view = null;             // current (zoomed/panned) extent
  let labels = { x: "", y: "" };

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
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag, step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return out;
  }
  function fmtNum(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
    return (Math.round(v * 1000) / 1000).toString();
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

    // grid + ticks
    ctx.lineWidth = r; ctx.font = `${11 * r}px system-ui, sans-serif`; ctx.textBaseline = "middle";
    ctx.strokeStyle = grid; ctx.fillStyle = muted;
    ctx.textAlign = "right";
    for (const ty of niceTicks(view.y0, view.y1, 6)) {
      const py = sy(ty);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(a.l, py); ctx.lineTo(a.l + a.w, py); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmtNum(ty), a.l - 6 * r, py);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tx of niceTicks(view.x0, view.x1, 7)) {
      const px = sx(tx);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(px, a.t); ctx.lineTo(px, a.t + a.h); ctx.stroke();
      ctx.globalAlpha = 1; ctx.fillText(fmtNum(tx), px, a.t + a.h + 6 * r);
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
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale); canvas.height = Math.floor(canvas.clientHeight * scale);
      draw();
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      canvas.width = ow; canvas.height = oh; draw();
      return blob;
    },
  };
}
