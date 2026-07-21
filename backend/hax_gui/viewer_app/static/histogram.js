// ChimeraX-style contour widget: draws the volume's value histogram (log counts)
// with a draggable vertical threshold line. Emits the chosen absolute level.
// No dependencies.
//
// `onLevel(level, dragging)` — `dragging` is true for every frame of a drag and false for
// the last one, when the pointer is released. Callers use it to show a cheap draft surface
// while the pointer is down and compute the full-resolution one only once, on release.
export function createHistogram(canvas, onLevel) {
  const ctx = canvas.getContext("2d");
  let data = null;   // { counts, edges, min, max, mean, std }
  let level = 0;
  let dragging = false;

  function dpr() { return Math.min(window.devicePixelRatio || 1, 2); }
  function resize() {
    const r = dpr();
    const w = Math.max(1, Math.floor(canvas.clientWidth * r)), h = Math.max(1, Math.floor(canvas.clientHeight * r));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  const PAD = 2;
  function xOf(v) {
    if (!data || data.max === data.min) return PAD;
    return PAD + (v - data.min) / (data.max - data.min) * (canvas.width - 2 * PAD);
  }
  function vOf(px) {
    if (!data || data.max === data.min) return data ? data.min : 0;
    return data.min + (px - PAD) / (canvas.width - 2 * PAD) * (data.max - data.min);
  }

  function draw() {
    resize();
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (!data) return;
    const n = data.counts.length;
    const maxc = Math.log1p(Math.max(1, ...data.counts));
    const bw = (w - 2 * PAD) / n;
    ctx.fillStyle = "rgba(148,163,184,0.45)";
    for (let i = 0; i < n; i++) {
      const bh = (Math.log1p(data.counts[i]) / maxc) * (h - 4);
      ctx.fillRect(PAD + i * bw, h - bh, Math.max(1, bw - 0.5), bh);
    }
    // threshold line + filled region to the right (kept density)
    const lx = xOf(level);
    ctx.fillStyle = "rgba(45,212,191,0.10)";
    ctx.fillRect(lx, 0, w - lx, h);
    ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 2 * dpr();
    ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, h); ctx.stroke();
    // handle
    ctx.fillStyle = "#2dd4bf";
    ctx.beginPath(); ctx.arc(lx, 8 * dpr(), 4 * dpr(), 0, Math.PI * 2); ctx.fill();
  }

  function pick(clientX, live) {
    const rect = canvas.getBoundingClientRect();
    const px = (clientX - rect.left) * dpr();
    level = Math.max(data.min, Math.min(data.max, vOf(px)));
    draw();
    onLevel(level, live);
  }
  canvas.addEventListener("pointerdown", (e) => { if (!data) return; dragging = true; canvas.setPointerCapture(e.pointerId); pick(e.clientX, true); });
  canvas.addEventListener("pointermove", (e) => { if (dragging) pick(e.clientX, true); });
  canvas.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
    onLevel(level, false);        // settled: the caller now computes the full-res surface
  });
  window.addEventListener("resize", draw);

  return {
    setData(d, lvl) { data = d; level = lvl != null ? lvl : (d.mean + 2.5 * d.std); draw(); },
    setLevel(lvl) { level = lvl; draw(); },
    get level() { return level; },
    get sigma() { return data ? (level - data.mean) / (data.std || 1) : 2.5; },
  };
}
