// Metadata (.xmd/.star/.cs) table viewer controller: a windowed table (table.js)
// plus a column plotter (plot.js). Everything loads lazily from the backend —
// row windows, per-row image thumbnails and decimated/​binned plot data — so big
// files open fast and stay light.
import { createTable } from "./table.js";
import { createPlot } from "./plot.js";

const $ = (id) => document.getElementById(id);
const PALETTE = ["#2dd4bf", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#34d399", "#fb7185", "#38bdf8"];

const state = { meta: null, imgSize: 72, ySel: new Set(), mode: "table" };
let table = null;
const plot = createPlot($("plot"));

function status(msg, cls = "") { const el = $("status"); el.textContent = msg; el.title = msg; el.className = "status" + (cls ? " " + cls : ""); }

// ---- theme (mirrors the other viewers) ----
function applyTheme(theme) { document.documentElement.dataset.theme = theme === "light" ? "light" : "dark"; plot.redraw(); }
applyTheme(new URLSearchParams(location.search).get("theme"));
window.addEventListener("message", (e) => { if (e.data && e.data.type === "hax-theme") applyTheme(e.data.theme); });

// ---- boot ----
(async function boot() {
  try {
    status("Parsing metadata…", "busy");
    const meta = await (await fetch("./api/meta")).json();
    state.meta = meta;
    $("title").textContent = meta.title || "Metadata";
    $("meta").textContent = `${meta.n_rows.toLocaleString()} rows · ${meta.columns.length} cols` + (meta.has_images ? " · images" : "");
    if (meta.filename) document.title = "Hax — " + meta.filename.split("/").pop();
    $("rowInfo").textContent = `${meta.n_rows.toLocaleString()} rows × ${meta.columns.length} columns`;

    buildTable();
    buildPlotControls();
    if (meta.has_images) { $("imgToggleWrap").style.display = ""; $("imgSizeWrap").style.display = ""; }
    status("Ready");
  } catch (err) { status("Failed to load metadata: " + err.message, "err"); }
})();

// ---- table ----
function buildTable() {
  const meta = state.meta;
  const imageCol = meta.has_images ? meta.columns.indexOf("image") : -1;
  table = createTable($("tableWrap"), {
    columns: meta.columns,
    nRows: meta.n_rows,
    imageCol,
    getRows: (start, count) => fetch(`./api/rows?start=${start}&count=${count}`).then((r) => r.json()),
    imageURL: (row) => `./api/image?row=${row}&size=${state.imgSize}`,
  });
}

$("renderImages").addEventListener("change", (e) => table && table.setRenderImages(e.target.checked));
$("imgSize").addEventListener("input", (e) => { state.imgSize = +e.target.value; table && table.setImageSize(state.imgSize); });
$("download").addEventListener("click", () => { window.location.href = "./api/download"; });

// ---- plot controls ----
function buildPlotControls() {
  const nums = state.meta.numeric_columns;
  const xsel = $("xcol");
  xsel.innerHTML = "";
  xsel.appendChild(new Option("(row index)", "__index__"));
  for (const c of nums) xsel.appendChild(new Option(c, c));

  const chips = $("ycols");
  chips.innerHTML = "";
  for (const c of nums) {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = c;
    b.onclick = () => { b.classList.toggle("active"); if (state.ySel.has(c)) state.ySel.delete(c); else state.ySel.add(c); };
    chips.appendChild(b);
  }
  if (!nums.length) chips.appendChild(Object.assign(document.createElement("span"), { className: "muted small", textContent: "no numeric columns" }));
  onTypeChange();
}

function onTypeChange() {
  const t = $("plotType").value;
  $("xwrap").style.display = t === "scatter" || t === "line" ? "" : "none";
  $("binswrap").style.display = t === "histogram" ? "" : "none";
}
$("plotType").addEventListener("change", onTypeChange);
$("replot").addEventListener("click", doPlot);
$("resetZoom").addEventListener("click", () => plot.resetZoom());
$("bins").addEventListener("change", () => { if (state.mode === "plot") doPlot(); });
$("savePlot").addEventListener("click", async () => {
  const blob = await plot.snapshot({ scale: 2 }); if (!blob) return;
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = (state.meta.filename ? state.meta.filename.split("/").pop().replace(/\.[^.]+$/, "") : "metadata") + "_plot.png";
  a.click(); URL.revokeObjectURL(a.href);
});

async function doPlot() {
  const t = $("plotType").value;
  const ys = [...state.ySel];
  if (!ys.length) { status("Pick one or more Y columns", "warn"); return; }
  status("Loading plot data…", "busy");
  try {
    if (t === "histogram") {
      const bins = +$("bins").value;
      const hs = await Promise.all(ys.map((c) => fetch(`./api/histogram?name=${encodeURIComponent(c)}&bins=${bins}`).then((r) => r.json())));
      const series = hs.map((h, i) => ({
        name: h.name, color: PALETTE[i % PALETTE.length],
        bars: h.counts.map((c, k) => [h.edges[k], h.edges[k + 1], c]),
      }));
      plot.setSeries("histogram", series, { x: ys.length === 1 ? ys[0] : "value", y: "count" });
      legend(series);
      status(`Histogram · ${hs[0].n.toLocaleString()} values`);
    } else {
      const xcol = $("xcol").value;
      const useIndex = xcol === "__index__";
      const names = [...new Set([...(useIndex ? [] : [xcol]), ...ys])];
      const r = await (await fetch(`./api/columns?names=${encodeURIComponent(names.join(","))}`)).json();
      const xvals = useIndex ? r.indices : r.columns[xcol];
      const series = ys.map((c, i) => ({
        name: c, color: PALETTE[i % PALETTE.length],
        points: r.columns[c].map((v, k) => [xvals[k], v]),
      }));
      plot.setSeries(t, series, { x: useIndex ? "row index" : xcol, y: ys.length === 1 ? ys[0] : "value" });
      legend(series);
      status((r.decimated ? `Showing ${r.indices.length.toLocaleString()} of ${r.total.toLocaleString()} rows (decimated)` : `${r.total.toLocaleString()} rows`));
    }
    $("plotInfo").textContent = "wheel = zoom · drag = pan · double-click = reset";
  } catch (err) { status("Plot failed: " + err.message, "err"); }
}

function legend(series) {
  const el = $("legend"); el.innerHTML = "";
  if (series.length < 2) return;
  for (const s of series) {
    const row = document.createElement("div"); row.className = "leg-item";
    const sw = document.createElement("span"); sw.className = "leg-swatch"; sw.style.background = s.color;
    row.appendChild(sw); row.appendChild(document.createTextNode(s.name));
    el.appendChild(row);
  }
}

// ---- mode toggle ----
for (const b of $("modeSeg").querySelectorAll(".seg-btn")) {
  b.addEventListener("click", () => setMode(b.dataset.mode));
}
function setMode(mode) {
  state.mode = mode;
  for (const b of $("modeSeg").querySelectorAll(".seg-btn")) b.classList.toggle("active", b.dataset.mode === mode);
  const isTable = mode === "table";
  $("tableTools").style.display = isTable ? "" : "none";
  $("plotTools").style.display = isTable ? "none" : "";
  $("tableWrap").style.display = isTable ? "" : "none";
  $("plotWrap").style.display = isTable ? "none" : "flex";
  if (!isTable) plot.redraw();
}
