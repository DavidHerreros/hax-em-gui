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
// The rail holds the type toggle, X select and a searchable Y checklist; every change
// re-plots automatically (debounced). Y is multi-select, and each picked column shows
// its series colour in the list, so the checklist doubles as the legend.
const yItems = new Map();   // column name -> { row, sw }, for in-place refresh
let plotTimer = null;
let hasSeries = false;

function buildPlotControls() {
  const nums = state.meta.numeric_columns;

  const xsel = $("xcol");
  xsel.innerHTML = "";
  xsel.appendChild(new Option("(row index)", "__index__"));
  for (const c of nums) xsel.appendChild(new Option(c, c));

  const list = $("ycols");
  list.innerHTML = "";
  yItems.clear();
  if (!nums.length) {
    list.appendChild(Object.assign(document.createElement("div"), { className: "ynote", textContent: "no numeric columns" }));
  } else {
    for (const c of nums) {
      const row = document.createElement("div"); row.className = "yitem";
      const sw = document.createElement("span"); sw.className = "yitem-sw";
      const nm = document.createElement("span"); nm.className = "yitem-name"; nm.textContent = c;
      row.append(sw, nm);
      row.addEventListener("click", () => {
        if (state.ySel.has(c)) state.ySel.delete(c); else state.ySel.add(c);
        refreshY(); schedulePlot();
      });
      list.appendChild(row);
      yItems.set(c, { row, sw });
    }
    state.ySel.add(nums[0]);   // start on the first column so Plot mode isn't blank
  }

  $("plotInfo").textContent = "wheel = zoom · drag = pan · double-click = reset";
  refreshY();
  onTypeChange();
}

// Reflect the selection into the checklist: on-state + each picked column's series
// colour (colour == its position in the selection, matching doPlot's PALETTE index).
function refreshY() {
  const order = [...state.ySel];
  for (const [c, { row, sw }] of yItems) {
    const on = state.ySel.has(c);
    row.classList.toggle("on", on);
    sw.style.background = on ? PALETTE[order.indexOf(c) % PALETTE.length] : "";
  }
  $("ycount").textContent = state.ySel.size ? `${state.ySel.size} selected` : "";
  applyYFilter();
  updateCaption();
}

function applyYFilter() {
  const q = $("yfilter").value.trim().toLowerCase();
  for (const [c, { row }] of yItems) row.style.display = (!q || c.toLowerCase().includes(q)) ? "" : "none";
}

function currentType() { return $("plotType").querySelector(".type-btn.active").dataset.type; }

function onTypeChange() {
  const t = currentType();
  $("xgrp").style.display = t === "histogram" ? "none" : "";   // X is meaningless for a histogram
  $("binsgrp").style.display = t === "histogram" ? "" : "none";
  updateCaption();
}

function updateCaption() {
  const t = currentType();
  const yl = state.ySel.size ? [...state.ySel].join(", ") : "—";
  const el = $("plotCaption"); el.innerHTML = "";
  const strong = (txt) => { const b = document.createElement("b"); b.textContent = txt; return b; };
  if (t === "histogram") {
    el.append(strong(yl), document.createTextNode(" distribution"));
  } else {
    const xl = $("xcol").value === "__index__" ? "row index" : $("xcol").value;
    el.append(strong(yl), document.createTextNode(" vs "), strong(xl));
  }
}

// Re-plot after a short quiet period so rapid changes (toggling several Y columns,
// sliding bins) collapse into one backend fetch.
function schedulePlot() {
  updateCaption();
  if (state.mode !== "plot") return;
  clearTimeout(plotTimer);
  status("Updating…", "busy");
  plotTimer = setTimeout(doPlot, 260);
}

$("plotType").addEventListener("click", (e) => {
  const b = e.target.closest(".type-btn"); if (!b) return;
  for (const el of $("plotType").children) el.classList.toggle("active", el === b);
  onTypeChange(); schedulePlot();
});
$("xcol").addEventListener("change", () => { updateCaption(); schedulePlot(); });
$("yfilter").addEventListener("input", applyYFilter);
$("yclear").addEventListener("click", () => { state.ySel.clear(); refreshY(); schedulePlot(); });
$("bins").addEventListener("input", () => { $("binsVal").textContent = $("bins").value; });
$("bins").addEventListener("change", schedulePlot);
$("resetZoom").addEventListener("click", () => plot.resetZoom());
$("savePlot").addEventListener("click", async () => {
  const blob = await plot.snapshot({ scale: 2 }); if (!blob) return;
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = (state.meta.filename ? state.meta.filename.split("/").pop().replace(/\.[^.]+$/, "") : "metadata") + "_plot.png";
  a.click(); URL.revokeObjectURL(a.href);
});

async function doPlot() {
  const t = currentType();
  const ys = [...state.ySel];
  if (!ys.length) { plot.setSeries(t, [], { x: "", y: "" }); legend([]); hasSeries = false; status("Pick a Y column", "warn"); return; }
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
    hasSeries = true;
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
  // Table has its own toolbar; the plot controls live inside plotWrap (the rail), so
  // only the table toolbar toggles here.
  $("tableTools").style.display = isTable ? "" : "none";
  $("tableWrap").style.display = isTable ? "" : "none";
  $("plotWrap").style.display = isTable ? "none" : "flex";
  if (!isTable) {
    plot.redraw();
    if (!hasSeries && state.ySel.size) schedulePlot();   // first entry: draw the preselected column
  }
}
