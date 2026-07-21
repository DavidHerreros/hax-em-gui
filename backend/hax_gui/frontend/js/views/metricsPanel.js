// One run's metrics, drawn natively from its TensorBoard event files (no iframe).
//
// Three sub-views over the same log directory:
//   • Scalars — a card per tag, a series per run (which is also how add_scalars lands),
//     with TB-style EMA smoothing, log-y and step/wall-time axes.
//   • Images — hax rewrites the same image tags every epoch (volume slices, predicted
//     batches, angular distribution), so they are shown as a step slider you can play
//     through: training as a movie, instead of TB's per-step accordion.
//   • Latent space — the projector, replaced. The per-epoch tensors.tsv is projected to
//     2D or 3D in the browser and drawn with the same scatter / WebGL point cloud the
//     landscape viewer uses. It is re-read on every request, so new epochs appear on
//     their own — the exact thing TB's projector cannot do for a JAX run.
//
// The plotter, the scatter and the point cloud are the viewer app's own modules, served
// at /shared, so there is one canonical copy of each rather than a fork drifting out of
// sync — and a latent space looks the same here as it does in the full landscape.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";
import { pca } from "../pca.js";
import { launchViewerForJob } from "./viewerLaunch.js";
import { createPlot } from "/shared/plot.js";
import { createScatter } from "/shared/scatter.js";
import { createCloud } from "/shared/cloud.js";

const PALETTE = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f87171", "#22d3ee", "#fb923c"];
const POLL_MS = 5000;
const FRAME_MS = 500;

const colorAt = (i) => PALETTE[i % PALETTE.length];
// cloud.js wants float RGB in 0..1, not a CSS colour string.
const hexToRgb01 = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// A run is the directory holding the event file. add_scalars gives each key a folder of
// its own under the tag ("Reconstruction loss (HetSIREN)/train"), so the last component
// is the series name ("train"). For plain add_scalar the run folder is the writer's
// timestamp, which is what distinguishes a resumed run's curves from the original's.
function runLabel(run, tag) {
  const base = (run || ".").split("/").pop();
  const prefix = tag.replace(/\//g, "_") + "_";
  if (base.startsWith(prefix)) return base.slice(prefix.length) || base;
  return base === "." ? "run" : base;
}

// TensorBoard's exponential moving average, debiased so the curve starts at the data
// rather than at zero.
function smooth(points, weight) {
  if (!weight) return points;
  let acc = 0, norm = 0;
  return points.map(([x, y]) => {
    if (y == null || !isFinite(y)) return [x, null];
    acc = acc * weight + (1 - weight) * y;
    norm = norm * weight + (1 - weight);
    return [x, acc / norm];
  });
}

export function metricsPanel({ logdir, jobId }) {
  const root = h("div", { class: "mx" });
  const body = h("div", { class: "mx-body" });

  let manifest = null;
  let tab = "scalars";
  let live = true;
  let timer = null;
  let destroyed = false;

  // Export settings live on the panel so a plot and the latent landscape come out of the
  // same run looking alike — same idea (and same wording) as the annotate viewer's.
  const exportOpts = { scale: 2, transparent: false };
  const dpiSel = h("select", {
    class: "input mx-select", title: "Image scale / DPI",
    onchange: (e) => (exportOpts.scale = +e.target.value),
  }, ...[1, 2, 3, 4].map((s) => h("option", { value: String(s), selected: s === 2 }, `${s}×`)));
  const transpBox = h("input", {
    type: "checkbox",
    onchange: (e) => (exportOpts.transparent = e.target.checked),
  });

  // -- head: sub-tabs + export + live toggle + the TensorBoard escape hatch --
  const subtabs = h("div", { class: "mx-subtabs" });
  const liveBox = h("input", { type: "checkbox", checked: true, onchange: (e) => { live = e.target.checked; schedule(); } });
  const tbBtn = h("button", {
    class: "btn ghost", title: "Open this run in real TensorBoard, in a new browser tab",
    onclick: openTensorBoard,
  }, icon("external", 15), "TensorBoard");

  root.appendChild(h("div", { class: "mx-head" },
    subtabs,
    h("div", { class: "mx-spacer" }),
    h("label", { class: "mx-live", title: "Resolution of saved PNGs" }, "Export", dpiSel),
    h("label", { class: "mx-live", title: "Save with no background, to drop onto a slide or a figure" }, transpBox, "Transparent"),
    h("label", { class: "mx-live", title: "Re-read the event files every few seconds" }, liveBox, "Live"),
    tbBtn));
  root.appendChild(body);

  function renderSubtabs() {
    clear(subtabs);
    const defs = [
      { key: "scalars", label: "Scalars", ic: "chart", n: manifest?.scalars.length || 0 },
      { key: "images", label: "Images", ic: "image", n: manifest?.images.length || 0 },
      { key: "latent", label: "Latent space", ic: "scatter", n: manifest?.embeddings.length || 0 },
    ];
    for (const d of defs) {
      subtabs.appendChild(h("button", {
        class: "mx-subtab" + (tab === d.key ? " active" : ""),
        disabled: !d.n,
        title: d.n ? "" : "nothing of this kind was logged in this run",
        onclick: () => { tab = d.key; renderSubtabs(); showTab(); },
      }, icon(d.ic, 15), d.label, d.n ? h("span", { class: "mx-count" }, String(d.n)) : null));
    }
  }

  // Opened in its own browser tab, not an iframe: TB then has the full window to lay
  // itself out in, and its lifetime is the user's to manage (the manager reuses one
  // process per logdir and kills them all on GUI shutdown).
  async function openTensorBoard() {
    tbBtn.disabled = true;
    try {
      const inst = await api.startTensorBoard(logdir);
      window.open(inst.url, "_blank", "noopener");
    } catch (err) {
      mount(body, errorBox(err));
    } finally {
      tbBtn.disabled = false;
    }
  }

  // -- the three views, built once and kept mounted so their state survives ---
  const views = {};
  function showTab() {
    for (const [k, v] of Object.entries(views)) v.el.style.display = k === tab ? "" : "none";
    views[tab]?.onShow?.();
  }

  // ======================= scalars =======================================
  function scalarsView() {
    const el = h("div", { class: "mx-scalars" });
    const rail = h("aside", { class: "mx-rail" });
    const cards = h("div", { class: "mx-cards" });
    el.appendChild(rail);
    el.appendChild(cards);

    const selected = new Set(manifest.scalars.map((s) => s.tag));
    const known = new Set(selected);   // tags a live run starts logging later get selected too
    let smoothing = 0.6, logY = false, xAxis = "step";
    let data = {};                       // {tag: {run: [[step, wall, value], …]}}
    const plots = new Map();             // tag -> { card, plot, legend, styleBox }
    const curveStyles = new Map();       // "tag\0run" -> { color, dash, width, alpha }
    let filter = "";

    const search = h("input", {
      class: "input", placeholder: "filter tags…",
      oninput: (e) => { filter = e.target.value.toLowerCase(); buildRail(); },
    });
    const tagList = h("div", { class: "mx-taglist" });
    const smoothVal = h("span", { class: "mono small" }, "0.60");

    function buildRail() {
      clear(tagList);
      const tags = manifest.scalars.map((s) => s.tag).filter((t) => t.toLowerCase().includes(filter));
      for (const tag of tags) {
        const cb = h("input", {
          type: "checkbox", checked: selected.has(tag),
          onchange: (e) => { e.target.checked ? selected.add(tag) : selected.delete(tag); renderCards(); },
        });
        tagList.appendChild(h("label", { class: "mx-tagitem", title: tag }, cb, h("span", null, tag)));
      }
      if (!tags.length) tagList.appendChild(h("p", { class: "muted small" }, "No tag matches."));
    }

    const allBtns = h("div", { class: "mx-railrow" },
      h("button", { class: "btn ghost small", onclick: () => { manifest.scalars.forEach((s) => selected.add(s.tag)); buildRail(); renderCards(); } }, "All"),
      h("button", { class: "btn ghost small", onclick: () => { selected.clear(); buildRail(); renderCards(); } }, "None"));

    rail.appendChild(h("div", { class: "mx-railgrp" }, h("div", { class: "mx-raillabel" }, "Tags"), search, tagList, allBtns));
    rail.appendChild(h("div", { class: "mx-railgrp" },
      h("div", { class: "mx-raillabel" }, "Display"),
      h("label", { class: "mx-ctl" }, h("span", null, "Smoothing"), smoothVal),
      h("input", {
        type: "range", min: "0", max: "0.99", step: "0.01", value: "0.6", class: "mx-range",
        oninput: (e) => { smoothing = parseFloat(e.target.value); smoothVal.textContent = smoothing.toFixed(2); refresh(); },
      }),
      h("label", { class: "mx-ctl" },
        h("input", { type: "checkbox", onchange: (e) => { logY = e.target.checked; refresh(); } }),
        h("span", null, "Log scale (y)")),
      h("label", { class: "mx-ctl" },
        h("span", null, "X axis"),
        h("select", { class: "input mx-select", onchange: (e) => { xAxis = e.target.value; refresh(); } },
          h("option", { value: "step" }, "Step"),
          h("option", { value: "time" }, "Time (min)")))));

    buildRail();

    // t0 = the earliest wall-clock stamp in the run, so "time" is minutes into training.
    function firstWall() {
      let t0 = Infinity;
      for (const runs of Object.values(data)) {
        for (const pts of Object.values(runs)) if (pts.length) t0 = Math.min(t0, pts[0][1]);
      }
      return isFinite(t0) ? t0 : 0;
    }

    // Per-curve appearance, remembered per (tag, run) so it survives re-polls, tag
    // re-selection and a smoothing change. Seeded from the palette on first sight.
    function styleFor(tag, run, i) {
      const key = `${tag}\u0000${run}`;
      let st = curveStyles.get(key);
      if (!st) {
        st = { color: colorAt(i), dash: "solid", width: 1.5, alpha: 1, hidden: false };
        curveStyles.set(key, st);
      }
      return st;
    }

    // Every curve of the tag, hidden ones included: the legend needs them all (that is
    // where you switch one back on), the plot takes only the visible subset.
    function seriesFor(tag, t0) {
      const runs = data[tag] || {};
      return Object.entries(runs).map(([run, pts], i) => {
        const st = styleFor(tag, run, i);
        let p = pts.map(([step, wall, v]) => [xAxis === "step" ? step : (wall - t0) / 60, v]);
        if (logY) p = p.map(([x, y]) => [x, y != null && y > 0 ? Math.log10(y) : null]);
        return {
          name: runLabel(run, tag), points: smooth(p, smoothing),
          color: st.color, dash: st.dash, width: st.width, alpha: st.alpha,
          hidden: st.hidden, style: st,
        };
      });
    }

    // The style editor: one row per curve on the card it belongs to, so you tweak what
    // you are looking at instead of hunting for it in a global settings pane.
    function buildStyleBox(tag, box) {
      clear(box);
      const runs = Object.keys(data[tag] || {});
      if (!runs.length) return;
      runs.forEach((run, i) => {
        const st = styleFor(tag, run, i);
        const apply = () => refresh();
        box.appendChild(h("div", { class: "mx-styrow" },
          h("input", {
            type: "checkbox", checked: !st.hidden, title: "Plot this curve",
            onchange: (e) => { st.hidden = !e.target.checked; apply(); },
          }),
          h("input", {
            type: "color", class: "mx-color", value: st.color, title: "Curve colour",
            oninput: (e) => { st.color = e.target.value; apply(); },
          }),
          h("span", { class: "mx-styname", title: runLabel(run, tag) }, runLabel(run, tag)),
          h("select", {
            class: "input mx-select", title: "Line style",
            onchange: (e) => { st.dash = e.target.value; apply(); },
          }, ...["solid", "dashed", "dotted"].map((d) =>
            h("option", { value: d, selected: d === st.dash }, d))),
          h("select", {
            class: "input mx-select", title: "Line width",
            onchange: (e) => { st.width = +e.target.value; apply(); },
          }, ...[1, 1.5, 2, 3].map((w) =>
            h("option", { value: String(w), selected: w === st.width }, `${w}px`))),
          h("input", {
            type: "range", min: "0.1", max: "1", step: "0.05", value: String(st.alpha),
            class: "mx-range mx-alpha", title: "Opacity",
            oninput: (e) => { st.alpha = +e.target.value; apply(); },
          })));
      });
    }

    function renderCards() {
      const tags = manifest.scalars.map((s) => s.tag).filter((t) => selected.has(t));
      // Drop cards for tags no longer selected.
      for (const [tag, entry] of plots) {
        if (!tags.includes(tag)) { entry.card.remove(); plots.delete(tag); }
      }
      if (!tags.length) {
        if (!cards.querySelector(".mx-empty")) {
          cards.appendChild(h("p", { class: "muted mx-empty" }, "Pick a tag on the left to plot it."));
        }
      } else cards.querySelector(".mx-empty")?.remove();

      for (const tag of tags) {
        if (plots.has(tag)) continue;
        const canvas = h("canvas", { class: "mx-canvas" });
        const legend = h("div", { class: "mx-legend" });
        const styleBox = h("div", { class: "mx-stylebox", style: { display: "none" } });
        const styleBtn = h("button", {
          class: "btn-icon", title: "Customise the curves (colour, style, width, opacity)",
          onclick: () => {
            const open = styleBox.style.display === "none";
            styleBox.style.display = open ? "" : "none";
            styleBtn.classList.toggle("active", open);
            if (open) buildStyleBox(tag, styleBox);
          },
        }, icon("sliders", 14));
        const card = h("div", { class: "mx-card" },
          h("div", { class: "mx-card-head" },
            h("span", { class: "mx-card-title", title: tag }, tag),
            legend,
            styleBtn,
            h("button", {
              class: "btn-icon", title: "Save as PNG (uses the export settings above)",
              onclick: async () => download(
                await plots.get(tag).plot.snapshot({ ...exportOpts }),
                `${tag.replace(/[^\w.-]+/g, "_")}${exportOpts.scale}x.png`),
            }, icon("image", 14)),
            h("button", { class: "btn-icon", title: "Download CSV", onclick: () => exportCsv(tag) }, icon("fileText", 14))),
          styleBox,
          canvas);
        cards.appendChild(card);
        plots.set(tag, { card, plot: createPlot(canvas), legend, styleBox });
      }
      refresh();
    }

    function exportCsv(tag) {
      const rows = [["run", "step", "wall_time", "value"]];
      for (const [run, pts] of Object.entries(data[tag] || {})) {
        for (const [step, wall, v] of pts) rows.push([runLabel(run, tag), step, wall, v]);
      }
      const csv = rows.map((r) => r.map((c) => (typeof c === "string" && /[",]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n");
      download(new Blob([csv], { type: "text/csv" }), `${tag.replace(/[^\w.-]+/g, "_")}.csv`);
    }

    function refresh() {
      const t0 = firstWall();
      const ylab = logY ? "log₁₀(value)" : "value";
      const xlab = xAxis === "step" ? "step" : "minutes";
      for (const [tag, entry] of plots) {
        const ss = seriesFor(tag, t0);
        entry.plot.setSeries("line", ss.filter((s) => !s.hidden), { x: xlab, y: ylab });
        clear(entry.legend);
        if (ss.length > 1) {
          // The legend doubles as the on/off switch — clicking a name drops that curve
          // from the plot (and from the autoscale), clicking it again brings it back.
          for (const s of ss) {
            entry.legend.appendChild(h("button", {
              class: "mx-legitem" + (s.hidden ? " off" : ""),
              title: s.hidden ? `Show "${s.name}"` : `Hide "${s.name}"`,
              onclick: () => {
                s.style.hidden = !s.style.hidden;
                refresh();
                if (entry.styleBox.style.display !== "none") buildStyleBox(tag, entry.styleBox);
              },
            }, h("i", { class: "mx-swatch", style: { background: s.color, opacity: String(s.alpha) } }), s.name));
          }
        }
        if (entry.styleBox.style.display !== "none" && entry.styleBox.childElementCount !== ss.length) {
          buildStyleBox(tag, entry.styleBox);   // only when the curve set changed: a rebuild
        }                                       // mid-drag would replace the slider you hold
      }
    }

    async function load() {
      const res = await api.metricsScalars(logdir);
      data = res.tags || {};
      refresh();
    }

    renderCards();
    load().catch(() => { /* the panel's poller surfaces errors */ });

    return {
      el,
      onShow() { for (const { plot } of plots.values()) plot.redraw(); },
      async poll() { await load(); },
      onManifest() {
        // A tag first logged mid-run (HetSIREN's inverse phase, say) should appear
        // plotted, not sitting unticked in the rail where nobody would look for it.
        for (const s of manifest.scalars) {
          if (!known.has(s.tag)) { known.add(s.tag); selected.add(s.tag); }
        }
        buildRail();
        renderCards();
      },
    };
  }

  // ======================= images =======================================
  // Frames, not steps: hax logs its image panels with no global_step, so every epoch
  // lands on step 0 and only wall-clock order tells them apart. Indexing by frame is
  // what turns "one picture" back into the training movie.
  function imagesView() {
    const el = h("div", { class: "mx-images" });
    const chips = h("div", { class: "mx-chips" });
    const img = h("img", { class: "mx-img", alt: "" });
    const frameLabel = h("span", { class: "mono small" }, "");
    const range = h("input", { type: "range", min: "0", max: "0", value: "0", class: "mx-range grow", oninput: (e) => { idx = +e.target.value; show(); } });
    const playBtn = h("button", { class: "btn-icon", title: "Play through the epochs", onclick: togglePlay }, icon("play", 15));
    const saveBtn = h("button", {
      class: "btn-icon", title: "Save this frame",
      onclick: async () => {
        const t = current();
        if (t) download(await api.metricsImage(logdir, t.runs[0], t.tag, idx), `${t.tag.replace(/[^\w.-]+/g, "_")}_${idx}.png`);
      },
    }, icon("save", 15));

    let tagName = manifest.images[0]?.tag || null;
    let idx = 0, playing = false, ticker = null;
    const cache = new Map();      // "tag@frame" -> object URL

    el.appendChild(chips);
    el.appendChild(h("div", { class: "mx-imgwrap" }, img));
    el.appendChild(h("div", { class: "mx-transport" }, playBtn, range, frameLabel, saveBtn));

    const current = () => manifest.images.find((t) => t.tag === tagName) || null;
    const frames = () => current()?.frames || [];

    function buildChips() {
      clear(chips);
      for (const t of manifest.images) {
        chips.appendChild(h("button", {
          class: "mx-chip" + (t.tag === tagName ? " active" : ""),
          onclick: () => { tagName = t.tag; idx = frames().length - 1; syncRange(); show(); buildChips(); },
        }, t.tag, h("span", { class: "mx-count" }, String(t.frames.length))));
      }
    }

    function syncRange() {
      const n = frames().length;
      range.max = String(Math.max(0, n - 1));
      idx = Math.min(idx, Math.max(0, n - 1));
      range.value = String(idx);
      range.disabled = n < 2;
      playBtn.disabled = n < 2;
    }

    // Steps are all 0 here, so the honest x-label is elapsed time into the run.
    function label(fs, i) {
      const useStep = fs.some((f) => f.step !== fs[0].step);
      if (useStep) return `step ${fs[i].step}  (${i + 1}/${fs.length})`;
      const secs = Math.max(0, Math.round(fs[i].wall - fs[0].wall));
      const mm = String(Math.floor(secs / 60)).padStart(2, "0");
      const ss = String(secs % 60).padStart(2, "0");
      return `+${mm}:${ss} into the run  (${i + 1}/${fs.length})`;
    }

    async function show() {
      const t = current(), fs = frames();
      if (!t || !fs.length) return;
      range.value = String(idx);
      frameLabel.textContent = label(fs, idx);
      const key = `${t.tag}@${fs[idx].wall}`;
      let url = cache.get(key);
      if (!url) {
        try {
          url = URL.createObjectURL(await api.metricsImage(logdir, t.runs[0], t.tag, idx));
        } catch { return; }
        cache.set(key, url);
      }
      img.src = url;
    }

    function togglePlay() {
      playing = !playing;
      clear(playBtn);
      playBtn.appendChild(icon(playing ? "pause" : "play", 15));
      if (ticker) { clearInterval(ticker); ticker = null; }
      if (playing) {
        ticker = setInterval(() => {
          const n = frames().length;
          if (!n) return;
          idx = (idx + 1) % n;
          show();
        }, FRAME_MS);
      }
    }

    buildChips();
    idx = Math.max(0, frames().length - 1);   // open on the newest frame
    syncRange();
    show();

    return {
      el,
      onManifest() {
        const atEnd = idx >= frames().length - 1;
        buildChips();
        syncRange();
        if (atEnd) { idx = Math.max(0, frames().length - 1); show(); }   // a live run keeps following
      },
      destroy() {
        if (ticker) clearInterval(ticker);
        for (const url of cache.values()) URL.revokeObjectURL(url);
      },
    };
  }

  // ======================= latent space =================================
  // 2D uses the Canvas2D scatter; 3D reuses the landscape viewer's WebGL point cloud
  // (cloud.js, orbit camera and all), so both views of a latent space look and behave
  // the same whether you reach them from here or from the full landscape.
  function latentView() {
    const el = h("div", { class: "mx-latent" });
    const side = h("div", { class: "mx-latent-side" });
    const canvas2d = h("canvas", { class: "mx-scatter" });
    const canvas3d = h("canvas", { class: "mx-scatter", style: { display: "none" } });
    const stepLabel = h("span", { class: "mono small" }, "");
    const range = h("input", { type: "range", min: "0", max: "0", value: "0", class: "mx-range grow", oninput: (e) => { idx = +e.target.value; show(); } });
    const playBtn = h("button", { class: "btn-icon", title: "Play through the epochs", onclick: togglePlay }, icon("play", 15));
    const resetBtn = h("button", { class: "btn-icon", title: "Reset the camera", style: { display: "none" }, onclick: () => cloud?.resetView() }, icon("refresh", 15));
    const saveBtn = h("button", {
      class: "btn-icon", title: "Save the landscape as PNG (uses the export settings above)",
      onclick: async () => {
        const src = dims === 3 ? cloud : scatter;
        const blob = await src?.snapshot({ ...exportOpts });
        if (blob) download(blob, `latent_${dims}d_epoch${steps()[idx]}_${exportOpts.scale}x.png`);
      },
    }, icon("image", 15));
    const info = h("div", { class: "mx-info" });
    const colorSel = h("select", { class: "input mx-select", onchange: (e) => { colorBy = +e.target.value; paintColors(); } });
    const colorRow = h("label", { class: "mx-ctl", style: { display: "none" } }, h("span", null, "Colour by"), colorSel);
    const spriteCanvas = h("canvas", { class: "mx-sprite", width: 96, height: 96 });
    const spriteBox = h("div", { class: "mx-spritebox", style: { display: "none" } },
      h("div", { class: "mx-raillabel" }, "Selected particle"), spriteCanvas);

    const tags = [...new Set(manifest.embeddings.map((e) => e.tag))];
    let tagName = tags[0] || null;
    let idx = 0, playing = false, ticker = null, colorBy = -1, dims = 2;
    let cloud = null;                    // built lazily: WebGL2 is only needed if 3D is asked for
    let selectedPoint = -1;
    const bases = { 2: null, 3: null };  // last basis per dimensionality — keeps the movie stable
    const raws = new Map();              // step -> server payload (raw latent vectors)
    const projs = new Map();             // "step|k" -> { points, variance }
    const sprites = new Map();           // step -> HTMLImageElement

    // The renderers only *report* a click — neither highlights the point on its own, that
    // is the caller's job. Drive both from one place so the selection survives a 2D↔3D
    // toggle instead of being whichever renderer happened to be up when you clicked.
    function pickPoint(i) {
      selectedPoint = i;
      scatter.select(i);
      cloud?.select(i);
      drawSprite();
    }

    const scatter = createScatter(canvas2d, pickPoint);

    const dimBtns = [2, 3].map((k) => h("button", {
      class: "mx-dimbtn" + (k === dims ? " active" : ""),
      title: k === 2 ? "Project onto the first 2 principal components" : "Project onto the first 3 principal components and orbit it",
      onclick: () => setDims(k),
    }, `${k}D`));
    const dimSeg = h("div", { class: "mx-dimseg" }, ...dimBtns);

    const landscapeBtn = h("button", {
      class: "btn secondary", style: { width: "100%", justifyContent: "center" },
      title: "Open the full conformational landscape: UMAP, clustering and volume decoding",
      onclick: openLandscape,
    }, icon("box", 15), "Open full landscape");

    side.appendChild(h("div", { class: "mx-railgrp" },
      h("div", { class: "mx-raillabel" }, "View"),
      h("label", { class: "mx-ctl" }, h("span", null, "Dimensions"), dimSeg),
      colorRow));
    side.appendChild(h("div", { class: "mx-railgrp" }, h("div", { class: "mx-raillabel" }, "Epoch"), info));
    side.appendChild(spriteBox);
    if (jobId) side.appendChild(h("div", { class: "mx-railgrp" }, landscapeBtn,
      h("p", { class: "field-help", style: { margin: "8px 0 0" } },
        "This view is a live PCA of what training logged. The landscape viewer decodes volumes from the trained network.")));

    const steps = () => manifest.embeddings.filter((e) => e.tag === tagName).map((e) => e.step);

    const tagSel = h("select", { class: "input mx-select", onchange: (e) => { tagName = e.target.value; resetCaches(); idx = steps().length - 1; syncRange(); show(); } },
      ...tags.map((t) => h("option", { value: t }, t)));
    if (tags.length > 1) side.insertBefore(h("div", { class: "mx-railgrp" }, h("div", { class: "mx-raillabel" }, "Space"), tagSel), side.firstChild);

    el.appendChild(side);
    el.appendChild(h("div", { class: "mx-latent-main" },
      h("div", { class: "mx-scatterwrap" }, canvas2d, canvas3d),
      h("div", { class: "mx-transport" }, playBtn, range, stepLabel, saveBtn, resetBtn)));

    function resetCaches() {
      bases[2] = bases[3] = null;
      raws.clear();
      projs.clear();
      pickPoint(-1);   // a different latent space: the old point index means nothing in it
    }

    // WebGL2 may simply not be there (remote X, software GL). Fall back to 2D and say so,
    // rather than leaving a dead black canvas.
    function setDims(k) {
      if (k === 3 && !cloud) {
        try {
          cloud = createCloud(canvas3d, pickPoint);
          const panel = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim();
          if (panel.startsWith("#")) cloud.setBackground(hexToRgb01(panel));
        } catch (err) {
          mount(info, errorBox("3D needs WebGL2, which this browser did not provide: " + err.message));
          return;
        }
      }
      dims = k;
      dimBtns.forEach((b, i) => b.classList.toggle("active", [2, 3][i] === k));
      canvas2d.style.display = k === 2 ? "" : "none";
      canvas3d.style.display = k === 3 ? "" : "none";
      resetBtn.style.display = k === 3 ? "" : "none";
      show();
    }

    async function openLandscape() {
      landscapeBtn.disabled = true;
      try {
        const job = await api.job(jobId);
        await launchViewerForJob(job);
      } catch (err) {
        mount(info, errorBox(err));
      } finally {
        landscapeBtn.disabled = false;
      }
    }

    function syncRange() {
      const n = steps().length;
      range.max = String(Math.max(0, n - 1));
      idx = Math.min(idx, Math.max(0, n - 1));
      range.value = String(idx);
      range.disabled = n < 2;
      playBtn.disabled = n < 2;
    }

    async function fetchEpoch(step) {
      let raw = raws.get(step);
      if (!raw) {
        raw = await api.metricsEmbedding(logdir, step, tagName);
        raws.set(step, raw);
      }
      const key = `${step}|${dims}`;
      let proj = projs.get(key);
      if (!proj) {
        // Project against the previous epoch's basis so the cloud doesn't flip sign
        // between frames (eigenvectors are only defined up to a sign).
        const { points, basis, variance } = pca(raw.points, dims, bases[dims]);
        bases[dims] = basis || bases[dims];
        proj = { points, variance };
        projs.set(key, proj);
      }
      return { raw, proj };
    }

    function buildColorSel(meta) {
      const cols = meta?.columns || [];
      if (!cols.length) { colorRow.style.display = "none"; return; }
      colorRow.style.display = "";
      if (colorSel.options.length !== cols.length + 1) {
        clear(colorSel);
        colorSel.appendChild(h("option", { value: "-1" }, "Nothing"));
        cols.forEach((c, i) => colorSel.appendChild(h("option", { value: String(i) }, c)));
      }
      colorSel.value = String(colorBy);
    }

    // The 2D scatter takes CSS colour strings; the point cloud takes a packed float RGB
    // buffer. Same colouring, two upload paths.
    function paintColors() {
      const raw = raws.get(steps()[idx]);
      if (!raw) return;
      const n = raw.points.length;
      const none = colorBy < 0 || !raw.meta;
      const values = none ? null : raw.meta.rows.map((r) => r[colorBy]);
      const uniq = none ? null : [...new Set(values)];

      if (dims === 2) {
        scatter.setPointColors(none ? null : values.map((v) => colorAt(uniq.indexOf(v))));
        return;
      }
      if (!cloud) return;
      const buf = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const rgb = none ? [0.58, 0.64, 0.74] : hexToRgb01(colorAt(uniq.indexOf(values[i])));
        buf[i * 3] = rgb[0]; buf[i * 3 + 1] = rgb[1]; buf[i * 3 + 2] = rgb[2];
      }
      cloud.setColors(buf);
    }

    async function drawSprite() {
      const step = steps()[idx];
      const raw = raws.get(step);
      const ctx = spriteCanvas.getContext("2d");
      ctx.clearRect(0, 0, spriteCanvas.width, spriteCanvas.height);
      if (!raw?.sprite || selectedPoint < 0) { spriteBox.style.display = raw?.sprite ? "" : "none"; return; }
      spriteBox.style.display = "";
      let im = sprites.get(step);
      if (!im) {
        try {
          const url = URL.createObjectURL(await api.metricsSprite(logdir, step, tagName));
          im = await new Promise((res, rej) => {
            const x = new Image();
            x.onload = () => res(x);
            x.onerror = rej;
            x.src = url;
          });
          sprites.set(step, im);
        } catch { return; }
      }
      const { tile, per_side: per } = raw.sprite;
      if (!tile || !per) return;
      const orig = raw.indices[selectedPoint] ?? 0;   // plotted index -> original row
      const tx = (orig % per) * tile, ty = Math.floor(orig / per) * tile;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(im, tx, ty, tile, tile, 0, 0, spriteCanvas.width, spriteCanvas.height);
    }

    async function show() {
      const all = steps();
      if (!all.length) return;
      const step = all[idx];
      range.value = String(idx);
      stepLabel.textContent = `epoch ${step}  (${idx + 1}/${all.length})`;
      let raw, proj;
      try { ({ raw, proj } = await fetchEpoch(step)); } catch (err) { mount(info, errorBox(err)); return; }

      if (dims === 2) scatter.setPoints(proj.points);
      else cloud?.setPoints(proj.points);
      buildColorSel(raw.meta);
      paintColors();
      // Re-assert the selection on both: setPoints on a newly built cloud starts unmarked,
      // and the same particle index is meaningful in either projection.
      scatter.select(selectedPoint);
      cloud?.select(selectedPoint);
      drawSprite();

      const pv = proj.variance.map((v) => `${(100 * v).toFixed(0)}%`).join(" / ");
      mount(info, h("div", null,
        h("div", { class: "mono" }, `step ${step}`),
        h("p", { class: "muted small", style: { margin: "6px 0 0" } },
          `${raw.n_full.toLocaleString()} points · ${raw.dim}-D latent`),
        h("p", { class: "muted small", style: { margin: "2px 0 0" } },
          `PC 1–${dims} explain ${pv} of the variance`),
        raw.dim < dims ? h("p", { class: "muted small", style: { margin: "2px 0 0" } },
          `latent space is only ${raw.dim}-D — the extra axis is flat`) : null));
    }

    function togglePlay() {
      playing = !playing;
      clear(playBtn);
      playBtn.appendChild(icon(playing ? "pause" : "play", 15));
      if (ticker) { clearInterval(ticker); ticker = null; }
      if (playing) {
        ticker = setInterval(() => {
          const n = steps().length;
          if (!n) return;
          idx = (idx + 1) % n;
          show();
        }, FRAME_MS * 1.4);
      }
    }

    syncRange();
    idx = Math.max(0, steps().length - 1);
    show();

    return {
      el,
      // Both renderers size themselves off clientWidth, which is 0 while the tab is
      // hidden — so redraw on the way back in.
      onShow() { if (dims === 2) scatter.draw(); else cloud?.render(); },
      onManifest() {
        const atEnd = idx >= steps().length - 1;
        syncRange();
        if (atEnd) { idx = Math.max(0, steps().length - 1); show(); }   // follow a live run
      },
      destroy() {
        if (ticker) clearInterval(ticker);
        for (const im of sprites.values()) URL.revokeObjectURL(im.src);
      },
    };
  }

  // -- load, then poll ------------------------------------------------------
  async function loadManifest() {
    const next = await api.metricsManifest(logdir);
    const first = manifest === null;
    manifest = next;
    renderSubtabs();

    if (first) {
      if (!manifest.scalars.length && !manifest.images.length && !manifest.embeddings.length) {
        mount(body, h("p", { class: "muted", style: { padding: "24px 0" } },
          "This run has event files but nothing logged yet. Metrics appear as training writes them."));
        return;
      }
      // Open on whichever view has something in it.
      tab = manifest.scalars.length ? "scalars" : manifest.images.length ? "images" : "latent";
      clear(body);
      if (manifest.scalars.length) views.scalars = scalarsView();
      if (manifest.images.length) views.images = imagesView();
      if (manifest.embeddings.length) views.latent = latentView();
      for (const v of Object.values(views)) body.appendChild(v.el);
      renderSubtabs();
      showTab();
      if (manifest.texts && Object.keys(manifest.texts).length) body.appendChild(legends(manifest.texts));
      return;
    }
    for (const v of Object.values(views)) v.onManifest?.();
  }

  // The colour legends hax logs with add_text (what blue/red mean on the variation map).
  function legends(texts) {
    const box = h("details", { class: "mx-legends" }, h("summary", null, "Colour legends"));
    for (const [tag, html] of Object.entries(texts)) {
      box.appendChild(h("div", { class: "mx-legendtext", html }));
    }
    return box;
  }

  function schedule() {
    if (timer) { clearInterval(timer); timer = null; }
    if (!live) return;
    timer = setInterval(async () => {
      if (destroyed || document.hidden) return;
      try {
        await loadManifest();
        await views.scalars?.poll?.();
      } catch { /* transient — the next tick retries */ }
    }, POLL_MS);
  }

  mount(body, h("p", { class: "muted", style: { padding: "24px 0" } }, "Reading event files…"));
  loadManifest().then(schedule).catch((err) => mount(body, errorBox(err)));

  return {
    el: root,
    // Called when this panel's tab becomes the visible one. Its canvases were laid out
    // at zero width while hidden, so they need to be told to draw themselves again now
    // that they actually have a box.
    onShow() { showTab(); },
    // `closing` distinguishes the tab being closed from the view merely unmounting on
    // navigation: only the former releases the server's event cache. A TensorBoard the
    // user opened is deliberately left alone — it lives in a browser tab of its own.
    destroy(closing = false) {
      destroyed = true;
      if (timer) clearInterval(timer);
      for (const v of Object.values(views)) v.destroy?.();
      if (closing) api.dropMetrics(logdir).catch(() => {});
    },
  };
}
