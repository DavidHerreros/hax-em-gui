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
import { sanitizedHtml } from "../markdown.js";
import { api } from "../api.js";
import { errorBox } from "./ui.js";
import { pca } from "../pca.js";
import { launchViewerForJob } from "./viewerLaunch.js";
import { prefsFor, changeWriter } from "../metricsPrefs.js";
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

  // How this run was left the last time it was open. The panel is destroyed on every
  // navigation away from /metrics, so without this the whole configuration — cards in
  // Split, a soloed curve, an unticked tag — is rebuilt from defaults each time.
  const prefs = prefsFor(logdir);
  const panelPrefs = prefs.get("panel");
  const savePanel = changeWriter(prefs, "panel");
  const notePanel = () => savePanel(() => ({ tab, live, scale: exportOpts.scale, transparent: exportOpts.transparent }), 0);

  let manifest = null;
  let tab = panelPrefs.tab || "scalars";
  let live = panelPrefs.live !== false;
  let timer = null;
  let destroyed = false;

  // Export settings live on the panel so a plot and the latent landscape come out of the
  // same run looking alike — same idea (and same wording) as the annotate viewer's.
  const exportOpts = {
    scale: [1, 2, 3, 4].includes(panelPrefs.scale) ? panelPrefs.scale : 2,
    transparent: !!panelPrefs.transparent,
  };
  const dpiSel = h("select", {
    class: "input mx-select", title: "Image scale / DPI",
    onchange: (e) => { exportOpts.scale = +e.target.value; notePanel(); },
  }, ...[1, 2, 3, 4].map((s) => h("option", { value: String(s), selected: s === exportOpts.scale }, `${s}×`)));
  const transpBox = h("input", {
    type: "checkbox", checked: exportOpts.transparent,
    onchange: (e) => { exportOpts.transparent = e.target.checked; notePanel(); },
  });

  // -- head: sub-tabs + export + live toggle + the TensorBoard escape hatch --
  const subtabs = h("div", { class: "mx-subtabs" });
  const liveBox = h("input", { type: "checkbox", checked: live, onchange: (e) => { live = e.target.checked; notePanel(); schedule(); } });
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
        onclick: () => { tab = d.key; notePanel(); renderSubtabs(); showTab(); },
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

    // Everything the view is configured by comes out of the saved slice first, and falls
    // back to the defaults for a run nobody has opened yet.
    const saved = prefs.get("scalars");
    const saveScalars = changeWriter(prefs, "scalars");
    const allTags = manifest.scalars.map((s) => s.tag);
    const savedTags = Array.isArray(saved.selected) ? saved.selected.filter((t) => allTags.includes(t)) : null;
    const savedKnown = Array.isArray(saved.known) ? saved.known : null;

    // A restored selection means the tags left out were left out on purpose — except for
    // tags this run has started logging since, which nobody has had the chance to untick.
    const selected = new Set(
      savedTags
        ? [...savedTags, ...allTags.filter((t) => !savedKnown || !savedKnown.includes(t))]
        : allTags);
    // Seeded from every tag the manifest already has, not from the selection: an unticked
    // tag must not come back as "new" on the next poll (see onManifest).
    const known = new Set(allTags);
    let smoothing = typeof saved.smoothing === "number" ? saved.smoothing : 0.6;
    let logY = !!saved.logY;
    let xAxis = saved.xAxis === "time" ? "time" : "step";
    let endLabels = saved.endLabels !== false;
    let data = {};                       // {tag: {run: [[step, wall, value], …]}}
    const plots = new Map();             // tag -> card entry (see buildCard)
    const curveStyles = new Map();       // "tag\0run" -> { color, dash, width, alpha }
    // Per-card display state, kept out of the entry so it survives a card being dropped
    // and re-created when its tag is unticked and ticked again.
    const cardOpts = new Map();          // tag -> { mode, wide, solo }
    let filter = "";

    for (const [key, st] of Object.entries(saved.styles || {})) {
      if (st && typeof st === "object") curveStyles.set(key, { ...st });
    }
    for (const [tag, o] of Object.entries(saved.cards || {})) {
      if (o && typeof o === "object") {
        cardOpts.set(tag, {
          mode: ["shared", "norm", "split"].includes(o.mode) ? o.mode : "shared",
          wide: !!o.wide,
          solo: typeof o.solo === "string" ? o.solo : null,
        });
      }
    }

    function optsFor(tag) {
      let o = cardOpts.get(tag);
      if (!o) { o = { mode: "shared", wide: false, solo: null }; cardOpts.set(tag, o); }
      return o;
    }

    // Called from refresh(), so every path that changes anything lands here; the writer
    // drops the write when the settings are byte-identical, which is what keeps a live
    // run's five-second poll from rewriting localStorage forever.
    function noteScalars() {
      saveScalars(() => ({
        selected: [...selected], known: [...known], smoothing, logY, xAxis, endLabels,
        cards: Object.fromEntries([...cardOpts].map(([t, o]) => [t, { mode: o.mode, wide: o.wide, solo: o.solo }])),
        styles: Object.fromEntries(curveStyles),
      }));
    }

    const search = h("input", {
      class: "input", placeholder: "filter tags…",
      oninput: (e) => { filter = e.target.value.toLowerCase(); buildRail(); },
    });
    const tagList = h("div", { class: "mx-taglist" });
    const smoothVal = h("span", { class: "mono small" }, smoothing.toFixed(2));
    const smoothRange = h("input", {
      type: "range", min: "0", max: "0.99", step: "0.01", value: String(smoothing), class: "mx-range",
      oninput: (e) => { smoothing = parseFloat(e.target.value); smoothVal.textContent = smoothing.toFixed(2); refresh(); },
    });
    const logBox = h("input", { type: "checkbox", checked: logY, onchange: (e) => { logY = e.target.checked; refresh(); } });
    const endBox = h("input", { type: "checkbox", checked: endLabels, onchange: (e) => { endLabels = e.target.checked; refresh(); } });
    const xSel = h("select", { class: "input mx-select", onchange: (e) => { xAxis = e.target.value; refresh(); } },
      h("option", { value: "step", selected: xAxis === "step" }, "Step"),
      h("option", { value: "time", selected: xAxis === "time" }, "Time (min)"));

    // Everything above is remembered per run, which is worth an escape hatch: this puts
    // the whole view — tags, smoothing, card modes, curve colours — back to defaults.
    function resetView() {
      selected.clear();
      known.forEach((t) => selected.add(t));
      curveStyles.clear();
      cardOpts.clear();
      smoothing = 0.6; logY = false; xAxis = "step"; endLabels = true;
      smoothRange.value = "0.6"; smoothVal.textContent = "0.60";
      logBox.checked = false; endBox.checked = true; xSel.value = "step";
      for (const e of plots.values()) e.card.remove();
      plots.clear();
      buildRail();
      renderCards();
    }

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
      smoothRange,
      h("label", { class: "mx-ctl" }, logBox, h("span", null, "Log scale (y)")),
      h("label", { class: "mx-ctl", title: "Write each curve's name where its line ends, instead of making you match it to a legend swatch" },
        endBox, h("span", null, "End-of-line labels")),
      h("label", { class: "mx-ctl" }, h("span", null, "X axis"), xSel),
      h("button", {
        class: "btn ghost small", style: { width: "100%", justifyContent: "center" },
        title: "This run's view — tags, smoothing, card scales, curve colours — is remembered between visits. This puts it back to the defaults.",
        onclick: resetView,
      }, icon("restore", 14), "Reset view")));

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
    //
    // `hidden` is what the plot obeys and folds in the solo state; `off` is the curve's
    // own switch, which is what the legend strikes through — a curve you soloed past is
    // not a curve you turned off.
    function seriesFor(tag, t0) {
      const runs = data[tag] || {};
      const o = optsFor(tag);
      return Object.entries(runs).map(([run, pts], i) => {
        const st = styleFor(tag, run, i);
        const name = runLabel(run, tag);
        let p = pts.map(([step, wall, v]) => [xAxis === "step" ? step : (wall - t0) / 60, v]);
        if (logY) p = p.map(([x, y]) => [x, y != null && y > 0 ? Math.log10(y) : null]);
        p = smooth(p, smoothing);
        const plain = p;                       // pre-normalisation: the value to report
        let last = null;
        for (const [, y] of plain) if (y != null && isFinite(y)) last = y;
        if (o.mode === "norm") {
          // Min-max onto 0–1 by the series' own range, so five curves three decades apart
          // can be compared by shape. The raw value rides along as the point's 3rd slot,
          // which is what the plotter's hover box reports.
          let lo = Infinity, hi = -Infinity;
          for (const [, y] of plain) if (y != null && isFinite(y)) { if (y < lo) lo = y; if (y > hi) hi = y; }
          const span = (hi - lo) || 1;
          p = plain.map(([x, y]) => [x, y == null ? null : (y - lo) / span, y]);
        }
        return {
          name, points: p, plain, last,
          color: st.color, dash: st.dash, width: st.width, alpha: st.alpha,
          off: st.hidden, hidden: st.hidden || (o.solo != null && o.solo !== name),
          style: st,
        };
      });
    }

    // TensorBoard sanitises "Reconstruction loss (ReconSIREN)" down to
    // "Reconstruction_loss__ReconSIREN_", which is what a card title would otherwise
    // read. The raw tag stays in the tooltip and in exported file names.
    const prettyTag = (tag) => tag.replace(/_+/g, " ").trim();

    const fmtNum = (v) => {
      if (v == null || !isFinite(v)) return "—";
      const a = Math.abs(v);
      if (a !== 0 && (a >= 1e6 || a < 1e-4)) return v.toExponential(2);
      return String(+v.toPrecision(5));
    };

    // A 52×14 trace of the whole curve, drawn on its own scale. On a shared axis a flat
    // line tells you nothing about whether the metric moved; this does.
    function sparkFor(s) {
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", "0 0 52 14");
      svg.setAttribute("preserveAspectRatio", "none");
      svg.setAttribute("class", "mx-spark");
      const pts = s.plain.filter((p) => p[1] != null && isFinite(p[1]));
      if (pts.length > 1) {
        let lo = Infinity, hi = -Infinity, x0 = pts[0][0], x1 = pts[pts.length - 1][0];
        for (const [, y] of pts) { if (y < lo) lo = y; if (y > hi) hi = y; }
        const span = (hi - lo) || 1, xs = (x1 - x0) || 1;
        const step = Math.max(1, Math.floor(pts.length / 52));
        const d = pts.filter((_, i) => i % step === 0 || i === pts.length - 1)
          .map((p, i) => `${i ? "L" : "M"}${((p[0] - x0) / xs) * 52} ${13 - ((p[1] - lo) / span) * 12}`).join(" ");
        const path = document.createElementNS(ns, "path");
        path.setAttribute("d", d);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", s.color);
        path.setAttribute("stroke-width", "1.4");
        svg.appendChild(path);
      }
      return svg;
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

      for (const tag of tags) if (!plots.has(tag)) plots.set(tag, buildCard(tag));
      refresh();
    }

    // One card. The head is the tag's own row — the legend used to share it, which is
    // what squeezed a five-curve title down to "Heterog…" — and everything that reads a
    // curve (legend, values, sparks) sits under the plot where the curves are.
    function buildCard(tag) {
      const o = optsFor(tag);
      const body = h("div", { class: "mx-cardbody" });
      // A poll rebuilds the legend's rows, which can swallow the mouseleave of the row
      // the pointer was on and leave a curve spotlit for good. The container outlives the
      // rebuild, so its own mouseleave is the one that can be trusted to clear it.
      const legend = h("div", {
        class: "mx-legend",
        onmouseleave: () => {
          const entry = plots.get(tag);
          if (entry && entry.focus) { entry.focus = null; entry.plot?.setFocus(null); }
        },
      });
      const styleBox = h("div", { class: "mx-stylebox", style: { display: "none" } });
      const count = h("span", { class: "mx-count", title: "curves on this card" }, "");

      const styleBtn = h("button", {
        class: "btn-icon", title: "Customise the curves (colour, style, width, opacity)",
        onclick: () => {
          const open = styleBox.style.display === "none";
          styleBox.style.display = open ? "" : "none";
          styleBtn.classList.toggle("active", open);
          if (open) buildStyleBox(tag, styleBox);
        },
      }, icon("sliders", 14));

      // The buttons take their state from `o` in refresh(), not from the click that set
      // it — that is what lets a restored (or reset) card come up showing the truth.
      const expandBtn = h("button", {
        class: "btn-icon", onclick: () => { o.wide = !o.wide; refresh(); },
      }, icon(o.wide ? "shrink" : "expand", 14));

      // The y-scale control lives at the foot, next to the legend it changes the meaning
      // of, rather than crowding the head of a 380px card.
      const modeBtns = [
        ["shared", "Shared", "One y-axis for every curve on the card"],
        ["norm", "Norm", "Map each curve onto 0–1 by its own range: compare shapes, not magnitudes. Hover still reports real values."],
        ["split", "Split", "Give each curve its own strip and its own scale"],
      ].map(([k, lab, tip]) => h("button", {
        class: "mx-modebtn", title: tip, "data-mode": k,
        onclick: () => { o.mode = k; refresh(); },
      }, lab));
      const hint = h("span", { class: "mx-hint" }, "");
      const foot = h("div", { class: "mx-cardfoot" }, h("div", { class: "mx-modeseg" }, ...modeBtns), hint);

      const card = h("div", { class: "mx-card" + (o.wide ? " wide" : "") },
        h("div", { class: "mx-card-head" },
          h("span", { class: "mx-card-title", title: tag }, prettyTag(tag)),
          count,
          h("div", { class: "mx-card-tools" },
            expandBtn,
            styleBtn,
            h("button", {
              class: "btn-icon", title: "Save as PNG (uses the export settings above)",
              onclick: async () => {
                const entry = plots.get(tag);
                const r = entry.strips || entry.plot;
                if (r) download(await r.snapshot({ ...exportOpts }), `${tag.replace(/[^\w.-]+/g, "_")}${exportOpts.scale}x.png`);
              },
            }, icon("image", 14)),
            h("button", { class: "btn-icon", title: "Download CSV", onclick: () => exportCsv(tag) }, icon("fileText", 14)))),
        styleBox,
        body,
        legend,
        foot);
      cards.appendChild(card);
      return { tag, card, body, legend, styleBox, count, foot, hint, modeBtns, expandBtn, plot: null, strips: null, focus: null };
    }

    // Swap the renderer when the card's mode changes — and, in split mode, when the
    // number of strips changes, since that is what sets the canvas height.
    function ensureRenderer(entry, n) {
      const mode = optsFor(entry.tag).mode;
      if (entry.builtMode === mode && (mode !== "split" || entry.builtN === n)) return;
      clear(entry.body);
      entry.plot = entry.strips = null;
      const canvas = h("canvas", { class: "mx-canvas" });
      if (mode === "split") {
        canvas.style.height = `${34 + Math.max(1, n) * 46}px`;
        entry.body.appendChild(canvas);
        entry.strips = createStrips(canvas);
      } else {
        entry.body.appendChild(canvas);
        entry.plot = createPlot(canvas);
      }
      entry.builtMode = mode;
      entry.builtN = n;
    }

    // The legend, below the plot: a swatch, the curve's own trace, its name and what it
    // is worth right now. Ordered by that value, so the top row is the top line.
    function buildLegend(entry, ss) {
      const o = optsFor(entry.tag);
      clear(entry.legend);
      entry.hint.textContent = o.mode === "split"
        ? "One strip per curve, each on its own scale"
        : o.solo ? `Showing "${o.solo}" alone — click it again for the rest`
          : "Hover to spotlight · click to isolate · shift-click to hide";
      if (ss.length < 2 || o.mode === "split") { entry.legend.style.display = "none"; return; }
      entry.legend.style.display = "";
      const rows = [...ss].sort((a, b) => (b.last == null ? -Infinity : b.last) - (a.last == null ? -Infinity : a.last));
      for (const s of rows) {
        const dim = o.solo != null && o.solo !== s.name;
        const item = h("button", {
          class: "mx-legitem" + (s.off ? " off" : "") + (dim ? " dim" : "") + (o.solo === s.name ? " solo" : ""),
          title: `${s.name} = ${fmtNum(s.last)}\nclick to show this curve alone · shift-click to hide it`,
          onmouseenter: () => { if (!s.hidden) { entry.focus = s.name; entry.plot?.setFocus(s.name); } },
          onmouseleave: () => { entry.focus = null; entry.plot?.setFocus(null); },
          onclick: (e) => {
            if (e.shiftKey) {
              s.style.hidden = !s.style.hidden;
              if (o.solo === s.name) o.solo = null;
            } else o.solo = o.solo === s.name ? null : s.name;
            entry.focus = null;
            refresh();
            if (entry.styleBox.style.display !== "none") buildStyleBox(entry.tag, entry.styleBox);
          },
        },
          h("i", { class: "mx-swatch", style: { background: s.color, opacity: String(s.alpha) } }),
          sparkFor(s),
          h("span", { class: "mx-legname" }, s.name),
          h("span", { class: "mx-legval mono" }, fmtNum(s.last)));
        entry.legend.appendChild(item);
      }
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
        const o = optsFor(tag);
        const ss = seriesFor(tag, t0);
        const vis = ss.filter((s) => !s.hidden);
        ensureRenderer(entry, vis.length);
        entry.card.classList.toggle("wide", o.wide);
        for (const b of entry.modeBtns) b.classList.toggle("active", b.dataset.mode === o.mode);
        entry.expandBtn.classList.toggle("active", o.wide);
        entry.expandBtn.replaceChildren(icon(o.wide ? "shrink" : "expand", 14));
        entry.expandBtn.title = o.wide ? "Put this card back in the grid" : "Give this card the full width of the grid";
        entry.count.textContent = String(ss.length);
        entry.count.style.display = ss.length > 1 ? "" : "none";
        entry.foot.style.display = ss.length > 1 ? "" : "none";
        if (entry.strips) entry.strips.setSeries(vis, { x: xlab });
        else entry.plot.setSeries("line", vis,
          { x: xlab, y: o.mode === "norm" ? "0–1 per series" : ylab },
          { focus: entry.focus, endLabels });
        buildLegend(entry, ss);
        if (entry.styleBox.style.display !== "none" && entry.styleBox.childElementCount !== ss.length) {
          buildStyleBox(tag, entry.styleBox);   // only when the curve set changed: a rebuild
        }                                       // mid-drag would replace the slider you hold
      }
      noteScalars();
    }

    // The "Split" y-scale: one strip per curve, each on its own scale, sharing one x
    // axis. Deliberately a small renderer of its own rather than a mode inside plot.js —
    // the plotter's contract is a single pair of axes, and this is a stack of n of them,
    // which is a panel layout rather than a plot. It reads the same CSS variables and
    // answers the same snapshot({scale, transparent}) call, so the card's PNG button does
    // not care which of the two it is holding.
    function createStrips(canvas) {
      const ctx = canvas.getContext("2d");
      let series = [], xlab = "";
      let snapBg = null, snapping = false;

      function dpr() {
        if (snapping) return canvas.width / Math.max(1, canvas.clientWidth);
        return Math.min(window.devicePixelRatio || 1, 2);
      }
      function resize() {
        if (snapping) return;
        const r = dpr();
        const w = Math.max(1, Math.floor(canvas.clientWidth * r)), ht = Math.max(1, Math.floor(canvas.clientHeight * r));
        if (canvas.width !== w || canvas.height !== ht) { canvas.width = w; canvas.height = ht; }
      }
      function niceStep(span, n) {
        const s0 = (span || 1) / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(s0))), norm = s0 / mag;
        return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
      }
      const fmtX = (v) => (Math.abs(v) >= 1e6 ? +(v / 1e6).toFixed(1) + "M"
        : Math.abs(v) >= 1e4 ? +(v / 1e3).toFixed(1) + "k" : String(+v.toPrecision(4)));
      function shorten(text, maxPx) {
        if (ctx.measureText(text).width <= maxPx) return text;
        let s = text;
        while (s.length > 3 && ctx.measureText(s + "…").width > maxPx) s = s.slice(0, -1);
        return s + "…";
      }

      function draw() {
        resize();
        const r = dpr(), W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        if (snapBg) { ctx.fillStyle = snapBg; ctx.fillRect(0, 0, W, H); }
        const live = series.filter((s) => s.plain && s.plain.some((p) => p[1] != null && isFinite(p[1])));
        if (!live.length) return;

        const css = getComputedStyle(document.documentElement);
        const fg = css.getPropertyValue("--text").trim() || "#e5e7eb";
        const muted = css.getPropertyValue("--muted").trim() || "#94a3b8";
        const band = css.getPropertyValue("--hover").trim() || "rgba(30,41,59,.5)";

        const NAME = 116 * r, VAL = 58 * r, BOT = 20 * r, TOP = 4 * r, GAP = 6 * r;
        const bandH = (H - TOP - BOT - GAP * (live.length - 1)) / live.length;
        const l = NAME, w = Math.max(20, W - NAME - VAL);
        if (bandH < 8 * r) return;

        let x0 = Infinity, x1 = -Infinity;
        for (const s of live) for (const [x, y] of s.plain) {
          if (y == null || !isFinite(y)) continue;
          if (x < x0) x0 = x; if (x > x1) x1 = x;
        }
        if (!(x1 > x0)) { x1 = x0 + 1; }
        const sx = (x) => l + (x - x0) / (x1 - x0) * w;

        live.forEach((s, i) => {
          const bt = TOP + i * (bandH + GAP);
          let lo = Infinity, hi = -Infinity;
          for (const [, y] of s.plain) if (y != null && isFinite(y)) { if (y < lo) lo = y; if (y > hi) hi = y; }
          const span = (hi - lo) || Math.max(1e-12, Math.abs(hi) || 1);
          const sy = (y) => bt + bandH - 3 * r - (y - lo) / span * (bandH - 6 * r);
          const alpha = s.alpha == null ? 1 : s.alpha;

          ctx.globalAlpha = 1;
          ctx.fillStyle = band; ctx.fillRect(l, bt, w, bandH);

          ctx.textAlign = "right"; ctx.textBaseline = "middle";
          ctx.font = `${11 * r}px system-ui, sans-serif`; ctx.fillStyle = fg;
          ctx.fillText(shorten(s.name, NAME - 14 * r), l - 8 * r, bt + bandH / 2 - 6 * r);
          ctx.font = `${9.5 * r}px ui-monospace, Menlo, monospace`; ctx.fillStyle = muted;
          ctx.fillText(`${fmtNum(lo)} – ${fmtNum(hi)}`, l - 8 * r, bt + bandH / 2 + 6 * r);

          ctx.save();
          ctx.beginPath(); ctx.rect(l, bt, w, bandH); ctx.clip();
          ctx.beginPath();
          let started = false;
          for (const [x, y] of s.plain) {
            if (y == null || !isFinite(y)) { started = false; continue; }
            const px = sx(x), py = sy(y);
            if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
          }
          ctx.strokeStyle = s.color; ctx.lineWidth = (s.width || 1.5) * r;
          ctx.lineJoin = "round"; ctx.lineCap = "round";
          ctx.globalAlpha = alpha; ctx.stroke();
          // close the same path down to the baseline for a faint fill: on a 40px strip
          // the area is what carries the shape, the stroke alone is too thin to read
          ctx.lineTo(sx(x1), bt + bandH); ctx.lineTo(sx(x0), bt + bandH); ctx.closePath();
          ctx.fillStyle = s.color; ctx.globalAlpha = alpha * 0.14; ctx.fill();
          ctx.restore();

          ctx.globalAlpha = 1;
          ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.font = `${10 * r}px ui-monospace, Menlo, monospace`; ctx.fillStyle = s.color;
          ctx.fillText(shorten(fmtNum(s.last), VAL - 10 * r), l + w + 6 * r, bt + bandH / 2);
        });

        // one shared x axis under the stack
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.font = `${10.5 * r}px system-ui, sans-serif`; ctx.fillStyle = muted;
        const step = niceStep(x1 - x0, Math.max(2, Math.min(6, Math.round(w / (86 * r)))));
        for (let v = Math.ceil(x0 / step) * step; v <= x1; v += step) ctx.fillText(fmtX(v), sx(v), H - BOT + 4 * r);
        if (xlab) { ctx.textAlign = "right"; ctx.fillStyle = fg; ctx.fillText(xlab, W - 4 * r, H - BOT + 4 * r); }
      }

      if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(canvas);

      return {
        setSeries(ss, lbls) { series = ss || []; xlab = (lbls && lbls.x) || ""; draw(); },
        redraw: draw,
        setFocus() { /* nothing to spotlight: every curve already has its own strip */ },
        async snapshot({ scale = 2, transparent = false } = {}) {
          const ow = canvas.width, oh = canvas.height;
          if (!transparent) snapBg = getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || "#0f172a";
          snapping = true;
          canvas.width = Math.max(1, Math.floor(canvas.clientWidth * scale));
          canvas.height = Math.max(1, Math.floor(canvas.clientHeight * scale));
          draw();
          const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
          snapping = false; snapBg = null;
          canvas.width = ow; canvas.height = oh;
          draw();
          return blob;
        },
      };
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
      onShow() { for (const e of plots.values()) (e.strips || e.plot)?.redraw(); },
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
    const savedImg = prefs.get("images");
    const saveImages = changeWriter(prefs, "images");
    const noteImages = () => saveImages(() => ({
      tag: tagName, open: sheetPanel.style.display !== "none", sheet: { ...sheet },
    }), 0);

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
    const sheetBtn = h("button", {
      class: "btn-icon", title: "Progression sheet: N frames spanning the run, composed into one PNG",
      onclick: toggleSheet,
    }, icon("sheet", 15));

    let tagName = (manifest.images.some((t) => t.tag === savedImg.tag) ? savedImg.tag : manifest.images[0]?.tag) || null;
    let idx = 0, playing = false, ticker = null;
    const cache = new Map();      // "tag@wall" -> object URL
    const decoded = new Map();    // "tag@wall" -> HTMLImageElement, for compositing

    const sheetPanel = h("div", { class: "mx-sheet", style: { display: "none" } });

    el.appendChild(chips);
    el.appendChild(h("div", { class: "mx-imgwrap" }, img));
    el.appendChild(h("div", { class: "mx-transport" }, playBtn, range, frameLabel, saveBtn, sheetBtn));
    el.appendChild(sheetPanel);

    const current = () => manifest.images.find((t) => t.tag === tagName) || null;
    const frames = () => current()?.frames || [];

    function buildChips() {
      clear(chips);
      for (const t of manifest.images) {
        chips.appendChild(h("button", {
          class: "mx-chip" + (t.tag === tagName ? " active" : ""),
          onclick: () => {
            tagName = t.tag;
            idx = frames().length - 1;
            syncRange(); show(); buildChips();
            noteImages();
            if (sheetPanel.style.display !== "none") schedulePreview(0);
          },
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
      sheetBtn.disabled = n < 2;
    }

    // Steps are all 0 here, so the honest stamp is elapsed time into the run.
    function stamp(fs, i) {
      if (fs.some((f) => f.step !== fs[0].step)) return `step ${fs[i].step}`;
      const secs = Math.max(0, Math.round(fs[i].wall - fs[0].wall));
      const mm = String(Math.floor(secs / 60)).padStart(2, "0");
      const ss = String(secs % 60).padStart(2, "0");
      return `+${mm}:${ss}`;
    }
    function label(fs, i) {
      const s = stamp(fs, i);
      return `${s}${s.startsWith("+") ? " into the run" : ""}  (${i + 1}/${fs.length})`;
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

    // ---- progression sheet -------------------------------------------------
    // The transport plays the epochs as a movie; this takes the same run away as one
    // picture. Frames are chosen by index over the wall-clock-ordered list the manifest
    // already carries, which is what makes it work with hax's step-0 image logging, and
    // the sheet is composed here from frames the panel has (or fetches once), so nothing
    // is asked of the backend that the movie did not already ask.
    const sv = savedImg.sheet || {};
    const sheet = {
      n: Number.isFinite(sv.n) ? Math.min(24, Math.max(2, sv.n)) : 10,
      pick: ["even", "first", "last"].includes(sv.pick) ? sv.pick : "even",
      cols: Number.isFinite(sv.cols) && sv.cols >= 0 && sv.cols <= 6 ? sv.cols : 4,
      captions: sv.captions !== false,
      header: sv.header !== false,
      arrow: sv.arrow !== false,
    };
    const sheetPrev = h("canvas", { class: "mx-sheetprev" });
    const sheetStatus = h("div", { class: "mx-hint" }, "");
    let sheetBuilt = false, previewTimer = null;

    function toggleSheet() {
      const open = sheetPanel.style.display === "none";
      sheetPanel.style.display = open ? "" : "none";
      sheetBtn.classList.toggle("active", open);
      if (!open) { noteImages(); return; }
      if (!sheetBuilt) { buildSheetPanel(); sheetBuilt = true; }
      schedulePreview(0);
    }

    function sheetSelect(text, options, get, set) {
      return h("label", { class: "mx-ctl" }, h("span", null, text),
        h("select", {
          class: "input mx-select",
          onchange: (e) => { set(e.target.value); schedulePreview(); },
        }, ...options.map(([v, t]) => h("option", { value: v, selected: String(get()) === v }, t))));
    }
    function sheetCheck(text, get, set, title) {
      return h("label", { class: "mx-ctl", title: title || "" },
        h("input", { type: "checkbox", checked: get(), onchange: (e) => { set(e.target.checked); schedulePreview(); } }),
        h("span", null, text));
    }

    function buildSheetPanel() {
      const nOut = h("span", { class: "mono small" }, String(sheet.n));
      const form = h("div", { class: "mx-sheetform" },
        h("div", { class: "mx-raillabel" }, "Progression sheet"),
        h("label", { class: "mx-ctl" }, h("span", null, "Frames"), nOut),
        h("input", {
          type: "range", class: "mx-range", min: "2", max: "24", step: "1", value: String(sheet.n),
          oninput: (e) => { sheet.n = +e.target.value; nOut.textContent = String(sheet.n); schedulePreview(); },
        }),
        sheetSelect("Pick", [["even", "Evenly spaced"], ["first", "First N"], ["last", "Last N"]],
          () => sheet.pick, (v) => (sheet.pick = v)),
        sheetSelect("Columns", [["0", "Auto"], ["1", "1"], ["2", "2"], ["3", "3"], ["4", "4"], ["5", "5"], ["6", "6"]],
          () => sheet.cols, (v) => (sheet.cols = +v)),
        sheetCheck("Caption each frame", () => sheet.captions, (v) => (sheet.captions = v),
          "Write the step (or elapsed time) and the frame number under each panel"),
        sheetCheck("Header strip", () => sheet.header, (v) => (sheet.header = v),
          "Tag, run folder and the span the sheet covers"),
        sheetCheck("Direction arrow", () => sheet.arrow, (v) => (sheet.arrow = v),
          "A training→ arrow across the top, so the reading order is unambiguous"),
        h("p", { class: "field-help", style: { margin: "2px 0 8px" } },
          "Resolution and background follow the Export settings above."),
        h("button", { class: "btn", style: { width: "100%", justifyContent: "center" }, onclick: downloadSheet },
          icon("save", 15), "Download PNG"),
        sheetStatus);
      mount(sheetPanel, h("div", { class: "mx-sheetgrid" }, form, h("div", { class: "mx-sheetprevwrap" }, sheetPrev)));
    }

    function pickIndices() {
      const total = frames().length;
      const n = Math.max(1, Math.min(sheet.n, total));
      if (sheet.pick === "last") return Array.from({ length: n }, (_, i) => total - n + i);
      if (sheet.pick === "first") return Array.from({ length: n }, (_, i) => i);
      if (n === 1) return [total - 1];
      // evenly spaced, first and last always in
      return Array.from({ length: n }, (_, i) => Math.round(i * (total - 1) / (n - 1)));
    }

    // The movie caches object URLs; the sheet needs decoded bitmaps. Same keys, so a
    // frame you have already looked at costs nothing to put on the sheet.
    async function frameImage(i) {
      const t = current(), fs = frames();
      const key = `${t.tag}@${fs[i].wall}`;
      let im = decoded.get(key);
      if (im) return im;
      let url = cache.get(key);
      if (!url) {
        url = URL.createObjectURL(await api.metricsImage(logdir, t.runs[0], t.tag, i));
        cache.set(key, url);
      }
      im = await new Promise((res, rej) => {
        const x = new Image();
        x.onload = () => res(x);
        x.onerror = () => rej(new Error(`frame ${i + 1} could not be decoded`));
        x.src = url;
      });
      decoded.set(key, im);
      return im;
    }

    async function composeSheet(scale) {
      const t = current(), fs = frames();
      if (!t || fs.length < 2) throw new Error("this tag has only one frame — there is no progression to draw");
      const idxs = pickIndices();
      const imgs = await Promise.all(idxs.map(frameImage));
      const iw = imgs[0].naturalWidth || 1, ih = imgs[0].naturalHeight || 1;
      // A hax image panel can be 1500px wide; four of those side by side, times the
      // export scale, is a PNG no viewer wants. Cap the cell and let the scale do the DPI.
      const cw = Math.min(iw, 512), ch = Math.max(1, Math.round(ih * (cw / iw)));
      const cols = sheet.cols || Math.min(idxs.length, Math.max(2, Math.round(Math.sqrt(idxs.length * 1.6))));
      const rows = Math.ceil(idxs.length / cols);
      const capH = sheet.captions ? 15 : 0, headH = sheet.header ? 36 : 0, arrowH = sheet.arrow ? 16 : 0;
      const pad = 14, gap = 10;
      const W = pad * 2 + cols * cw + (cols - 1) * gap;
      const H = pad * 2 + headH + arrowH + rows * (ch + capH) + (rows - 1) * gap;

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(W * scale);
      canvas.height = Math.round(H * scale);
      const g = canvas.getContext("2d");
      g.scale(scale, scale);

      const css = getComputedStyle(document.documentElement);
      const fg = css.getPropertyValue("--text").trim() || "#f1f5f9";
      const dim = css.getPropertyValue("--faint").trim() || "#64748b";
      if (!exportOpts.transparent) {
        g.fillStyle = css.getPropertyValue("--panel").trim() || "#0f172a";
        g.fillRect(0, 0, W, H);
      }

      let y = pad;
      if (headH) {
        g.textAlign = "left"; g.textBaseline = "top";
        g.fillStyle = fg; g.font = "600 15px system-ui, sans-serif";
        g.fillText(t.tag, pad, y);
        g.fillStyle = dim; g.font = "11px ui-monospace, Menlo, monospace";
        g.fillText(`${logdir.replace(/\/+$/, "").split("/").pop()} · ${idxs.length} of ${fs.length} frames · ` +
          `${stamp(fs, idxs[0])} → ${stamp(fs, idxs[idxs.length - 1])}`, pad, y + 19);
        y += headH;
      }
      if (arrowH) {
        const ay = y + 5;
        g.globalAlpha = 0.8; g.strokeStyle = dim; g.fillStyle = dim; g.lineWidth = 1;
        g.beginPath(); g.moveTo(pad, ay); g.lineTo(W - pad - 8, ay); g.stroke();
        g.beginPath(); g.moveTo(W - pad - 8, ay - 3.5); g.lineTo(W - pad, ay); g.lineTo(W - pad - 8, ay + 3.5);
        g.closePath(); g.fill();
        g.font = "10px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "bottom";
        g.fillText("training", W / 2, ay - 3);
        g.globalAlpha = 1;
        y += arrowH;
      }

      idxs.forEach((fi, k) => {
        const cx = pad + (k % cols) * (cw + gap);
        const cy = y + Math.floor(k / cols) * (ch + capH + gap);
        g.drawImage(imgs[k], cx, cy, cw, ch);
        g.strokeStyle = exportOpts.transparent ? "rgba(148,163,184,.35)" : dim;
        g.globalAlpha = exportOpts.transparent ? 1 : 0.35;
        g.lineWidth = 1;
        g.strokeRect(cx + 0.5, cy + 0.5, cw - 1, ch - 1);
        g.globalAlpha = 1;
        if (capH) {
          g.fillStyle = dim; g.font = "10.5px ui-monospace, Menlo, monospace"; g.textBaseline = "top";
          g.textAlign = "left"; g.fillText(stamp(fs, fi), cx + 1, cy + ch + 3);
          g.textAlign = "right"; g.fillText(`${fi + 1}/${fs.length}`, cx + cw - 1, cy + ch + 3);
        }
      });
      return { canvas, W, H, n: idxs.length };
    }

    function schedulePreview(delay = 250) {
      noteImages();
      if (previewTimer) clearTimeout(previewTimer);
      previewTimer = setTimeout(() => { previewTimer = null; renderPreview(); }, delay);
    }

    async function renderPreview() {
      if (sheetPanel.style.display === "none" || destroyed) return;
      sheetStatus.textContent = "Reading frames…";
      try {
        const { canvas, W, H, n } = await composeSheet(1);
        sheetPrev.width = canvas.width;
        sheetPrev.height = canvas.height;
        sheetPrev.getContext("2d").drawImage(canvas, 0, 0);
        sheetPrev.style.width = `${Math.min(W, 720)}px`;
        sheetPrev.style.height = "auto";
        sheetStatus.textContent =
          `${n} frames · ${Math.round(W * exportOpts.scale)} × ${Math.round(H * exportOpts.scale)} px at ${exportOpts.scale}×`;
      } catch (err) {
        mount(sheetStatus, errorBox(err));
      }
    }

    async function downloadSheet() {
      try {
        const { canvas, n } = await composeSheet(exportOpts.scale);
        const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
        download(blob, `${current().tag.replace(/[^\w.-]+/g, "_")}_progression_${n}.png`);
      } catch (err) {
        mount(sheetStatus, errorBox(err));
      }
    }

    buildChips();
    idx = Math.max(0, frames().length - 1);   // open on the newest frame
    syncRange();
    show();

    // Reopen the sheet if that is how the run was left — but leave the preview to
    // onShow, so a panel sitting behind the Scalars tab doesn't fetch frames nobody
    // is looking at.
    if (savedImg.open && frames().length > 1) {
      sheetPanel.style.display = "";
      sheetBtn.classList.add("active");
      buildSheetPanel();
      sheetBuilt = true;
    }

    return {
      el,
      onShow() { if (sheetPanel.style.display !== "none") schedulePreview(0); },
      onManifest() {
        const atEnd = idx >= frames().length - 1;
        buildChips();
        syncRange();
        if (atEnd) { idx = Math.max(0, frames().length - 1); show(); }   // a live run keeps following
        if (sheetPanel.style.display !== "none") schedulePreview(600);   // new epochs land on the sheet too
      },
      destroy() {
        if (ticker) clearInterval(ticker);
        if (previewTimer) clearTimeout(previewTimer);
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
    const colorSel = h("select", { class: "input mx-select", onchange: (e) => { colorBy = +e.target.value; noteLatent(); paintColors(); } });
    const colorRow = h("label", { class: "mx-ctl", style: { display: "none" } }, h("span", null, "Colour by"), colorSel);
    const spriteCanvas = h("canvas", { class: "mx-sprite", width: 96, height: 96 });
    const spriteBox = h("div", { class: "mx-spritebox", style: { display: "none" } },
      h("div", { class: "mx-raillabel" }, "Selected particle"), spriteCanvas);

    const savedLat = prefs.get("latent");
    const saveLatent = changeWriter(prefs, "latent");
    const noteLatent = () => saveLatent(() => ({ tag: tagName, dims, colorBy }), 0);

    const tags = [...new Set(manifest.embeddings.map((e) => e.tag))];
    let tagName = (tags.includes(savedLat.tag) ? savedLat.tag : tags[0]) || null;
    // dims stays 2 here even when 3 was saved: the WebGL cloud is built lazily, so the
    // restore goes through setDims once the view exists (below).
    let idx = 0, playing = false, ticker = null, dims = 2;
    let colorBy = Number.isInteger(savedLat.colorBy) ? savedLat.colorBy : -1;
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

    const tagSel = h("select", { class: "input mx-select", onchange: (e) => { tagName = e.target.value; noteLatent(); resetCaches(); idx = steps().length - 1; syncRange(); show(); } },
      ...tags.map((t) => h("option", { value: t, selected: t === tagName }, t)));
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
      noteLatent();
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
      if (colorBy >= cols.length) colorBy = -1;   // a restored column this space doesn't have
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
    if (savedLat.dims === 3) setDims(3);   // builds the WebGL cloud, then shows the epoch
    else show();

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
      // Open where this run was left, or on whichever view has something in it.
      const has = { scalars: manifest.scalars.length, images: manifest.images.length, latent: manifest.embeddings.length };
      if (!has[tab]) tab = has.scalars ? "scalars" : has.images ? "images" : "latent";
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

  // The colour legends hax logs with add_text (what blue/red mean on the variation
  // map). The payload comes straight out of a run's event files — an arbitrary
  // server directory the user pointed the Metrics page at — so it is UNTRUSTED
  // HTML: sanitize it, never hand it to an `html:` prop (that was an XSS).
  function legends(texts) {
    const box = h("details", { class: "mx-legends" }, h("summary", null, "Colour legends"));
    for (const html of Object.values(texts)) {
      const div = h("div", { class: "mx-legendtext" });
      div.appendChild(sanitizedHtml(html));
      box.appendChild(div);
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
