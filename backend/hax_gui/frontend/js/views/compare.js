// Side-by-side comparison of two runs: parameters (diff-highlighted), status,
// timing, disk and the provenance-stamped package versions. Answers "what did I
// change between the run that worked and the one that didn't" from data the
// backend already stores — nothing here mutates anything.
import { h, mount, clear } from "../h.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { errorBox, formatSize } from "./ui.js";
import { badge, jobDisplayName } from "./jobs.js";

export function CompareView(params) {
  const root = h("div", { class: "view" });
  root.appendChild(h("a", { class: "back", href: "#/jobs" }, icon("arrowLeft", 16), "Jobs"));
  const body = h("div");
  root.appendChild(body);
  mount(body, h("p", { class: "muted" }, "Loading both runs…"));

  Promise.all([api.job(params.a), api.job(params.b)])
    .then(([a, b]) => render(a, b))
    .catch((err) => mount(body, errorBox(err)));

  function fmtWhen(iso) { return iso ? new Date(iso).toLocaleString() : "—"; }
  function duration(j) {
    if (!j.started_at || !j.finished_at) return "—";
    const s = (new Date(j.finished_at) - new Date(j.started_at)) / 1000;
    if (!isFinite(s) || s < 0) return "—";
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${(s / 60).toFixed(1)} min`;
    return `${(s / 3600).toFixed(1)} h`;
  }

  // One diff row: differing values get the highlight; a value only one side has
  // shows "—" on the other, so an added/removed flag is as visible as a change.
  function row(label, va, vb, { mono = true } = {}) {
    const differ = String(va ?? "—") !== String(vb ?? "—");
    const cell = (v) => h("td", { class: (mono ? "mono " : "") + (differ ? "cmp-diff" : "") },
                          v == null || v === "" ? "—" : String(v));
    return h("tr", { class: differ ? "cmp-row-diff" : "" },
             h("td", { class: "cmp-key" }, label), cell(va), cell(vb));
  }

  function render(a, b) {
    clear(body);
    let onlyDiffs = true;

    const title = (j) => h("div", { class: "cmp-title" },
      h("a", { href: `#/jobs/${j.id}`, class: "mono cmp-name", title: "Open this run" }, jobDisplayName(j)),
      badge(j.status),
      j.project_name ? h("span", { class: "proj-chip" }, icon("box", 12), j.project_name) : null);

    body.appendChild(h("h1", { style: { fontSize: "20px", marginBottom: "12px" } }, "Compare runs"));
    if (a.program !== b.program) {
      body.appendChild(h("p", { class: "muted", style: { marginBottom: "10px" } },
        icon("sliders", 14), ` Different programs (${a.program} vs ${b.program}) — parameters are matched by flag name where they overlap.`));
    }

    // -- run facts -------------------------------------------------------
    const facts = h("table", { class: "cmp-table" },
      h("thead", null, h("tr", null, h("th", null, ""), h("th", null, title(a)), h("th", null, title(b)))),
      h("tbody", null,
        row("program", a.program, b.program),
        row("status", a.status, b.status, { mono: false }),
        row("exit code", a.return_code, b.return_code),
        row("GPU", a.gpu || "default", b.gpu || "default"),
        row("launched", fmtWhen(a.started_at), fmtWhen(b.started_at), { mono: false }),
        row("duration", duration(a), duration(b), { mono: false })));

    // -- parameters ------------------------------------------------------
    const paramsBody = h("tbody");
    const paramsTable = h("table", { class: "cmp-table" },
      h("thead", null, h("tr", null,
        h("th", null, "parameter"),
        h("th", { class: "mono" }, jobDisplayName(a)),
        h("th", { class: "mono" }, jobDisplayName(b)))),
      paramsBody);
    const emptyNote = h("p", { class: "muted small", hidden: true }, "No differing parameters — the runs were launched with identical arguments.");

    function renderParams() {
      clear(paramsBody);
      const keys = [...new Set([...Object.keys(a.args || {}), ...Object.keys(b.args || {})])].sort();
      let shown = 0, differing = 0;
      for (const k of keys) {
        const va = (a.args || {})[k], vb = (b.args || {})[k];
        const differ = String(va ?? "—") !== String(vb ?? "—");
        if (differ) differing += 1;
        if (onlyDiffs && !differ) continue;
        paramsBody.appendChild(row("--" + k, va, vb));
        shown += 1;
      }
      emptyNote.hidden = !(onlyDiffs && shown === 0);
      countEl.textContent = `${differing} of ${keys.length} parameters differ`;
    }

    const countEl = h("span", { class: "muted small" });
    const toggle = h("label", { class: "cmp-toggle" },
      h("input", { type: "checkbox", checked: true,
                   onchange: (e) => { onlyDiffs = e.target.checked; renderParams(); } }),
      h("span", null, "only differences"));

    // -- environment (provenance) ---------------------------------------
    const pa = (a.provenance && a.provenance.packages) || {};
    const pb = (b.provenance && b.provenance.packages) || {};
    const pkgKeys = [...new Set([...Object.keys(pa), ...Object.keys(pb)])].sort();
    const envRows = pkgKeys.map((k) => row(k, pa[k], pb[k]));
    const envTable = pkgKeys.length
      ? h("table", { class: "cmp-table" },
          h("thead", null, h("tr", null, h("th", null, "package"), h("th", null, ""), h("th", null, ""))),
          h("tbody", null, ...envRows))
      : h("p", { class: "muted small" }, "No provenance recorded (runs created before version stamping).");

    // -- disk (async, cached server-side) --------------------------------
    const diskA = h("span", { class: "mono" }, "…"), diskB = h("span", { class: "mono" }, "…");
    for (const [j, el] of [[a, diskA], [b, diskB]]) {
      api.jobUsage(j.id).then((u) => { el.textContent = u && u.exists ? formatSize(u.bytes) : "—"; })
        .catch(() => { el.textContent = "—"; });
    }

    body.appendChild(h("div", { class: "cmp-section" }, facts));
    body.appendChild(h("div", { class: "row section-head", style: { margin: "18px 0 8px" } },
      h("span", { class: "muted small" }, icon("sliders", 13), " Parameters"),
      countEl, h("span", { class: "ml-auto" }), toggle));
    body.appendChild(paramsTable);
    body.appendChild(emptyNote);
    body.appendChild(h("div", { class: "row section-head", style: { margin: "18px 0 8px" } },
      h("span", { class: "muted small" }, icon("cpu", 13), " Environment")));
    body.appendChild(envTable);
    body.appendChild(h("div", { class: "row section-head", style: { margin: "18px 0 8px" } },
      h("span", { class: "muted small" }, icon("folder", 13), " Disk"),
      h("span", { class: "muted small" }, "run folder sizes")));
    body.appendChild(h("table", { class: "cmp-table" },
      h("tbody", null, h("tr", null, h("td", { class: "cmp-key" }, "run folder"),
                                     h("td", null, diskA), h("td", null, diskB)))));

    renderParams();
  }

  return root;
}
