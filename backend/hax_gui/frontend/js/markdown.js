// Minimal Markdown -> DOM renderer. Builds real nodes via `h` (textContent, never
// innerHTML), so rendered model output can't inject markup — safe by construction,
// no external dependency, no build step. Covers the constructs the assistant emits:
// headings, bold/italic, inline + fenced code, ordered/unordered/nested lists,
// tables, blockquotes, links and horizontal rules.
//
// Note: emphasis is asterisk-only (`*`/`**`); underscores are left as literal text
// so flag names like --dataset_split_fraction are never mangled. LaTeX math
// (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) is rendered by a small built-in
// transformer (mathToNodes) covering the subset the assistant emits — no bundled
// math engine: super/subscripts, \frac, \sqrt, \mathbf/\text, and common
// operators/Greek mapped to Unicode.
import { h } from "./h.js";

export function renderMarkdown(src) {
  const frag = document.createDocumentFragment();
  for (const node of blocksToNodes(String(src == null ? "" : src))) frag.appendChild(node);
  return frag;
}

// ---- block level ----
function isBlockStart(line) {
  return (
    /^\s*```/.test(line) ||
    /^\s*#{1,6}\s+/.test(line) ||
    /^\s*([-*_])\1\1+\s*$/.test(line) ||
    /^\s*>\s?/.test(line) ||
    /^\s*([-*+]|\d+[.)])\s+/.test(line)
  );
}

function blocksToNodes(src) {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i++; continue; }

    // fenced code
    if (/^\s*```/.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // consume closing fence (if present)
      out.push(h("pre", { class: "md-pre" }, h("code", null, buf.join("\n"))));
      continue;
    }

    // heading
    const hd = line.match(/^\s*(#{1,6})\s+(.*)$/);
    if (hd) {
      const tag = "h" + Math.min(6, hd[1].length);
      out.push(h(tag, { class: "md-h" }, ...inlineToNodes(hd[2].trim())));
      i++; continue;
    }

    // horizontal rule
    if (/^\s*([-*_])\1\1+\s*$/.test(line)) { out.push(h("hr", { class: "md-hr" })); i++; continue; }

    // blockquote
    if (/^\s*>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
      out.push(h("blockquote", { class: "md-quote" }, ...blocksToNodes(buf.join("\n"))));
      continue;
    }

    // table (header row + separator row)
    if (line.includes("|") && i + 1 < lines.length &&
        /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
      const headers = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      out.push(h("div", { class: "md-table-wrap" },
        h("table", { class: "md-table" },
          h("thead", null, h("tr", null, ...headers.map((c) => h("th", null, ...inlineToNodes(c))))),
          h("tbody", null, ...rows.map((r) => h("tr", null, ...r.map((c) => h("td", null, ...inlineToNodes(c)))))))));
      continue;
    }

    // list
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const [node, next] = parseList(lines, i);
      out.push(node);
      i = next;
      continue;
    }

    // paragraph
    const buf = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) { buf.push(lines[i]); i++; }
    out.push(h("p", { class: "md-p" }, ...inlineToNodes(buf.join("\n"))));
  }
  return out;
}

function parseList(lines, start) {
  const first = lines[start].match(/^(\s*)([-*+]|\d+[.)])\s+/);
  const baseIndent = first[1].length;
  const ordered = /\d/.test(first[2]);
  const items = [];
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) break;
    const m = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (!m) break;
    const indent = m[1].length;
    if (indent < baseIndent) break;
    if (indent > baseIndent) {           // nested list -> attach to previous item
      const [sub, next] = parseList(lines, i);
      if (items.length) items[items.length - 1].appendChild(sub);
      i = next;
      continue;
    }
    items.push(h("li", { class: "md-li" }, ...inlineToNodes(m[3])));
    i++;
  }
  return [h(ordered ? "ol" : "ul", { class: "md-list" }, ...items), i];
}

// ---- inline level ----
function safeLink(text, url) {
  const href = /^(https?:|mailto:)/i.test(url) ? url : "#";
  return h("a", { href, target: "_blank", rel: "noreferrer noopener", class: "md-link" },
    ...inlineToNodes(text));
}

const INLINE = [
  { re: /`([^`]+)`/, node: (m) => h("code", { class: "md-code" }, m[1]) },
  { re: /\[([^\]]+)\]\(([^)\s]+)\)/, node: (m) => safeLink(m[1], m[2]) },
  // LaTeX math — display first so `$$` beats `$`
  { re: /\$\$([\s\S]+?)\$\$/, node: (m) => h("span", { class: "md-math md-math-display" }, ...mathToNodes(m[1])) },
  { re: /\\\[([\s\S]+?)\\\]/, node: (m) => h("span", { class: "md-math md-math-display" }, ...mathToNodes(m[1])) },
  { re: /\$(?!\s)([^$\n]+?)\$/, node: (m) => h("span", { class: "md-math" }, ...mathToNodes(m[1])) },
  { re: /\\\(([\s\S]+?)\\\)/, node: (m) => h("span", { class: "md-math" }, ...mathToNodes(m[1])) },
  { re: /\*\*\*([\s\S]+?)\*\*\*/, node: (m) => h("strong", null, h("em", null, ...inlineToNodes(m[1]))) },
  { re: /\*\*([\s\S]+?)\*\*/, node: (m) => h("strong", null, ...inlineToNodes(m[1])) },
  { re: /~~([\s\S]+?)~~/, node: (m) => h("del", null, ...inlineToNodes(m[1])) },
  { re: /\*([\s\S]+?)\*/, node: (m) => h("em", null, ...inlineToNodes(m[1])) },
];

function findInline(text) {
  let best = null;
  for (const p of INLINE) {
    const m = p.re.exec(text);
    if (m && (best === null || m.index < best.m.index)) best = { p, m };
  }
  if (!best) return null;
  return { index: best.m.index, length: best.m[0].length, node: best.p.node(best.m) };
}

function inlineToNodes(text) {
  const nodes = [];
  let rest = text;
  let guard = 0;
  while (rest && guard++ < 5000) {
    const hit = findInline(rest);
    if (!hit) { nodes.push(document.createTextNode(rest)); break; }
    if (hit.index > 0) nodes.push(document.createTextNode(rest.slice(0, hit.index)));
    nodes.push(hit.node);
    rest = rest.slice(hit.index + hit.length);
  }
  return nodes;
}

// ---- LaTeX math (subset) ----
// Commands mapped to a single Unicode glyph (empty string = drop, e.g. spacing/\left).
const MATH_SYMBOLS = {
  "\\times": "×", "\\cdot": "·", "\\div": "÷", "\\ast": "∗",
  "\\pm": "±", "\\mp": "∓", "\\approx": "≈", "\\sim": "∼",
  "\\simeq": "≃", "\\cong": "≅", "\\equiv": "≡", "\\neq": "≠",
  "\\ne": "≠", "\\leq": "≤", "\\le": "≤", "\\geq": "≥", "\\ge": "≥",
  "\\ll": "≪", "\\gg": "≫", "\\to": "→", "\\rightarrow": "→",
  "\\leftarrow": "←", "\\leftrightarrow": "↔", "\\Rightarrow": "⇒",
  "\\Leftarrow": "⇐", "\\Leftrightarrow": "⇔", "\\mapsto": "↦",
  "\\infty": "∞", "\\propto": "∝", "\\partial": "∂", "\\nabla": "∇",
  "\\angle": "∠", "\\perp": "⊥", "\\parallel": "∥", "\\in": "∈",
  "\\notin": "∉", "\\subset": "⊂", "\\subseteq": "⊆", "\\cup": "∪",
  "\\cap": "∩", "\\forall": "∀", "\\exists": "∃", "\\sum": "∑",
  "\\prod": "∏", "\\int": "∫", "\\sqrt": "√",
  "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
  "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ", "\\eta": "η",
  "\\theta": "θ", "\\vartheta": "ϑ", "\\iota": "ι", "\\kappa": "κ",
  "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ", "\\pi": "π",
  "\\rho": "ρ", "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ",
  "\\phi": "φ", "\\varphi": "φ", "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
  "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
  "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Phi": "Φ", "\\Psi": "Ψ",
  "\\Omega": "Ω", "\\cdots": "⋯", "\\ldots": "…", "\\dots": "…",
  "\\vdots": "⋮", "\\prime": "′", "\\circ": "∘", "\\bullet": "•",
  "\\%": "%", "\\&": "&", "\\#": "#", "\\$": "$", "\\_": "_", "\\{": "{", "\\}": "}",
  "\\,": " ", "\\;": " ", "\\:": " ", "\\!": "", "\\ ": " ",
  "\\quad": " ", "\\qquad": "  ",
  "\\left": "", "\\right": "", "\\displaystyle": "", "\\,": " ",
};

// Render a LaTeX-math string into safe DOM nodes (super/subscripts, \frac, \sqrt,
// \mathbf/\text groups, and the MATH_SYMBOLS above). Unknown \commands degrade to
// their name; everything is textNodes/spans, so nothing can inject markup.
function mathToNodes(src, depth = 0) {
  const s = String(src == null ? "" : src);
  const out = [];
  let i = 0, guard = 0;
  const n = s.length;

  // read the argument at position i: a {..} group, a \command, or one char
  function readArg() {
    if (s[i] === "{") {
      let depth2 = 0, j = i;
      for (; j < n; j++) {
        if (s[j] === "{") depth2++;
        else if (s[j] === "}") { depth2--; if (depth2 === 0) { j++; break; } }
      }
      const inner = s.slice(i + 1, j - 1);
      i = j;
      return inner;
    }
    if (s[i] === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i));
      const tok = m ? m[0] : s[i];
      i += tok.length;
      return tok;
    }
    const ch = s[i] == null ? "" : s[i];
    i += 1;
    return ch;
  }

  while (i < n && guard++ < 10000) {
    const ch = s[i];
    if (ch === "\\") {
      const rest = s.slice(i);
      let m;
      if ((m = /^\\(mathbf|boldsymbol|mathbb)\s*\{/.exec(rest))) {
        i += m[0].length - 1;
        out.push(h("span", { class: "md-math-bf" }, ...mathToNodes(readArg(), depth + 1)));
        continue;
      }
      if ((m = /^\\(text|textrm|mathrm|mathit|mathsf|operatorname)\s*\{/.exec(rest))) {
        i += m[0].length - 1;
        out.push(h("span", { class: "md-math-rm" }, ...mathToNodes(readArg(), depth + 1)));
        continue;
      }
      if (/^\\frac\b/.test(rest) || /^\\dfrac\b/.test(rest) || /^\\tfrac\b/.test(rest)) {
        i += rest.indexOf("frac") + 4;
        const num = readArg(), den = readArg();
        out.push(h("span", { class: "md-frac" },
          h("span", { class: "md-frac-n" }, ...mathToNodes(num, depth + 1)),
          h("span", { class: "md-frac-d" }, ...mathToNodes(den, depth + 1))));
        continue;
      }
      if (/^\\sqrt\b/.test(rest)) {
        i += 5;
        out.push(document.createTextNode("√"));
        out.push(h("span", { class: "md-sqrt" }, ...mathToNodes(readArg(), depth + 1)));
        continue;
      }
      const cm = /^\\[a-zA-Z]+|^\\./.exec(rest);
      const cmd = cm ? cm[0] : "\\";
      if (Object.prototype.hasOwnProperty.call(MATH_SYMBOLS, cmd)) {
        if (MATH_SYMBOLS[cmd]) out.push(document.createTextNode(MATH_SYMBOLS[cmd]));
        i += cmd.length;
        continue;
      }
      // unknown command: keep its name (drop the backslash)
      const um = /^\\([a-zA-Z]+)/.exec(rest);
      if (um) { out.push(document.createTextNode(um[1])); i += um[0].length; continue; }
      out.push(document.createTextNode(rest[1] || "")); i += 2; continue;
    }
    if (ch === "^" && depth < 40) { i++; out.push(h("sup", null, ...mathToNodes(readArg(), depth + 1))); continue; }
    if (ch === "_" && depth < 40) { i++; out.push(h("sub", null, ...mathToNodes(readArg(), depth + 1))); continue; }
    if (ch === "{") { out.push(...mathToNodes(readArg(), depth + 1)); continue; }
    if (ch === "}") { i++; continue; }
    out.push(document.createTextNode(ch));
    i++;
  }
  return out;
}
