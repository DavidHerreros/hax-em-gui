// Landing page: logo hero, documentation links, and a "how to use the GUI" guide.
import { h } from "../h.js";
import { icon } from "../icons.js";
import { UpdateBanner } from "../updates.js";
import { SetupNotice } from "./setupNotice.js";

const DOCS = "https://davidherreros.github.io/hax-em-docs/";

const DOC_CARDS = [
  { icon: "rocket", title: "Getting started", desc: "Install Hax, set up your environment, and learn the core concepts behind conformational heterogeneity analysis.", href: DOCS + "getting-started/" },
  { icon: "fileText", title: "Programs reference", desc: "What each tool does and every parameter it accepts — the same options the GUI generates its forms from.", href: DOCS + "programs/" },
  { icon: "workflow", title: "Workflows & examples", desc: "End-to-end pipelines and real-case walkthroughs combining several programs.", href: DOCS + "workflows/" },
];

const STEPS = [
  { icon: "box", title: "Create a project", desc: "Set a projects folder in Settings, then create or open a project. Each project keeps all of its runs together." },
  { icon: "sliders", title: "Pick a program", desc: "Browse programs in the right-hand menu and fill the auto-generated form. Output folders are created for you." },
  { icon: "terminal", title: "Run & follow logs", desc: "Launch on a chosen GPU into the active project and watch the live log stream with an animated progress line." },
  { icon: "chart", title: "Inspect metrics", desc: "Open one or more TensorBoard tabs in Metrics to explore training and validation curves." },
];

export function HomeView() {
  return h(
    "div",
    { class: "view" },
    UpdateBanner(),
    SetupNotice(),
    hero(),
    section("Documentation", h("div", { class: "grid", style: { marginBottom: "36px" } }, ...DOC_CARDS.map(docCard))),
    section("Using the GUI", h("div", { class: "steps" }, ...STEPS.map(step)))
  );
}

function hero() {
  return h(
    "div",
    { class: "hero" },
    h("div", { class: "logo-tile" }, h("img", { src: "/assets/logo.png", alt: "Hax logo" })),
    h(
      "div",
      null,
      h("h1", null, "Hax"),
      h("p", { class: "tagline" }, "A suite of tools for studying conformational heterogeneity in Cryo-EM data with JAX-powered neural networks — now driveable from your browser, against a remote GPU server."),
      h(
        "div",
        { class: "hero-actions" },
        h("a", { class: "btn", href: "#/projects" }, icon("box", 18), "Go to Projects"),
        h("a", { class: "btn secondary", href: DOCS, target: "_blank", rel: "noreferrer" }, icon("book", 18), "Open documentation")
      )
    )
  );
}

function section(title, body) {
  return h("section", { style: { marginBottom: "8px" } }, h("h2", { class: "section" }, title), body);
}

function docCard(c, i) {
  return h(
    "a",
    { class: "feature", href: c.href, target: "_blank", rel: "noreferrer", style: { animationDelay: `${i * 0.04}s` } },
    ficon(c.icon),
    h("h3", null, c.title),
    h("p", null, c.desc),
    h("span", { class: "ext" }, "Read the docs", icon("external", 14))
  );
}

function step(s, i) {
  return h(
    "div",
    { class: "step", style: { animationDelay: `${i * 0.04}s` } },
    h("div", { class: "num" }),
    h("h3", null, h("span", { style: { verticalAlign: "middle" } }, s.title)),
    h("p", null, s.desc)
  );
}

function ficon(name) {
  const box = h("div", { class: "ficon" });
  box.appendChild(icon(name, 20));
  return box;
}
