// Auto-layout for the workflow builder: arrange a dependency DAG into tiers, one
// independent subworkflow at a time.
//
// The layout this replaced bucketed nodes by depth and then packed each tier
// left-to-right in *node-creation order*. Nothing knew which subworkflow a step
// belonged to and nothing looked at where its parent had ended up, so a step's
// position came from its index within its tier — which is why unrelated workflows
// interleaved and edges ran diagonally across the whole canvas.
//
// Three passes fix that:
//
//   1. tiers      — longest path from a root, so an edge never points backwards.
//   2. components — weakly connected subgraphs are laid out separately and parked
//                   side by side. Two steps with no path between them can then never
//                   push each other around. This is the change that matters most.
//   3. ordering   — barycenter sweeps put each tier in the order its neighbours
//                   suggest, then a coordinate pass pulls every step toward the mean
//                   of its neighbours, separated just enough not to overlap.
//
// Pass 3 is what centres a parent over its children. It is deliberately *not* a tidy
// tree: a step here has three input ports, so a step fed by two upstream runs is
// normal, and a tree layout would have to pick one parent and ignore the other edges.
// Working from the mean of all neighbours places a merge between its parents instead.
//
// Geometry is expressed as (main, cross): main is the dependency axis, cross is the
// one nodes are spread along. The caller maps those onto x/y, so the same code serves
// both orientations — in horizontal mode the cross axis is y, where nodes have
// *different* sizes, which is why cross positions are centres and separation is done
// with half-sizes rather than a constant pitch.

// Longest-path depth per node, or null if the graph has a cycle (nothing can be
// tiered then, and the caller wants to say so rather than draw nonsense).
function tierDepths(ids, kids, pars) {
  const depth = {}, indeg = {};
  for (const id of ids) { depth[id] = 0; indeg[id] = pars.get(id).length; }
  const queue = ids.filter((id) => indeg[id] === 0);
  let placed = 0;
  while (queue.length) {
    const id = queue.shift();
    placed++;
    for (const k of kids.get(id)) {
      depth[k] = Math.max(depth[k], depth[id] + 1);
      if (--indeg[k] === 0) queue.push(k);
    }
  }
  return placed === ids.length ? depth : null;
}

// Weakly connected components. Seeding from `ids` (node-creation order) keeps the
// left-to-right order of subworkflows stable as the graph is edited.
//
// Each component is returned in creation order too, not in the order the walk happened
// to reach its nodes. That matters more than it sounds: this order becomes the initial
// order of every tier, and ties in the barycenter sort below keep it — so without the
// sort, siblings come out reversed (the walk uses a stack) and, worse, the result
// depends on the order the *edges* were declared, which is not part of the graph.
function componentsOf(ids, kids, pars) {
  const rankOf = new Map(ids.map((id, i) => [id, i]));
  const seen = new Set(), out = [];
  for (const start of ids) {
    if (seen.has(start)) continue;
    const comp = [], stack = [start];
    seen.add(start);
    while (stack.length) {
      const id = stack.pop();
      comp.push(id);
      for (const m of kids.get(id)) if (!seen.has(m)) { seen.add(m); stack.push(m); }
      for (const m of pars.get(id)) if (!seen.has(m)) { seen.add(m); stack.push(m); }
    }
    comp.sort((a, b) => rankOf.get(a) - rankOf.get(b));
    out.push(comp);
  }
  return out;
}

const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);

const SWEEPS = 8;      // barycenter ordering passes; well past convergence at this size
const RELAX = 24;      // coordinate passes; likewise

/**
 * @param nodes  [{ id, ... }] in creation order
 * @param edges  [{ from, to }]
 * @param opts   { crossSize(node)->number, mainStep, pad, gapCross, gapComp }
 * @returns { main: {id->number}, cross: {id->number} } with `cross` the node's CENTRE,
 *          or null if the graph has a cycle.
 */
export function layeredLayout(nodes, edges, opts) {
  const { crossSize, mainStep, pad = 40, gapCross = 26, gapComp = 72 } = opts;
  const ids = nodes.map((n) => n.id);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const kids = new Map(ids.map((id) => [id, []]));
  const pars = new Map(ids.map((id) => [id, []]));
  for (const e of edges) {
    // Ignore edges to nodes that aren't here — the builder can hold a partial graph.
    if (!byId.has(e.from) || !byId.has(e.to) || e.from === e.to) continue;
    kids.get(e.from).push(e.to);
    pars.get(e.to).push(e.from);
  }

  const depth = tierDepths(ids, kids, pars);
  if (!depth) return null;

  const size = (id) => crossSize(byId.get(id));
  const cross = {};
  let compAt = pad;

  for (const comp of componentsOf(ids, kids, pars)) {
    const inComp = new Set(comp);
    const tiers = [];
    for (const id of comp) (tiers[depth[id]] ||= []).push(id);
    for (let i = 0; i < tiers.length; i++) tiers[i] ||= [];

    // ── 1. order each tier to follow its neighbours ──
    // Ordering has to happen before placement: the coordinate pass only nudges nodes
    // along, it can never drag one past another, so a tangle here stays tangled.
    const rank = {};
    tiers.forEach((t) => t.forEach((id, i) => (rank[id] = i)));
    for (let s = 0; s < SWEEPS; s++) {
      const down = s % 2 === 0;
      const order = tiers.map((_, i) => i);
      for (const k of down ? order.slice(1) : order.slice(0, -1).reverse()) {
        const tier = tiers[k], bary = {};
        for (const id of tier) {
          const nb = (down ? pars : kids).get(id).filter((x) => inComp.has(x));
          bary[id] = nb.length ? mean(nb.map((x) => rank[x])) : rank[id];
        }
        // Ties keep their previous order, so a stable graph lays out identically twice.
        tier.sort((a, b) => bary[a] - bary[b] || rank[a] - rank[b]);
        tier.forEach((id, i) => (rank[id] = i));
      }
    }

    // ── 2. place along the cross axis ──
    for (const tier of tiers) {
      let at = 0;
      for (const id of tier) { cross[id] = at + size(id) / 2; at += size(id) + gapCross; }
    }
    for (let s = 0; s < RELAX; s++) {
      const down = s % 2 === 0;
      const order = tiers.map((_, i) => i);
      for (const k of down ? order : order.slice().reverse()) {
        const tier = tiers[k];
        if (!tier.length) continue;
        // Where each node would sit if nothing were in the way.
        const want = tier.map((id) => {
          const nb = (down ? pars : kids).get(id).filter((y) => inComp.has(y));
          return nb.length ? mean(nb.map((y) => cross[y])) : cross[id];
        });
        // Separate left-to-right by half-sizes, so differently sized nodes still clear.
        const got = [];
        let prev = null;
        for (let i = 0; i < tier.length; i++) {
          let v = want[i];
          if (prev !== null) v = Math.max(v, prev + size(tier[i - 1]) / 2 + gapCross + size(tier[i]) / 2);
          got.push(v);
          prev = v;
        }
        // Separation only ever pushes right, so without this the tier would ratchet
        // away a little on every pass. Restoring the centre of mass keeps the tier
        // where the graph asked for it while honouring the gaps.
        const shift = mean(want) - mean(got);
        tier.forEach((id, i) => (cross[id] = got[i] + shift));
      }
    }

    // ── 3. park this subworkflow after the previous one ──
    let lo = Infinity, hi = -Infinity;
    for (const id of comp) {
      lo = Math.min(lo, cross[id] - size(id) / 2);
      hi = Math.max(hi, cross[id] + size(id) / 2);
    }
    const shift = compAt - lo;
    for (const id of comp) cross[id] += shift;
    compAt += hi - lo + gapComp;
  }

  const main = {};
  for (const id of ids) main[id] = pad + depth[id] * mainStep;
  return { main, cross };
}
