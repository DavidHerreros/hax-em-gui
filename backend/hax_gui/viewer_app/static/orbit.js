// Shared trackball camera (ChimeraX-style): the model is rotated by an accumulated
// rotation matrix and the camera sits at a fixed distance looking at the origin, so
// rotation is continuous in every axis (no gimbal lock / no pole flip) and dragging
// from any point feels consistent. Zoom by wheel. Click-vs-drag is reported via
// onClick so renderers can still pick.
//
// Panning (middle-mouse drag) translates the model in the view plane. Shift is left
// free for the app's shift-click multi-select (building a morph path).
//
// A drag that begins near the rim of the window spins the model in the plane of the
// screen instead — ChimeraX's virtual sphere. See `inEdgeZone`.
//
// Centre of rotation follows ChimeraX's `cofr frontCenter` (its default): while the
// whole model still fits in the window, rotate about the model's centre; once you
// have zoomed in past it, rotate about the front-most surface at the centre of the
// window, so the feature you are looking at stays under the crosshair instead of
// swinging away. Both halves need geometry the camera cannot see for itself, and the
// renderer supplies them: `opts.modelBounds` (what is drawn) and `opts.probeDepth`
// (how far away its front surface is).
import { identity, axisAngle, multiply, mulVec4, halfFovY, fitDistance, FOV_X_DEG } from "./mat4.js";

export function createOrbit(canvas, onChange, onClick, opts = {}) {
  // Radius of the sphere the camera frames on open (and on reset) — see fitDistance().
  // The eye distance is no longer a constant: the field is pinned horizontally
  // (ChimeraX-style), so how far back the camera must sit to frame the map depends on
  // the shape of the window. Resolved lazily, because the canvas is often still
  // unlaid-out (clientWidth 0) when the renderer is constructed.
  const fitRadius = opts.fitRadius || 1.4;
  const FOV_X_2 = (FOV_X_DEG * Math.PI) / 360;
  // Models are normalized into a [-1,1] cube (half-diagonal √3 ≈ 1.73), and every
  // renderer's frustum is near=0.01 far=100. So the camera can come right up to the
  // surface — and inside it — and still pull far enough back to see the whole box.
  const minDist = opts.minDist || 0.05;
  const maxDist = opts.maxDist || 60;
  const modelWidth = opts.modelWidth || 2;   // fallback extent, when no bounds are supplied
  // () -> distance from the eye to the front-most surface along the view axis, or
  // null when the centre of the window is empty (or the renderer cannot tell).
  const probeDepth = opts.probeDepth || null;
  // () -> { lo:[x,y,z], hi:[x,y,z] }: the model-space bounding box of what is actually
  // drawn, or null when nothing is. Only the renderer knows this, and without it the
  // "does it all fit?" test below has to assume the model fills the whole normalized
  // cube — which for a map is almost never true, so it reports "zoomed in" while the
  // map is still comfortably framed and the pivot jumps to the nearest blob.
  const modelBounds = opts.modelBounds || null;
  const clampDist = (d) => Math.max(minDist, Math.min(maxDist, d));
  let R = identity();
  let dist = null;          // null = "not fitted yet"; resolved on first use, once the canvas has a size
  let offset = [0, 0, 0];   // world-space model translation (pan + carried orbit pivot)
  let pivotZ = 0;           // world Z of the centre of rotation; it always lies on the view axis
  let pivotStale = true;
  let dragging = false, moved = false, mode = "rotate", lx = 0, ly = 0, downX = 0, downY = 0;

  const aspect = () => (canvas.clientWidth || canvas.width || 1) / Math.max(1, canvas.clientHeight || canvas.height || 1);
  // The narrower of the two half-fields: what actually decides whether the model fits.
  const halfFov = () => Math.min(FOV_X_2, halfFovY(aspect()));
  // Every read of the eye distance goes through here, so the initial fit happens as
  // late as possible — by which time the canvas has been laid out and we know its shape.
  function eyeDist() {
    if (dist === null) dist = clampDist(fitDistance(fitRadius, aspect()));
    return dist;
  }

  // Full model matrix = translate(offset) · R. R is a pure rotation (last column
  // [0,0,0,1]), so this is just R with its translation column set to `offset`.
  function modelMatrix() {
    const m = R.slice();
    m[12] = offset[0]; m[13] = offset[1]; m[14] = offset[2];
    return m;
  }

  // The eye sits on the world +Z axis looking at the origin, so the ray through the
  // centre of the window *is* the world Z axis. Renderers hold their geometry
  // untransformed, so hand them that ray in model space: p_model = R⁻¹·(p_world - offset),
  // and R⁻¹ = Rᵀ because R is orthonormal.
  function axisRay() {
    const rt = (v) => [
      R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
      R[4] * v[0] + R[5] * v[1] + R[6] * v[2],
      R[8] * v[0] + R[9] * v[1] + R[10] * v[2],
    ];
    return { origin: rt([-offset[0], -offset[1], eyeDist() - offset[2]]), dir: rt([0, 0, -1]) };
  }

  // World Z of the model's centre if every corner of `b` is currently inside the
  // frustum, else null. Testing the eight corners against the field is what ChimeraX
  // does, and it is the part that matters: comparing one width against another (as
  // this used to) ignores the window's shape, ignores where the model has been panned
  // to, and — with no real bounds to measure — ignores that a contoured map occupies a
  // small, off-centre part of its own padded grid.
  function framedCentreZ(b) {
    const M = modelMatrix();
    const d = eyeDist();
    const tx = Math.tan(FOV_X_2), ty = Math.tan(halfFovY(aspect()));
    let cz = 0;
    for (let i = 0; i < 8; i++) {
      const p = mulVec4(M, [i & 1 ? b.hi[0] : b.lo[0], i & 2 ? b.hi[1] : b.lo[1], i & 4 ? b.hi[2] : b.lo[2], 1]);
      const depth = d - p[2];                          // the eye is on +Z, looking down -Z
      if (depth <= minDist) return null;               // a corner is level with, or behind, the eye
      if (Math.abs(p[0]) > tx * depth || Math.abs(p[1]) > ty * depth) return null;
      cz += p[2];
    }
    // The transform is affine, so the mean of the corners is the centre of the box.
    return cz / 8;
  }

  // ChimeraX recomputes the centre of rotation after translations only, never during
  // a rotation — otherwise the pivot would crawl across the surface as you drag and
  // the model would wander. So this runs at most once per rotate gesture.
  function updatePivot() {
    pivotStale = false;
    const d = eyeDist();
    const b = modelBounds && modelBounds();
    if (b) {
      const cz = framedCentreZ(b);
      if (cz !== null) { pivotZ = cz; return; }        // zoomed out: it all fits
    } else {
      const centre = offset[2];                        // no bounds: assume the model is centred
      const depth = Math.max(minDist, d - centre);     // eye -> model centre
      if (2 * depth * Math.tan(halfFov()) >= modelWidth) { pivotZ = centre; return; }
    }
    const t = probeDepth && probeDepth();
    if (t == null || !isFinite(t)) return;             // nothing at the centre: keep the depth we had
    pivotZ = Math.min(d - t, d - minDist);             // never let the pivot fall behind the eye
  }

  // Turn the model about the pivot rather than the world origin:
  //     offset' = P + dR·(offset - P)
  // which makes P a fixed point of translate(offset')·dR·R. P lies on the view axis,
  // so it stays pinned to the centre of the window for the whole drag.
  function applyDelta(dR) {
    if (pivotStale) updatePivot();
    R = multiply(dR, R);   // pre-multiply, so each drag composes onto the current orientation
    const o = mulVec4(dR, [offset[0], offset[1], offset[2] - pivotZ, 1]);
    offset = [o[0], o[1], o[2] + pivotZ];
  }

  function rotateBy(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    // axis ⟂ to the drag in camera space (== world space: camera looks down -Z)
    applyDelta(axisAngle([dy / len, dx / len, 0], len * 0.008));
  }
  // Spin in the plane of the screen. The angle is how far the pointer has swept around
  // the centre of the window, so the model tracks the hand exactly — you are turning a
  // wheel, not nudging a rate.
  function rotateZBy(angle) {
    if (Math.abs(angle) < 1e-5) return;
    applyDelta(axisAngle([0, 0, 1], angle));
  }
  function panBy(dxCss, dyCss) {
    // CSS px -> world units on the plane through the centre of rotation, so whatever
    // you grabbed there tracks the cursor. Pixels are square, so measure across the
    // width, where the field is the fixed one.
    const depth = Math.max(minDist, eyeDist() - pivotZ);
    const wpp = (2 * depth * Math.tan(FOV_X_2)) / Math.max(1, canvas.clientWidth);
    offset = [offset[0] + dxCss * wpp, offset[1] - dyCss * wpp, offset[2]];   // screen y-down -> world y-up
    pivotStale = true;   // a translation changed what sits at the centre of the window
  }
  function zoomBy(deltaY) {
    dist = clampDist(eyeDist() * Math.exp(deltaY * 0.001));
    pivotStale = true;
  }

  // ChimeraX's virtual sphere: picture a ball inscribed in the window. Grab it anywhere
  // on its face and it turns in the two screen axes; grab it past the silhouette — the
  // corners and the edge strips — and all you can do is spin it, which is how a
  // two-axis input reaches the third axis at all. The region is decided once, at press,
  // so a gesture never changes character halfway through (dragging a spin *towards* the
  // middle must keep spinning).
  const EDGE_R = 0.9;   // fraction of the inscribed circle's radius that still tumbles
  function centreOf() {
    const r = canvas.getBoundingClientRect();
    return { cx: r.left + r.width / 2, cy: r.top + r.height / 2, rad: (EDGE_R * Math.min(r.width, r.height)) / 2 };
  }
  function inEdgeZone(e) {
    const { cx, cy, rad } = centreOf();
    return Math.hypot(e.clientX - cx, e.clientY - cy) > rad;
  }
  // Bearing of the pointer from the centre of the window, measured the way the world is
  // oriented (screen y runs down, world y runs up) so a clockwise sweep reads as a
  // clockwise turn about the axis pointing at the viewer.
  function screenAngle(e) {
    const { cx, cy } = centreOf();
    return Math.atan2(-(e.clientY - cy), e.clientX - cx);
  }
  let lastAngle = 0;

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; moved = false; lx = downX = e.clientX; ly = downY = e.clientY;
    if (pivotStale) updatePivot();   // once per gesture: a pan slides the plane through the pivot too
    // middle-mouse = pan; shift stays free for select
    mode = e.button === 1 ? "pan" : inEdgeZone(e) ? "zrotate" : "rotate";
    if (mode === "zrotate") lastAngle = screenAngle(e);
    if (e.button === 1) e.preventDefault();                     // suppress middle-click autoscroll
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) moved = true;
    if (mode === "pan") panBy(e.clientX - lx, e.clientY - ly);
    else if (mode === "zrotate") {
      const a = screenAngle(e);
      // Sweeping past due west flips atan2 between ±π, so the raw difference reads as a
      // near-full turn the other way. That particular error happens to be invisible —
      // a turn of −350° about Z *is* a turn of +10° — but it only stays invisible while
      // nothing scales or thresholds `d`, and both the early-out in rotateZBy and any
      // future damping would be wrong on it. Normalise once, here, rather than rely on
      // every consumer being modulo-safe.
      let d = a - lastAngle;
      if (d > Math.PI) d -= 2 * Math.PI; else if (d < -Math.PI) d += 2 * Math.PI;
      rotateZBy(d);
      lastAngle = a;
    } else rotateBy(e.clientX - lx, e.clientY - ly);
    lx = e.clientX; ly = e.clientY; onChange();
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {}
    // A click near the rim is still a click: only panning has never been a pick.
    if (!moved && mode !== "pan" && onClick) onClick(e);
  });
  canvas.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomBy(e.deltaY);
    onChange();
  }, { passive: false });

  return {
    get rotation() { return R; },
    get model() { return modelMatrix(); },   // translate(offset) · R — use for the model transform
    get dist() { return eyeDist(); },
    get eye() { return [0, 0, eyeDist()]; },
    get pivot() { return [0, 0, pivotZ]; },  // world-space centre of rotation
    axisRay,                                          // model-space ray through the window centre
    // Each renderer owns a camera, but the map is one object. Switching render mode
    // hands the outgoing camera to the incoming one, so the view survives the switch.
    // Safe across renderers: they all frame the same [-1,1] cube through the same FOV.
    getState() { return { R: R.slice(), dist: eyeDist(), offset: offset.slice(), pivotZ }; },
    setState(s) {
      if (!s) return;
      R = s.R.slice(); dist = clampDist(s.dist); offset = s.offset.slice(); pivotZ = s.pivotZ;
      pivotStale = true;   // a different renderer may put a different surface at the centre
      onChange();
    },
    drag(dx, dy) { rotateBy(dx, dy); onChange(); },   // externally driven rotation (e.g. an overlay)
    pan(dx, dy) { panBy(dx, dy); onChange(); },       // externally driven pan (CSS px)
    zoom(deltaY) { zoomBy(deltaY); onChange(); },
    invalidatePivot() { pivotStale = true; },         // the geometry changed under the camera
    // null, not a stored constant: re-fit to whatever shape the window is now — this is
    // ChimeraX's `view`, which reframes the model rather than restoring an old distance.
    reset() { R = identity(); dist = null; offset = [0, 0, 0]; pivotZ = 0; pivotStale = true; onChange(); },
  };
}
