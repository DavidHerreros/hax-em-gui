// Shared trackball camera (ChimeraX-style): the model is rotated by an accumulated
// rotation matrix and the camera sits at a fixed distance looking at the origin, so
// rotation is continuous in every axis (no gimbal lock / no pole flip) and dragging
// from any point feels consistent. Zoom by wheel. Click-vs-drag is reported via
// onClick so renderers can still pick.
//
// Panning (middle-mouse drag) translates the model in the view plane. Shift is left
// free for the app's shift-click multi-select (building a morph path).
//
// Centre of rotation follows ChimeraX's `cofr frontCenter` (its default): while the
// whole model still fits in the window, rotate about the model's centre; once you
// have zoomed in past it, rotate about the front-most surface at the centre of the
// window, so the feature you are looking at stays under the crosshair instead of
// swinging away. The depth of that surface is supplied by the renderer, which is the
// only thing that knows the geometry — see `opts.probeDepth`.
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
  const modelWidth = opts.modelWidth || 2;   // extent of the normalized model, for the zoomed-out test
  // () -> distance from the eye to the front-most surface along the view axis, or
  // null when the centre of the window is empty (or the renderer cannot tell).
  const probeDepth = opts.probeDepth || null;
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

  // ChimeraX recomputes the centre of rotation after translations only, never during
  // a rotation — otherwise the pivot would crawl across the surface as you drag and
  // the model would wander. So this runs at most once per rotate gesture.
  function updatePivot() {
    pivotStale = false;
    const d = eyeDist();
    const centre = offset[2];                          // the model's centre sits at `offset`
    const depth = Math.max(minDist, d - centre);       // eye -> model centre
    if (2 * depth * Math.tan(halfFov()) >= modelWidth) { pivotZ = centre; return; }   // zoomed out: it all fits
    const t = probeDepth && probeDepth();
    if (t == null || !isFinite(t)) return;             // nothing at the centre: keep the depth we had
    pivotZ = Math.min(d - t, d - minDist);             // never let the pivot fall behind the eye
  }

  function rotateBy(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    if (pivotStale) updatePivot();
    // axis ⟂ to the drag in camera space (== world space: camera looks down -Z);
    // pre-multiply so each drag composes onto the current orientation.
    const dR = axisAngle([dy / len, dx / len, 0], len * 0.008);
    R = multiply(dR, R);
    // Turn the model about the pivot rather than the world origin:
    //     offset' = P + dR·(offset - P)
    // which makes P a fixed point of translate(offset')·dR·R. P lies on the view axis,
    // so it stays pinned to the centre of the window for the whole drag.
    const o = mulVec4(dR, [offset[0], offset[1], offset[2] - pivotZ, 1]);
    offset = [o[0], o[1], o[2] + pivotZ];
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

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; moved = false; lx = downX = e.clientX; ly = downY = e.clientY;
    if (pivotStale) updatePivot();   // once per gesture: a pan slides the plane through the pivot too
    mode = (e.button === 1) ? "pan" : "rotate";                 // middle-mouse = pan; shift stays free for select
    if (e.button === 1) e.preventDefault();                     // suppress middle-click autoscroll
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) moved = true;
    if (mode === "pan") panBy(e.clientX - lx, e.clientY - ly);
    else rotateBy(e.clientX - lx, e.clientY - ly);
    lx = e.clientX; ly = e.clientY; onChange();
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {}
    if (!moved && mode === "rotate" && onClick) onClick(e);
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
