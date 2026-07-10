// Shared trackball camera (ChimeraX-style): the model is rotated by an accumulated
// rotation matrix and the camera sits at a fixed distance looking at the origin, so
// rotation is continuous in every axis (no gimbal lock / no pole flip) and dragging
// from any point feels consistent. Zoom by wheel. Click-vs-drag is reported via
// onClick so renderers can still pick.
//
// Panning (shift-drag or middle-mouse drag) translates the model in the view plane.
// The pan offset is carried through every subsequent rotation, so the camera keeps
// orbiting the point that is currently centred in the viewport — after you pan (or
// zoom) to a feature, rotation happens around that feature rather than the data origin.
import { identity, axisAngle, multiply, mulVec4 } from "./mat4.js";

export function createOrbit(canvas, onChange, onClick, opts = {}) {
  const dist0 = opts.dist || 3.4;
  const FOV = ((opts.fov || 45) * Math.PI) / 180;   // must match the renderer's perspective FOV
  // Models are normalized into a [-1,1] cube (half-diagonal √3 ≈ 1.73), and every
  // renderer's frustum is near=0.01 far=100. So the camera can come right up to the
  // surface — and inside it — and still pull far enough back to see the whole box.
  const minDist = opts.minDist || 0.05;
  const maxDist = opts.maxDist || 60;
  const clampDist = (d) => Math.max(minDist, Math.min(maxDist, d));
  let R = identity();
  let dist = dist0;
  let offset = [0, 0, 0];   // world-space model translation (pan + carried orbit pivot)
  let dragging = false, moved = false, mode = "rotate", lx = 0, ly = 0, downX = 0, downY = 0;

  // Full model matrix = translate(offset) · R. R is a pure rotation (last column
  // [0,0,0,1]), so this is just R with its translation column set to `offset`.
  function modelMatrix() {
    const m = R.slice();
    m[12] = offset[0]; m[13] = offset[1]; m[14] = offset[2];
    return m;
  }

  function rotateBy(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    // axis ⟂ to the drag in camera space (== world space: camera looks down -Z);
    // pre-multiply so each drag composes onto the current orientation.
    const dR = axisAngle([dy / len, dx / len, 0], len * 0.008);
    R = multiply(dR, R);
    // Carry the pan offset through the same incremental rotation. This keeps the
    // world origin (== the viewport centre) fixed under rotation, so the orbit
    // pivot tracks whatever the pan/zoom brought to the centre of the view.
    const o = mulVec4(dR, [offset[0], offset[1], offset[2], 1]);
    offset = [o[0], o[1], o[2]];
  }
  function panBy(dxCss, dyCss) {
    // CSS px -> world units on the plane through the origin (the orbit centre).
    const wpp = (2 * dist * Math.tan(FOV / 2)) / Math.max(1, canvas.clientHeight);
    offset = [offset[0] + dxCss * wpp, offset[1] - dyCss * wpp, offset[2]];   // screen y-down -> world y-up
  }

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; moved = false; lx = downX = e.clientX; ly = downY = e.clientY;
    mode = (e.button === 1 || e.shiftKey) ? "pan" : "rotate";   // middle-mouse or shift = pan
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
    dist = clampDist(dist * Math.exp(e.deltaY * 0.001));
    onChange();
  }, { passive: false });

  return {
    get rotation() { return R; },
    get model() { return modelMatrix(); },   // translate(offset) · R — use for the model transform
    get dist() { return dist; },
    get eye() { return [0, 0, dist]; },
    drag(dx, dy) { rotateBy(dx, dy); onChange(); },   // externally driven rotation (e.g. an overlay)
    pan(dx, dy) { panBy(dx, dy); onChange(); },       // externally driven pan (CSS px)
    zoom(deltaY) { dist = clampDist(dist * Math.exp(deltaY * 0.001)); onChange(); },
    reset() { R = identity(); dist = dist0; offset = [0, 0, 0]; onChange(); },
  };
}
