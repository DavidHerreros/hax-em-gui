// Shared trackball camera (ChimeraX-style): the model is rotated by an accumulated
// rotation matrix and the camera sits at a fixed distance looking at the origin, so
// rotation is continuous in every axis (no gimbal lock / no pole flip) and dragging
// from any point feels consistent. Zoom by wheel. Click-vs-drag is reported via
// onClick so renderers can still pick.
import { identity, axisAngle, multiply } from "./mat4.js";

export function createOrbit(canvas, onChange, onClick, opts = {}) {
  const dist0 = opts.dist || 3.4;
  let R = identity();
  let dist = dist0;
  let dragging = false, moved = false, lx = 0, ly = 0, downX = 0, downY = 0;

  function rotateBy(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    // axis ⟂ to the drag in camera space (== world space: camera looks down -Z);
    // pre-multiply so each drag composes onto the current orientation.
    R = multiply(axisAngle([dy / len, dx / len, 0], len * 0.008), R);
  }

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; moved = false; lx = downX = e.clientX; ly = downY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) moved = true;
    rotateBy(e.clientX - lx, e.clientY - ly);
    lx = e.clientX; ly = e.clientY; onChange();
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {}
    if (!moved && onClick) onClick(e);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(1.2, Math.min(16, dist * Math.exp(e.deltaY * 0.001)));
    onChange();
  }, { passive: false });

  return {
    get rotation() { return R; },
    get dist() { return dist; },
    get eye() { return [0, 0, dist]; },
    drag(dx, dy) { rotateBy(dx, dy); onChange(); },   // externally driven rotation (e.g. an overlay)
    zoom(deltaY) { dist = Math.max(1.2, Math.min(16, dist * Math.exp(deltaY * 0.001))); onChange(); },
    reset() { R = identity(); dist = dist0; onChange(); },
  };
}
