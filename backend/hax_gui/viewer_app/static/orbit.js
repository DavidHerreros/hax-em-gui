// Shared trackball camera. The rotation half of this is a transcription of ChimeraX's
// `rotate` mouse mode — MoveMouseMode in chimerax/mouse_modes/std_modes.py, plus
// View._front_center_cofr in chimerax/graphics/view.py — down to its constants, because
// "close to ChimeraX" and "ChimeraX" feel different in the hand. The three rules it
// takes from there:
//
//   * 0.5° of rotation per pixel of drag, measured per pointer event as 0.5·√(dx²+dy²)
//     and taken about the screen-plane axis (dy, dx, 0) perpendicular to the drag.
//   * A drag that *begins* near the rim spins the model in the plane of the screen
//     instead — the only way a two-axis input reaches the third axis. Same 0.5°/px;
//     only the sign comes from where the pointer is. See `inEdgeZone` and rotateZBy.
//   * Centre of rotation = `cofr frontCenter`, ChimeraX's default: while the model is
//     still framed, turn about the model's own centre — wherever it has been panned to;
//     once you are zoomed in past it, turn about the front-most surface at the centre of
//     the window, so the feature under the crosshair stays there instead of swinging
//     away. Recomputed after translations only, never during a rotation, or the pivot
//     would crawl across the surface as you drag.
//
// ChimeraX turns the camera about a still model; here the camera is still and the model
// carries an accumulated rotation. The two are the same transform read from opposite
// ends (see applyDelta), and the model-side reading is what the renderers want — but it
// means anything ChimeraX measures in scene coordinates has to be measured here in
// *model* coordinates to stay equally invariant under the drag.
//
// The camera cannot see the geometry it is orbiting, so the renderer supplies it:
// `opts.modelBounds` (the model-space box of what is drawn — ChimeraX's drawing_bounds)
// and `opts.probeDepth` (eye-to-front-surface distance down the view axis — its
// front_center_point). Zoom by wheel; panning (middle-mouse drag) translates the model in
// the view plane, leaving shift free for the app's shift-click multi-select. Click-vs-drag
// is reported via onClick so renderers can still pick.
import { identity, axisAngle, multiply, mulVec4, fitDistance, viewWidth } from "./mat4.js";

// ChimeraX's rotation rate: `angle = 0.5*sqrt(dx*dx+dy*dy)` degrees, for both the tumble
// and the screen-plane spin (MoveMouseMode._rotation_axis_angle). Its pixels are Qt
// logical pixels — device-independent, like CSS pixels — so the number carries over as is.
const RAD_PER_PX = (0.5 * Math.PI) / 180;

export function createOrbit(canvas, onChange, onClick, opts = {}) {
  // Radius of the sphere the camera frames on open (and on reset) — see fitDistance().
  // The eye distance is no longer a constant: the field is pinned horizontally
  // (ChimeraX-style), so how far back the camera must sit to frame the map depends on
  // the shape of the window. Resolved lazily, because the canvas is often still
  // unlaid-out (clientWidth 0) when the renderer is constructed.
  const fitRadius = opts.fitRadius || 1.4;
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
  // World-space centre of rotation. Not constrained to the view axis: ChimeraX turns a
  // framed model about its own centre, which is off-axis as soon as you pan it aside, and
  // pinning it back to the middle of the window is the difference between a model that
  // spins where you left it and one that swings around the crosshair.
  let pivot = [0, 0, 0];
  let pivotStale = true;
  let dragging = false, moved = false, mode = "rotate", lx = 0, ly = 0, downX = 0, downY = 0;

  const aspect = () => (canvas.clientWidth || canvas.width || 1) / Math.max(1, canvas.clientHeight || canvas.height || 1);
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

  // Distance from the eye to a world-space point. The eye is on the +Z axis looking down
  // -Z, so this is the true distance, not the depth along the axis — the two part company
  // once the pivot is off-centre, and ChimeraX measures the true one (view_width and
  // pixel_size both take `distance(camera_origin, point)`).
  const eyeRange = (p) => Math.hypot(p[0], p[1], eyeDist() - p[2]);

  // ChimeraX's `cofr frontCenter`, its default (View._front_center_cofr):
  //
  //     b = drawing_bounds()
  //     if camera.view_width(b.center()) >= b.width(): cofr = b.center()
  //     else:                                          cofr = front_center_point()
  //
  // — one width against another, not the eight box corners against the frustum, which is
  // what this used to test. The corner test is the more careful-looking reading of "does
  // it all fit?", and it is the wrong one: ChimeraX's `b` is the *scene* box of a model
  // the camera orbits, so it never turns and the test cannot flip part-way through a drag.
  // Here the model turns instead, so the box that stays put is the model-space one — same
  // comparison, same stability. Test the corners against a rotating box and the verdict
  // flips as the diagonal swings into view, and the pivot jumps from the model's centre to
  // the nearest blob and back, mid-gesture.
  //
  // Recomputed after translations only, never during a rotation, so this runs at most once
  // per rotate gesture.
  function updatePivot() {
    pivotStale = false;
    const d = eyeDist();
    const b = modelBounds && modelBounds();
    // b.center() / b.width(): the box centre carried into the world by the model
    // transform, and the largest of its three axis extents (ChimeraX's Bounds.width).
    const c = b
      ? mulVec4(modelMatrix(), [(b.lo[0] + b.hi[0]) / 2, (b.lo[1] + b.hi[1]) / 2, (b.lo[2] + b.hi[2]) / 2, 1])
      : offset;                                        // no bounds: assume the model is centred on its own origin
    const width = b ? Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]) : modelWidth;
    if (viewWidth(eyeRange(c)) >= width) { pivot = [c[0], c[1], c[2]]; return; }   // still framed
    const t = probeDepth && probeDepth();
    // Nothing at the centre of the window: ChimeraX keeps the depth it had and slides the
    // pivot back onto the view axis (_center_point_matching_depth), which is what dropping
    // x and y does. Never let it fall behind the eye.
    pivot = t == null || !isFinite(t) ? [0, 0, pivot[2]] : [0, 0, Math.min(d - t, d - minDist)];
  }

  // Turn the model about the pivot rather than the world origin:
  //     offset' = P + dR·(offset - P)
  // which makes P a fixed point of translate(offset')·dR·R. This is ChimeraX's
  // `camera.position = rotation(axis, angle, cofr).inverse() * camera.position` read from
  // the other end: rotating the camera by r⁻¹ about a fixed point and rotating the model
  // by r about it put the same pixels on the screen.
  function applyDelta(dR) {
    if (pivotStale) updatePivot();
    R = multiply(dR, R);   // pre-multiply, so each drag composes onto the current orientation
    const o = mulVec4(dR, [offset[0] - pivot[0], offset[1] - pivot[1], offset[2] - pivot[2], 1]);
    offset = [o[0] + pivot[0], o[1] + pivot[1], o[2] + pivot[2]];
  }

  function rotateBy(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    // Axis ⟂ to the drag in camera space, which is also world space here: the camera looks
    // down -Z from +Z with world Y up, exactly ChimeraX's camera frame, so its (dy, dx, 0)
    // carries over verbatim — dy measured downwards, as it is there.
    applyDelta(axisAngle([dy, dx, 0], len * RAD_PER_PX));
  }
  // Spin in the plane of the screen — ChimeraX's `rotate z` branch, taken when the drag
  // began near the rim. The angle is the drag *distance*, at the same 0.5°/px as a tumble;
  // it is not the angle the pointer sweeps about the centre of the window, which is what
  // this used to do. Tracking the sweep is the more natural-looking reading — you turn a
  // wheel, the model follows your hand — but out at the rim where this mode lives an arc of
  // one pixel at radius r subtends only 1/r radians, so it runs 2-3x slower than ChimeraX
  // on any normally-sized window. That gap is most of why the spin has never felt right.
  //
  // Only the *sign* comes from where the pointer is: the sense of the sweep, as the 2-D
  // cross product of the pointer's offset from the window centre with its motion. Both are
  // in screen coordinates (y down), as in ChimeraX, so the test transfers unchanged.
  function rotateZBy(dx, dy, ex, ey) {
    const len = Math.hypot(dx, dy);
    if (len < 1e-4) return;
    const angle = len * RAD_PER_PX;
    applyDelta(axisAngle([0, 0, 1], -dy * ex + dx * ey < 0 ? -angle : angle));
  }
  function panBy(dxCss, dyCss) {
    // ChimeraX's View.pixel_size: the width of the field at the centre of rotation over the
    // width of the window. So whatever sits at the pivot tracks the cursor, and pixels stay
    // square because the horizontal field is the pinned one.
    const wpp = viewWidth(Math.max(minDist, eyeRange(pivot))) / Math.max(1, canvas.clientWidth);
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
  // two-axis input reaches the third axis at all. The region is decided once, at press
  // (MoveMouseMode._set_z_rotation runs from mouse_down), so a gesture never changes
  // character halfway through — dragging a spin *towards* the middle must keep spinning.
  const EDGE_R = 0.9;   // fperim in _set_z_rotation: fraction of the inscribed circle that still tumbles
  function centreOf() {
    const r = canvas.getBoundingClientRect();
    return { cx: r.left + r.width / 2, cy: r.top + r.height / 2, rad: (EDGE_R * Math.min(r.width, r.height)) / 2 };
  }
  function inEdgeZone(e) {
    const { cx, cy, rad } = centreOf();
    return Math.hypot(e.clientX - cx, e.clientY - cy) > rad;
  }

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; moved = false; lx = downX = e.clientX; ly = downY = e.clientY;
    if (pivotStale) updatePivot();   // once per gesture: a pan slides the plane through the pivot too
    // middle-mouse = pan; shift stays free for select
    mode = e.button === 1 ? "pan" : inEdgeZone(e) ? "zrotate" : "rotate";
    if (e.button === 1) e.preventDefault();                     // suppress middle-click autoscroll
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) moved = true;
    if (mode === "pan") panBy(e.clientX - lx, e.clientY - ly);
    else if (mode === "zrotate") {
      // The offset is read fresh each event, not from the press: ChimeraX takes it from
      // event.position() inside the drag handler, so sweeping across the centre reverses
      // the spin part-way through the gesture.
      const { cx, cy } = centreOf();
      rotateZBy(e.clientX - lx, e.clientY - ly, e.clientX - cx, e.clientY - cy);
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
    get pivot() { return pivot.slice(); },   // world-space centre of rotation
    axisRay,                                          // model-space ray through the window centre
    // Each renderer owns a camera, but the map is one object. Switching render mode
    // hands the outgoing camera to the incoming one, so the view survives the switch.
    // Safe across renderers: they all frame the same [-1,1] cube through the same FOV.
    getState() { return { R: R.slice(), dist: eyeDist(), offset: offset.slice(), pivot: pivot.slice() }; },
    setState(s) {
      if (!s) return;
      R = s.R.slice(); dist = clampDist(s.dist); offset = s.offset.slice(); pivot = s.pivot.slice();
      pivotStale = true;   // a different renderer may put a different surface at the centre
      onChange();
    },
    drag(dx, dy) { rotateBy(dx, dy); onChange(); },   // externally driven rotation (e.g. an overlay)
    pan(dx, dy) { panBy(dx, dy); onChange(); },       // externally driven pan (CSS px)
    zoom(deltaY) { zoomBy(deltaY); onChange(); },
    invalidatePivot() { pivotStale = true; },         // the geometry changed under the camera
    // null, not a stored constant: re-fit to whatever shape the window is now — this is
    // ChimeraX's `view`, which reframes the model rather than restoring an old distance.
    reset() { R = identity(); dist = null; offset = [0, 0, 0]; pivot = [0, 0, 0]; pivotStale = true; onChange(); },
  };
}
