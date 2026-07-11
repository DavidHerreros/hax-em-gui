// Minimal column-major 4x4 matrix + vec3 helpers (no gl-matrix dependency).
// Only what the renderers need: the camera, lookAt, multiply.

// ChimeraX's camera, so that a map looks here the way it looks there.
// `MonoCamera.field_of_view = 30` and that is the *horizontal* field: its
// perspective_projection_matrix() sizes the frustum from the width
// (w = 2·near·tan(fov/2)) and derives the height from the aspect ratio
// (h = w·height/width). See chimerax/graphics/camera.py.
//
// The number is the whole story. Fixing 45° *vertically* instead — as this
// viewer used to — comes out at ~64° across a landscape canvas, more than twice
// ChimeraX's angular field. A wide field means the camera must sit close to keep
// the map framed, and a close camera foreshortens hard: the near side of the
// map balloons relative to the far side. Hence the "zoomed / fish-eyed" look.
export const FOV_X_DEG = 30;
const FOV_X_2 = (FOV_X_DEG * Math.PI) / 360;   // half the horizontal field, in radians

// Half the vertical field, for a canvas of the given aspect (width / height).
// Renderers need this to turn pixels into world units; it is no longer a constant,
// because with the field pinned horizontally the vertical one follows the window.
export function halfFovY(aspect) {
  return Math.atan(Math.tan(FOV_X_2) / Math.max(1e-6, aspect));
}

// The projection matrix for that camera. aspect = width / height.
export function projection(aspect, near, far) {
  const f = 1 / Math.tan(FOV_X_2);   // horizontal, not vertical: m[0] carries it, m[5] scales by aspect
  const nf = 1 / (near - far);
  return new Float32Array([
    f, 0, 0, 0,
    0, f * aspect, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

// Eye distance at which a sphere of the given radius about the origin exactly fills
// the narrower of the two fields — ChimeraX's `view_all`, fitted to a bounding sphere
// rather than the box corners so the framing does not breathe as the trackball turns.
// Narrowing the field without pulling the eye back would just shrink the map on screen;
// this is what keeps it the same size while the perspective relaxes.
export function fitDistance(radius, aspect) {
  return radius / Math.sin(Math.min(FOV_X_2, halfFovY(aspect)));
}

export function lookAt(eye, center, up) {
  const z = normalize(sub(eye, center));        // forward (points from center to eye)
  const x = normalize(cross(up, z));             // right
  const y = cross(z, x);                         // true up
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

// column-major mat4 * vec4 -> vec4
export function mulVec4(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

// a * b (both column-major).
export function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

export function identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

// Rotation about an arbitrary axis (column-major). Used by the trackball.
export function axisAngle(axis, angle) {
  const [x, y, z] = normalize(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return new Float32Array([
    t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0,
    t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0,
    t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0,
    0, 0, 0, 1,
  ]);
}

// Full 4x4 inverse (column-major). Returns null if singular.
export function invert(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
  inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
  inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
  inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
  inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
  inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
  inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
  inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
  inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
  inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
  inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
  inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
  inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
  let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  if (!det) return null;
  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

// vec3 helpers
export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function normalize(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
