// WebGL2 isosurface renderer, shaded as ChimeraX shades a map surface.
//
// The lighting here is not "ChimeraX-like": it is a transcription of ChimeraX 1.8's
// fragment shader (chimerax/graphics/fragmentShader.txt) driven by the constants its
// `lighting` presets set (chimerax/std_commands/lighting.py) and its Material defaults
// (chimerax/graphics/opengl.py, class Material). The three modes this viewer offers map
// onto three ChimeraX presets:
//
//     standard  -> lighting simple   (the ChimeraX default: key + fill + ambient, no shadows)
//     smooth    -> lighting soft     (no directional light at all; ambient occlusion only)
//     ray traced-> lighting full     (key + fill + ambient, key casts a shadow, plus AO)
//
// The ambient occlusion in the latter two is ChimeraX's "multishadow": 64 real shadow maps
// cast from uniformly distributed directions, tiled into one depth texture, and summed per
// fragment with a cosine weight. It is not screen-space AO — this used to use SSAO, and the
// difference is not subtle, because SSAO can only see occluders that are on screen and
// within a small radius, so it draws thin creases where ChimeraX draws a whole domain
// shadowing its neighbour.
//
// Pipeline:
//   1. multishadow atlas  -> 8x8 tiles of 128x128 depth, in MODEL space (see below), rebuilt
//                            only when the geometry changes. Skipped in `standard`.
//   2. key shadow map     -> one 2048x2048 depth map, rebuilt every frame (the key light is
//                            fixed to the camera, so it moves over the model). `raytraced` only.
//   3. beauty pass        -> ChimeraX's lighting equation into the default (MSAA) framebuffer.
//
// Why model space: ChimeraX orbits a camera around a still scene, so its ambient shadows are
// computed once in scene coordinates and survive any camera move. Here the camera is still
// and the model carries the rotation (see orbit.js), so the coordinates that stay put under a
// drag are the mesh's own — compute the atlas there and it is equally view-invariant, and the
// occlusion stays welded to the surface exactly as it does in ChimeraX.
//
// Hand-written, no dependencies.
import { projection, lookAt, multiply, invert, mulVec4, normalize, cross } from "./mat4.js";
import { createOrbit } from "./orbit.js";

// ---------------------------------------------------------------------------
// ChimeraX constants
// ---------------------------------------------------------------------------

// class Material, set_default_parameters(). These multiply the light intensities to give
// the colours the shader actually receives — see Render._light_shader_parameter_values().
const MATERIAL = { ambient: 0.8, diffuse: 0.8, specular: 0.3, exponent: 30 };

// class Lighting, set_default_parameters(). Unit vectors in camera coordinates (x right,
// y up, z towards the viewer) pointing in the direction each light *shines*, so the key
// light comes from the upper left and slightly in front, and the fill is nearly a headlight.
// `move_lights_with_camera` is true by default, which is why these are camera-space
// constants and not model-space ones.
const KEY_DIR = normalize([0.577, -0.577, -0.577]);
const FILL_DIR = normalize([-0.2, -0.2, -0.959]);

// The presets, as `lighting.py` writes them. Everything else (directions, depth cue,
// material) stays at the defaults above, which is also what ChimeraX does: a preset only
// touches the intensities and the shadow switches.
const LIGHTING = {
  standard:  { key: 1.0, fill: 0.5, ambient: 0.4, shadows: false, multishadow: 0  },  // "simple"
  smooth:    { key: 0.0, fill: 0.0, ambient: 1.5, shadows: false, multishadow: 64 },  // "soft"
  raytraced: { key: 0.7, fill: 0.3, ambient: 0.8, shadows: true,  multishadow: 64 },  // "full"
};

// Lighting.depth_cue_{start,end}: fractions of the near-to-far clip range. depth_cue_color
// tracks the background (View.background_color keeps the two in step), so distant geometry
// fades *into the background* rather than darkening towards black.
const DEPTH_CUE_START = 0.5, DEPTH_CUE_END = 1.0;
// View._near_far_pad and View._min_near_fraction.
const NEAR_FAR_PAD = 0.01, MIN_NEAR_FRACTION = 0.001;

// Lighting.multishadow_map_size / multishadow_depth_bias, and the same pair for the key
// light's shadow (shadow_map_size / shadow_depth_bias). The biases are fractions of the
// scene diameter, which is the unit the shadow texture's depth axis is scaled into.
const MS_COUNT = 64, MS_MAP_SIZE = 1024, MS_DEPTH_BIAS = 0.01;
const SHADOW_MAP_SIZE = 2048, SHADOW_DEPTH_BIAS = 0.005;

// The 64 shadow transforms travel as a uniform block rather than a plain array: 64 mat4 is
// 256 uniform vectors, well past the 224 GLSL ES 3.0 guarantees, but exactly the 16 KB a
// uniform block is guaranteed to hold. ChimeraX makes the same trade for the same reason.
const MS_BLOCK_BINDING = 0;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

// Mesh vertex shader. `vViewPos`/`vNormal` are ChimeraX's `v` and `N` — camera-space
// position and normal, straight out of its vertexShader.txt. The model-space pair is the
// extra this renderer needs, because its shadow transforms are model->texture (see the
// header note on coordinate spaces) rather than ChimeraX's camera->texture.
const VERT = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
// Optional per-vertex colour (surfaces painted by another map). When the attribute
// array is disabled the generic attribute feeds a constant and uUseVColor gates it off.
layout(location=2) in vec3 aColor;
uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vViewPos;
out vec3 vModelPos;
out vec3 vModelNormal;
out vec3 vColor;
void main() {
  vec4 vp = uView * uModel * vec4(aPos, 1.0);
  vViewPos = vp.xyz;
  // The model transform is rigid (rotation + translation), as ChimeraX's always is, so the
  // plain matrix transforms normals correctly and no inverse-transpose is needed.
  vNormal = normalize(mat3(uView * uModel) * aNormal);
  vModelPos = aPos;
  vModelNormal = aNormal;
  vColor = aColor;
  gl_Position = uProj * vp;
}`;

// Depth-only pass, used for both shadow maps. No colour attachment is bound, so the
// fragment stage writes nothing — it exists only to apply the clip slab, which it must,
// or geometry the beauty pass has cut away would go on casting shadows onto the interior
// the cut just revealed. Both shadow passes render with model-space view matrices, so
// aPos is already the coordinate the slab is expressed in.
const VERT_DEPTH = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uProj;
uniform mat4 uView;
out vec3 vModelPos;
void main() {
  vModelPos = aPos;
  gl_Position = uProj * uView * vec4(aPos, 1.0);
}`;
const FRAG_DEPTH = `#version 300 es
precision highp float;
in vec3 vModelPos;
uniform int uClipOn;
uniform vec3 uClipN;
uniform vec2 uClipRange;
void main() {
  if (uClipOn == 1) {
    float d = dot(vModelPos, uClipN);
    if (d < uClipRange.x || d > uClipRange.y) discard;
  }
}`;

// Unlit line overlay (PDB backbone traces). Depth-tested against the surfaces so a
// trace threads through its map instead of floating in front of it.
const VERT_LINE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uProj;
uniform mat4 uView;
uniform mat4 uModel;
void main() { gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); }`;
const FRAG_LINE = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() { outColor = vec4(uColor, 1.0); }`;

// Camera-space position prepass, read back a few pixels at a time to find what is under
// the centre of the window. That is the only thing the camera needs from the renderer
// (orbit.js turns it into a ChimeraX `cofr frontCenter` pivot), so it runs on demand —
// once per gesture — rather than every frame.
const FRAG_GEOM = `#version 300 es
precision highp float;
in vec3 vViewPos;
in vec3 vModelPos;
uniform int uClipOn;
uniform vec3 uClipN;
uniform vec2 uClipRange;
out vec4 outPos;
void main() {
  // Same clip as the beauty pass, so the pivot lands on a surface you can see.
  if (uClipOn == 1) {
    float d = dot(vModelPos, uClipN);
    if (d < uClipRange.x || d > uClipRange.y) discard;
  }
  outPos = vec4(vViewPos, 1.0);
}`;

// The beauty pass: ChimeraX's fragmentShader.txt, USE_LIGHTING branch, with
// USE_LIGHTING_NORMALS, USE_SHADOW, USE_MULTISHADOW and USE_DEPTH_CUE all enabled and
// selected at run time by uniforms instead of by #define.
const FRAG_BEAUTY = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 vNormal;
in vec3 vViewPos;
in vec3 vModelPos;
in vec3 vModelNormal;
in vec3 vColor;

uniform vec3 uColor;
uniform int uUseVColor;   // 1 = per-vertex colours (painted surface), 0 = uColor

// The lighting_block uniforms, already multiplied by the material reflectivities.
uniform vec3 uKeyDir;             // camera space, direction the key light shines
uniform vec3 uFillDir;
uniform vec3 uKeyDiffuse;
uniform vec3 uKeySpecular;
uniform float uSpecExponent;
uniform vec3 uFillDiffuse;
uniform vec3 uAmbient;

// Key-light shadow map. uShadowXf maps model space to its texture coordinates.
uniform int uUseShadow;
uniform mat4 uShadowXf;
uniform sampler2DShadow uShadowMap;

// Ambient ("multi") shadows. uMSCount is 0 or 64; uMSDepth is the scene diameter, which
// recovers each light's direction from the third row of its transform exactly as ChimeraX
// does, so the two never drift apart.
uniform int uMSCount;
uniform float uMSDepth;
uniform sampler2DShadow uMSMap;
layout(std140) uniform shadow_matrix_block { mat4 uMSXf[${MS_COUNT}]; };

// Depth cue: fade towards uDepthCueColor between the two camera-space distances in
// uDepthCueRange.
uniform vec3 uDepthCueColor;
uniform vec2 uDepthCueRange;

// Model-space clip slab: a pair of parallel planes with unit normal uClipN, keeping the
// fragments whose signed distance along it lands inside uClipRange. Clipping to a
// coordinate axis is the case where the normal is a coordinate axis, like ChimeraX's
// per-axis clipping of a surface; an arbitrary normal is what the viewer's "view" cut
// uses. The shadow passes apply the same slab, so a cutaway is lit like an exterior
// rather than sitting in the shadow of the geometry it removed — but the 64-pass ambient
// atlas only catches up on a debounce (see setClip), since rebuilding it on every slider
// tick would make the slab unusable to drag. The one-pass key shadow keeps up live.
uniform int uClipOn;
uniform vec3 uClipN;
uniform vec2 uClipRange;

out vec4 frag_color;

void main() {
  if (uClipOn == 1) {
    float clipD = dot(vModelPos, uClipN);
    if (clipD < uClipRange.x || clipD > uClipRange.y) discard;
  }
  vec4 fcolor = vec4(uUseVColor == 1 ? vColor : uColor, 1.0);

  // Unit normal, with two-sided lighting.
  float side = gl_FrontFacing ? 1.0 : -1.0;
  vec3 N1 = normalize(vNormal) * side;
  // Unit vector from camera to surface point.
  vec3 v1 = normalize(vViewPos);

  float key_factor = max(-dot(N1, uKeyDir), 0.0);
  float fill_factor = max(-dot(N1, uFillDir), 0.0);
  // Phong, not Blinn-Phong: ChimeraX reflects the light about the normal and dots that
  // against the view vector. Swapping in a halfway vector would need roughly 4x the
  // exponent to put the highlight at the same size, and would still fall off differently.
  vec3 R = normalize(reflect(uKeyDir, N1));
  float specular_factor = pow(max(-dot(R, v1), 0.0), uSpecExponent);

  float shadow = 1.0;
  if (uUseShadow == 1) shadow = texture(uShadowMap, (uShadowXf * vec4(vModelPos, 1.0)).stp);

  // Diffuse: radiated equally in all directions, strength from the light direction and the
  // surface normal. Specular: the key light reflected towards the camera. Both are killed
  // by the key light's shadow; only the ambient term survives it.
  vec3 Idiff = fcolor.rgb * (uKeyDiffuse * key_factor * shadow + uFillDiffuse * fill_factor);
  vec3 Ispec = uKeySpecular * specular_factor * shadow;

  vec3 Iamb;
  if (uMSCount > 0) {
    // Cosine-weighted fraction of the sphere of directions from which this point can see
    // out. The 0.25 in the normalisation is the mean of max(cos,0) over a uniformly
    // sampled sphere, so an unoccluded point comes out at 1 and the ambient term is
    // untouched there.
    vec3 Nm = normalize(vModelNormal) * side;
    float mshadow = 0.0;
    for (int i = 0; i < ${MS_COUNT}; i++) {
      if (i >= uMSCount) break;
      mat4 stf = uMSXf[i];
      vec3 light_direction = uMSDepth * vec3(stf[0][2], stf[1][2], stf[2][2]);
      float diffuse = max(-dot(Nm, light_direction), 0.0);
      mshadow += diffuse * texture(uMSMap, (stf * vec4(vModelPos, 1.0)).stp);
    }
    mshadow /= 0.25 * float(uMSCount);
    Iamb = fcolor.rgb * uAmbient * mshadow;
  } else {
    Iamb = fcolor.rgb * uAmbient;
  }

  // Total colour. Deliberately not tone-mapped or normalised: with the "soft" preset the
  // ambient intensity is 1.5, so a bright unoccluded face lands above 1 and clips against
  // the framebuffer. That clipping is what the preset looks like in ChimeraX.
  vec3 Ifrag = Iamb + Idiff + Ispec;

  float d = -vViewPos.z;   // distance from eye to fragment
  float dim = (uDepthCueRange.y - d) / (uDepthCueRange.y - uDepthCueRange.x);
  Ifrag = mix(uDepthCueColor, Ifrag, clamp(dim, 0.0, 1.0));

  frag_color = vec4(Ifrag, 1.0);
}`;

export function createRenderer(canvas) {
  // preserveDrawingBuffer lets us read the canvas back for snapshots/recording.
  const gl = canvas.getContext("webgl2", { antialias: true, depth: true, preserveDrawingBuffer: true, premultipliedAlpha: false });
  if (!gl) throw new Error("WebGL2 is not available in this browser.");

  // The pivot probe writes camera-space positions into a float target; shadows do not need
  // one. So a driver without float colour buffers loses the front-centre centre of
  // rotation and nothing else.
  const floatTargets = !!gl.getExtension("EXT_color_buffer_float");

  const beauty = makeProg(gl, VERT, FRAG_BEAUTY);
  const depthP = makeProg(gl, VERT_DEPTH, FRAG_DEPTH);
  const geomP = floatTargets ? makeProg(gl, VERT, FRAG_GEOM) : null;
  const lineP = makeProg(gl, VERT_LINE, FRAG_LINE);

  // Line overlays (PDB traces): id -> { vao, buf, count, color }. Drawn unlit after
  // the beauty pass, depth-tested against the surfaces. GL_LINES pairs, model space.
  const lineModels = new Map();

  // Mesh geometry, one entry per model in the scene. A lone map is just the case where
  // there is one of them, so the single-map path is not a separate code path.
  //   { vao, posBuf, normBuf, idxBuf, count, color|null, visible, bounds }
  // Insertion order is draw order; `color: null` means "use the viewer's global colour",
  // which is what a single unnamed map has always used.
  const models = new Map();

  function makeModel() {
    const m = { vao: gl.createVertexArray(), posBuf: gl.createBuffer(), normBuf: gl.createBuffer(),
                colBuf: gl.createBuffer(), idxBuf: gl.createBuffer(),
                count: 0, color: null, visible: true, bounds: null, hasVColor: false };
    gl.bindVertexArray(m.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.normBuf); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    // Colour attribute stays disabled until a painted mesh arrives.
    gl.bindBuffer(gl.ARRAY_BUFFER, m.colBuf); gl.vertexAttribPointer(2, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idxBuf);
    gl.bindVertexArray(null);
    return m;
  }
  const drawable = () => [...models.values()].filter((m) => m.visible && m.count);
  // Whether *anything* is on screen. The shadow passes and the snapshot path key off this.
  const anyGeometry = () => drawable().length > 0;
  gl.enable(gl.DEPTH_TEST);
  // The shadow-transform block is bound for the life of the renderer, zero-filled until an
  // atlas is built. Drawing with an active uniform block that has no buffer bound to its
  // binding point is an INVALID_OPERATION, even in `standard`, where the shader's loop over
  // the block never executes — the block is still active because the compiler cannot prove
  // uMSCount is zero.
  const msUbo = gl.createBuffer();
  {
    gl.bindBuffer(gl.UNIFORM_BUFFER, msUbo);
    gl.bufferData(gl.UNIFORM_BUFFER, MS_COUNT * 64, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    const bi = gl.getUniformBlockIndex(beauty.prog, "shadow_matrix_block");
    if (bi !== gl.INVALID_INDEX) gl.uniformBlockBinding(beauty.prog, bi, MS_BLOCK_BINDING);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, MS_BLOCK_BINDING, msUbo);
  }

  let color = [0.863, 0.863, 0.863];   // ChimeraX-style near-white neutral gray
  let bg = [0.016, 0.024, 0.06];
  let light = { ...LIGHTING.standard };
  // Model-space clip slab: keep the fragments with clipRange[0] <= dot(p, clipN) <= clipRange[1].
  // A null normal is no clipping at all.
  let clipN = null, clipRange = [0, 0];
  let clipSettle = 0;                  // the pending "rebuild the ambient atlas" timer

  // Every program that honours the slab reads the same three uniforms, and all of them
  // have to be written whenever the program is bound — the depth passes included.
  function uploadClip(p) {
    gl.uniform1i(p.u("uClipOn"), clipN ? 1 : 0);
    if (!clipN) return;
    gl.uniform3fv(p.u("uClipN"), clipN);
    gl.uniform2fv(p.u("uClipRange"), clipRange);
  }

  // ---------------------------------------------------------------------------
  // Shadow maps
  // ---------------------------------------------------------------------------

  // A depth texture set up the way ChimeraX's Texture.initialize_depth does: 24-bit depth
  // with comparison filtering, so a sampler2DShadow lookup returns "is this point lit"
  // rather than a raw depth, with the hardware doing the 2x2 PCF for free.
  function makeDepthTex(size) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, size, size, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    return t;
  }
  // Depth-only framebuffer: legal in WebGL2 with no colour attachment at all.
  function makeDepthFbo(tex) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, tex, 0);
    gl.drawBuffers([gl.NONE]);
    gl.readBuffer(gl.NONE);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok ? f : null;
  }

  // Bound to every shadow sampler that is not in use this frame. Sampling an unbound
  // sampler2DShadow is undefined behaviour on some drivers even where the branch guarding
  // it never runs.
  const dummyDepthTex = makeDepthTex(1);

  let msTex = null, msFbo = null;
  let msReady = false;          // atlas matches the current geometry
  let msDepth = 1;              // scene diameter, the unit the transforms' depth axis uses
  const msXf = new Float32Array(MS_COUNT * 16);
  let shadowTex = null, shadowFbo = null;
  let shadowXf = null;          // model -> key-light shadow texture
  let shadowKey = null;         // the (geometry, light direction) the map above was built for
  let geomGen = 0;              // bumped whenever the set of drawn triangles changes

  // ChimeraX's geometry.sphere.sphere_points(n) — the Bauer spiral. The exact point set
  // matters: 64 directions is few enough that a different distribution puts the soft
  // shadows in visibly different places.
  const MS_DIRS = (() => {
    const p = [], snp = Math.sqrt(MS_COUNT * Math.PI);
    for (let i = 0; i < MS_COUNT; i++) {
      const phi = Math.acos(-1 + (2 * i + 1) / MS_COUNT);
      const theta = snp * phi, s = Math.sin(phi);
      p.push([s * Math.cos(theta), s * Math.sin(theta), Math.cos(phi)]);
    }
    return p;
  })();

  // A rotation with the given z axis; the rotation about that axis is free (ChimeraX's
  // geometry.orthonormal_frame says so, and the shadow map only cares which way it looks).
  function orthoFrame(z) {
    const zz = normalize(z);
    const a = Math.abs(zz[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const x = normalize(cross(a, zz));
    const y = cross(zz, x);
    return new Float32Array([x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, zz[0], zz[1], zz[2], 0, 0, 0, 0, 1]);
  }

  // ChimeraX Shadow._shadow_transforms: the view matrix the depth pass renders with, and
  // the model->texture-coordinate matrix the beauty pass looks the result up with.
  //
  //   lv    = translation(center - radius*ld) * orthonormal_frame(-ld)
  //   lvinv = lv^-1
  //   stf   = translation(0.5, 0.5, -bias) * scale(0.5/r, 0.5/r, -0.5/r) * lvinv
  //
  // The light sits a radius back along its own direction, the scene spans light-space z in
  // [-2r, 0], and the scale maps that onto the [0,1] the depth buffer stores. `bias` is
  // subtracted in those same units, i.e. as a fraction of the scene diameter.
  function shadowTransforms(ld, center, radius, bias) {
    const d = normalize(ld);
    const lv = orthoFrame([-d[0], -d[1], -d[2]]);
    lv[12] = center[0] - radius * d[0];
    lv[13] = center[1] - radius * d[1];
    lv[14] = center[2] - radius * d[2];
    const lvinv = invert(lv);
    if (!lvinv) return null;
    const S = [0.5 / radius, 0.5 / radius, -0.5 / radius], T = [0.5, 0.5, -bias];
    const stf = lvinv.slice();
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 3; r++)
        stf[c * 4 + r] = lvinv[c * 4 + r] * S[r] + (c === 3 ? T[r] : 0);
    return { lvinv, stf };
  }

  // ChimeraX Render.start_depth_render's projection: orthographic along z, spanning
  // +-radius across and the full [0, -2r] depth. Written out in the column-major order
  // WebGL wants.
  const shadowProjection = (r) =>
    new Float32Array([1 / r, 0, 0, 0, 0, 1 / r, 0, 0, 0, 0, -1 / r, 0, 0, 0, -1, 1]);

  // Bounding sphere of the scene, in model space. `radius` is ChimeraX's Bounds.radius:
  // the sphere around the bounding *box*, i.e. half its diagonal — not the tightest sphere
  // around the vertices. Both shadow volumes and the depth-cue range are sized from it, so
  // using anything tighter would shift the fade and clip the shadows.
  function boundingSphere() {
    const b = sceneBounds();
    if (!b) return null;
    const c = [0, 0, 0], s = [0, 0, 0];
    for (let a = 0; a < 3; a++) { c[a] = (b.lo[a] + b.hi[a]) / 2; s[a] = b.hi[a] - b.lo[a]; }
    return { center: c, radius: 0.5 * Math.hypot(s[0], s[1], s[2]) };
  }

  function drawDepth(proj, view) {
    gl.useProgram(depthP.prog);
    gl.uniformMatrix4fv(depthP.u("uProj"), false, proj);
    gl.uniformMatrix4fv(depthP.u("uView"), false, view);
    uploadClip(depthP);
    for (const m of drawable()) {
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
  }

  // The 64-direction ambient shadow atlas. Model space, so it is valid until the geometry
  // itself changes — which is what makes 64 depth passes affordable: they do not run again
  // when you rotate, pan or zoom.
  function ensureMultishadow() {
    if (msReady) return true;
    const bs = boundingSphere();
    if (!bs || bs.radius === 0) return false;
    if (!msTex) {
      msTex = makeDepthTex(MS_MAP_SIZE);
      msFbo = makeDepthFbo(msTex);
      if (!msFbo) return false;
    }
    const d = Math.ceil(Math.sqrt(MS_COUNT));   // tiles per axis: 8
    const s = Math.floor(MS_MAP_SIZE / d);      // tile size: 128
    gl.bindFramebuffer(gl.FRAMEBUFFER, msFbo);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.viewport(0, 0, MS_MAP_SIZE, MS_MAP_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    const proj = shadowProjection(bs.radius);
    for (let l = 0; l < MS_COUNT; l++) {
      const x = (l % d) * s, y = Math.floor(l / d) * s;
      gl.viewport(x, y, s, s);
      const t = shadowTransforms(MS_DIRS[l], bs.center, bs.radius, MS_DEPTH_BIAS);
      if (!t) continue;
      drawDepth(proj, t.lvinv);
      // Fold the sub-viewport into the transform, so the lookup lands in this tile:
      // translation(x/w, y/w, 0) * scale(vw/w, vh/h, 1) * stf. Only the x and y rows move,
      // which is what lets the fragment shader keep reading the light direction off the
      // untouched z row.
      const stf = t.stf;
      for (let c = 0; c < 4; c++) {
        stf[c * 4 + 0] = stf[c * 4 + 0] * (s / MS_MAP_SIZE) + (c === 3 ? x / MS_MAP_SIZE : 0);
        stf[c * 4 + 1] = stf[c * 4 + 1] * (s / MS_MAP_SIZE) + (c === 3 ? y / MS_MAP_SIZE : 0);
      }
      msXf.set(stf, l * 16);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindBuffer(gl.UNIFORM_BUFFER, msUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, msXf);
    gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    msDepth = 2 * bs.radius;
    msReady = true;
    return true;
  }

  // The key light's shadow. Unlike the ambient atlas this cannot be cached: the key light
  // is fixed to the camera (Lighting.move_lights_with_camera), so every rotation swings it
  // over the model. Its model-space direction is the camera-space constant carried back
  // through the rotation part of view*model.
  function buildKeyShadow(view, model) {
    const bs = boundingSphere();
    if (!bs || bs.radius === 0) { shadowXf = null; shadowKey = null; return; }
    if (!shadowTex) {
      shadowTex = makeDepthTex(SHADOW_MAP_SIZE);
      shadowFbo = makeDepthFbo(shadowTex);
      if (!shadowFbo) { shadowTex = null; shadowXf = null; return; }
    }
    const mv = multiply(view, model);
    // Rigid transform, so the inverse rotation is the transpose: pull the camera-space
    // direction back into model space by multiplying by the 3x3 from the left.
    const ld = [
      mv[0] * KEY_DIR[0] + mv[1] * KEY_DIR[1] + mv[2] * KEY_DIR[2],
      mv[4] * KEY_DIR[0] + mv[5] * KEY_DIR[1] + mv[6] * KEY_DIR[2],
      mv[8] * KEY_DIR[0] + mv[9] * KEY_DIR[1] + mv[10] * KEY_DIR[2],
    ];
    // Only the model-space light direction, the geometry and the clip slab can change what
    // this map holds, so a redraw that changed none of them (a recolour, a resize, a zoom,
    // a pan — the last two move the camera without turning the model) reuses it. Rotation
    // does change it, which is inherent: a camera-fixed light sweeps over the model as it
    // turns. Dragging the slab changes it too, and unlike the 64-pass ambient atlas one
    // extra depth pass per tick is cheap enough to just pay.
    const clipKey = clipN ? `${clipN.join()}/${clipRange.join()}` : "-";
    const key = `${geomGen}|${ld[0].toFixed(6)},${ld[1].toFixed(6)},${ld[2].toFixed(6)}|${clipKey}`;
    if (key === shadowKey && shadowXf) return;

    const t = shadowTransforms(ld, bs.center, bs.radius, SHADOW_DEPTH_BIAS);
    if (!t) { shadowXf = null; shadowKey = null; return; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFbo);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.viewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    drawDepth(shadowProjection(bs.radius), t.lvinv);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    shadowXf = t.stf;
    shadowKey = key;
  }

  // ---------------------------------------------------------------------------
  // Camera pivot probe
  // ---------------------------------------------------------------------------

  let posTex = null, posRbo = null, posFbo = null, pw = 0, ph = 0;

  // Distance from the eye to the front-most surface at the centre of the window, which is
  // what the camera turns into a ChimeraX-style centre of rotation. Renders a camera-space
  // position pass and reads back a few pixels; called from orbit.js at most once per
  // gesture, never per frame. Null on a driver without float targets, or over background,
  // which leaves the camera on the pivot it already had.
  function centreDepth() {
    const W = canvas.width, H = canvas.height, N = 5;   // a few px wide, so a 1px gap doesn't read as "empty"
    if (!floatTargets || !anyGeometry() || W < N || H < N) return null;
    if (pw !== W || ph !== H || !posFbo) {
      if (posTex) gl.deleteTexture(posTex);
      if (posRbo) gl.deleteRenderbuffer(posRbo);
      if (posFbo) gl.deleteFramebuffer(posFbo);
      pw = W; ph = H;
      posTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, posTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, W, H, 0, gl.RGBA, gl.HALF_FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      posRbo = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, posRbo);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, W, H);
      posFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, posFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, posTex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, posRbo);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null); posFbo = null; return null;
      }
    }
    const { proj, view, model } = camera();
    gl.bindFramebuffer(gl.FRAMEBUFFER, posFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, W, H);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.clearColor(0, 0, 0, 0);   // pos.w = 0 => background
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(geomP.prog);
    gl.uniformMatrix4fv(geomP.u("uProj"), false, proj);
    gl.uniformMatrix4fv(geomP.u("uView"), false, view);
    gl.uniformMatrix4fv(geomP.u("uModel"), false, model);
    uploadClip(geomP);
    for (const m of drawable()) {
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    const x = (W >> 1) - (N >> 1), y = (H >> 1) - (N >> 1);
    let px;
    try {
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      // An RGBA16F attachment is only readable through the pair the driver advertises.
      if (gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) !== gl.RGBA) return null;
      const type = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE);
      if (type === gl.FLOAT) {
        px = new Float32Array(4 * N * N);
        gl.readPixels(x, y, N, N, gl.RGBA, gl.FLOAT, px);
      } else if (type === gl.HALF_FLOAT) {
        const half = new Uint16Array(4 * N * N);
        gl.readPixels(x, y, N, N, gl.RGBA, gl.HALF_FLOAT, half);
        px = Float32Array.from(half, fromHalf);
      } else return null;
    } catch {
      return null;
    } finally {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    let best = null;
    for (let i = 0; i < N * N; i++) {
      if (px[i * 4 + 3] < 0.5) continue;   // pos.w = 0 => background
      const t = -px[i * 4 + 2];            // view-space z is negative in front of the eye
      if (t > 0 && (best === null || t < best)) best = t;
    }
    return best;
  }

  // fitRadius 1.33 keeps the map the on-screen size it has always been (the old
  // 45°-vertical camera at dist 3.2 showed 3.2·tan(22.5°) ≈ 1.33 world units of height);
  // the camera now reaches that framing from further back, through ChimeraX's field.
  const orbit = createOrbit(canvas, () => render(), null,
    { fitRadius: 1.33, probeDepth: centreDepth, modelBounds: sceneBounds });

  let renderScale = null;   // pinned backing-store scale while a video is being recorded
  function resize() {
    const dpr = renderScale != null ? renderScale : Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function camera() {
    const aspect = canvas.width / Math.max(1, canvas.height);
    const proj = projection(aspect, 0.01, 100);
    const view = lookAt(orbit.eye, [0, 0, 0], [0, 1, 0]);   // camera fixed; model carries the trackball rotation + pan
    return { proj, view, model: orbit.model };
  }

  // ChimeraX View.near_far_distances: the scene's bounding sphere projected onto the view
  // axis, padded by 1%, then clamped so the near plane stays in front of the eye. The
  // depth cue fades between fixed fractions of that range, which is why the fade tracks the
  // model as you zoom instead of being pinned to the GL frustum (fixed here at 0.01..100).
  function depthCueRange(view, model) {
    const bs = boundingSphere();
    if (!bs) return [1, 2];
    const c = mulVec4(multiply(view, model), [bs.center[0], bs.center[1], bs.center[2], 1]);
    const d = -c[2];                       // camera looks down -z, so this is the axial distance
    const r = (1 + NEAR_FAR_PAD) * bs.radius;
    let near = d - r, far = d + r;
    const nearMin = far > near ? MIN_NEAR_FRACTION * (far - near) : 1;
    near = Math.max(near, nearMin);
    if (far <= near) far = 2 * near;
    return [near + (far - near) * DEPTH_CUE_START, near + (far - near) * DEPTH_CUE_END];
  }

  function bindDepthTex(unit, tex) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex || dummyDepthTex);
  }

  function paint(alpha) {
    const { proj, view, model } = camera();
    const useMS = light.multishadow > 0 && anyGeometry() && ensureMultishadow();
    if (light.shadows && anyGeometry()) buildKeyShadow(view, model); else shadowXf = null;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.clearColor(bg[0], bg[1], bg[2], alpha);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!anyGeometry()) { drawLines(proj, view, model); return; }

    gl.useProgram(beauty.prog);
    gl.uniformMatrix4fv(beauty.u("uProj"), false, proj);
    gl.uniformMatrix4fv(beauty.u("uView"), false, view);
    gl.uniformMatrix4fv(beauty.u("uModel"), false, model);

    // Render._light_shader_parameter_values(): the intensities meet the material's
    // reflectivities here, and the shader only ever sees the product.
    const kd = MATERIAL.diffuse * light.key, ks = MATERIAL.specular * light.key;
    const fd = MATERIAL.diffuse * light.fill, am = MATERIAL.ambient * light.ambient;
    gl.uniform3fv(beauty.u("uKeyDir"), KEY_DIR);
    gl.uniform3fv(beauty.u("uFillDir"), FILL_DIR);
    gl.uniform3f(beauty.u("uKeyDiffuse"), kd, kd, kd);
    gl.uniform3f(beauty.u("uKeySpecular"), ks, ks, ks);
    gl.uniform1f(beauty.u("uSpecExponent"), MATERIAL.exponent);
    gl.uniform3f(beauty.u("uFillDiffuse"), fd, fd, fd);
    gl.uniform3f(beauty.u("uAmbient"), am, am, am);

    gl.uniform1i(beauty.u("uUseShadow"), shadowXf ? 1 : 0);
    if (shadowXf) gl.uniformMatrix4fv(beauty.u("uShadowXf"), false, shadowXf);
    gl.uniform1i(beauty.u("uMSCount"), useMS ? MS_COUNT : 0);
    gl.uniform1f(beauty.u("uMSDepth"), msDepth);
    bindDepthTex(0, useMS ? msTex : null); gl.uniform1i(beauty.u("uMSMap"), 0);
    bindDepthTex(1, shadowXf ? shadowTex : null); gl.uniform1i(beauty.u("uShadowMap"), 1);

    // Depth cue fades towards the background, as ChimeraX's does: View.background_color
    // keeps Lighting.depth_cue_color in step with it.
    gl.uniform3fv(beauty.u("uDepthCueColor"), bg);
    gl.uniform2fv(beauty.u("uDepthCueRange"), depthCueRange(view, model));

    uploadClip(beauty);

    const cLoc = beauty.u("uColor"), vcLoc = beauty.u("uUseVColor");
    for (const m of drawable()) {
      gl.uniform3fv(cLoc, m.color || color);
      gl.uniform1i(vcLoc, m.hasVColor ? 1 : 0);
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
    drawLines(proj, view, model);
  }

  function drawLines(proj, view, model) {
    if (!lineModels.size) return;
    gl.useProgram(lineP.prog);
    gl.uniformMatrix4fv(lineP.u("uProj"), false, proj);
    gl.uniformMatrix4fv(lineP.u("uView"), false, view);
    gl.uniformMatrix4fv(lineP.u("uModel"), false, model);
    for (const m of lineModels.values()) {
      if (!m.count) continue;
      gl.uniform3fv(lineP.u("uColor"), m.color);
      gl.bindVertexArray(m.vao);
      gl.drawArrays(gl.LINES, 0, m.count);
    }
    gl.bindVertexArray(null);
  }

  // The multishadow atlas lives in model space, so a canvas resize does not touch it; only
  // the pivot probe's targets follow the canvas, and those rebuild themselves on demand.
  function render() { resize(); paint(1); }

  // Axis-aligned box enclosing the vertices, for the camera's centre-of-rotation rule and
  // for sizing the shadow volumes. A single pass over the positions, once per contour.
  function aabb(positions) {
    if (!positions.length) return null;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        const v = positions[i + a];
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
    }
    return isFinite(lo[0]) ? { lo, hi } : null;
  }

  // Union of what is on screen — the camera frames and pivots on the whole scene, not on
  // whichever model happens to be first, and the shadows are cast over all of it.
  function sceneBounds() {
    let lo = null, hi = null;
    for (const m of drawable()) {
      if (!m.bounds) continue;
      if (!lo) { lo = m.bounds.lo.slice(); hi = m.bounds.hi.slice(); continue; }
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], m.bounds.lo[a]);
        hi[a] = Math.max(hi[a], m.bounds.hi[a]);
      }
    }
    return lo ? { lo, hi } : null;
  }

  function setModelMesh(id, positions, normals, indices, colors) {
    let m = models.get(id);
    if (!m) models.set(id, (m = makeModel()));
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf); gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.normBuf); gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    // Optional per-vertex uint8 RGB (painted surfaces). A mesh delivered without
    // colours turns painting off again — re-contouring clears the paint by design.
    m.hasVColor = !!(colors && colors.length === positions.length);
    gl.bindVertexArray(m.vao);
    if (m.hasVColor) {
      gl.bindBuffer(gl.ARRAY_BUFFER, m.colBuf); gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2);
    } else gl.disableVertexAttribArray(2);
    gl.bindVertexArray(null);
    m.count = indices.length;
    m.bounds = indices.length ? aabb(positions) : null;
    orbit.invalidatePivot();   // a new contour is a new front surface
    msReady = false; geomGen++;   // ...and a new set of occluders
    render();
  }

  function dropModel(id) {
    const m = models.get(id);
    if (!m) return;
    gl.deleteVertexArray(m.vao);
    gl.deleteBuffer(m.posBuf); gl.deleteBuffer(m.normBuf); gl.deleteBuffer(m.idxBuf);
    models.delete(id);
    orbit.invalidatePivot();
    msReady = false; geomGen++;
    render();
  }

  window.addEventListener("resize", render);

  return {
    render,
    // Single-map callers (the landscape viewer) address the one implicit model.
    setMesh: (positions, normals, indices) => setModelMesh(0, positions, normals, indices),
    setModelMesh, dropModel,
    // Line overlay (GL_LINES vertex pairs in model space). Same id space as nothing
    // else — line models live in their own map and never collide with mesh ids.
    setLineModel(id, positions, rgb) {
      let m = lineModels.get(id);
      if (!m) {
        m = { vao: gl.createVertexArray(), buf: gl.createBuffer(), count: 0, color: [1, 1, 1] };
        gl.bindVertexArray(m.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, m.buf);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
        lineModels.set(id, m);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, m.buf);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      m.count = positions.length / 3;
      if (rgb) m.color = rgb;
      render();
    },
    dropLineModel(id) {
      const m = lineModels.get(id);
      if (!m) return;
      gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.buf);
      lineModels.delete(id);
      render();
    },
    clearLineModels() { for (const id of [...lineModels.keys()]) this.dropLineModel(id); },
    setModelColor(id, rgb) { const m = models.get(id); if (m) { m.color = rgb; render(); } },
    setModelVisible(id, on) {
      const m = models.get(id);
      if (!m || m.visible === !!on) return;
      m.visible = !!on;
      orbit.invalidatePivot();   // hiding the front map exposes a different front surface
      msReady = false; geomGen++;   // and removes an occluder from the ambient shadows
      render();
    },
    clear() { for (const id of [...models.keys()]) dropModel(id); render(); },
    setColor(rgb) { color = rgb; render(); },
    // The background is also what the depth cue fades towards, so changing it changes the
    // shading of the far side of the map — as it does in ChimeraX.
    setBackground(rgb) { bg = rgb; render(); },
    setLighting(m) { light = { ...(LIGHTING[m] || LIGHTING.standard) }; render(); },
    // Model-space clip slab: keep lo <= dot(p, n) <= hi, with `n` a unit normal in the
    // [-1,1] cube coordinates. Pass a null normal to clear. The pivot probe honours it
    // too, so rotation centres on what is actually visible.
    //
    // The ambient atlas is 64 depth passes, far too many to redo per slider tick, so it
    // is left standing (a freshly revealed interior stays dark for a moment) and rebuilt
    // once the slab has been still for a beat. Clearing the clip takes the same route:
    // the standing atlas was built cut, and has to be rebuilt whole.
    setClip(n, lo, hi) {
      clipN = n ? [n[0], n[1], n[2]] : null;
      clipRange = [lo, hi];
      orbit.invalidatePivot();
      clearTimeout(clipSettle);
      clipSettle = setTimeout(() => { msReady = false; render(); }, 150);
      render();
    },
    resetView() { orbit.reset(); },
    getCamera() { return orbit.getState(); },   // handed to the other renderer on a render-mode switch
    setCamera(cam) { orbit.setState(cam); },
    captureStream(fps) { return canvas.captureStream(fps); },
    canvas,
    setRenderScale(s) { renderScale = s; render(); },
    async snapshot({ scale = 2, transparent = false } = {}) {
      const ow = canvas.width, oh = canvas.height;
      canvas.width = Math.floor(canvas.clientWidth * scale);
      canvas.height = Math.floor(canvas.clientHeight * scale);
      paint(transparent ? 0 : 1);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      canvas.width = ow; canvas.height = oh; render();
      return blob;
    },
  };
}

// IEEE half -> float. readPixels only hands back the type the driver advertises, and
// on some drivers that is HALF_FLOAT rather than FLOAT.
function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

function makeProg(gl, vs, fs) {
  const prog = link(gl, vs, fs);
  const cache = {};
  return { prog, u(n) { if (!(n in cache)) cache[n] = gl.getUniformLocation(prog, n); return cache[n]; } };
}
function link(gl, vsrc, fsrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("shader link failed: " + gl.getProgramInfoLog(p));
  return p;
}
function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader compile: " + gl.getShaderInfoLog(s));
  return s;
}
