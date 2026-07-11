// PCA down to 2 or 3 components for the latent-space view. Latent dims are small (a
// handful to a few dozen), so the covariance is a tiny d×d matrix and power iteration +
// deflation is plenty — no linear-algebra dependency, and it runs per epoch on a slider
// drag.
//
// `prev` is the previous epoch's basis. Eigenvectors are only defined up to a sign, so
// without it every epoch would be free to flip and the training "movie" would jitter
// meaninglessly. Flipping each new axis to agree with the last one keeps the landscape
// stable frame to frame — and in 3D, where you have also chosen a viewing angle, a flip
// is far more disorienting than in 2D.

function covariance(points, mean, n, d) {
  const cov = Array.from({ length: d }, () => new Float64Array(d));
  for (const p of points) {
    for (let i = 0; i < d; i++) {
      const a = p[i] - mean[i];
      for (let j = i; j < d; j++) cov[i][j] += a * (p[j] - mean[j]);
    }
  }
  const norm = Math.max(1, n - 1);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) { cov[i][j] /= norm; cov[j][i] = cov[i][j]; }
  }
  return cov;
}

function topEigen(cov, d, seed) {
  let v = new Float64Array(d);
  for (let i = 0; i < d; i++) v[i] = seed ? seed[i] : Math.sin(i + 1) + 0.5;   // deterministic start
  let lambda = 0;
  for (let it = 0; it < 128; it++) {
    const w = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += cov[i][j] * v[j];
      w[i] = s;
    }
    let norm = 0;
    for (let i = 0; i < d; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return { vec: v, val: 0 };
    for (let i = 0; i < d; i++) w[i] /= norm;
    let delta = 0;
    for (let i = 0; i < d; i++) delta += Math.abs(w[i] - v[i]);
    v = w;
    lambda = norm;
    if (delta < 1e-9) break;
  }
  return { vec: v, val: lambda };
}

function deflate(cov, v, lambda, d) {
  const out = Array.from({ length: d }, () => new Float64Array(d));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) out[i][j] = cov[i][j] - lambda * v[i] * v[j];
  }
  return out;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// -> { points: [[x, y(, z)], …], basis: [v1, …, vk], variance: [f1, …, fk] }
//    variance entries are the fraction of total variance each component explains.
export function pca(rows, k = 2, prev = null) {
  const n = rows.length;
  const zeros = () => Array.from({ length: k }, () => 0);
  if (!n) return { points: [], basis: null, variance: zeros() };
  const d = rows[0].length;

  const mean = new Float64Array(d);
  for (const p of rows) for (let i = 0; i < d; i++) mean[i] += p[i];
  for (let i = 0; i < d; i++) mean[i] /= n;

  const cov = covariance(rows, mean, n, d);
  let total = 0;
  for (let i = 0; i < d; i++) total += cov[i][i];

  // A latent space with fewer dims than components asked for still has to render:
  // the missing axes are flat, not absent.
  const axes = [];
  const vals = [];
  let work = cov;
  for (let c = 0; c < k; c++) {
    if (c >= d) { axes.push(new Float64Array(d)); vals.push(0); continue; }
    const e = topEigen(work, d, prev?.[c]);
    axes.push(e.vec);
    vals.push(e.val);
    work = deflate(work, e.vec, e.val, d);
  }

  // Sign-align each axis to the previous epoch's (eigenvectors are sign-ambiguous).
  if (prev) {
    for (let c = 0; c < k; c++) {
      if (prev[c] && dot(axes[c], prev[c]) < 0) {
        for (let i = 0; i < d; i++) axes[c][i] = -axes[c][i];
      }
    }
  }

  const points = rows.map((p) => {
    const out = zeros();
    for (let i = 0; i < d; i++) {
      const cen = p[i] - mean[i];
      for (let c = 0; c < k; c++) out[c] += cen * axes[c][i];
    }
    return out;
  });

  return {
    points,
    basis: axes.map((a) => Array.from(a)),
    variance: vals.map((v) => (total > 0 ? v / total : 0)),
  };
}
