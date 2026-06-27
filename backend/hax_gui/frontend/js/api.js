// REST + WebSocket client. Same-origin (served by the backend), so paths are
// relative. Shared bearer token (single-tenant) kept in localStorage.

const TOKEN_KEY = "hax_gui_token";
export const getToken = () => localStorage.getItem(TOKEN_KEY) || "";
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function req(path, init = {}) {
  const token = getToken();
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init.headers || {}),
    },
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const api = {
  health: () => req("/api/health"),
  programs: () => req("/api/programs"),
  program: (name) => req(`/api/programs/${name}`),
  gpus: () => req("/api/gpus"),
  files: (path) => req(`/api/files${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  jobs: () => req("/api/jobs"),
  job: (id) => req(`/api/jobs/${id}`),
  jobLog: (id) => req(`/api/jobs/${id}/log`),
  renameJob: (id, name) => req(`/api/jobs/${id}`, { method: "PUT", body: JSON.stringify({ name }) }),
  startJob: (id) => req(`/api/jobs/${id}/start`, { method: "POST" }),
  cancelJob: (id) => req(`/api/jobs/${id}/cancel`, { method: "POST" }),
  rerunJob: (id) => req(`/api/jobs/${id}/rerun`, { method: "POST" }),
  deleteJob: (id) => req(`/api/jobs/${id}`, { method: "DELETE" }),
  startTensorBoard: (logdir) =>
    req("/api/tensorboard", { method: "POST", body: JSON.stringify({ logdir }) }),
  tensorboards: () => req("/api/tensorboard"),
  metricsJobs: () => req("/api/metrics/jobs"),

  // WebGL viewer sessions
  viewers: () => req("/api/viewer"),
  viewerFromJob: (jobId) => req(`/api/viewer/from-job/${jobId}`, { method: "POST" }),
  stopViewer: (id) => req(`/api/viewer/${id}`, { method: "DELETE" }),

  // settings + projects
  getSettings: () => req("/api/settings"),
  putSettings: (data) => req("/api/settings", { method: "PUT", body: JSON.stringify(data) }),
  projects: () => req("/api/projects"),
  createProject: (name) => req("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  renameProject: (id, name) => req(`/api/projects/${id}`, { method: "PUT", body: JSON.stringify({ name }) }),
  deleteProject: (id) => req(`/api/projects/${id}`, { method: "DELETE" }),
  project: (id) => req(`/api/projects/${id}`),
  projectJobs: (id) => req(`/api/projects/${id}/jobs`),
  projectSources: (id, kind) => req(`/api/projects/${id}/sources?kind=${encodeURIComponent(kind)}`),
  createProjectJob: (projectId, program, args, gpu, name, bindings, launch = true) =>
    req(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      body: JSON.stringify({ program, args, gpu, name: name || null, bindings: bindings || {}, launch }),
    }),
};

// Subscribe to a job's live log. Returns a close() function.
export function subscribeLog(jobId, onMessage) {
  const token = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${proto}://${location.host}/ws/jobs/${jobId}/log${q}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return () => { try { ws.close(); } catch { /* already closed */ } };
}
