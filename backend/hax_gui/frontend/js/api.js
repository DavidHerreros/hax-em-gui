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
  updates: () => req("/api/updates"),
  checkUpdates: () => req("/api/updates/check", { method: "POST" }),
  updateRunState: () => req("/api/updates/run"),
  runUpdate: (pkg, mode = "full") => req("/api/updates/run", { method: "POST", body: JSON.stringify({ package: pkg, mode }) }),
  health: () => req("/api/health"),
  // developer: git branches on editable (devel) installs
  branches: () => req("/api/branches"),
  fetchBranches: (pkg) => req("/api/branches/fetch", { method: "POST", body: JSON.stringify({ package: pkg }) }),
  switchBranch: (pkg, branch) => req("/api/branches/switch", { method: "POST", body: JSON.stringify({ package: pkg, branch }) }),
  setBranchSource: (pkg, path) => req("/api/branches/source", { method: "POST", body: JSON.stringify({ package: pkg, path: path || null }) }),
  restartServer: () => req("/api/server/restart", { method: "POST" }),
  files: (path) => req(`/api/files${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  jobs: () => req("/api/jobs"),
  job: (id) => req(`/api/jobs/${id}`),
  jobLog: (id) => req(`/api/jobs/${id}/log`),
  renameJob: (id, name) => req(`/api/jobs/${id}`, { method: "PUT", body: JSON.stringify({ name }) }),
  startJob: (id) => req(`/api/jobs/${id}/start`, { method: "POST" }),
  cancelJob: (id) => req(`/api/jobs/${id}/cancel`, { method: "POST" }),
  rerunJob: (id) => req(`/api/jobs/${id}/rerun`, { method: "POST" }),
  deleteJob: (id) => req(`/api/jobs/${id}`, { method: "DELETE" }),
  // labels (colored tags on jobs)
  labels: () => req("/api/labels"),
  createLabel: (name, color) => req("/api/labels", { method: "POST", body: JSON.stringify({ name, color: color || null }) }),
  updateLabel: (id, body) => req(`/api/labels/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteLabel: (id) => req(`/api/labels/${id}`, { method: "DELETE" }),
  setJobLabels: (jobId, labels) => req(`/api/jobs/${jobId}/labels`, { method: "PUT", body: JSON.stringify({ labels }) }),

  startTensorBoard: (logdir) =>
    req("/api/tensorboard", { method: "POST", body: JSON.stringify({ logdir }) }),
  tensorboards: () => req("/api/tensorboard"),
  metricsJobs: () => req("/api/metrics/jobs"),

  // workflows (visual pipeline builder)
  workflowCatalog: () => req("/api/workflows/catalog"),
  workflows: (projectId) => req(`/api/workflows${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  workflow: (id) => req(`/api/workflows/${id}`),
  createWorkflow: (body) => req("/api/workflows", { method: "POST", body: JSON.stringify(body) }),
  updateWorkflow: (id, body) => req(`/api/workflows/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteWorkflow: (id) => req(`/api/workflows/${id}`, { method: "DELETE" }),
  // runWorkflow with no `nodes` runs the whole graph; pass an array of node ids
  // to run just that subset (a single node or a selection).
  runWorkflow: (id, nodes) => req(`/api/workflows/${id}/run`, { method: "POST", body: JSON.stringify(nodes ? { nodes } : {}) }),
  workflowFromProject: (projectId, name) => req(`/api/workflows/from-project/${projectId}`, { method: "POST", body: JSON.stringify({ name: name || null }) }),
  workflowSaveAsProject: (id, name) => req(`/api/workflows/${id}/save-as-project`, { method: "POST", body: JSON.stringify({ name }) }),

  // WebGL viewer sessions
  viewers: () => req("/api/viewer"),
  viewerFromJob: (jobId, space) => req(`/api/viewer/from-job/${jobId}${space != null ? `?space=${space}` : ""}`, { method: "POST" }),
  flexconsensusSpaces: (jobId) => req(`/api/viewer/flexconsensus/${jobId}`),
  stopViewer: (id) => req(`/api/viewer/${id}`, { method: "DELETE" }),

  // Map (.mrc) viewer sessions — inspect any volume a job produced
  jobMrcs: (jobId) => req(`/api/jobs/${jobId}/mrcs`),
  mrcViewerFromJob: (jobId, path) => req(`/api/mrcviewer/from-job/${jobId}?path=${encodeURIComponent(path)}`, { method: "POST" }),
  mrcViewer: (path, title) => req("/api/mrcviewer", { method: "POST", body: JSON.stringify({ path, title: title || null }) }),
  stopMrcViewer: (id) => req(`/api/mrcviewer/${id}`, { method: "DELETE" }),

  // Files a job took as inputs (maps + metadata referenced in its args)
  jobInputs: (jobId) => req(`/api/jobs/${jobId}/inputs`),

  // Metadata (.xmd/.star/.cs) viewer sessions — browse any table a job produced
  jobMetadata: (jobId) => req(`/api/jobs/${jobId}/metadata`),
  mdViewerFromJob: (jobId, path) => req(`/api/mdviewer/from-job/${jobId}?path=${encodeURIComponent(path)}`, { method: "POST" }),
  mdViewer: (path, title) => req("/api/mdviewer", { method: "POST", body: JSON.stringify({ path, title: title || null }) }),
  stopMdViewer: (id) => req(`/api/mdviewer/${id}`, { method: "DELETE" }),

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
  createProjectJob: (projectId, program, args, gpu, name, bindings, launch = true, recreateSsd = false) =>
    req(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      body: JSON.stringify({ program, args, gpu, name: name || null, bindings: bindings || {}, launch, recreate_ssd: recreateSsd }),
    }),
};

// Subscribe to a job's live log. Returns a close() function.
export function subscribeLog(jobId, onMessage) {
  return subscribeWS(`/ws/jobs/${jobId}/log`, onMessage);
}

// Subscribe to the live update-installer log. Returns a close() function.
export function subscribeUpdateLog(onMessage) {
  return subscribeWS(`/ws/updates/log`, onMessage);
}

function subscribeWS(path, onMessage) {
  const token = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${proto}://${location.host}${path}${q}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return () => { try { ws.close(); } catch { /* already closed */ } };
}
