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

// Same as `req`, for endpoints that answer with binary (image/png).
async function blob(path) {
  const token = getToken();
  const res = await fetch(path, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new ApiError(res.status, detail);
  }
  return res.blob();
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
  // `fallback` degrades an unusable start path to the nearest listable folder
  // instead of erroring — used when opening the picker at a field's current value.
  files: (path, fallback) =>
    req(`/api/files${path ? `?path=${encodeURIComponent(path)}${fallback ? "&fallback=true" : ""}` : ""}`),
  mkdir: (path, name) => req("/api/files/mkdir", { method: "POST", body: JSON.stringify({ path: path || null, name }) }),
  jobs: () => req("/api/jobs"),
  job: (id) => req(`/api/jobs/${id}`),
  jobLog: (id) => req(`/api/jobs/${id}/log`),
  renameJob: (id, name) => req(`/api/jobs/${id}`, { method: "PUT", body: JSON.stringify({ name }) }),
  startJob: (id) => req(`/api/jobs/${id}/start`, { method: "POST" }),
  cancelJob: (id) => req(`/api/jobs/${id}/cancel`, { method: "POST" }),
  // Re-running rebuilds this job's run folder, which is where everything downstream
  // reads its inputs — so `cascade` says what happens to those: "stale" flags them,
  // "queue" also re-runs them in order. `cancel_running` is the user's explicit yes to
  // stopping downstream runs still executing (without it the server answers 409).
  rerunJob: (id, cascade) => req(`/api/jobs/${id}/rerun`, { method: "POST", body: JSON.stringify(cascade || {}) }),
  // The runs this one feeds, transitively — what a rerun would invalidate.
  jobDependents: (id) => req(`/api/jobs/${id}/dependents`),
  // Recipe for copying a job into another project (program + params, inputs cleared).
  jobCopyPrefill: (id) => req(`/api/jobs/${id}/copy-prefill`),
  continueJob: (id, cascade) => req(`/api/jobs/${id}/continue`, { method: "POST", body: JSON.stringify(cascade || {}) }),
  // Disk usage. Measured on demand (a folder walk), cached server-side for a couple
  // of minutes; `refresh` forces a re-measure.
  jobUsage: (id, refresh) => req(`/api/jobs/${id}/usage${refresh ? "?refresh=true" : ""}`),
  projectUsage: (id, refresh) => req(`/api/projects/${id}/usage${refresh ? "?refresh=true" : ""}`),
  projectsUsage: (refresh) => req(`/api/projects/usage${refresh ? "?refresh=true" : ""}`),
  deleteJob: (id) => req(`/api/jobs/${id}`, { method: "DELETE" }),
  // labels (colored tags on jobs)
  labels: () => req("/api/labels"),
  createLabel: (name, color) => req("/api/labels", { method: "POST", body: JSON.stringify({ name, color: color || null }) }),
  updateLabel: (id, body) => req(`/api/labels/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteLabel: (id) => req(`/api/labels/${id}`, { method: "DELETE" }),
  setJobLabels: (jobId, labels) => req(`/api/jobs/${jobId}/labels`, { method: "PUT", body: JSON.stringify({ labels }) }),

  // TensorBoard is no longer the metrics UI — the GUI reads the same event files
  // itself (see below). These stay so a run can still be opened in real TB, in its
  // own browser tab, for anything the native view doesn't cover.
  startTensorBoard: (logdir) =>
    req("/api/tensorboard", { method: "POST", body: JSON.stringify({ logdir }) }),
  stopTensorBoard: (id) => req(`/api/tensorboard/${id}`, { method: "DELETE" }),
  restartTensorBoard: (id) => req(`/api/tensorboard/${id}/restart`, { method: "POST" }),
  tensorboards: () => req("/api/tensorboard"),

  // native metrics: the GUI's own reader over the run's tfevents + projector TSVs
  metricsJobs: () => req("/api/metrics/jobs"),
  metricsManifest: (logdir) => req(`/api/metrics/manifest?logdir=${encodeURIComponent(logdir)}`),
  metricsScalars: (logdir, tags) =>
    req(`/api/metrics/scalars?logdir=${encodeURIComponent(logdir)}` +
      (tags && tags.length ? `&tags=${encodeURIComponent(tags.join("\n"))}` : "")),
  metricsEmbedding: (logdir, step, tag) =>
    req(`/api/metrics/embedding?logdir=${encodeURIComponent(logdir)}&step=${step}` +
      (tag ? `&tag=${encodeURIComponent(tag)}` : "")),
  dropMetrics: (logdir) => req(`/api/metrics/cache?logdir=${encodeURIComponent(logdir)}`, { method: "DELETE" }),
  // PNGs come back as blobs, not JSON: an <img src> can't carry the bearer header,
  // so they are fetched here and handed to the DOM as object URLs.
  // `frame` indexes the wall-clock-ordered images, not the step: hax logs every epoch
  // without a global_step, so they all carry step 0.
  metricsImage: (logdir, run, tag, frame) =>
    blob(`/api/metrics/image?logdir=${encodeURIComponent(logdir)}&run=${encodeURIComponent(run)}` +
      `&tag=${encodeURIComponent(tag)}${frame != null ? `&frame=${frame}` : ""}`),
  metricsSprite: (logdir, step, tag) =>
    blob(`/api/metrics/sprite?logdir=${encodeURIComponent(logdir)}&step=${step}` +
      (tag ? `&tag=${encodeURIComponent(tag)}` : "")),

  // workflows (visual pipeline builder)
  workflowCatalog: () => req("/api/workflows/catalog"),
  workflows: (projectId) => req(`/api/workflows${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  workflow: (id) => req(`/api/workflows/${id}`),
  createWorkflow: (body) => req("/api/workflows", { method: "POST", body: JSON.stringify(body) }),
  updateWorkflow: (id, body) => req(`/api/workflows/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteWorkflow: (id) => req(`/api/workflows/${id}`, { method: "DELETE" }),
  // runWorkflow with no `nodes` runs the whole graph; pass an array of node ids
  // to run just that subset (a single node or a selection). If a run is already in
  // flight the steps are appended to its queue rather than rejected.
  // `cascade` carries the same rerun choice as rerunJob, for the steps this subset
  // feeds; ignored when the whole graph runs. It also carries `restart_running`: the
  // user's yes to stopping steps in the request that are still executing (without it a
  // busy step answers 409 rather than quietly letting its old run continue).
  runWorkflow: (id, nodes, cascade) =>
    req(`/api/workflows/${id}/run`, { method: "POST", body: JSON.stringify({ ...(nodes ? { nodes } : {}), ...(cascade || {}) }) }),
  stopWorkflow: (id) => req(`/api/workflows/${id}/stop`, { method: "POST" }),
  // Runs the server was restarted out from under, whose next step never started.
  resumableWorkflows: () => req("/api/workflows/resumable"),
  resumeWorkflow: (id) => req(`/api/workflows/${id}/resume`, { method: "POST" }),
  dismissWorkflowResume: (id) => req(`/api/workflows/${id}/dismiss-resume`, { method: "POST" }),
  workflowFromProject: (projectId, name) => req(`/api/workflows/from-project/${projectId}`, { method: "POST", body: JSON.stringify({ name: name || null }) }),
  workflowSaveAsProject: (id, name) => req(`/api/workflows/${id}/save-as-project`, { method: "POST", body: JSON.stringify({ name }) }),
  // Clone a workflow's graph into a new standalone workflow under another project.
  copyWorkflow: (id, projectId, name) => req(`/api/workflows/${id}/copy`, { method: "POST", body: JSON.stringify({ project_id: projectId, name: name || null }) }),

  // WebGL viewer sessions
  viewers: () => req("/api/viewer"),
  // gpu: "0" / "cpu" pins the viewer's device; omit to fall back to the saved default.
  viewerFromJob: (jobId, space, gpu) => {
    const q = new URLSearchParams();
    if (space != null) q.set("space", space);
    if (gpu) q.set("gpu", gpu);
    const qs = q.toString();
    return req(`/api/viewer/from-job/${jobId}${qs ? `?${qs}` : ""}`, { method: "POST" });
  },
  flexconsensusSpaces: (jobId) => req(`/api/viewer/flexconsensus/${jobId}`),
  stopViewer: (id) => req(`/api/viewer/${id}`, { method: "DELETE" }),

  // Map (.mrc) viewer sessions — inspect any volume a job produced
  jobMrcs: (jobId) => req(`/api/jobs/${jobId}/mrcs`),
  mrcViewerFromJob: (jobId, path) => req(`/api/mrcviewer/from-job/${jobId}?path=${encodeURIComponent(path)}`, { method: "POST" }),
  mrcViewer: (path, title) => req("/api/mrcviewer", { method: "POST", body: JSON.stringify({ path, title: title || null }) }),
  // Several maps in ONE viewer (a model panel + morph), not one tab each.
  mrcViewerMulti: (paths, title) => req("/api/mrcviewer", { method: "POST", body: JSON.stringify({ paths, title: title || null }) }),
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
  setupNotice: () => req("/api/setup-notice"),
  silenceSetupNotice: () => req("/api/setup-notice/silence", { method: "POST" }),
  projects: () => req("/api/projects"),
  createProject: (name) => req("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  renameProject: (id, name) => req(`/api/projects/${id}`, { method: "PUT", body: JSON.stringify({ name }) }),
  deleteProject: (id) => req(`/api/projects/${id}`, { method: "DELETE" }),
  project: (id) => req(`/api/projects/${id}`),
  projectJobs: (id) => req(`/api/projects/${id}/jobs`),
  // Recover input bindings an edit-and-rerun flattened to literal paths, so the
  // workflow edges they back reappear. Returns { relinked: [...], count }.
  relinkProject: (id) => req(`/api/projects/${id}/relink`, { method: "POST" }),
  projectSources: (id, kind) => req(`/api/projects/${id}/sources?kind=${encodeURIComponent(kind)}`),
  // `cascade` ({cascade, cancel_running}) only bites alongside replaceJobId: editing a
  // run in place re-makes its outputs, so it invalidates what it feeds just as a rerun does.
  createProjectJob: (projectId, program, args, gpu, name, bindings, launch = true, recreateSsd = false, replaceJobId = null, resume = false, cascade = null) =>
    req(`/api/projects/${projectId}/jobs`, {
      method: "POST",
      // The name is always sent as a string, never null: on a replace the server reads
      // null as "not specified" and keeps the run's current name, so clearing the field
      // in the form has to arrive as "" to actually clear it.
      body: JSON.stringify({ program, args, gpu, name: name || "", bindings: bindings || {}, launch, recreate_ssd: recreateSsd, replace_job_id: replaceJobId, resume, ...(cascade || {}) }),
    }),
};

// Optional AI assistant. status() is always safe to call (reports
// {available:false} when the hax-em-assistant package isn't installed).
api.assistantStatus = () => req("/api/assistant/status");
api.diagnoseJob = (id) => req(`/api/jobs/${id}/diagnose`, { method: "POST" });
// Rebuild the docs vector index from the current hax-em-docs checkout (live,
// no GUI restart). Returns {ok, rebuilt, chunks, ...}.
api.assistantReindexDocs = (force = true) =>
  req(`/api/assistant/reindex-docs?force=${force ? "true" : "false"}`, { method: "POST" });
// Which artifacts a job exposes to the assistant (metrics/latents/volumes).
api.assistantRunPreview = (jobId) => req(`/api/assistant/run-preview/${jobId}`);
// Read-only plan + visible GPUs for preparing a run's deformation field.
api.assistantPrepareMotionPlan = (jobId) =>
  req(`/api/assistant/actions/prepare-motion/${jobId}`);
// Run the (confirmed) prepare-motion pipeline, streaming progress lines.
api.assistantPrepareMotion = ({ jobId, gpu, onStatus, onDone, onError }) => {
  const token = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${proto}://${location.host}/ws/assistant/prepare-motion${q}`);
  let settled = false;   // done/error already delivered (or caller closed)
  ws.onmessage = (e) => {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.status != null) { if (onStatus) onStatus(m.status); }
    else if (m.done) { settled = true; if (onDone) onDone(m.deformation_dir); ws.close(); }
    else if (m.error != null) { settled = true; if (onError) onError(m.error); ws.close(); }
  };
  ws.onopen = () => ws.send(JSON.stringify({ job_id: jobId, gpu }));
  // One-shot pipeline: no auto-reconnect (the server-side run can't be
  // rejoined), but a drop must not leave the dialog spinning forever.
  ws.onclose = () => { if (!settled) { settled = true; if (onError) onError("connection lost"); } };
  return { close: () => { settled = true; ws.close(); } };
};
// Expand an assistant job proposal (program + path) into a draft spec + path facts.
api.assistantPreviewJob = (program, path) =>
  req("/api/assistant/actions/preview-job", { method: "POST", body: JSON.stringify({ program, path: path || null }) });

// Generate a PDF run report and trigger a browser download. Returns nothing;
// throws ApiError on failure. Blocking on the server (figures + narration).
api.assistantRunReport = async (jobId, narrate = true) => {
  const token = getToken();
  const res = await fetch(`/api/assistant/run-report/${jobId}?narrate=${narrate ? "true" : "false"}`, {
    method: "POST",
    headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new ApiError(res.status, detail);
  }
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") || "";
  const m = /filename="?([^"]+)"?/.exec(cd);
  const name = m ? m[1] : `report_${jobId}.pdf`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
};

// Persisted assistant chats (optionally grouped by project).
api.chats = () => req("/api/chats");
api.chat = (id) => req(`/api/chats/${id}`);
api.createChat = (projectId, title) =>
  req("/api/chats", { method: "POST", body: JSON.stringify({ project_id: projectId || null, title: title || null }) });
api.updateChat = (id, body) => req(`/api/chats/${id}`, { method: "PUT", body: JSON.stringify(body) });
api.deleteChat = (id) => req(`/api/chats/${id}`, { method: "DELETE" });

// Open a chat socket. Send full-history requests via send({messages, grounded});
// tokens arrive through onToken, then onDone; onError on failure. Returns
// { send, close }.
export function assistantChat({ onToken, onStatus, onAction, onDone, onError, onClose }) {
  // Reconnecting: the old socket was created once and cached for the app's
  // lifetime with no onclose — after a server restart every Send silently
  // dropped its payload and the composer locked on "Stop" forever. Now a drop
  // notifies once via onClose (so the view can unlock a stuck turn) and the
  // socket reopens with backoff; later sends queue until it is OPEN again.
  let ws = null;
  let closed = false;
  let attempt = 0;
  let timer = null;
  let pending = [];

  const open = () => {
    if (closed) return;
    ws = new WebSocket(wsUrl("/ws/assistant/chat"));
    ws.onopen = () => {
      attempt = 0;
      const queued = pending; pending = [];
      for (const d of queued) ws.send(d);
    };
    ws.onmessage = (e) => {
      let m; try { m = JSON.parse(e.data); } catch { return; }
      if (m.token != null) onToken(m.token);
      else if (m.status != null) { if (onStatus) onStatus(m.status); }
      else if (m.action != null) { if (onAction) onAction(m.action); }
      else if (m.error) { if (onError) onError(m.error); }
      else if (m.done) onDone(m.turn);
    };
    ws.onclose = () => {
      if (closed) return;
      // A mid-turn drop loses the turn server-side; tell the view once so it
      // can unlock, then quietly reopen for the next question.
      if (onClose) onClose();
      attempt += 1;
      timer = setTimeout(open, Math.min(15000, 500 * 2 ** Math.min(attempt, 5)));
    };
  };
  open();

  // `new WebSocket()` returns in CONNECTING state; calling send() before the
  // socket is OPEN throws InvalidStateError. Queue payloads until onopen
  // flushes (also covers the reconnecting window).
  const raw = (payload) => {
    const data = JSON.stringify(payload);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(data);
    else pending.push(data);
  };
  return {
    send: (payload) => raw(payload),
    stop: () => raw({ stop: true }),
    close: () => {
      closed = true; pending = [];
      if (timer) clearTimeout(timer);
      try { ws.close(); } catch { /* already closed */ }
    },
  };
}

// Subscribe to a job's live log. Returns a close() function.
// `shouldReconnect` (from the caller, which knows the job's status) gates the
// reconnect: a finished job's socket is closed by the server on purpose after
// the replay, and blindly reopening it would loop forever.
export function subscribeLog(jobId, onMessage, opts) {
  return subscribeWS(`/ws/jobs/${jobId}/log`, onMessage, opts);
}

// Subscribe to the live update-installer log. Returns a close() function.
// Always reconnects: the update restarts the server, and reconnecting through
// that restart is exactly how the modal shows the tail end of the install.
export function subscribeUpdateLog(onMessage) {
  return subscribeWS(`/ws/updates/log`, onMessage);
}

function wsUrl(path) {
  const token = getToken();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${proto}://${location.host}${path}${q}`;
}

// A WebSocket subscription that survives the other end going away. The old
// version set only onmessage: a server restart (which the GUI itself triggers
// on update) or a proxy idle-timeout silently froze the stream — the log
// stopped mid-training with no indication and no recovery short of a reload.
// Reconnects with exponential backoff (0.5 s → 15 s), resets on success, and
// guards JSON.parse so one malformed frame can't kill the handler.
function subscribeWS(path, onMessage, { shouldReconnect } = {}) {
  let ws = null;
  let closed = false;
  let attempt = 0;
  let timer = null;
  const open = () => {
    if (closed) return;
    ws = new WebSocket(wsUrl(path));
    ws.onopen = () => { attempt = 0; };
    ws.onmessage = (e) => {
      let m; try { m = JSON.parse(e.data); } catch { return; }
      onMessage(m);
    };
    ws.onclose = () => {
      if (closed) return;
      if (shouldReconnect && !shouldReconnect()) { closed = true; return; }
      attempt += 1;
      timer = setTimeout(open, Math.min(15000, 500 * 2 ** Math.min(attempt, 5)));
    };
  };
  open();
  return () => {
    closed = true;
    if (timer) clearTimeout(timer);
    try { ws.close(); } catch { /* already closed */ }
  };
}
