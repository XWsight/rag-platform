"use strict";

const state = {apiKey: sessionStorage.getItem("rag-platform-api-key") || "", projects: [], applications: []};
const byId = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", state.apiKey);
  const response = await fetch(path, {...options, headers});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error?.message || `请求失败（${response.status}）`);
  return payload;
}

async function refresh() {
  state.apiKey = byId("platform-key").value.trim();
  sessionStorage.setItem("rag-platform-api-key", state.apiKey);
  const [projects, bases] = await Promise.all([api("/v1/projects"), api("/v1/knowledge-bases?limit=100")]);
  state.projects = projects.items || [];
  state.applications = [];
  for (const project of state.projects) {
    const payload = await api(`/v1/applications?project_id=${encodeURIComponent(project.id)}&limit=100`);
    state.applications.push(...(payload.items || []));
  }
  const kb = byId("application-kb");
  kb.replaceChildren(...(bases.items || []).filter((item) => item.status === "ready").map((item) => {
    const option = document.createElement("option"); option.value = item.id; option.textContent = item.name; return option;
  }));
  await renderApplications();
  byId("platform-status").textContent = "已连接";
}

async function renderApplications() {
  const list = byId("application-list");
  const answerSelect = byId("answer-application");
  list.replaceChildren(); answerSelect.replaceChildren();
  for (const application of state.applications) {
    const [revisions, deployments] = await Promise.all([
      api(`/v1/applications/${application.id}/revisions`),
      api(`/v1/applications/${application.id}/deployments`),
    ]);
    const entry = document.createElement("article"); entry.className = "application-entry";
    const title = document.createElement("strong"); title.textContent = application.display_name;
    const detail = document.createElement("small");
    detail.textContent = `${application.active_revision_id ? "已发布" : "未发布"} · ${revisions.count} 个版本 · ${deployments.count} 次部署`;
    entry.append(title, detail);
    for (const revision of revisions.items) {
      const row = document.createElement("small");
      row.textContent = `v${revision.revision_number} · ${revision.knowledge_base_ids.join(", ")}${revision.id === application.active_revision_id ? " · 当前" : ""}`;
      entry.append(row);
      if (revision.id !== application.active_revision_id) {
        const rollback = document.createElement("button"); rollback.className = "secondary-button";
        rollback.textContent = `发布 v${revision.revision_number}`;
        rollback.addEventListener("click", async () => {
          await api(`/v1/applications/${application.id}/deployments`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({revision_id: revision.id, expected_active_revision_id: application.active_revision_id})});
          await refresh();
        });
        entry.append(rollback);
      }
    }
    list.append(entry);
    if (application.active_revision_id) {
      const option = document.createElement("option"); option.value = application.id; option.textContent = application.display_name; answerSelect.append(option);
    }
  }
  if (!state.applications.length) list.textContent = "尚未创建应用。";
}

async function createAndPublish(event) {
  event.preventDefault(); byId("application-error").textContent = "";
  try {
    let project = state.projects[0];
    if (!project) project = await api("/v1/projects", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({display_name: "Default", description: "Default application project"})});
    const application = await api("/v1/applications", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({project_id: project.id, display_name: byId("application-name").value, application_kind: "knowledge_chat"})});
    const draft = await api(`/v1/applications/${application.id}/draft`, {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({expected_version: 0, knowledge_base_ids: [byId("application-kb").value], retrieval_profile: "default", answer_policy: {require_citations: true, allow_cloud: byId("application-cloud").checked, allow_web: false, allow_research: false}, session_policy: {enabled: true, ttl_seconds: 3600}, change_summary: "Initial release"})});
    const revision = await api(`/v1/applications/${application.id}/draft/revisions`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({expected_version: draft.version})});
    await api(`/v1/applications/${application.id}/deployments`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({revision_id: revision.id, expected_active_revision_id: application.active_revision_id})});
    await refresh();
  } catch (error) { byId("application-error").textContent = error.message; }
}

async function answer(event) {
  event.preventDefault();
  try {
    const result = await api(`/v1/apps/${byId("answer-application").value}/answer`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question: byId("application-question").value, session_id: `ui_${Date.now()}`})});
    byId("application-answer").textContent = `${result.answer}\n\nRevision: ${result.revision_id}\nTrace: ${result.trace_id}`;
  } catch (error) { byId("application-answer").textContent = error.message; }
}

byId("platform-key").value = state.apiKey;
byId("load-apps").addEventListener("click", () => refresh().catch((error) => { byId("platform-status").textContent = error.message; }));
byId("refresh-apps").addEventListener("click", () => refresh().catch(() => {}));
byId("application-form").addEventListener("submit", createAndPublish);
byId("application-answer-form").addEventListener("submit", answer);
if (state.apiKey) refresh().catch(() => {});
