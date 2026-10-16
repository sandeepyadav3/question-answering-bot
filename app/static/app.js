"use strict";
const form = document.querySelector("#qa-form");
const submit = document.querySelector("#submit-button");
const notice = document.querySelector("#notice");
const section = document.querySelector("#results-section");
const list = document.querySelector("#results-list");
const download = document.querySelector("#download-button");
const documentInput = document.querySelector("#document-file");
const tenantToken = document.querySelector("#tenant-token");
const savedDocument = document.querySelector("#saved-document");
let latestResponse = null;
let savedDocumentId = null;
let savedFileKey = null;
let savedToken = "";
let savedTenantId = "";

function fileKey(file) {
  return file ? `${file.name}:${file.size}:${file.lastModified}` : "";
}
function forgetSavedDocument() {
  savedDocumentId = null;
  savedFileKey = null;
  savedToken = "";
  savedTenantId = "";
  savedDocument.hidden = true;
}
async function postForm(url, body) {
  const response = await fetch(url, {
    method: "POST",
    body,
    headers: { Authorization: `Bearer ${tenantToken.value}` },
  });
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`The server returned an unreadable response (${response.status}). Please try again.`); }
  if (!response.ok) {
    const message = data.error?.code === "PROVIDER_NOT_CONFIGURED"
      ? "The answer service is not configured yet. Please ask the server administrator to finish setup."
      : data.error?.message || `Request failed (${response.status}).`;
    const requestId = data.error?.request_id || response.headers.get("X-Request-ID");
    throw new Error(message + (requestId ? ` Request ID: ${requestId}` : ""));
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function showNotice(message, kind) {
  notice.textContent = message;
  notice.dataset.kind = kind;
  notice.hidden = false;
}
function renderResults(data) {
  list.replaceChildren();
  const labels = { answered: "ANSWERED", not_found: "NOT FOUND", error: "ERROR" };
  data.results.forEach((result, index) => {
    const card = element("article", "result-card");
    const heading = element("div", "result-heading");
    heading.append(element("h3", "", `${index + 1}. ${result.question}`),
      element("span", `badge ${result.status}`, labels[result.status] || "UNKNOWN"));
    let answerText = result.answer;
    if (result.status === "error") answerText = result.error?.message || "This question could not be processed.";
    else if (result.status === "not_found" && result.explanation) answerText = `${result.answer}. ${result.explanation}`;
    card.append(heading, element("p", "answer", answerText));
    if (result.citations?.length) {
      const details = element("details", "citations");
      details.append(element("summary", "", `View evidence (${result.citations.length})`));
      result.citations.forEach((citation) => {
        const location = citation.source_type === "pdf" ? `Page ${citation.page}` : citation.json_path;
        details.append(element("p", "citation-label", location), element("blockquote", "", citation.snippet));
      });
      card.append(details);
    }
    list.append(card);
  });
  const metadata = data.metadata;
  document.querySelector("#results-summary").textContent =
    `${metadata.questions_answered} answered · ${metadata.questions_not_found} not found · ${metadata.questions_failed} failed · ${(metadata.latency_ms / 1000).toFixed(1)}s`;
  document.querySelector("#request-reference").textContent = `Request ID: ${data.request_id}`;
  section.hidden = false;
  document.querySelector("#results-title").focus();
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled || !form.reportValidity()) return;
  latestResponse = null;
  section.hidden = true;
  list.replaceChildren();
  const selected = documentInput.files[0];
  const selectedKey = fileKey(selected);
  submit.disabled = true;
  form.setAttribute("aria-busy", "true");
  showNotice("Reading your document and preparing answers…", "loading");
  try {
    if (tenantToken.value !== savedToken) forgetSavedDocument();
    if (!savedDocumentId || selectedKey !== savedFileKey) {
      const upload = new FormData();
      upload.append("document_file", selected);
      const created = await postForm("/api/v1/documents", upload);
      savedDocumentId = created.document_id;
      savedFileKey = selectedKey;
      savedToken = tenantToken.value;
      savedTenantId = created.tenant_id;
    }
    const questions = new FormData();
    questions.append("questions_file", form.elements.questions_file.files[0]);
    showNotice("Preparing answers from the saved document…", "loading");
    const data = await postForm(`/api/v1/documents/${savedDocumentId}/questions`, questions);
    if (!Array.isArray(data.results) || !data.metadata || !data.request_id) {
      throw new Error("The server returned an unexpected response. Please try again.");
    }
    savedDocument.hidden = false;
    savedDocument.textContent = `Saved document for ${savedTenantId}: ${savedDocumentId}`;
    renderResults(data);
    latestResponse = data;
    notice.hidden = true;
  } catch (error) {
    section.hidden = true;
    showNotice(error instanceof TypeError ? "Could not reach the server. Check your connection and try again." : error.message, "error");
  } finally {
    submit.disabled = false;
    form.removeAttribute("aria-busy");
  }
});
documentInput.addEventListener("change", () => {
  if (fileKey(documentInput.files[0]) !== savedFileKey) {
    savedDocumentId = null;
    savedFileKey = null;
    savedDocument.hidden = true;
  }
});
tenantToken.addEventListener("input", () => {
  if (tenantToken.value !== savedToken) forgetSavedDocument();
});
download.addEventListener("click", () => {
  if (!latestResponse) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(latestResponse, null, 2)], { type: "application/json" }));
  const anchor = element("a");
  anchor.href = url;
  anchor.download = "document-qa-results.json";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
