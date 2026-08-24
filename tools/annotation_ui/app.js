"use strict";

const STORAGE_KEY = "vcoco-v3-annotator-id";
const GUIDE_KEY = "vcoco-v3-guide-seen";
const FIELDS = ["posture", "visible_translation", "gait", "visibility"];
const SHORTCUTS = {
  "1": ["posture", "seated"],
  "2": ["posture", "upright"],
  "3": ["posture", "other"],
  "4": ["posture", "indeterminate"],
  q: ["visible_translation", "stationary"],
  w: ["visible_translation", "locomoting"],
  e: ["visible_translation", "transition"],
  r: ["visible_translation", "not_inferable"],
  a: ["gait", "walking"],
  s: ["gait", "running"],
  d: ["gait", "not_applicable"],
  f: ["gait", "indeterminate"],
  z: ["visibility", "clear"],
  x: ["visibility", "occluded"],
  c: ["visibility", "truncated"],
  v: ["visibility", "too_small"],
};

const elements = {
  app: document.querySelector("#app"),
  workspace: document.querySelector("#workspace"),
  loadingState: document.querySelector("#loading-state"),
  completeState: document.querySelector("#complete-state"),
  form: document.querySelector("#annotation-form"),
  formError: document.querySelector("#form-error"),
  contextImage: document.querySelector("#context-image"),
  personImage: document.querySelector("#person-image"),
  taskPosition: document.querySelector("#task-position"),
  progressLabel: document.querySelector("#progress-label"),
  progressFill: document.querySelector("#progress-fill"),
  annotatorLabel: document.querySelector("#annotator-label"),
  saveState: document.querySelector("#save-state"),
  notes: document.querySelector("#notes"),
  previousButton: document.querySelector("#previous-button"),
  skipButton: document.querySelector("#skip-button"),
  saveButton: document.querySelector("#save-button"),
  exportButton: document.querySelector("#export-button"),
  completeExportButton: document.querySelector("#complete-export-button"),
  reviewButton: document.querySelector("#review-button"),
  changeAnnotatorButton: document.querySelector("#change-annotator-button"),
  instructionsButton: document.querySelector("#instructions-button"),
  instructionsDialog: document.querySelector("#instructions-dialog"),
  closeInstructionsButton: document.querySelector("#close-instructions-button"),
  guideDoneButton: document.querySelector("#guide-done-button"),
  annotatorDialog: document.querySelector("#annotator-dialog"),
  annotatorForm: document.querySelector("#annotator-form"),
  annotatorInput: document.querySelector("#annotator-input"),
  annotatorError: document.querySelector("#annotator-error"),
  toast: document.querySelector("#toast"),
};

const session = {
  annotatorId: "",
  guideVersion: "v3-pilot-1",
  tasks: [],
  currentIndex: 0,
  saving: false,
};

let toastTimer;

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 2600);
}

function setError(element, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function setSaveState(text, kind = "") {
  elements.saveState.textContent = text;
  elements.saveState.className = `save-state ${kind}`.trim();
}

function isEditingText() {
  const tag = document.activeElement?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

function selectChoice(field, value) {
  const input = elements.form.querySelector(
    `input[name="${field}"][value="${value}"]`,
  );
  if (input && !input.disabled) {
    input.checked = true;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

function synchronizeGaitControls() {
  const motion = elements.form.elements.visible_translation.value;
  const gaitInputs = [...elements.form.querySelectorAll('input[name="gait"]')];
  const locomoting = motion === "locomoting";
  gaitInputs.forEach((input) => {
    input.disabled = !locomoting && ["walking", "running"].includes(input.value);
  });
  if (!locomoting && ["walking", "running"].includes(elements.form.elements.gait.value)) {
    selectChoice("gait", motion ? "not_applicable" : "");
  }
}

function completedCount() {
  return session.tasks.filter((task) => task.annotation).length;
}

function updateProgress() {
  const complete = completedCount();
  const total = session.tasks.length;
  const percentage = total ? (complete / total) * 100 : 0;
  elements.progressLabel.textContent = `${complete} of ${total} complete`;
  elements.progressFill.style.width = `${percentage}%`;
  elements.annotatorLabel.textContent = session.annotatorId
    ? `Annotator: ${session.annotatorId}`
    : "Not signed in";
}

function clearForm() {
  elements.form.reset();
  elements.notes.value = "";
  setError(elements.formError);
  synchronizeGaitControls();
}

function populateForm(annotation) {
  clearForm();
  if (!annotation) return;
  FIELDS.forEach((field) => selectChoice(field, annotation[field]));
  elements.notes.value = annotation.notes || "";
  synchronizeGaitControls();
}

function loadImage(element, source) {
  element.classList.remove("loaded");
  element.onload = () => element.classList.add("loaded");
  element.onerror = () => {
    element.classList.remove("loaded");
    setSaveState("Image failed to load", "error");
  };
  element.src = source;
}

function showTask(index) {
  if (!session.tasks.length) return;
  session.currentIndex = Math.max(0, Math.min(index, session.tasks.length - 1));
  const task = session.tasks[session.currentIndex];
  const cacheKey = task.annotation?.revision || 0;

  elements.loadingState.hidden = true;
  elements.completeState.hidden = true;
  elements.workspace.hidden = false;
  elements.taskPosition.textContent = `Example ${session.currentIndex + 1} of ${session.tasks.length}`;
  elements.previousButton.disabled = session.currentIndex === 0;
  populateForm(task.annotation);
  loadImage(elements.contextImage, `/media/${encodeURIComponent(task.task_id)}/context?v=${cacheKey}`);
  loadImage(elements.personImage, `/media/${encodeURIComponent(task.task_id)}/person?v=${cacheKey}`);
  setSaveState(task.annotation ? "Saved response" : "Not yet saved");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function firstUnansweredIndex() {
  const index = session.tasks.findIndex((task) => !task.annotation);
  return index === -1 ? session.tasks.length : index;
}

function showComplete() {
  elements.loadingState.hidden = true;
  elements.workspace.hidden = true;
  elements.completeState.hidden = false;
  updateProgress();
}

function showNextUnanswered(afterIndex) {
  for (let index = afterIndex + 1; index < session.tasks.length; index += 1) {
    if (!session.tasks[index].annotation) {
      showTask(index);
      return;
    }
  }
  for (let index = 0; index <= afterIndex; index += 1) {
    if (!session.tasks[index].annotation) {
      showTask(index);
      return;
    }
  }
  showComplete();
}

async function loadSession(annotatorId) {
  elements.app.setAttribute("aria-busy", "true");
  elements.loadingState.hidden = false;
  elements.workspace.hidden = true;
  elements.completeState.hidden = true;
  setError(elements.annotatorError);

  const response = await fetch(`/api/state?annotator_id=${encodeURIComponent(annotatorId)}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to start the session");

  session.annotatorId = payload.annotator_id;
  session.guideVersion = payload.guide_version;
  session.tasks = payload.tasks;
  localStorage.setItem(STORAGE_KEY, session.annotatorId);
  elements.app.setAttribute("aria-busy", "false");
  updateProgress();

  const nextIndex = firstUnansweredIndex();
  if (nextIndex === session.tasks.length) showComplete();
  else showTask(nextIndex);

}

function showGuideOnce() {
  if (!localStorage.getItem(GUIDE_KEY)) elements.instructionsDialog.showModal();
}

function formPayload() {
  const values = Object.fromEntries(new FormData(elements.form).entries());
  return {
    task_id: session.tasks[session.currentIndex].task_id,
    annotator_id: session.annotatorId,
    posture: values.posture || "",
    visible_translation: values.visible_translation || "",
    gait: values.gait || "",
    visibility: values.visibility || "",
    notes: elements.notes.value.trim(),
    guide_version: session.guideVersion,
  };
}

function validatePayload(payload) {
  const missing = FIELDS.filter((field) => !payload[field]);
  if (missing.length) return "Complete all four questions before saving.";
  if (
    payload.visible_translation !== "locomoting" &&
    ["walking", "running"].includes(payload.gait)
  ) {
    return "Walking or running can only be selected when visible translation is locomoting.";
  }
  return "";
}

async function saveCurrent() {
  if (session.saving) return;
  const payload = formPayload();
  const validationError = validatePayload(payload);
  if (validationError) {
    setError(elements.formError, validationError);
    return;
  }

  session.saving = true;
  elements.saveButton.disabled = true;
  setError(elements.formError);
  setSaveState("Saving…", "saving");
  try {
    const response = await fetch("/api/annotation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The response could not be saved");
    session.tasks[session.currentIndex].annotation = result.saved;
    setSaveState("Saved", "saving");
    updateProgress();
    showNextUnanswered(session.currentIndex);
  } catch (error) {
    setError(elements.formError, error.message);
    setSaveState("Save failed", "error");
  } finally {
    session.saving = false;
    elements.saveButton.disabled = false;
  }
}

function navigate(delta) {
  if (!session.tasks.length || session.saving) return;
  showTask(session.currentIndex + delta);
}

function exportAnnotations() {
  if (!session.annotatorId) return;
  window.location.assign(`/api/export?annotator_id=${encodeURIComponent(session.annotatorId)}`);
}

elements.annotatorForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const annotatorId = elements.annotatorInput.value.trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(annotatorId)) {
    setError(
      elements.annotatorError,
      "Use letters, numbers, dots, underscores, or hyphens; begin with a letter or number.",
    );
    return;
  }
  try {
    await loadSession(annotatorId);
    elements.annotatorDialog.close();
    showGuideOnce();
  } catch (error) {
    setError(elements.annotatorError, error.message);
  }
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  saveCurrent();
});

elements.form.addEventListener("change", (event) => {
  if (event.target.name === "visible_translation") synchronizeGaitControls();
  setError(elements.formError);
});

elements.previousButton.addEventListener("click", () => navigate(-1));
elements.skipButton.addEventListener("click", () => showNextUnanswered(session.currentIndex));
elements.exportButton.addEventListener("click", exportAnnotations);
elements.completeExportButton.addEventListener("click", exportAnnotations);
elements.reviewButton.addEventListener("click", () => showTask(0));
elements.changeAnnotatorButton.addEventListener("click", () => {
  elements.annotatorInput.value = "";
  setError(elements.annotatorError);
  elements.annotatorDialog.showModal();
  elements.annotatorInput.focus();
});

function openGuide() {
  elements.instructionsDialog.showModal();
}

function closeGuide() {
  localStorage.setItem(GUIDE_KEY, "true");
  elements.instructionsDialog.close();
}

elements.instructionsButton.addEventListener("click", openGuide);
elements.closeInstructionsButton.addEventListener("click", closeGuide);
elements.guideDoneButton.addEventListener("click", closeGuide);

document.addEventListener("keydown", (event) => {
  if (elements.annotatorDialog.open || elements.instructionsDialog.open || isEditingText()) return;
  const shortcut = SHORTCUTS[event.key.toLowerCase()];
  if (shortcut) {
    event.preventDefault();
    selectChoice(...shortcut);
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    saveCurrent();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    navigate(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    navigate(1);
  }
});

window.addEventListener("beforeunload", (event) => {
  if (session.saving) {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function start() {
  const remembered = localStorage.getItem(STORAGE_KEY);
  if (!remembered) {
    elements.loadingState.hidden = true;
    elements.annotatorDialog.showModal();
    elements.annotatorInput.focus();
    return;
  }
  try {
    await loadSession(remembered);
    showGuideOnce();
  } catch (error) {
    elements.loadingState.hidden = true;
    setError(elements.annotatorError, error.message);
    elements.annotatorInput.value = remembered;
    elements.annotatorDialog.showModal();
  }
}

start();
