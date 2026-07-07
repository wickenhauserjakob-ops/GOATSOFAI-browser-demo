const MODEL_SIZE = 416;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const TFLITE_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.3/dist/";
const BUILD_ID = "debug-v4-2026-07-07";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const inputCanvas = document.getElementById("inputCanvas");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const captureButton = document.getElementById("capture");
const toggleButton = document.getElementById("toggleCamera");
const loadModelButton = document.getElementById("loadModel");
const autoScanButton = document.getElementById("autoScan");
const copyDebugButton = document.getElementById("copyDebug");
const debugEl = document.getElementById("debug");

let facingMode = "environment";
let model = null;
let labels = [];
let letterbox = { scale: 1, padX: 0, padY: 0, sourceWidth: 1, sourceHeight: 1 };
let autoScanTimer = null;
let loadingModel = false;

captureButton.disabled = true;
autoScanButton.disabled = true;

function setStatus(message) {
  statusEl.textContent = message;
  debugEl.textContent = message;
}

function appendDebug(message) {
  debugEl.textContent = `${debugEl.textContent}\n${message}`.trim();
}

function errorText(error) {
  if (!error) {
    return "unknown error";
  }
  if (error.stack) {
    return error.stack;
  }
  if (error.message) {
    return error.message;
  }
  return String(error);
}

window.addEventListener("error", (event) => {
  appendDebug(`window error: ${event.message || "unknown"} at ${event.filename || ""}:${event.lineno || ""}`);
});

window.addEventListener("unhandledrejection", (event) => {
  appendDebug(`unhandled rejection: ${errorText(event.reason)}`);
});

async function loadLabels() {
  const response = await fetch("../labels.txt?v=4", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`labels.txt failed: HTTP ${response.status}`);
  }
  const text = await response.text();
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function waitForRuntime() {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = setInterval(() => {
      if (window.tflite && typeof window.tflite.loadTFLiteModel === "function") {
        clearInterval(timer);
        resolve(window.tflite);
        return;
      }
      if (performance.now() - started > 10000) {
        clearInterval(timer);
        reject(new Error("TFLite runtime did not load. Reload once or try Chrome."));
      }
    }, 100);
  });
}

async function loadModel() {
  if (model || loadingModel) {
    return;
  }
  loadingModel = true;
  loadModelButton.disabled = true;
  setStatus("Loading labels and TFLite runtime...");
  try {
    appendDebug(`build: ${BUILD_ID}`);
    appendDebug(`browser: ${navigator.userAgent}`);
    appendDebug(`tf global: ${typeof window.tf}`);
    appendDebug(`tflite global: ${typeof window.tflite}`);
    labels = await loadLabels();
    appendDebug(`labels loaded: ${labels.length}`);
    const runtime = await waitForRuntime();
    appendDebug(`runtime keys: ${Object.keys(runtime).slice(0, 12).join(", ")}`);
    if (typeof runtime.setWasmPath === "function") {
      runtime.setWasmPath(TFLITE_CDN);
      appendDebug(`wasm path: ${TFLITE_CDN}`);
    }
    setStatus("Downloading model.tflite...");
    const modelResponse = await fetch("../model.tflite?v=4", { method: "HEAD", cache: "no-store" });
    appendDebug(`model HEAD: ${modelResponse.status} ${modelResponse.headers.get("content-length") || "unknown"} bytes`);
    try {
      model = await runtime.loadTFLiteModel("../model.tflite?v=4");
    } catch (directError) {
      appendDebug(`direct model URL load failed: ${errorText(directError)}`);
      appendDebug("trying blob URL fallback...");
      const modelBody = await fetch("../model.tflite?v=4", { cache: "no-store" });
      if (!modelBody.ok) {
        throw new Error(`model download failed: HTTP ${modelBody.status}`);
      }
      const blob = await modelBody.blob();
      const blobUrl = URL.createObjectURL(blob);
      model = await runtime.loadTFLiteModel(blobUrl);
    }
    setStatus("Model ready. Press Run Scan.");
    captureButton.disabled = false;
    autoScanButton.disabled = false;
  } finally {
    loadingModel = false;
    loadModelButton.disabled = Boolean(model);
  }
}

async function startCamera() {
  if (video.srcObject) {
    for (const track of video.srcObject.getTracks()) {
      track.stop();
    }
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode },
    audio: false,
  });
  video.srcObject = stream;
  setStatus(model ? "Camera ready. Press Run Scan." : "Camera ready. Press Load Model.");
}

function drawLetterboxedFrame() {
  const ctx = inputCanvas.getContext("2d", { willReadFrequently: true });
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const scale = Math.min(MODEL_SIZE / sourceWidth, MODEL_SIZE / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const padX = (MODEL_SIZE - drawWidth) / 2;
  const padY = (MODEL_SIZE - drawHeight) / 2;

  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(video, padX, padY, drawWidth, drawHeight);
  letterbox = { scale, padX, padY, sourceWidth, sourceHeight };
}

function tensorFromCanvas() {
  return tf.tidy(() => tf.browser.fromPixels(inputCanvas).toFloat().div(255).expandDims(0));
}

function toDetections(output) {
  const data = output.dataSync();
  const channels = output.shape[1];
  const anchors = output.shape[2];
  const detections = [];

  for (let anchor = 0; anchor < anchors; anchor += 1) {
    const cx = data[anchor];
    const cy = data[anchors + anchor];
    const width = data[anchors * 2 + anchor];
    const height = data[anchors * 3 + anchor];

    let bestClass = -1;
    let bestScore = 0;
    for (let channel = 4; channel < channels; channel += 1) {
      const score = data[anchors * channel + anchor];
      if (score > bestScore) {
        bestScore = score;
        bestClass = channel - 4;
      }
    }

    if (bestScore < CONF_THRESHOLD) {
      continue;
    }

    const x1 = (cx - width / 2 - letterbox.padX) / letterbox.scale;
    const y1 = (cy - height / 2 - letterbox.padY) / letterbox.scale;
    const x2 = (cx + width / 2 - letterbox.padX) / letterbox.scale;
    const y2 = (cy + height / 2 - letterbox.padY) / letterbox.scale;

    detections.push({
      classId: bestClass,
      label: labels[bestClass] || `class ${bestClass}`,
      score: bestScore,
      box: [
        Math.max(0, Math.min(letterbox.sourceWidth, x1)),
        Math.max(0, Math.min(letterbox.sourceHeight, y1)),
        Math.max(0, Math.min(letterbox.sourceWidth, x2)),
        Math.max(0, Math.min(letterbox.sourceHeight, y2)),
      ],
    });
  }
  return nonMaxSuppression(detections.sort((a, b) => b.score - a.score)).slice(0, 5);
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(1, areaA + areaB - intersection);
}

function nonMaxSuppression(detections) {
  const kept = [];
  for (const detection of detections) {
    if (kept.every((other) => iou(detection.box, other.box) < IOU_THRESHOLD)) {
      kept.push(detection);
    }
  }
  return kept;
}

function drawDetections(detections) {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const scaleX = overlay.width / letterbox.sourceWidth;
  const scaleY = overlay.height / letterbox.sourceHeight;
  ctx.lineWidth = 3;
  ctx.font = "16px Arial";

  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.box;
    const left = x1 * scaleX;
    const top = y1 * scaleY;
    const width = (x2 - x1) * scaleX;
    const height = (y2 - y1) * scaleY;
    const text = `${detection.label} ${(detection.score * 100).toFixed(1)}%`;

    ctx.strokeStyle = "#10b981";
    ctx.fillStyle = "#10b981";
    ctx.strokeRect(left, top, width, height);
    const textWidth = ctx.measureText(text).width + 12;
    ctx.fillRect(left, Math.max(0, top - 24), textWidth, 24);
    ctx.fillStyle = "#061512";
    ctx.fillText(text, left + 6, Math.max(16, top - 7));
  }
}

async function runInference() {
  if (!model) {
    setStatus("Model is not loaded yet. Press Load Model first.");
    return;
  }
  if (video.readyState < 2) {
    setStatus("Camera is not ready yet.");
    return;
  }

  setStatus("Running scan...");
  const started = performance.now();
  drawLetterboxedFrame();
  const input = tensorFromCanvas();
  let tensor = null;
  try {
    const output = model.predict(input);
    tensor = Array.isArray(output) ? output[0] : output;
    const detections = toDetections(tensor);
    drawDetections(detections);
    const elapsed = performance.now() - started;
    if (detections.length === 0) {
      labelEl.textContent = "No aircraft detected";
      confidenceEl.textContent = "-";
    } else {
      labelEl.textContent = detections[0].label;
      confidenceEl.textContent = `${(detections[0].score * 100).toFixed(1)}%`;
    }
    setStatus(`Done in ${elapsed.toFixed(0)} ms`);
  } catch (error) {
    console.error(error);
    setStatus(`Scan failed: ${error.message || error}`);
  } finally {
    input.dispose();
    if (tensor && typeof tensor.dispose === "function") {
      tensor.dispose();
    }
  }
}

toggleButton.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
});

captureButton.addEventListener("click", runInference);

loadModelButton.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (error) {
    console.error(error);
    setStatus(`Model setup failed: ${error.message || error}`);
    appendDebug(errorText(error));
  }
});

copyDebugButton.addEventListener("click", async () => {
  const text = debugEl.textContent || "";
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Debug copied.");
  } catch (_) {
    window.prompt("Copy debug text", text);
  }
});

autoScanButton.addEventListener("click", () => {
  if (autoScanTimer) {
    clearInterval(autoScanTimer);
    autoScanTimer = null;
    autoScanButton.textContent = "Auto Scan";
    setStatus("Auto scan stopped.");
    return;
  }
  runInference();
  autoScanTimer = setInterval(runInference, 1200);
  autoScanButton.textContent = "Stop Auto";
});

window.addEventListener("load", async () => {
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus(`Camera setup failed: ${error.message || error}`);
  }
});
