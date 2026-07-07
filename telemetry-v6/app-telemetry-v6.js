const MODEL_SIZE = 416;
const DEFAULT_CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const TFLITE_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.3/dist/";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const inputCanvas = document.getElementById("inputCanvas");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const bestCandidateEl = document.getElementById("bestCandidate");
const a320ScoreEl = document.getElementById("a320Score");
const a320Vs737El = document.getElementById("a320Vs737");
const topClassesEl = document.getElementById("topClasses");
const confidenceThresholdInput = document.getElementById("confidenceThreshold");
const thresholdValueEl = document.getElementById("thresholdValue");
const captureButton = document.getElementById("capture");
const toggleButton = document.getElementById("toggleCamera");
const loadModelButton = document.getElementById("loadModel");
const autoScanButton = document.getElementById("autoScan");
const copyDebugButton = document.getElementById("copyDebug");
const copyReportButton = document.getElementById("copyReport");
const resetStatsButton = document.getElementById("resetStats");
const debugEl = document.getElementById("debug");
const telemetryEls = {
  lastScanMs: document.getElementById("lastScanMs"),
  avgScanMs: document.getElementById("avgScanMs"),
  throughput: document.getElementById("throughput"),
  runCount: document.getElementById("runCount"),
  modelLoadMs: document.getElementById("modelLoadMs"),
  modelSize: document.getElementById("modelSize"),
  cameraInfo: document.getElementById("cameraInfo"),
  batteryInfo: document.getElementById("batteryInfo"),
  memoryInfo: document.getElementById("memoryInfo"),
  networkInfo: document.getElementById("networkInfo"),
  deviceInfo: document.getElementById("deviceInfo"),
};

let facingMode = "environment";
let model = null;
let labels = [];
let letterbox = { scale: 1, padX: 0, padY: 0, sourceWidth: 1, sourceHeight: 1 };
let autoScanTimer = null;
let loadingModel = false;
let battery = null;
let confidenceThreshold = DEFAULT_CONF_THRESHOLD;

const telemetry = {
  build: "telemetry-v6-2026-07-07",
  startedAt: new Date().toISOString(),
  labelsLoadMs: null,
  runtimeWaitMs: null,
  modelHeadMs: null,
  modelLoadMs: null,
  modelBytes: null,
  modelLoadPath: null,
  runs: 0,
  successfulRuns: 0,
  failedRuns: 0,
  lastScanMs: null,
  avgScanMs: null,
  minScanMs: null,
  maxScanMs: null,
  lastPreprocessMs: null,
  lastTensorMs: null,
  lastPredictDecodeMs: null,
  lastDrawMs: null,
  lastDetections: 0,
  lastLabel: "-",
  lastConfidence: null,
  lastBestCandidate: null,
  lastA320Score: null,
  lastWatchScores: {},
  lastTopClasses: [],
  confidenceThreshold,
  firstBattery: null,
  latestBattery: null,
  camera: null,
};

captureButton.disabled = true;
autoScanButton.disabled = true;

function setStatus(message) {
  statusEl.textContent = message;
  debugEl.textContent = message;
}

function appendDebug(message) {
  debugEl.textContent = `${debugEl.textContent}\n${message}`.trim();
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(0)} ms` : "-";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "-";
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

function formatBattery(snapshot) {
  if (!snapshot) {
    return "not exposed";
  }
  const level = `${(snapshot.level * 100).toFixed(0)}%`;
  const charging = snapshot.charging ? "charging" : "discharging";
  const rate = batteryDischargeRate();
  return rate ? `${level}, ${charging}, ${rate}` : `${level}, ${charging}`;
}

function batteryDischargeRate() {
  const first = telemetry.firstBattery;
  const latest = telemetry.latestBattery;
  if (!first || !latest || first.charging || latest.charging) {
    return "";
  }
  const hours = (latest.time - first.time) / 3600000;
  const drop = first.level - latest.level;
  if (hours <= 0 || drop <= 0) {
    return "";
  }
  return `${(drop * 100 / hours).toFixed(1)}%/h`;
}

function getMemoryInfo() {
  const memory = performance.memory;
  const tfMemory = window.tf && typeof tf.memory === "function" ? tf.memory() : null;
  const parts = [];
  if (memory) {
    parts.push(`${formatBytes(memory.usedJSHeapSize)} JS`);
  }
  if (tfMemory) {
    parts.push(`${tfMemory.numTensors} tensors`);
  }
  return parts.length ? parts.join(", ") : "not exposed";
}

function getNetworkInfo() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!connection) {
    return "not exposed";
  }
  const parts = [];
  if (connection.effectiveType) {
    parts.push(connection.effectiveType);
  }
  if (Number.isFinite(connection.downlink)) {
    parts.push(`${connection.downlink} Mbps`);
  }
  if (Number.isFinite(connection.rtt)) {
    parts.push(`${connection.rtt} ms RTT`);
  }
  if (connection.saveData) {
    parts.push("save-data");
  }
  return parts.length ? parts.join(", ") : "available";
}

function getDeviceInfo() {
  const parts = [];
  if (navigator.hardwareConcurrency) {
    parts.push(`${navigator.hardwareConcurrency} cores`);
  }
  if (navigator.deviceMemory) {
    parts.push(`${navigator.deviceMemory} GB RAM hint`);
  }
  parts.push(`${screen.width}x${screen.height}@${window.devicePixelRatio || 1}`);
  return parts.join(", ");
}

function updateTelemetryDisplay() {
  telemetryEls.lastScanMs.textContent = formatMs(telemetry.lastScanMs);
  telemetryEls.avgScanMs.textContent = formatMs(telemetry.avgScanMs);
  telemetryEls.throughput.textContent = telemetry.avgScanMs ? `${(1000 / telemetry.avgScanMs).toFixed(2)} fps` : "-";
  telemetryEls.runCount.textContent = `${telemetry.successfulRuns}/${telemetry.runs}`;
  telemetryEls.modelLoadMs.textContent = formatMs(telemetry.modelLoadMs);
  telemetryEls.modelSize.textContent = formatBytes(telemetry.modelBytes);
  telemetryEls.cameraInfo.textContent = telemetry.camera || "-";
  telemetryEls.batteryInfo.textContent = formatBattery(telemetry.latestBattery);
  telemetryEls.memoryInfo.textContent = getMemoryInfo();
  telemetryEls.networkInfo.textContent = getNetworkInfo();
  telemetryEls.deviceInfo.textContent = getDeviceInfo();
}

function formatClassScore(item) {
  if (!item) {
    return "-";
  }
  return `${item.label} ${(item.score * 100).toFixed(1)}%`;
}

function updateDiagnosticDisplay() {
  bestCandidateEl.textContent = formatClassScore(telemetry.lastBestCandidate);
  a320ScoreEl.textContent = Number.isFinite(telemetry.lastA320Score)
    ? `${(telemetry.lastA320Score * 100).toFixed(1)}%`
    : "-";
  const watch = telemetry.lastWatchScores || {};
  const a320 = Number.isFinite(watch.A320) ? `${(watch.A320 * 100).toFixed(1)}%` : "-";
  const b737200 = Number.isFinite(watch["737-200"]) ? `${(watch["737-200"] * 100).toFixed(1)}%` : "-";
  a320Vs737El.textContent = `A320 ${a320} | 737-200 ${b737200}`;
  topClassesEl.textContent = telemetry.lastTopClasses.length
    ? telemetry.lastTopClasses.map(formatClassScore).join(" | ")
    : "-";
  thresholdValueEl.textContent = `${Math.round(confidenceThreshold * 100)}%`;
}

function updateScanStats(totalMs, timings, analysis) {
  const detections = analysis.detections;
  telemetry.runs += 1;
  telemetry.successfulRuns += 1;
  telemetry.lastScanMs = totalMs;
  telemetry.avgScanMs = telemetry.avgScanMs === null
    ? totalMs
    : ((telemetry.avgScanMs * (telemetry.successfulRuns - 1)) + totalMs) / telemetry.successfulRuns;
  telemetry.minScanMs = telemetry.minScanMs === null ? totalMs : Math.min(telemetry.minScanMs, totalMs);
  telemetry.maxScanMs = telemetry.maxScanMs === null ? totalMs : Math.max(telemetry.maxScanMs, totalMs);
  telemetry.lastPreprocessMs = timings.preprocessMs;
  telemetry.lastTensorMs = timings.tensorMs;
  telemetry.lastPredictDecodeMs = timings.predictDecodeMs;
  telemetry.lastDrawMs = timings.drawMs;
  telemetry.lastDetections = detections.length;
  telemetry.lastLabel = detections[0]?.label || "No aircraft detected";
  telemetry.lastConfidence = detections[0]?.score ?? null;
  telemetry.lastBestCandidate = analysis.bestCandidate;
  telemetry.lastA320Score = analysis.a320Score;
  telemetry.lastWatchScores = analysis.watchScores;
  telemetry.lastTopClasses = analysis.topClasses;
  telemetry.confidenceThreshold = confidenceThreshold;
  updateDiagnosticDisplay();
  updateTelemetryDisplay();
}

function markFailedRun() {
  telemetry.runs += 1;
  telemetry.failedRuns += 1;
  updateTelemetryDisplay();
}

function batterySnapshot(source) {
  if (!source) {
    return null;
  }
  return {
    level: source.level,
    charging: source.charging,
    chargingTime: source.chargingTime,
    dischargingTime: source.dischargingTime,
    time: Date.now(),
  };
}

async function initBatteryTelemetry() {
  if (!navigator.getBattery) {
    updateTelemetryDisplay();
    return;
  }
  try {
    battery = await navigator.getBattery();
    telemetry.firstBattery = batterySnapshot(battery);
    telemetry.latestBattery = telemetry.firstBattery;
    const updateBattery = () => {
      telemetry.latestBattery = batterySnapshot(battery);
      updateTelemetryDisplay();
    };
    battery.addEventListener("levelchange", updateBattery);
    battery.addEventListener("chargingchange", updateBattery);
    updateBattery();
  } catch (error) {
    appendDebug(`battery telemetry unavailable: ${error.message || error}`);
  }
}

function buildReport() {
  const report = {
    model: "GOATSOFAI unified 416 baseline TFLite",
    measured_accuracy: "3019/3333 = 90.58%",
    build: telemetry.build,
    timestamp: new Date().toISOString(),
    browser: navigator.userAgent,
    device: getDeviceInfo(),
    camera: telemetry.camera,
    network: getNetworkInfo(),
    battery: formatBattery(telemetry.latestBattery),
    memory: getMemoryInfo(),
    model_size: formatBytes(telemetry.modelBytes),
    model_load_ms: telemetry.modelLoadMs,
    model_head_ms: telemetry.modelHeadMs,
    labels_load_ms: telemetry.labelsLoadMs,
    runtime_wait_ms: telemetry.runtimeWaitMs,
    model_load_path: telemetry.modelLoadPath,
    runs: telemetry.runs,
    successful_runs: telemetry.successfulRuns,
    failed_runs: telemetry.failedRuns,
    last_scan_ms: telemetry.lastScanMs,
    average_scan_ms: telemetry.avgScanMs,
    min_scan_ms: telemetry.minScanMs,
    max_scan_ms: telemetry.maxScanMs,
    estimated_fps_from_average: telemetry.avgScanMs ? 1000 / telemetry.avgScanMs : null,
    last_preprocess_ms: telemetry.lastPreprocessMs,
    last_tensor_ms: telemetry.lastTensorMs,
    last_predict_decode_ms: telemetry.lastPredictDecodeMs,
    last_draw_ms: telemetry.lastDrawMs,
    last_result: telemetry.lastLabel,
    last_confidence: telemetry.lastConfidence,
    confidence_threshold: telemetry.confidenceThreshold,
    best_raw_candidate: telemetry.lastBestCandidate,
    a320_raw_score: telemetry.lastA320Score,
    watched_raw_scores: telemetry.lastWatchScores,
    top_raw_classes: telemetry.lastTopClasses,
    note: "Browser APIs do not expose real CPU/GPU wattage. Battery rate is only shown when the browser exposes Battery Status and the battery level changes while unplugged.",
  };
  return JSON.stringify(report, null, 2);
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
  const response = await fetch("../labels.txt?v=6", { cache: "no-store" });
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
    const loadStarted = performance.now();
    appendDebug(`browser: ${navigator.userAgent}`);
    appendDebug(`device: ${getDeviceInfo()}`);
    appendDebug(`network: ${getNetworkInfo()}`);
    appendDebug(`battery: ${formatBattery(telemetry.latestBattery)}`);
    appendDebug(`tf global: ${typeof window.tf}`);
    appendDebug(`tflite global: ${typeof window.tflite}`);
    const labelsStarted = performance.now();
    labels = await loadLabels();
    telemetry.labelsLoadMs = performance.now() - labelsStarted;
    appendDebug(`labels loaded: ${labels.length}`);
    const runtimeStarted = performance.now();
    const runtime = await waitForRuntime();
    telemetry.runtimeWaitMs = performance.now() - runtimeStarted;
    appendDebug(`runtime keys: ${Object.keys(runtime).slice(0, 12).join(", ")}`);
    if (typeof runtime.setWasmPath === "function") {
      runtime.setWasmPath(TFLITE_CDN);
      appendDebug(`wasm path: ${TFLITE_CDN}`);
    }
    setStatus("Downloading model.tflite...");
    const headStarted = performance.now();
    const modelResponse = await fetch("../model.tflite?v=6", { method: "HEAD", cache: "no-store" });
    telemetry.modelHeadMs = performance.now() - headStarted;
    const contentLength = Number(modelResponse.headers.get("content-length"));
    telemetry.modelBytes = Number.isFinite(contentLength) ? contentLength : null;
    appendDebug(`model HEAD: ${modelResponse.status} ${modelResponse.headers.get("content-length") || "unknown"} bytes`);
    try {
      const directStarted = performance.now();
      model = await runtime.loadTFLiteModel("../model.tflite?v=6");
      telemetry.modelLoadMs = performance.now() - directStarted;
      telemetry.modelLoadPath = "direct URL";
    } catch (directError) {
      appendDebug(`direct model URL load failed: ${errorText(directError)}`);
      appendDebug("trying blob URL fallback...");
      const blobStarted = performance.now();
      const modelBody = await fetch("../model.tflite?v=6", { cache: "no-store" });
      if (!modelBody.ok) {
        throw new Error(`model download failed: HTTP ${modelBody.status}`);
      }
      if (!telemetry.modelBytes) {
        telemetry.modelBytes = Number(modelBody.headers.get("content-length")) || null;
      }
      const blob = await modelBody.blob();
      const blobUrl = URL.createObjectURL(blob);
      model = await runtime.loadTFLiteModel(blobUrl);
      telemetry.modelLoadMs = performance.now() - blobStarted;
      telemetry.modelLoadPath = "blob fallback";
    }
    appendDebug(`model load: ${formatMs(telemetry.modelLoadMs)} via ${telemetry.modelLoadPath}`);
    appendDebug(`setup total: ${formatMs(performance.now() - loadStarted)}`);
    setStatus("Model ready. Press Run Scan.");
    captureButton.disabled = false;
    autoScanButton.disabled = false;
    updateTelemetryDisplay();
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
  await new Promise((resolve) => {
    if (video.readyState >= 2) {
      resolve();
      return;
    }
    video.onloadedmetadata = resolve;
  });
  const track = stream.getVideoTracks()[0];
  const settings = track?.getSettings ? track.getSettings() : {};
  const width = settings.width || video.videoWidth;
  const height = settings.height || video.videoHeight;
  const frameRate = settings.frameRate ? ` ${settings.frameRate} fps` : "";
  telemetry.camera = width && height ? `${width}x${height}${frameRate}` : "active";
  updateTelemetryDisplay();
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

function analyzeOutput(output) {
  const data = output.dataSync();
  const channels = output.shape[1];
  const anchors = output.shape[2];
  const detections = [];
  const classScores = new Array(Math.max(0, channels - 4)).fill(0);
  let bestCandidate = null;

  for (let anchor = 0; anchor < anchors; anchor += 1) {
    const cx = data[anchor];
    const cy = data[anchors + anchor];
    const width = data[anchors * 2 + anchor];
    const height = data[anchors * 3 + anchor];

    let bestClass = -1;
    let bestScore = 0;
    for (let channel = 4; channel < channels; channel += 1) {
      const score = data[anchors * channel + anchor];
      const classId = channel - 4;
      if (score > classScores[classId]) {
        classScores[classId] = score;
      }
      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
    }

    const x1 = (cx - width / 2 - letterbox.padX) / letterbox.scale;
    const y1 = (cy - height / 2 - letterbox.padY) / letterbox.scale;
    const x2 = (cx + width / 2 - letterbox.padX) / letterbox.scale;
    const y2 = (cy + height / 2 - letterbox.padY) / letterbox.scale;

    const candidate = {
      classId: bestClass,
      label: labels[bestClass] || `class ${bestClass}`,
      score: bestScore,
      box: [
        Math.max(0, Math.min(letterbox.sourceWidth, x1)),
        Math.max(0, Math.min(letterbox.sourceHeight, y1)),
        Math.max(0, Math.min(letterbox.sourceWidth, x2)),
        Math.max(0, Math.min(letterbox.sourceHeight, y2)),
      ],
    };

    if (!bestCandidate || candidate.score > bestCandidate.score) {
      bestCandidate = candidate;
    }

    if (bestScore >= confidenceThreshold) {
      detections.push(candidate);
    }
  }

  const topClasses = classScores
    .map((score, classId) => ({
      classId,
      label: labels[classId] || `class ${classId}`,
      score,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  const a320Index = labels.indexOf("A320");
  const a320Score = a320Index >= 0 ? classScores[a320Index] : null;
  const watchScores = {};
  for (const name of ["A320", "737-200", "A319", "A321", "737-300"]) {
    const index = labels.indexOf(name);
    watchScores[name] = index >= 0 ? classScores[index] : null;
  }

  return {
    detections: nonMaxSuppression(detections.sort((a, b) => b.score - a.score)).slice(0, 5),
    bestCandidate,
    topClasses,
    a320Score,
    watchScores,
  };
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
  const timings = {
    preprocessMs: null,
    tensorMs: null,
    predictDecodeMs: null,
    drawMs: null,
  };
  let mark = performance.now();
  drawLetterboxedFrame();
  timings.preprocessMs = performance.now() - mark;
  mark = performance.now();
  const input = tensorFromCanvas();
  timings.tensorMs = performance.now() - mark;
  let tensor = null;
  try {
    mark = performance.now();
    const output = model.predict(input);
    tensor = Array.isArray(output) ? output[0] : output;
    const analysis = analyzeOutput(tensor);
    const detections = analysis.detections;
    timings.predictDecodeMs = performance.now() - mark;
    mark = performance.now();
    drawDetections(detections);
    timings.drawMs = performance.now() - mark;
    const elapsed = performance.now() - started;
    if (detections.length === 0) {
      labelEl.textContent = "No aircraft detected";
      confidenceEl.textContent = "-";
    } else {
      labelEl.textContent = detections[0].label;
      confidenceEl.textContent = `${(detections[0].score * 100).toFixed(1)}%`;
    }
    updateScanStats(elapsed, timings, analysis);
    setStatus(`Done in ${elapsed.toFixed(0)} ms`);
  } catch (error) {
    console.error(error);
    markFailedRun();
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

copyReportButton.addEventListener("click", async () => {
  const text = buildReport();
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Report copied.");
  } catch (_) {
    window.prompt("Copy report", text);
  }
});

resetStatsButton.addEventListener("click", () => {
  telemetry.runs = 0;
  telemetry.successfulRuns = 0;
  telemetry.failedRuns = 0;
  telemetry.lastScanMs = null;
  telemetry.avgScanMs = null;
  telemetry.minScanMs = null;
  telemetry.maxScanMs = null;
  telemetry.lastPreprocessMs = null;
  telemetry.lastTensorMs = null;
  telemetry.lastPredictDecodeMs = null;
  telemetry.lastDrawMs = null;
  telemetry.lastDetections = 0;
  telemetry.lastLabel = "-";
  telemetry.lastConfidence = null;
  telemetry.lastBestCandidate = null;
  telemetry.lastA320Score = null;
  telemetry.lastWatchScores = {};
  telemetry.lastTopClasses = [];
  updateDiagnosticDisplay();
  updateTelemetryDisplay();
  setStatus("Stats reset.");
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
  updateTelemetryDisplay();
  initBatteryTelemetry();
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (connection && typeof connection.addEventListener === "function") {
    connection.addEventListener("change", updateTelemetryDisplay);
  }
  setInterval(updateTelemetryDisplay, 5000);
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus(`Camera setup failed: ${error.message || error}`);
  }
});
