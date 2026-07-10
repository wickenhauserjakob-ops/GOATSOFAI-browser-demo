const TRACKER_SIZE = 960;
const VARIANT_SIZE = 416;
const TRACKER_CONF_THRESHOLD = 0.08;
const VARIANT_CONF_THRESHOLD = 0.20;
const IOU_THRESHOLD = 0.45;
const CROP_EXPAND = 2.5;
const MIN_CROP_SOURCE_PX = 416;
const VOTE_BURST_SIZE = 10;
const MAX_UPLOAD_VOTE_IMAGES = 10;
const VOTE_GAP_MS = 150;
const ASSET_VERSION = "v10robust416vote10upload10expand25";
const TFLITE_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.3/dist/";

const video = document.getElementById("video");
const uploadedImage = document.getElementById("uploadedImage");
const overlay = document.getElementById("overlay");
const inputCanvas = document.getElementById("inputCanvas");
const cropCanvas = document.getElementById("cropCanvas");
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
const imageUploadInput = document.getElementById("imageUpload");
const useCameraButton = document.getElementById("useCamera");
const autoScanButton = document.getElementById("autoScan");
const copyDebugButton = document.getElementById("copyDebug");
const copyReportButton = document.getElementById("copyReport");
const downloadLogButton = document.getElementById("downloadLog");
const resetStatsButton = document.getElementById("resetStats");
const debugEl = document.getElementById("debug");
const telemetryEls = {
  lastScanMs: document.getElementById("lastScanMs"),
  avgScanMs: document.getElementById("avgScanMs"),
  throughput: document.getElementById("throughput"),
  runCount: document.getElementById("runCount"),
  logCount: document.getElementById("logCount"),
  modelLoadMs: document.getElementById("modelLoadMs"),
  modelSize: document.getElementById("modelSize"),
  cameraInfo: document.getElementById("cameraInfo"),
  batteryInfo: document.getElementById("batteryInfo"),
  memoryInfo: document.getElementById("memoryInfo"),
  networkInfo: document.getElementById("networkInfo"),
  deviceInfo: document.getElementById("deviceInfo"),
};

let trackerModel = null;
let variantModel = null;
let labels = [];
let facingMode = "environment";
let cameraDevices = [];
let activeCameraDeviceId = null;
let cameraRestartingForRear = false;
let trackerLetterbox = { scale: 1, padX: 0, padY: 0, sourceWidth: 1, sourceHeight: 1 };
let cropLetterbox = { scale: 1, padX: 0, padY: 0, cropWidth: 1, cropHeight: 1 };
let inferenceRunning = false;
let voteBurstRunning = false;
let voteBurstCancel = false;
let autoScanRunning = false;
let loadingModel = false;
let sourceMode = "camera";
let uploadedImageUrl = null;
let uploadedImageMeta = null;
let uploadedBatch = [];
let uploadedBatchIndex = 0;
let scanLog = [];
let debugLog = [];
const LOG_STORAGE_KEY = "goatsofai-airport-tracker-robust-classifier-vote10-log-v1";
const MAX_LOG_ENTRIES = 500;

const telemetry = {
  build: "airport-pipeline-tracker-robust-classifier-vote10-upload10-expand25-v3-2026-07-10",
  startedAt: new Date().toISOString(),
  modelBytes: null,
  trackerLoadMs: null,
  variantLoadMs: null,
  labelsLoadMs: null,
  runtimeWaitMs: null,
  runs: 0,
  successfulRuns: 0,
  failedRuns: 0,
  avgScanMs: null,
  minScanMs: null,
  maxScanMs: null,
  lastScanMs: null,
  lastTrackerMs: null,
  lastCropMs: null,
  lastVariantMs: null,
  lastDrawMs: null,
  lastLabel: "-",
  lastConfidence: null,
  lastTrackerBox: null,
  lastTrackerScore: null,
  lastVariantBestCandidate: null,
  lastA320Score: null,
  lastWatchScores: {},
  lastTopClasses: [],
  lastVoteResult: null,
  uploadVoteCount: 0,
  camera: "",
};

captureButton.disabled = true;
autoScanButton.disabled = true;
confidenceThresholdInput.value = String(Math.round(VARIANT_CONF_THRESHOLD * 100));
thresholdValueEl.textContent = `${Math.round(VARIANT_CONF_THRESHOLD * 100)}%`;

function appendDebug(message) {
  debugLog.push(`${new Date().toISOString()} ${message}`);
  if (debugLog.length > 160) debugLog = debugLog.slice(-160);
  debugEl.textContent = debugLog.join("\n");
}

function setStatus(message) {
  statusEl.textContent = message;
  appendDebug(`status: ${message}`);
}

function errorText(error) {
  if (!error) return "unknown";
  return error.stack || error.message || String(error);
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(0)} ms` : "-";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "-";
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function getDeviceInfo() {
  const parts = [];
  if (navigator.hardwareConcurrency) parts.push(`${navigator.hardwareConcurrency} cores`);
  if (navigator.deviceMemory) parts.push(`${navigator.deviceMemory} GB RAM hint`);
  parts.push(`${screen.width}x${screen.height}@${window.devicePixelRatio || 1}`);
  return parts.join(", ");
}

function getNetworkInfo() {
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!connection) return "not exposed";
  return [connection.effectiveType, connection.downlink && `${connection.downlink} Mbps`, connection.rtt && `${connection.rtt} ms`]
    .filter(Boolean)
    .join(", ") || "available";
}

function getMemoryInfo() {
  const tfMemory = window.tf && typeof tf.memory === "function" ? tf.memory() : null;
  return tfMemory ? `${tfMemory.numTensors} tensors` : "not exposed";
}

function formatBox(box) {
  if (!box) return "-";
  const w = box[2] - box[0];
  const h = box[3] - box[1];
  return `${w.toFixed(0)}x${h.toFixed(0)} @ ${telemetry.lastTrackerScore ? (telemetry.lastTrackerScore * 100).toFixed(1) : "-"}%`;
}

function updateTelemetryDisplay() {
  telemetryEls.lastScanMs.textContent = formatMs(telemetry.lastScanMs);
  telemetryEls.avgScanMs.textContent = formatMs(telemetry.avgScanMs);
  telemetryEls.throughput.textContent = telemetry.avgScanMs ? `${(1000 / telemetry.avgScanMs).toFixed(2)} fps` : "-";
  telemetryEls.runCount.textContent = `${telemetry.successfulRuns}/${telemetry.runs}`;
  telemetryEls.logCount.textContent = String(scanLog.length);
  telemetryEls.modelLoadMs.textContent = formatMs((telemetry.trackerLoadMs || 0) + (telemetry.variantLoadMs || 0));
  telemetryEls.modelSize.textContent = formatBytes(telemetry.modelBytes);
  telemetryEls.cameraInfo.textContent = telemetry.camera || "-";
  telemetryEls.batteryInfo.textContent = "not exposed";
  telemetryEls.memoryInfo.textContent = getMemoryInfo();
  telemetryEls.networkInfo.textContent = getNetworkInfo();
  telemetryEls.deviceInfo.textContent = getDeviceInfo();
  bestCandidateEl.textContent = `track ${formatBox(telemetry.lastTrackerBox)}`;
  a320ScoreEl.textContent = Number.isFinite(telemetry.lastA320Score) ? `${(telemetry.lastA320Score * 100).toFixed(1)}%` : "-";
  const a320 = Number.isFinite(telemetry.lastWatchScores.A320) ? `${(telemetry.lastWatchScores.A320 * 100).toFixed(1)}%` : "-";
  const b737 = Number.isFinite(telemetry.lastWatchScores["737-200"]) ? `${(telemetry.lastWatchScores["737-200"] * 100).toFixed(1)}%` : "-";
  a320Vs737El.textContent = `A320 ${a320} | 737-200 ${b737}`;
  topClassesEl.textContent = telemetry.lastTopClasses.length
    ? telemetry.lastTopClasses.map((x) => `${x.label} ${(x.score * 100).toFixed(1)}%`).join(" | ")
    : "-";
}

function loadScanLog() {
  try {
    const saved = localStorage.getItem(LOG_STORAGE_KEY);
    scanLog = saved ? JSON.parse(saved) : [];
    if (!Array.isArray(scanLog)) scanLog = [];
  } catch (_) {
    scanLog = [];
  }
}

function saveScanLog() {
  if (scanLog.length > MAX_LOG_ENTRIES) scanLog = scanLog.slice(-MAX_LOG_ENTRIES);
  try {
    localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(scanLog));
  } catch (_) {
    scanLog = scanLog.slice(-Math.floor(MAX_LOG_ENTRIES / 2));
  }
}

function appendScanLog(entry) {
  scanLog.push({ index: scanLog.length + 1, timestamp: new Date().toISOString(), ...entry });
  saveScanLog();
  updateTelemetryDisplay();
}

function clearUploadedBatch() {
  for (const item of uploadedBatch) {
    if (item?.url) URL.revokeObjectURL(item.url);
  }
  uploadedBatch = [];
  uploadedBatchIndex = 0;
  uploadedImageUrl = null;
  uploadedImageMeta = null;
}

function selectedUploadCount() {
  return sourceMode === "upload" ? Math.max(1, uploadedBatch.length) : VOTE_BURST_SIZE;
}

function loadUploadedBatchImage(index) {
  const item = uploadedBatch[index];
  if (!item) return Promise.reject(new Error("No uploaded image at this index."));
  uploadedBatchIndex = index;
  uploadedImageUrl = item.url;
  uploadedImageMeta = item.meta;
  return new Promise((resolve, reject) => {
    uploadedImage.onload = () => {
      item.meta.width = uploadedImage.naturalWidth;
      item.meta.height = uploadedImage.naturalHeight;
      resolve(item.meta);
    };
    uploadedImage.onerror = () => reject(new Error(`Could not load uploaded image ${item.meta.name}`));
    uploadedImage.src = item.url;
  });
}

function getActiveSource() {
  if (sourceMode === "upload" && uploadedImage.complete && uploadedImage.naturalWidth > 0) {
    return { element: uploadedImage, width: uploadedImage.naturalWidth, height: uploadedImage.naturalHeight, kind: "upload" };
  }
  return { element: video, width: video.videoWidth, height: video.videoHeight, kind: "camera" };
}

function setPreviewMode(mode) {
  sourceMode = mode;
  if (mode === "upload") {
    video.style.display = "none";
    uploadedImage.style.display = "block";
    stopAutoScan("Auto scan stopped for uploaded image.");
    const count = uploadedBatch.length || 1;
    setStatus(trackerModel && variantModel ? `${count} image(s) loaded. Press Run Vote Scan.` : `${count} image(s) loaded. Press Load Model.`);
    return;
  }
  uploadedImage.style.display = "none";
  video.style.display = "block";
  setStatus(trackerModel && variantModel ? "Camera ready. Press Run Vote Scan." : "Camera ready. Press Load Model.");
}

function waitForRuntime() {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = setInterval(() => {
      if (window.tflite && typeof window.tflite.loadTFLiteModel === "function") {
        clearInterval(timer);
        resolve(window.tflite);
      } else if (performance.now() - started > 10000) {
        clearInterval(timer);
        reject(new Error("TFLite runtime not found"));
      }
    }, 50);
  });
}

async function loadLabels() {
  const response = await fetch(`labels.txt?${ASSET_VERSION}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`labels.txt failed: HTTP ${response.status}`);
  return (await response.text()).split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
}

async function loadTflite(runtime, path) {
  const head = await fetch(`${path}?${ASSET_VERSION}`, { method: "HEAD", cache: "no-store" });
  const bytes = Number(head.headers.get("content-length"));
  telemetry.modelBytes = (telemetry.modelBytes || 0) + (Number.isFinite(bytes) ? bytes : 0);
  appendDebug(`${path} HEAD ${head.status} ${head.headers.get("content-length") || "unknown"} bytes`);
  return runtime.loadTFLiteModel(`${path}?${ASSET_VERSION}`);
}

async function loadModel() {
  if ((trackerModel && variantModel) || loadingModel) return;
  loadingModel = true;
  loadModelButton.disabled = true;
  try {
    setStatus("Loading tracker and classifier...");
    appendDebug(`browser: ${navigator.userAgent}`);
    appendDebug(`device: ${getDeviceInfo()}`);
    const labelsStarted = performance.now();
    labels = await loadLabels();
    telemetry.labelsLoadMs = performance.now() - labelsStarted;
    const runtimeStarted = performance.now();
    const runtime = await waitForRuntime();
    telemetry.runtimeWaitMs = performance.now() - runtimeStarted;
    if (typeof runtime.setWasmPath === "function") runtime.setWasmPath(TFLITE_CDN);
    let mark = performance.now();
    trackerModel = await loadTflite(runtime, "tracker.tflite");
    telemetry.trackerLoadMs = performance.now() - mark;
    mark = performance.now();
    variantModel = await loadTflite(runtime, "classifier.tflite");
    telemetry.variantLoadMs = performance.now() - mark;
    setStatus("Pipeline ready. Press Run Vote Scan.");
    captureButton.disabled = false;
    autoScanButton.disabled = false;
    updateTelemetryDisplay();
  } finally {
    loadingModel = false;
    loadModelButton.disabled = Boolean(trackerModel && variantModel);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera API not available.");
    return;
  }
  try {
    if (video.srcObject) video.srcObject.getTracks().forEach((track) => track.stop());
    const stream = await openPreferredCamera();
    video.srcObject = stream;
    await video.play();
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings?.() || {};
    activeCameraDeviceId = settings.deviceId || activeCameraDeviceId;
    await refreshCameraDevices();
    const rearDevice = findCameraDevice("environment");
    const activeDevice = cameraDevices.find((device) => device.deviceId === activeCameraDeviceId);
    if (
      facingMode === "environment" &&
      rearDevice &&
      activeDevice &&
      rearDevice.deviceId !== activeDevice.deviceId &&
      !isRearCameraLabel(activeDevice.label) &&
      !cameraRestartingForRear
    ) {
      cameraRestartingForRear = true;
      activeCameraDeviceId = rearDevice.deviceId;
      setStatus("Switching to rear camera...");
      await startCamera();
      cameraRestartingForRear = false;
      return;
    }
    cameraRestartingForRear = false;
    const label = activeDevice?.label ? ` ${activeDevice.label}` : "";
    const mode = settings.facingMode ? ` ${settings.facingMode}` : "";
    telemetry.camera = `${video.videoWidth || settings.width}x${video.videoHeight || settings.height} ${settings.frameRate || ""} fps${mode}${label}`.trim();
    setPreviewMode("camera");
    updateTelemetryDisplay();
  } catch (error) {
    cameraRestartingForRear = false;
    setStatus(`Camera failed: ${error.message || error}`);
    appendDebug(errorText(error));
  }
}

async function refreshCameraDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    cameraDevices = devices.filter((device) => device.kind === "videoinput");
    if (!activeCameraDeviceId && cameraDevices.length) {
      activeCameraDeviceId = cameraDevices[0].deviceId;
    }
    toggleButton.textContent = cameraDevices.length > 1 ? "Switch Camera" : "Switch Camera";
    appendDebug(`cameras: ${cameraDevices.map((device) => device.label || device.deviceId || "camera").join(" | ") || "not labelled"}`);
  } catch (error) {
    appendDebug(`camera enumeration failed: ${errorText(error)}`);
  }
}

function isRearCameraLabel(label) {
  return /back|rear|environment|world|rueck|rück|hinten|kamera 0|camera 0/i.test(label || "");
}

function isFrontCameraLabel(label) {
  return /front|user|face|selfie|facetime|true depth|truedepth|vorne/i.test(label || "");
}

function findCameraDevice(mode) {
  if (!cameraDevices.length) return null;
  const matcher = mode === "environment" ? isRearCameraLabel : isFrontCameraLabel;
  return cameraDevices.find((device) => matcher(device.label)) || null;
}

function buildCameraConstraints(deviceId = null, mode = facingMode, exactFacing = false) {
  const videoConstraints = {
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  };
  if (deviceId) {
    videoConstraints.deviceId = { exact: deviceId };
  } else {
    videoConstraints.facingMode = exactFacing ? { exact: mode } : { ideal: mode };
  }
  return { video: videoConstraints, audio: false };
}

async function openPreferredCamera() {
  await refreshCameraDevices();
  const labelledDevice = findCameraDevice(facingMode);
  if (labelledDevice) activeCameraDeviceId = labelledDevice.deviceId;
  const attempts = [];
  if (activeCameraDeviceId) attempts.push(buildCameraConstraints(activeCameraDeviceId));
  attempts.push(buildCameraConstraints(null, facingMode, true));
  attempts.push(buildCameraConstraints(null, facingMode, false));
  attempts.push({ video: true, audio: false });
  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      appendDebug(`camera attempt failed: ${errorText(error)}`);
    }
  }
  throw lastError || new Error("No camera stream available.");
}

function selectNextCameraDevice() {
  if (cameraDevices.length < 2) {
    facingMode = facingMode === "environment" ? "user" : "environment";
    activeCameraDeviceId = null;
    return;
  }
  const currentIndex = cameraDevices.findIndex((device) => device.deviceId === activeCameraDeviceId);
  const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % cameraDevices.length : 0;
  const nextDevice = cameraDevices[nextIndex];
  activeCameraDeviceId = nextDevice.deviceId;
  facingMode = isRearCameraLabel(nextDevice.label) ? "environment" : "user";
}

function drawTrackerFrame() {
  if (inputCanvas.width !== TRACKER_SIZE || inputCanvas.height !== TRACKER_SIZE) {
    inputCanvas.width = TRACKER_SIZE;
    inputCanvas.height = TRACKER_SIZE;
  }
  const source = getActiveSource();
  const ctx = inputCanvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(TRACKER_SIZE / source.width, TRACKER_SIZE / source.height);
  const drawWidth = source.width * scale;
  const drawHeight = source.height * scale;
  const padX = (TRACKER_SIZE - drawWidth) / 2;
  const padY = (TRACKER_SIZE - drawHeight) / 2;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, TRACKER_SIZE, TRACKER_SIZE);
  ctx.drawImage(source.element, padX, padY, drawWidth, drawHeight);
  trackerLetterbox = { scale, padX, padY, sourceWidth: source.width, sourceHeight: source.height };
}

function tensorFromCanvas(canvas) {
  return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0));
}

function decodeDetections(output, classLabels, threshold, coordSize) {
  const data = output.dataSync();
  const channels = output.shape[1];
  const anchors = output.shape[2];
  const detections = [];
  const classCount = channels - 4;
  const classScores = new Array(Math.max(0, classCount)).fill(0);
  for (let a = 0; a < anchors; a += 1) {
    let bestClass = 0;
    let bestScore = 0;
    for (let c = 0; c < classCount; c += 1) {
      const score = data[(4 + c) * anchors + a];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
      if (score > classScores[c]) classScores[c] = score;
    }
    if (bestScore < threshold) continue;
    let cx = data[a];
    let cy = data[anchors + a];
    let w = data[2 * anchors + a];
    let h = data[3 * anchors + a];
    const maxCoord = Math.max(Math.abs(cx), Math.abs(cy), Math.abs(w), Math.abs(h));
    if (coordSize && maxCoord <= 4) {
      cx *= coordSize;
      cy *= coordSize;
      w *= coordSize;
      h *= coordSize;
    }
    detections.push({
      classId: bestClass,
      label: classLabels[bestClass] || `class ${bestClass}`,
      score: bestScore,
      box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
    });
  }
  return { detections: nonMaxSuppression(detections), classScores };
}

function iou(a, b) {
  const x1 = Math.max(a.box[0], b.box[0]);
  const y1 = Math.max(a.box[1], b.box[1]);
  const x2 = Math.min(a.box[2], b.box[2]);
  const y2 = Math.min(a.box[3], b.box[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.box[2] - a.box[0]) * Math.max(0, a.box[3] - a.box[1]);
  const areaB = Math.max(0, b.box[2] - b.box[0]) * Math.max(0, b.box[3] - b.box[1]);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nonMaxSuppression(detections) {
  const sorted = detections.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const det of sorted) {
    if (kept.every((other) => iou(det, other) < IOU_THRESHOLD)) kept.push(det);
    if (kept.length >= 10) break;
  }
  return kept;
}

function trackerBoxToSource(box) {
  const lb = trackerLetterbox;
  const x1 = (box[0] - lb.padX) / lb.scale;
  const y1 = (box[1] - lb.padY) / lb.scale;
  const x2 = (box[2] - lb.padX) / lb.scale;
  const y2 = (box[3] - lb.padY) / lb.scale;
  const left = Math.max(0, Math.min(lb.sourceWidth, Math.min(x1, x2)));
  const top = Math.max(0, Math.min(lb.sourceHeight, Math.min(y1, y2)));
  const right = Math.max(0, Math.min(lb.sourceWidth, Math.max(x1, x2)));
  const bottom = Math.max(0, Math.min(lb.sourceHeight, Math.max(y1, y2)));
  if (right - left < 8 || bottom - top < 8) {
    return [0, 0, lb.sourceWidth, lb.sourceHeight];
  }
  return [left, top, right, bottom];
}

function expandCrop(box, sourceWidth, sourceHeight) {
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  const halfWidth = Math.max(((box[2] - box[0]) * CROP_EXPAND) / 2, MIN_CROP_SOURCE_PX / 2);
  const halfHeight = Math.max(((box[3] - box[1]) * CROP_EXPAND) / 2, MIN_CROP_SOURCE_PX / 2);
  return [
    Math.max(0, cx - halfWidth),
    Math.max(0, cy - halfHeight),
    Math.min(sourceWidth, cx + halfWidth),
    Math.min(sourceHeight, cy + halfHeight),
  ];
}

function drawCropFromOriginal(cropBox) {
  if (cropCanvas.width !== VARIANT_SIZE || cropCanvas.height !== VARIANT_SIZE) {
    cropCanvas.width = VARIANT_SIZE;
    cropCanvas.height = VARIANT_SIZE;
  }
  const source = getActiveSource();
  const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
  const cropW = Math.max(1, cropBox[2] - cropBox[0]);
  const cropH = Math.max(1, cropBox[3] - cropBox[1]);
  const scale = Math.min(VARIANT_SIZE / cropW, VARIANT_SIZE / cropH);
  const drawW = cropW * scale;
  const drawH = cropH * scale;
  const padX = (VARIANT_SIZE - drawW) / 2;
  const padY = (VARIANT_SIZE - drawH) / 2;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, VARIANT_SIZE, VARIANT_SIZE);
  ctx.drawImage(source.element, cropBox[0], cropBox[1], cropW, cropH, padX, padY, drawW, drawH);
  cropLetterbox = { scale, padX, padY, cropWidth: cropW, cropHeight: cropH };
}

function analyzeVariant(output) {
  const rawScores = Array.from(output.dataSync());
  const needsSoftmax = rawScores.some((score) => score < 0 || score > 1)
    || Math.abs(rawScores.reduce((sum, score) => sum + score, 0) - 1) > 0.05;
  const classScores = needsSoftmax ? softmax(rawScores) : rawScores;
  const topClasses = classScores
    .map((score, classId) => ({ classId, label: labels[classId] || `class ${classId}`, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  const watchScores = {};
  for (const name of ["A320", "737-200", "A319", "A321", "737-300"]) {
    const index = labels.indexOf(name);
    watchScores[name] = index >= 0 ? classScores[index] : null;
  }
  const best = topClasses[0] || null;
  const detections = best && best.score >= VARIANT_CONF_THRESHOLD
    ? [{ classId: best.classId, label: best.label, score: best.score, box: [0, 0, VARIANT_SIZE, VARIANT_SIZE] }]
    : [];
  return {
    detections,
    topClasses,
    watchScores,
    a320Score: watchScores.A320,
    bestCandidate: best,
  };
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const exp = scores.map((score) => Math.exp(score - maxScore));
  const sum = exp.reduce((total, value) => total + value, 0) || 1;
  return exp.map((value) => value / sum);
}

function drawOverlay(trackerBox, variantDetection) {
  const source = getActiveSource();
  overlay.width = overlay.clientWidth * (window.devicePixelRatio || 1);
  overlay.height = overlay.clientHeight * (window.devicePixelRatio || 1);
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!trackerBox) return;
  const scaleX = overlay.width / source.width;
  const scaleY = overlay.height / source.height;
  const [x1, y1, x2, y2] = trackerBox;
  const left = x1 * scaleX;
  const top = y1 * scaleY;
  const width = (x2 - x1) * scaleX;
  const height = (y2 - y1) * scaleY;
  const text = variantDetection ? `${variantDetection.label} ${(variantDetection.score * 100).toFixed(1)}%` : "tracked aircraft";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#10b981";
  ctx.fillStyle = "#10b981";
  ctx.strokeRect(left, top, width, height);
  ctx.font = "16px system-ui, sans-serif";
  const tw = ctx.measureText(text).width + 12;
  ctx.fillRect(left, Math.max(0, top - 24), tw, 24);
  ctx.fillStyle = "#061512";
  ctx.fillText(text, left + 6, Math.max(16, top - 7));
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runInference() {
  if (inferenceRunning) {
    appendDebug("scan skipped: previous scan still running");
    return null;
  }
  if (!trackerModel || !variantModel) {
    setStatus("Pipeline is not loaded yet. Press Load Model first.");
    return null;
  }
  if (sourceMode === "camera" && video.readyState < 2) {
    setStatus("Camera is not ready yet.");
    return null;
  }
  inferenceRunning = true;
  const started = performance.now();
  const timings = {};
  let trackerInput = null;
  let trackerOutputTensor = null;
  let variantInput = null;
  let variantOutputTensor = null;
  let trackerBox = null;
  let cropBox = null;
  let variantAnalysis = null;
  try {
    setStatus("Running pipeline...");
    let mark = performance.now();
    drawTrackerFrame();
    trackerInput = tensorFromCanvas(inputCanvas);
    const trackerOutput = trackerModel.predict(trackerInput);
    trackerOutputTensor = Array.isArray(trackerOutput) ? trackerOutput[0] : trackerOutput;
    const trackerDecoded = decodeDetections(trackerOutputTensor, ["aircraft"], TRACKER_CONF_THRESHOLD, TRACKER_SIZE);
    timings.trackerMs = performance.now() - mark;
    if (trackerDecoded.detections.length) {
      const rawSourceBox = trackerBoxToSource(trackerDecoded.detections[0].box);
      trackerBox = rawSourceBox;
      telemetry.lastTrackerScore = trackerDecoded.detections[0].score;
      cropBox = expandCrop(rawSourceBox, trackerLetterbox.sourceWidth, trackerLetterbox.sourceHeight);
    } else {
      trackerBox = [0, 0, trackerLetterbox.sourceWidth, trackerLetterbox.sourceHeight];
      telemetry.lastTrackerScore = 0;
      cropBox = trackerBox;
    }
    mark = performance.now();
    drawCropFromOriginal(cropBox);
    variantInput = tensorFromCanvas(cropCanvas);
    timings.cropMs = performance.now() - mark;
    mark = performance.now();
    const variantOutput = variantModel.predict(variantInput);
    variantOutputTensor = Array.isArray(variantOutput) ? variantOutput[0] : variantOutput;
    variantAnalysis = analyzeVariant(variantOutputTensor);
    timings.variantMs = performance.now() - mark;
    mark = performance.now();
    drawOverlay(trackerBox, variantAnalysis.detections[0]);
    timings.drawMs = performance.now() - mark;
    const elapsed = performance.now() - started;
    const result = variantAnalysis.detections[0];
    labelEl.textContent = result ? result.label : "No aircraft detected";
    confidenceEl.textContent = result ? `${(result.score * 100).toFixed(1)}%` : "-";
    telemetry.runs += 1;
    telemetry.successfulRuns += 1;
    telemetry.lastScanMs = elapsed;
    telemetry.avgScanMs = telemetry.avgScanMs === null ? elapsed : ((telemetry.avgScanMs * (telemetry.successfulRuns - 1)) + elapsed) / telemetry.successfulRuns;
    telemetry.minScanMs = telemetry.minScanMs === null ? elapsed : Math.min(telemetry.minScanMs, elapsed);
    telemetry.maxScanMs = telemetry.maxScanMs === null ? elapsed : Math.max(telemetry.maxScanMs, elapsed);
    telemetry.lastTrackerMs = timings.trackerMs;
    telemetry.lastCropMs = timings.cropMs;
    telemetry.lastVariantMs = timings.variantMs;
    telemetry.lastDrawMs = timings.drawMs;
    telemetry.lastLabel = result ? result.label : "No aircraft detected";
    telemetry.lastConfidence = result?.score ?? null;
    telemetry.lastTrackerBox = trackerBox;
    telemetry.lastVariantBestCandidate = variantAnalysis.bestCandidate;
    telemetry.lastA320Score = variantAnalysis.a320Score;
    telemetry.lastWatchScores = variantAnalysis.watchScores;
    telemetry.lastTopClasses = variantAnalysis.topClasses;
    appendScanLog({
      failed: false,
      result: telemetry.lastLabel,
      confidence: telemetry.lastConfidence,
      scan_ms: elapsed,
      tracker_ms: timings.trackerMs,
      crop_ms: timings.cropMs,
      variant_ms: timings.variantMs,
      draw_ms: timings.drawMs,
      tracker_box: trackerBox,
      crop_box: cropBox,
      tracker_score: telemetry.lastTrackerScore,
      top_raw_classes: variantAnalysis.topClasses,
      watched_raw_scores: variantAnalysis.watchScores,
      source: sourceMode,
      source_description: getSourceDescription(),
      camera: telemetry.camera,
      memory: getMemoryInfo(),
      error: null,
    });
    setStatus(`Done in ${elapsed.toFixed(0)} ms`);
    return {
      failed: false,
      label: telemetry.lastLabel,
      confidence: telemetry.lastConfidence || 0,
      elapsed,
      trackerScore: telemetry.lastTrackerScore || 0,
      source: sourceMode,
      sourceDescription: getSourceDescription(),
    };
  } catch (error) {
    console.error(error);
    telemetry.runs += 1;
    telemetry.failedRuns += 1;
    appendDebug(`pipeline failed: ${errorText(error)}`);
    appendScanLog({ failed: true, result: "pipeline failed", error: errorText(error), camera: telemetry.camera, memory: getMemoryInfo() });
    setStatus(`Pipeline failed: ${error.message || error}`);
    return { failed: true, label: "pipeline failed", confidence: 0, error: errorText(error), source: sourceMode, sourceDescription: getSourceDescription() };
  } finally {
    if (trackerInput) trackerInput.dispose();
    if (trackerOutputTensor && typeof trackerOutputTensor.dispose === "function") trackerOutputTensor.dispose();
    if (variantInput) variantInput.dispose();
    if (variantOutputTensor && typeof variantOutputTensor.dispose === "function") variantOutputTensor.dispose();
    inferenceRunning = false;
    updateTelemetryDisplay();
  }
}

function chooseVoteResult(results) {
  const votes = new Map();
  for (const result of results) {
    if (!result || result.failed) continue;
    const label = result.label || "No aircraft detected";
    if (!votes.has(label)) votes.set(label, { label, count: 0, confidenceSum: 0 });
    const vote = votes.get(label);
    vote.count += 1;
    vote.confidenceSum += result.confidence || 0;
  }
  if (!votes.size) return null;
  return Array.from(votes.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return b.confidenceSum - a.confidenceSum;
  })[0];
}

async function runVoteBurst() {
  if (sourceMode === "upload" && uploadedBatch.length === 0) {
    setStatus("Upload 1-10 images first.");
    return;
  }
  voteBurstRunning = true;
  voteBurstCancel = false;
  const results = [];
  const sampleCount = sourceMode === "upload" ? Math.min(uploadedBatch.length, MAX_UPLOAD_VOTE_IMAGES) : VOTE_BURST_SIZE;
  telemetry.uploadVoteCount = sourceMode === "upload" ? sampleCount : 0;
  try {
    for (let i = 0; i < sampleCount; i += 1) {
      if (voteBurstCancel) break;
      if (sourceMode === "upload") {
        await loadUploadedBatchImage(i);
        setStatus(`Upload vote ${i + 1}/${sampleCount}: ${uploadedImageMeta?.name || "image"}...`);
      } else {
        setStatus(`Vote scan ${i + 1}/${sampleCount}...`);
      }
      const result = await runInference();
      if (result) {
        results.push({
          ...result,
          voteIndex: i + 1,
          uploadName: sourceMode === "upload" ? uploadedImageMeta?.name || null : null,
        });
      }
      if (sourceMode !== "upload" && i < sampleCount - 1 && !voteBurstCancel) await delay(VOTE_GAP_MS);
    }
    const winner = chooseVoteResult(results);
    telemetry.lastVoteResult = winner ? { ...winner, sampleCount: results.length } : null;
    if (winner) {
      labelEl.textContent = winner.label;
      confidenceEl.textContent = winner.confidenceSum > 0 ? `${((winner.confidenceSum / winner.count) * 100).toFixed(1)}% avg` : "-";
      setStatus(`Vote result: ${winner.label} (${winner.count}/${results.length})`);
      appendScanLog({
        failed: false,
        result: `vote: ${winner.label}`,
        confidence: winner.confidenceSum / Math.max(1, winner.count),
        vote_mode: sourceMode === "upload" ? "upload_batch" : "camera_burst",
        vote_size: sampleCount,
        vote_samples: results,
        vote_count: winner.count,
        upload_vote_count: sourceMode === "upload" ? sampleCount : null,
        source: sourceMode,
        source_description: getSourceDescription(),
        camera: telemetry.camera,
        memory: getMemoryInfo(),
        error: null,
      });
    } else if (!voteBurstCancel) {
      setStatus("Vote scan finished without a valid result.");
    }
  } finally {
    voteBurstRunning = false;
    voteBurstCancel = false;
    updateTelemetryDisplay();
  }
}

function stopAutoScan(message = "Auto scan stopped.") {
  if (autoScanRunning || voteBurstRunning) {
    autoScanRunning = false;
    voteBurstCancel = true;
    autoScanButton.textContent = "Auto Vote";
    captureButton.disabled = false;
    setStatus(message);
  }
}

async function runAutoScanLoop() {
  if (autoScanRunning) {
    stopAutoScan();
    return;
  }
  if (!trackerModel || !variantModel) {
    setStatus("Load model first.");
    return;
  }
  autoScanRunning = true;
  voteBurstCancel = false;
  autoScanButton.textContent = "Stop Auto";
  captureButton.disabled = true;
  try {
    while (autoScanRunning) {
      await runVoteBurst();
      if (!autoScanRunning) break;
      await delay(VOTE_GAP_MS);
    }
  } finally {
    autoScanRunning = false;
    voteBurstCancel = false;
    captureButton.disabled = false;
    autoScanButton.textContent = "Auto Vote";
    updateTelemetryDisplay();
  }
}

function getSourceDescription() {
  if (sourceMode === "upload" && uploadedBatch.length > 1) {
    const activeName = uploadedBatch[uploadedBatchIndex]?.meta?.name || "active image";
    return `upload batch ${uploadedBatch.length} image(s), active ${uploadedBatchIndex + 1}/${uploadedBatch.length}: ${activeName}`;
  }
  if (sourceMode === "upload" && uploadedImageMeta) {
    return `upload ${uploadedImageMeta.name} ${uploadedImageMeta.width}x${uploadedImageMeta.height}`;
  }
  return `camera ${telemetry.camera || "active"}`;
}

function buildReport() {
  return JSON.stringify({
    model: "GOATSOFAI airport tracker plus robust classifier browser pipeline with temporal voting",
    pipeline: "tracker 960 TFLite -> crop original frame -> robust classifier 416 TFLite -> camera 10-frame vote or upload 1-10 image vote",
    build: telemetry.build,
    timestamp: new Date().toISOString(),
    browser: navigator.userAgent,
    device: getDeviceInfo(),
    camera: telemetry.camera,
    model_size: formatBytes(telemetry.modelBytes),
    tracker_load_ms: telemetry.trackerLoadMs,
    variant_load_ms: telemetry.variantLoadMs,
    labels_load_ms: telemetry.labelsLoadMs,
    runtime_wait_ms: telemetry.runtimeWaitMs,
    runs: telemetry.runs,
    successful_runs: telemetry.successfulRuns,
    failed_runs: telemetry.failedRuns,
    average_scan_ms: telemetry.avgScanMs,
    min_scan_ms: telemetry.minScanMs,
    max_scan_ms: telemetry.maxScanMs,
    last_tracker_ms: telemetry.lastTrackerMs,
    last_crop_ms: telemetry.lastCropMs,
    last_variant_ms: telemetry.lastVariantMs,
    last_draw_ms: telemetry.lastDrawMs,
    last_result: telemetry.lastLabel,
    last_confidence: telemetry.lastConfidence,
    last_tracker_box: telemetry.lastTrackerBox,
    last_tracker_score: telemetry.lastTrackerScore,
    last_vote_result: telemetry.lastVoteResult,
    vote_burst_size: VOTE_BURST_SIZE,
    max_upload_vote_images: MAX_UPLOAD_VOTE_IMAGES,
    upload_vote_count: telemetry.uploadVoteCount,
    uploaded_images: uploadedBatch.map((item, index) => ({ index: index + 1, ...item.meta })),
    scan_log_entries: scanLog.length,
    scan_log: scanLog,
    debug_log: debugLog,
  }, null, 2);
}

function downloadReport() {
  const blob = new Blob([buildReport()], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `goatsofai-airport-pipeline-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

window.addEventListener("error", (event) => appendDebug(`window error: ${event.message || "unknown"}`));
window.addEventListener("unhandledrejection", (event) => appendDebug(`unhandled rejection: ${errorText(event.reason)}`));

toggleButton.addEventListener("click", async () => {
  await refreshCameraDevices();
  selectNextCameraDevice();
  setPreviewMode("camera");
  await startCamera();
});
useCameraButton.addEventListener("click", async () => {
  setPreviewMode("camera");
  await startCamera();
});
loadModelButton.addEventListener("click", async () => {
  try {
    await loadModel();
  } catch (error) {
    setStatus(`Model setup failed: ${error.message || error}`);
    appendDebug(errorText(error));
  }
});
captureButton.addEventListener("click", runVoteBurst);
autoScanButton.addEventListener("click", runAutoScanLoop);
imageUploadInput.addEventListener("change", async () => {
  const files = Array.from(imageUploadInput.files || []).filter((file) => file.type.startsWith("image/"));
  if (!files.length) return;
  clearUploadedBatch();
  const selected = files.slice(0, MAX_UPLOAD_VOTE_IMAGES);
  uploadedBatch = selected.map((file) => ({
    file,
    url: URL.createObjectURL(file),
    meta: { name: file.name, size: file.size, type: file.type, width: null, height: null },
  }));
  if (files.length > MAX_UPLOAD_VOTE_IMAGES) {
    appendDebug(`upload limited: selected ${files.length}, using first ${MAX_UPLOAD_VOTE_IMAGES}`);
  }
  try {
    await loadUploadedBatchImage(0);
    setPreviewMode("upload");
  } catch (error) {
    setStatus(`Image upload failed: ${error.message || error}`);
    appendDebug(errorText(error));
  }
});
copyDebugButton.addEventListener("click", async () => {
  const text = debugLog.join("\n");
  try { await navigator.clipboard.writeText(text); setStatus("Debug copied."); }
  catch (_) { window.prompt("Copy debug text", text); }
});
copyReportButton.addEventListener("click", async () => {
  const text = buildReport();
  try { await navigator.clipboard.writeText(text); setStatus("Report copied."); }
  catch (_) { window.prompt("Copy report JSON", text); }
});
downloadLogButton.addEventListener("click", downloadReport);
resetStatsButton.addEventListener("click", () => {
  scanLog = [];
  saveScanLog();
  telemetry.runs = 0;
  telemetry.successfulRuns = 0;
  telemetry.failedRuns = 0;
  telemetry.avgScanMs = null;
  telemetry.minScanMs = null;
  telemetry.maxScanMs = null;
  telemetry.lastScanMs = null;
  updateTelemetryDisplay();
  setStatus("Stats reset.");
});

loadScanLog();
updateTelemetryDisplay();
startCamera();
