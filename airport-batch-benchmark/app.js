const ASSET_VERSION = "batch-v4-rectcrop";
const TFLITE_CDN = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.3/dist/";
const AIRPORT_MANIFEST_URL = "airport_manifest.json";
const IOU_THRESHOLD = 0.45;
const TRACKER_CONF_THRESHOLD = 0.08;
const CROP_EXPAND = 1.8;
const MIN_CROP_SOURCE_PX = 416;
const CLASSIFIER_SIZE = 416;
const TRACKER_SIZE = 960;

const MODELS = [
  {
    id: "trusted_baseline_416",
    name: "Trusted 416 baseline",
    type: "detector",
    size: 416,
    modelUrl: "../model.tflite",
    defaultEnabled: true,
    note: "90.58% clean FGVC held-out; direct detector.",
  },
  {
    id: "all_original_10k_416",
    name: "All-original 10k 416",
    type: "detector",
    size: 416,
    modelUrl: "../all-original-10k-416/model.tflite",
    defaultEnabled: true,
    note: "All FGVC originals, scraped validation 73.98%.",
  },
  {
    id: "original_8k_416",
    name: "Original-8k 416",
    type: "detector",
    size: 416,
    modelUrl: "../original-8k-416/model.tflite",
    defaultEnabled: true,
    note: "Original-only 8k comparison detector.",
  },
  {
    id: "final_phone_416",
    name: "Final-phone 416",
    type: "detector",
    size: 416,
    modelUrl: "../final-phone-416/model.tflite",
    defaultEnabled: true,
    note: "All-original phone candidate at 416.",
  },
  {
    id: "final_phone_640",
    name: "Final-phone 640",
    type: "detector",
    size: 640,
    modelUrl: "../final-phone-640/model.tflite",
    defaultEnabled: true,
    note: "Higher-res direct detector.",
  },
  {
    id: "final_phone_960",
    name: "Final-phone 960",
    type: "detector",
    size: 960,
    modelUrl: "../final-phone-960/model.tflite",
    defaultEnabled: true,
    note: "Heavy direct detector, slower.",
  },
  {
    id: "airport_pipeline_960_classifier_416",
    name: "Pipeline 960 -> classifier 416",
    type: "pipeline",
    trackerUrl: "../airport-pipeline-960-classifier-416/tracker.tflite",
    classifierUrl: "../airport-pipeline-960-classifier-416/classifier.tflite",
    defaultEnabled: true,
    note: "Tracker zooms/crops first, then classifier predicts type.",
  },
  {
    id: "airport_pipeline_960_detector_416",
    name: "Pipeline 960 -> detector 416",
    type: "pipeline_detector",
    trackerUrl: "../airport-pipeline-960-416/tracker.tflite",
    detectorUrl: "../airport-pipeline-960-416/variant.tflite",
    detectorSize: 416,
    defaultEnabled: true,
    note: "Older two-stage pipeline: tracker crop, then 416 detector predicts type.",
  },
];

const LABELS_URL = "../labels.txt";
const AIRCRAFT_TYPES = [
  "A220",
  "A320",
  "A321",
  "A380",
  "CRJ-900",
  "A319",
  "CRJ-700",
  "737-800",
  "747-400",
  "777-300",
  "Eurofighter Typhoon",
  "Cessna 172",
];

const statusEl = document.getElementById("status");
const datasetModeEl = document.getElementById("datasetMode");
const groundTruthFieldEl = document.getElementById("groundTruthField");
const groundTruthEl = document.getElementById("groundTruth");
const fileButtonEl = document.getElementById("fileButton");
const imageFilesEl = document.getElementById("imageFiles");
const fileCountEl = document.getElementById("fileCount");
const modelListEl = document.getElementById("modelList");
const detectorThresholdEl = document.getElementById("detectorThreshold");
const thresholdTextEl = document.getElementById("thresholdText");
const runButton = document.getElementById("runBenchmark");
const stopButton = document.getElementById("stopBenchmark");
const progressTextEl = document.getElementById("progressText");
const bestModelEl = document.getElementById("bestModel");
const totalRuntimeEl = document.getElementById("totalRuntime");
const deviceInfoEl = document.getElementById("deviceInfo");
const summaryBody = document.querySelector("#summaryTable tbody");
const debugEl = document.getElementById("debug");
const copyReportButton = document.getElementById("copyReport");
const downloadJsonButton = document.getElementById("downloadJson");
const downloadCsvButton = document.getElementById("downloadCsv");
const workCanvas = document.getElementById("workCanvas");
const cropCanvas = document.getElementById("cropCanvas");

let labels = [];
let runtime = null;
let loadedModels = new Map();
let selectedFiles = [];
let airportManifest = [];
let benchmarkStopRequested = false;
let latestReport = null;
let debugLines = [];

function setStatus(message) {
  statusEl.textContent = message;
  appendDebug(message);
}

function appendDebug(message) {
  debugLines.push(`${new Date().toISOString()} ${message}`);
  if (debugLines.length > 220) debugLines = debugLines.slice(-220);
  debugEl.textContent = debugLines.join("\n");
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(0)} ms` : "-";
}

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} s` : "-";
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

function getDeviceInfo() {
  const parts = [];
  if (navigator.hardwareConcurrency) parts.push(`${navigator.hardwareConcurrency} cores`);
  if (navigator.deviceMemory) parts.push(`${navigator.deviceMemory} GB RAM hint`);
  parts.push(`${screen.width}x${screen.height}@${window.devicePixelRatio || 1}`);
  return parts.join(", ");
}

function initUi() {
  for (const type of AIRCRAFT_TYPES) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type === "A220" ? "A220 (out of FGVC label set)" : type;
    groundTruthEl.appendChild(option);
  }
  groundTruthEl.value = "A320";
  for (const model of MODELS) {
    const card = document.createElement("div");
    card.className = "model-card";
    card.innerHTML = `
      <label>
        <input type="checkbox" value="${model.id}" ${model.defaultEnabled ? "checked" : ""}>
        ${model.name}
      </label>
      <p>${model.note}</p>
    `;
    modelListEl.appendChild(card);
  }
  detectorThresholdEl.addEventListener("input", () => {
    thresholdTextEl.textContent = `${detectorThresholdEl.value}%`;
  });
  datasetModeEl.addEventListener("change", updateDatasetUi);
  imageFilesEl.addEventListener("change", () => {
    selectedFiles = Array.from(imageFilesEl.files || []).filter((file) => file.type.startsWith("image/"));
    updateDatasetUi();
    if (datasetModeEl.value === "upload") setStatus(`${selectedFiles.length} image(s) selected.`);
  });
  runButton.addEventListener("click", runBenchmark);
  stopButton.addEventListener("click", () => {
    benchmarkStopRequested = true;
    setStatus("Stopping after current scan...");
  });
  copyReportButton.addEventListener("click", copyReport);
  downloadJsonButton.addEventListener("click", () => downloadText("goatsofai-airport-batch-report.json", JSON.stringify(latestReport, null, 2), "application/json"));
  downloadCsvButton.addEventListener("click", () => downloadText("goatsofai-airport-batch-results.csv", buildCsv(latestReport), "text/csv"));
  deviceInfoEl.textContent = getDeviceInfo();
  updateDatasetUi();
  fetchAirportManifest()
    .then(() => updateDatasetUi())
    .catch((error) => appendDebug(`Airport manifest pre-load failed: ${error.message || error}`));
}

function updateDatasetUi() {
  const bundled = datasetModeEl.value === "airport";
  groundTruthFieldEl.classList.toggle("muted", bundled);
  groundTruthEl.disabled = bundled;
  imageFilesEl.disabled = bundled;
  fileButtonEl.classList.toggle("disabled", bundled);
  fileButtonEl.textContent = bundled ? "Bundled Set Active" : "Select Images";
  if (bundled) {
    const count = airportManifest.length || 48;
    const unsupported = airportManifest.filter((entry) => entry.label === "A220").length || 8;
    const supported = count - unsupported;
    fileCountEl.textContent = `${count} bundled, ${supported} scored`;
  } else {
    fileCountEl.textContent = String(selectedFiles.length);
  }
}

async function waitForRuntime() {
  if (runtime) return runtime;
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const timer = setInterval(() => {
      if (window.tflite && typeof window.tflite.loadTFLiteModel === "function") {
        clearInterval(timer);
        runtime = window.tflite;
        if (typeof runtime.setWasmPath === "function") runtime.setWasmPath(TFLITE_CDN);
        resolve(runtime);
      } else if (performance.now() - started > 10000) {
        clearInterval(timer);
        reject(new Error("TFLite runtime not available."));
      }
    }, 50);
  });
}

async function fetchLabels() {
  if (labels.length) return labels;
  const response = await fetch(`${LABELS_URL}?${ASSET_VERSION}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`labels.txt failed: HTTP ${response.status}`);
  labels = (await response.text()).split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  return labels;
}

async function fetchAirportManifest() {
  if (airportManifest.length) return airportManifest;
  const response = await fetch(`${AIRPORT_MANIFEST_URL}?${ASSET_VERSION}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`airport_manifest.json failed: HTTP ${response.status}`);
  airportManifest = await response.json();
  return airportManifest;
}

async function loadModelConfig(config) {
  if (loadedModels.has(config.id)) return loadedModels.get(config.id);
  const rt = await waitForRuntime();
  const started = performance.now();
  if (config.type === "detector") {
    setStatus(`Loading ${config.name}...`);
    const model = await rt.loadTFLiteModel(`${config.modelUrl}?${ASSET_VERSION}`);
    const loaded = { ...config, model, loadMs: performance.now() - started };
    loadedModels.set(config.id, loaded);
    return loaded;
  }
  setStatus(`Loading ${config.name} tracker...`);
  const tracker = await rt.loadTFLiteModel(`${config.trackerUrl}?${ASSET_VERSION}`);
  if (config.type === "pipeline_detector") {
    setStatus(`Loading ${config.name} detector...`);
    const detector = await rt.loadTFLiteModel(`${config.detectorUrl}?${ASSET_VERSION}`);
    const loaded = { ...config, tracker, detector, loadMs: performance.now() - started };
    loadedModels.set(config.id, loaded);
    return loaded;
  }
  setStatus(`Loading ${config.name} classifier...`);
  const classifier = await rt.loadTFLiteModel(`${config.classifierUrl}?${ASSET_VERSION}`);
  const loaded = { ...config, tracker, classifier, loadMs: performance.now() - started };
  loadedModels.set(config.id, loaded);
  return loaded;
}

function selectedModelConfigs() {
  const selected = new Set(Array.from(modelListEl.querySelectorAll("input:checked")).map((input) => input.value));
  return MODELS.filter((model) => selected.has(model.id));
}

async function getBenchmarkItems() {
  if (datasetModeEl.value === "airport") {
    const manifest = await fetchAirportManifest();
    return manifest.map((entry, index) => ({
      id: `airport_${index + 1}`,
      source: "bundled_munich_airport_2026_07_09",
      name: entry.file.split("/").pop(),
      url: entry.file,
      size: entry.size_bytes || 0,
      type: "image/jpeg",
      gtClass: entry.label,
      sourceFolder: entry.source_folder,
      sourceName: entry.source_name,
    }));
  }
  return selectedFiles.map((file, index) => ({
    id: `upload_${index + 1}`,
    source: "manual_upload",
    name: file.name,
    file,
    size: file.size,
    type: file.type,
    gtClass: groundTruthEl.value,
  }));
}

function loadImage(item) {
  return new Promise((resolve, reject) => {
    const url = item.file ? URL.createObjectURL(item.file) : item.url;
    const image = new Image();
    image.onload = () => resolve({ image, url, revoke: Boolean(item.file) });
    image.onerror = () => {
      if (item.file) URL.revokeObjectURL(url);
      reject(new Error(`Cannot load image: ${item.name}`));
    };
    image.src = url;
  });
}

function drawLetterboxToCanvas(image, canvas, size) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const scale = Math.min(size / image.naturalWidth, size / image.naturalHeight);
  const drawWidth = image.naturalWidth * scale;
  const drawHeight = image.naturalHeight * scale;
  const padX = (size - drawWidth) / 2;
  const padY = (size - drawHeight) / 2;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(image, padX, padY, drawWidth, drawHeight);
  return { scale, padX, padY, sourceWidth: image.naturalWidth, sourceHeight: image.naturalHeight };
}

function tensorFromCanvas(canvas) {
  return tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0));
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
  for (const detection of sorted) {
    if (kept.every((other) => iou(detection, other) < IOU_THRESHOLD)) kept.push(detection);
    if (kept.length >= 20) break;
  }
  return kept;
}

function decodeDetectorOutput(output, classLabels, threshold, coordSize) {
  const data = output.dataSync();
  const channels = output.shape[1];
  const anchors = output.shape[2];
  const classCount = channels - 4;
  const detections = [];
  const classScores = new Array(Math.max(0, classCount)).fill(0);
  for (let anchor = 0; anchor < anchors; anchor += 1) {
    let bestClass = 0;
    let bestScore = 0;
    for (let classId = 0; classId < classCount; classId += 1) {
      const score = data[(4 + classId) * anchors + anchor];
      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
      if (score > classScores[classId]) classScores[classId] = score;
    }
    if (bestScore < threshold) continue;
    let cx = data[anchor];
    let cy = data[anchors + anchor];
    let w = data[2 * anchors + anchor];
    let h = data[3 * anchors + anchor];
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
  const rankedClasses = classScores
    .map((score, classId) => ({ classId, label: classLabels[classId] || `class ${classId}`, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  return { detections: nonMaxSuppression(detections), rankedClasses };
}

function trackerBoxToSource(box, letterbox) {
  const x1 = (box[0] - letterbox.padX) / letterbox.scale;
  const y1 = (box[1] - letterbox.padY) / letterbox.scale;
  const x2 = (box[2] - letterbox.padX) / letterbox.scale;
  const y2 = (box[3] - letterbox.padY) / letterbox.scale;
  const left = Math.max(0, Math.min(letterbox.sourceWidth, Math.min(x1, x2)));
  const top = Math.max(0, Math.min(letterbox.sourceHeight, Math.min(y1, y2)));
  const right = Math.max(0, Math.min(letterbox.sourceWidth, Math.max(x1, x2)));
  const bottom = Math.max(0, Math.min(letterbox.sourceHeight, Math.max(y1, y2)));
  if (right - left < 8 || bottom - top < 8) {
    return [0, 0, letterbox.sourceWidth, letterbox.sourceHeight];
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

function drawCrop(image, cropBox) {
  cropCanvas.width = CLASSIFIER_SIZE;
  cropCanvas.height = CLASSIFIER_SIZE;
  const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
  const cropW = Math.max(1, cropBox[2] - cropBox[0]);
  const cropH = Math.max(1, cropBox[3] - cropBox[1]);
  const scale = Math.min(CLASSIFIER_SIZE / cropW, CLASSIFIER_SIZE / cropH);
  const drawW = cropW * scale;
  const drawH = cropH * scale;
  const padX = (CLASSIFIER_SIZE - drawW) / 2;
  const padY = (CLASSIFIER_SIZE - drawH) / 2;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  ctx.drawImage(image, cropBox[0], cropBox[1], cropW, cropH, padX, padY, drawW, drawH);
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const exp = scores.map((score) => Math.exp(score - maxScore));
  const sum = exp.reduce((total, value) => total + value, 0) || 1;
  return exp.map((value) => value / sum);
}

function analyzeClassifier(output, classLabels) {
  const raw = Array.from(output.dataSync());
  const sum = raw.reduce((total, value) => total + value, 0);
  const needsSoftmax = raw.some((score) => score < 0 || score > 1) || Math.abs(sum - 1) > 0.05;
  const scores = needsSoftmax ? softmax(raw) : raw;
  const ranked = scores
    .map((score, classId) => ({ classId, label: classLabels[classId] || `class ${classId}`, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  return ranked;
}

async function runDetectorModel(loaded, image, gtClass, threshold) {
  const started = performance.now();
  let input = null;
  let outputTensor = null;
  try {
    drawLetterboxToCanvas(image, workCanvas, loaded.size);
    input = tensorFromCanvas(workCanvas);
    const output = loaded.model.predict(input);
    outputTensor = Array.isArray(output) ? output[0] : output;
    const decoded = decodeDetectorOutput(outputTensor, labels, threshold, loaded.size);
    const best = decoded.detections[0] || null;
    const top5 = decoded.detections.slice(0, 5).map((detection) => ({ label: detection.label, score: detection.score }));
    return buildResult(loaded, gtClass, best ? best.label : "NO_DETECTION", best ? best.score : 0, top5, performance.now() - started, best ? best.box : null);
  } finally {
    if (input) input.dispose();
    if (outputTensor && typeof outputTensor.dispose === "function") outputTensor.dispose();
  }
}

async function runPipelineModel(loaded, image, gtClass) {
  const started = performance.now();
  let trackerInput = null;
  let trackerOutputTensor = null;
  let classifierInput = null;
  let classifierOutputTensor = null;
  try {
    const letterbox = drawLetterboxToCanvas(image, workCanvas, TRACKER_SIZE);
    trackerInput = tensorFromCanvas(workCanvas);
    const trackerOutput = loaded.tracker.predict(trackerInput);
    trackerOutputTensor = Array.isArray(trackerOutput) ? trackerOutput[0] : trackerOutput;
    const trackerDecoded = decodeDetectorOutput(trackerOutputTensor, ["aircraft"], TRACKER_CONF_THRESHOLD, TRACKER_SIZE);
    const trackerBest = trackerDecoded.detections[0] || null;
    if (!trackerBest) {
      return buildResult(loaded, gtClass, "NO_TRACKER_DETECTION", 0, [], performance.now() - started, null, null, null);
    }
    const sourceBox = trackerBoxToSource(trackerBest.box, letterbox);
    const cropBox = expandCrop(sourceBox, letterbox.sourceWidth, letterbox.sourceHeight);
    drawCrop(image, cropBox);
    classifierInput = tensorFromCanvas(cropCanvas);
    const classifierOutput = loaded.classifier.predict(classifierInput);
    classifierOutputTensor = Array.isArray(classifierOutput) ? classifierOutput[0] : classifierOutput;
    const top5 = analyzeClassifier(classifierOutputTensor, labels);
    const best = top5[0] || null;
    return buildResult(loaded, gtClass, best ? best.label : "NO_CLASSIFIER_OUTPUT", best ? best.score : 0, top5, performance.now() - started, sourceBox, trackerBest.score, cropBox);
  } finally {
    if (trackerInput) trackerInput.dispose();
    if (trackerOutputTensor && typeof trackerOutputTensor.dispose === "function") trackerOutputTensor.dispose();
    if (classifierInput) classifierInput.dispose();
    if (classifierOutputTensor && typeof classifierOutputTensor.dispose === "function") classifierOutputTensor.dispose();
  }
}

async function runPipelineDetectorModel(loaded, image, gtClass, threshold) {
  const started = performance.now();
  let trackerInput = null;
  let trackerOutputTensor = null;
  let detectorInput = null;
  let detectorOutputTensor = null;
  try {
    const letterbox = drawLetterboxToCanvas(image, workCanvas, TRACKER_SIZE);
    trackerInput = tensorFromCanvas(workCanvas);
    const trackerOutput = loaded.tracker.predict(trackerInput);
    trackerOutputTensor = Array.isArray(trackerOutput) ? trackerOutput[0] : trackerOutput;
    const trackerDecoded = decodeDetectorOutput(trackerOutputTensor, ["aircraft"], TRACKER_CONF_THRESHOLD, TRACKER_SIZE);
    const trackerBest = trackerDecoded.detections[0] || null;
    if (!trackerBest) {
      return buildResult(loaded, gtClass, "NO_TRACKER_DETECTION", 0, [], performance.now() - started, null, null, null);
    }
    const sourceBox = trackerBoxToSource(trackerBest.box, letterbox);
    const cropBox = expandCrop(sourceBox, letterbox.sourceWidth, letterbox.sourceHeight);
    drawCrop(image, cropBox);
    detectorInput = tensorFromCanvas(cropCanvas);
    const detectorOutput = loaded.detector.predict(detectorInput);
    detectorOutputTensor = Array.isArray(detectorOutput) ? detectorOutput[0] : detectorOutput;
    const decoded = decodeDetectorOutput(detectorOutputTensor, labels, threshold, loaded.detectorSize || CLASSIFIER_SIZE);
    const best = decoded.detections[0] || null;
    const top5 = decoded.detections.slice(0, 5).map((detection) => ({ label: detection.label, score: detection.score }));
    return buildResult(loaded, gtClass, best ? best.label : "NO_DETECTION", best ? best.score : 0, top5, performance.now() - started, sourceBox, trackerBest.score, cropBox);
  } finally {
    if (trackerInput) trackerInput.dispose();
    if (trackerOutputTensor && typeof trackerOutputTensor.dispose === "function") trackerOutputTensor.dispose();
    if (detectorInput) detectorInput.dispose();
    if (detectorOutputTensor && typeof detectorOutputTensor.dispose === "function") detectorOutputTensor.dispose();
  }
}

function buildResult(model, gtClass, predClass, confidence, top5, scanMs, box, trackerScore = null, cropBox = null) {
  const gtSupported = labels.includes(gtClass);
  const top5Classes = top5.map((item) => item.label);
  return {
    model_id: model.id,
    model_name: model.name,
    model_type: model.type,
    gt_class: gtClass,
    gt_supported: gtSupported,
    pred_class: predClass,
    confidence,
    top1_correct: gtSupported && predClass === gtClass,
    top5_correct: gtSupported && top5Classes.includes(gtClass),
    no_detection: predClass === "NO_DETECTION" || predClass === "NO_TRACKER_DETECTION" || predClass === "NO_CLASSIFIER_OUTPUT",
    top5,
    scan_ms: scanMs,
    box,
    tracker_score: trackerScore,
    crop_box: cropBox,
  };
}

function summarize(results, items, datasetSource, startedAt, elapsedMs) {
  const byModel = new Map();
  for (const result of results) {
    if (!byModel.has(result.model_id)) byModel.set(result.model_id, []);
    byModel.get(result.model_id).push(result);
  }
  const summaries = [];
  const perClass = [];
  for (const [modelId, rows] of byModel.entries()) {
    const model = MODELS.find((item) => item.id === modelId);
    const supportedRows = rows.filter((row) => row.gt_supported);
    const denom = supportedRows.length || rows.length || 1;
    const top1 = supportedRows.filter((row) => row.top1_correct).length;
    const top5 = supportedRows.filter((row) => row.top5_correct).length;
    const noDetection = supportedRows.filter((row) => row.no_detection).length;
    const avgScanMs = rows.reduce((sum, row) => sum + row.scan_ms, 0) / rows.length;
    summaries.push({
      model_id: modelId,
      model_name: model?.name || modelId,
      model_type: model?.type || "",
      images: rows.length,
      supported_images: supportedRows.length,
      unsupported_images: rows.length - supportedRows.length,
      top1_correct: top1,
      top1_accuracy: top1 / denom,
      top5_correct: top5,
      top5_accuracy: top5 / denom,
      no_detection: noDetection,
      no_detection_rate: noDetection / denom,
      avg_scan_ms: avgScanMs,
    });
    const classes = Array.from(new Set(rows.map((row) => row.gt_class))).sort();
    for (const className of classes) {
      const classRows = rows.filter((row) => row.gt_class === className);
      const classSupportedRows = classRows.filter((row) => row.gt_supported);
      const classDenom = classSupportedRows.length || classRows.length || 1;
      const classTop1 = classSupportedRows.filter((row) => row.top1_correct).length;
      const classTop5 = classSupportedRows.filter((row) => row.top5_correct).length;
      const classNoDetection = classSupportedRows.filter((row) => row.no_detection).length;
      perClass.push({
        model_id: modelId,
        model_name: model?.name || modelId,
        gt_class: className,
        gt_supported: labels.includes(className),
        images: classRows.length,
        supported_images: classSupportedRows.length,
        top1_correct: classTop1,
        top1_accuracy: classTop1 / classDenom,
        top5_correct: classTop5,
        top5_accuracy: classTop5 / classDenom,
        no_detection: classNoDetection,
        no_detection_rate: classNoDetection / classDenom,
      });
    }
  }
  summaries.sort((a, b) => b.top1_accuracy - a.top1_accuracy || b.top5_accuracy - a.top5_accuracy);
  perClass.sort((a, b) => a.model_name.localeCompare(b.model_name) || a.gt_class.localeCompare(b.gt_class));
  return {
    build: ASSET_VERSION,
    created_at: new Date().toISOString(),
    started_at: startedAt,
    elapsed_ms: elapsedMs,
    dataset_source: datasetSource,
    browser: navigator.userAgent,
    device: getDeviceInfo(),
    files: items.map((item) => ({
      id: item.id,
      name: item.name,
      size: item.size,
      type: item.type,
      gt_class: item.gtClass,
      source: item.source,
      source_folder: item.sourceFolder || "",
      source_name: item.sourceName || "",
    })),
    selected_models: selectedModelConfigs().map((model) => model.id),
    summaries,
    per_class: perClass,
    results,
  };
}

async function runBenchmark() {
  const configs = selectedModelConfigs();
  if (!configs.length) {
    setStatus("Select at least one model.");
    return;
  }
  benchmarkStopRequested = false;
  runButton.disabled = true;
  stopButton.disabled = false;
  copyReportButton.disabled = true;
  downloadJsonButton.disabled = true;
  downloadCsvButton.disabled = true;
  summaryBody.innerHTML = "";
  progressTextEl.textContent = "starting";
  bestModelEl.textContent = "-";
  const threshold = Number(detectorThresholdEl.value) / 100;
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const results = [];
  try {
    await fetchLabels();
    await waitForRuntime();
    const items = await getBenchmarkItems();
    if (!items.length) {
      setStatus(datasetModeEl.value === "airport" ? "Bundled airport manifest is empty." : "Select images first.");
      return;
    }
    const datasetSource = datasetModeEl.value === "airport" ? "bundled_munich_airport_2026_07_09" : "manual_upload";
    for (const config of configs) {
      if (benchmarkStopRequested) break;
      const loaded = await loadModelConfig(config);
      for (let index = 0; index < items.length; index += 1) {
        if (benchmarkStopRequested) break;
        const item = items[index];
        progressTextEl.textContent = `${config.name}: ${index + 1}/${items.length}`;
        setStatus(`Running ${config.name} on ${item.name} (${item.gtClass})`);
        const loadedImage = await loadImage(item);
        try {
          const result = config.type === "pipeline"
            ? await runPipelineModel(loaded, loadedImage.image, item.gtClass)
            : config.type === "pipeline_detector"
              ? await runPipelineDetectorModel(loaded, loadedImage.image, item.gtClass, threshold)
              : await runDetectorModel(loaded, loadedImage.image, item.gtClass, threshold);
          result.file_id = item.id;
          result.file_name = item.name;
          result.file_size = item.size;
          result.dataset_source = item.source;
          result.source_folder = item.sourceFolder || "";
          result.source_name = item.sourceName || "";
          result.image_width = loadedImage.image.naturalWidth;
          result.image_height = loadedImage.image.naturalHeight;
          results.push(result);
        } finally {
          if (loadedImage.revoke) URL.revokeObjectURL(loadedImage.url);
          await tf.nextFrame();
        }
      }
      latestReport = summarize(results, items, datasetSource, startedAt, performance.now() - started);
      renderSummary(latestReport);
    }
    latestReport = summarize(results, items, datasetSource, startedAt, performance.now() - started);
    renderSummary(latestReport);
    progressTextEl.textContent = benchmarkStopRequested ? "stopped" : "done";
    totalRuntimeEl.textContent = formatSeconds(latestReport.elapsed_ms / 1000);
    setStatus(benchmarkStopRequested ? "Benchmark stopped." : "Benchmark complete.");
    copyReportButton.disabled = false;
    downloadJsonButton.disabled = false;
    downloadCsvButton.disabled = false;
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message || error}`);
    appendDebug(error.stack || String(error));
  } finally {
    runButton.disabled = false;
    stopButton.disabled = true;
  }
}

function renderSummary(report) {
  summaryBody.innerHTML = "";
  for (const summary of report.summaries) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${summary.model_name}</td>
      <td>${summary.supported_images || summary.images} scored / ${summary.images} run</td>
      <td>${summary.top1_correct}/${summary.supported_images || summary.images} (${formatPercent(summary.top1_accuracy)})</td>
      <td>${summary.top5_correct}/${summary.supported_images || summary.images} (${formatPercent(summary.top5_accuracy)})</td>
      <td>${summary.no_detection}/${summary.supported_images || summary.images} (${formatPercent(summary.no_detection_rate)})</td>
      <td>${formatMs(summary.avg_scan_ms)}</td>
    `;
    summaryBody.appendChild(row);
  }
  const best = report.summaries[0];
  bestModelEl.textContent = best ? `${best.model_name} (${formatPercent(best.top1_accuracy)} top-1)` : "-";
  totalRuntimeEl.textContent = formatSeconds(report.elapsed_ms / 1000);
}

async function copyReport() {
  if (!latestReport) return;
  const text = JSON.stringify(latestReport, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    setStatus("JSON report copied.");
  } catch (_) {
    window.prompt("Copy JSON report", text);
  }
}

function downloadText(filename, text, type) {
  if (!latestReport) return;
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function buildCsv(report) {
  if (!report) return "";
  const fields = [
    "file_id",
    "file_name",
    "dataset_source",
    "source_folder",
    "source_name",
    "gt_class",
    "gt_supported",
    "model_id",
    "model_name",
    "model_type",
    "pred_class",
    "confidence",
    "top1_correct",
    "top5_correct",
    "no_detection",
    "scan_ms",
    "top5_classes",
  ];
  const lines = [fields.join(",")];
  for (const row of report.results) {
    const values = {
      ...row,
      top5_classes: row.top5.map((item) => `${item.label}:${item.score.toFixed(4)}`).join("|"),
    };
    lines.push(fields.map((field) => csvEscape(values[field])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

initUi();
