const MODEL_SIZE = 416;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const inputCanvas = document.getElementById("inputCanvas");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const captureButton = document.getElementById("capture");
const toggleButton = document.getElementById("toggleCamera");

let facingMode = "environment";
let model = null;
let labels = [];
let letterbox = { scale: 1, padX: 0, padY: 0, sourceWidth: 1, sourceHeight: 1 };

captureButton.disabled = true;

async function loadLabels() {
  const response = await fetch("labels.txt");
  const text = await response.text();
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

async function loadModel() {
  labels = await loadLabels();
  model = await tflite.loadTFLiteModel("model.tflite");
  statusEl.textContent = "Model ready. Point the camera at an aircraft.";
  captureButton.disabled = false;
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
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(inputCanvas)
      .toFloat()
      .div(255)
      .expandDims(0);
  });
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
  if (!model || video.readyState < 2) {
    return;
  }
  statusEl.textContent = "Running inference...";
  const started = performance.now();
  drawLetterboxedFrame();
  const input = tensorFromCanvas();
  const output = model.predict(input);
  const tensor = Array.isArray(output) ? output[0] : output;
  const detections = toDetections(tensor);
  input.dispose();
  tensor.dispose();

  drawDetections(detections);
  const elapsed = performance.now() - started;
  if (detections.length === 0) {
    labelEl.textContent = "No aircraft detected";
    confidenceEl.textContent = "-";
  } else {
    labelEl.textContent = detections[0].label;
    confidenceEl.textContent = `${(detections[0].score * 100).toFixed(1)}%`;
  }
  statusEl.textContent = `Done in ${elapsed.toFixed(0)} ms`;
}

toggleButton.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
});

captureButton.addEventListener("click", runInference);

window.addEventListener("load", async () => {
  try {
    await Promise.all([loadModel(), startCamera()]);
  } catch (error) {
    console.error(error);
    statusEl.textContent = `Setup failed: ${error.message}`;
  }
});
