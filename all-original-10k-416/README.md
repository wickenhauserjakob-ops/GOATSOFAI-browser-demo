# Browser Demo: All-Original 10k Unified Variant AI 416

This folder packages the 416 TFLite detector trained on all 10,000 original
FGVC-Aircraft images with real human boxes. Scraped Wikimedia images were not
used for training; they were used only for aircraft-type validation without
box metrics.

- model: `model.tflite`
- source weights: `variant_unified_yolov8n_416_100ep_all_original_10k_scrapedval_sgd`
- input size: `416 x 416`
- scraped validation top-1: `6434 / 8697 = 73.98%`
- scraped validation top-5: `6830 / 8697 = 78.53%`
- no-detection rate on scraped validation: `3 / 8697 = 0.0345%`
- important caveat: official FGVC test images are included in this model's
  training set, so this is a deployment/phone candidate, not a fair FGVC
  held-out benchmark
- output decoder: YOLOv8 raw output `[1, 104, 3549]`
- runtime telemetry: model load time, model size, per-scan latency, average
  latency, estimated FPS, run count, camera resolution, browser/device hints,
  memory/tensor state, network hints, and battery status when supported

Run from the repository root:

```powershell
C:\Users\jarja\miniconda3\python.exe -m http.server 8000
```

Then open:

```text
http://localhost:8000/browser_inference/all_original_10k_scrapedval_416/
```

On a phone, camera access requires `localhost` or HTTPS in most browsers. For
Android over USB:

```powershell
adb reverse tcp:8000 tcp:8000
```

Then open:

```text
http://localhost:8000/browser_inference/all_original_10k_scrapedval_416/
```

Browser JavaScript cannot read real CPU/GPU wattage. The page reports battery
level and estimated discharge rate only when the browser exposes that data.
