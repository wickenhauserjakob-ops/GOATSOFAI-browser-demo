# Browser Demo: Final-Phone Unified Variant AI 416

This folder packages the final-phone 416 TFLite detector for browser inference.

- model: `model.tflite`
- source weights: `variant_unified_final_phone_all_original_416_from_baseline_sgd`
- input size: `416 x 416`
- audit status: deployment smoke only; old test images are included in training
- output decoder: YOLOv8 raw output `[1, 104, 3549]`
- runtime telemetry: model load time, model size, per-scan latency, average
  latency, estimated FPS, run count, camera resolution, browser/device hints,
  memory/tensor state, network hints, and battery status when supported
- live failure diagnostics: best raw candidate, top raw classes, A320 raw
  score, and direct A320 vs 737-200 score comparison
- automatic session logging: every Run Scan / Auto Scan result is stored in the
  phone browser and included in `Copy Report`; `Download Log` saves JSON
- image upload: `Upload Image` previews a local photo and runs the same TFLite
  model on it; `Use Camera` switches back to the webcam stream

Run from the repository root:

```powershell
C:\Users\jarja\miniconda3\python.exe -m http.server 8000
```

Then open:

```text
http://localhost:8000/browser_inference/final_phone_416_all_original/
```

On a phone, open the same URL using the computer's local network IP address.
Camera access requires `localhost` or HTTPS in most browsers. For Android over
USB, the clean quick test is:

```powershell
adb reverse tcp:8000 tcp:8000
```

Then open this on the phone:

```text
http://localhost:8000/browser_inference/final_phone_416_all_original/
```

If testing over Wi-Fi with the PC's local IP address, use an HTTPS tunnel or an
HTTPS local server; plain LAN HTTP will usually load the page but block camera
access.

For the public HTTPS phone demo, use:

```text
https://wickenhauserjakob-ops.github.io/GOATSOFAI-browser-demo/telemetry-v8/
```

Use `Copy Report` after several scans to export a poster-ready JSON block.
Browser JavaScript cannot read real CPU/GPU wattage. The page therefore reports
battery level and estimated discharge rate only if the browser exposes the
Battery Status API and the battery level changes during the test.

The page can automatically collect logs locally, but it cannot automatically
send them to Codex from GitHub Pages because there is no writable backend.
