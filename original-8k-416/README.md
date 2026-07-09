# Browser Demo: Original-8k Unified Variant AI 416

This folder packages the 416 TFLite detector trained for the original-only
comparison study.

- model: `model.tflite`
- source weights: `variant_unified_yolov8n_416_100ep_original_train_8k_sgd`
- input size: `416 x 416`
- official FGVC test exact aircraft-type accuracy: `2968 / 3333 = 89.05%`
- scraped training images: `0`
- comparison target: `variant_unified_yolov8n_416_100ep_full_cleaned_scrapes_sgd`
- output decoder: YOLOv8 raw output `[1, 104, 3549]`
- runtime telemetry: same telemetry-enabled UI as the baseline browser demo

Run from the repository root:

```powershell
C:\Users\jarja\miniconda3\python.exe -m http.server 8000
```

Then open:

```text
http://localhost:8000/browser_inference/original_8k_416/
```

On a phone, camera access requires `localhost` or HTTPS in most browsers.
