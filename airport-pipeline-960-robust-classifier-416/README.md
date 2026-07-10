# GOATSOFAI Tracker + Robust Classifier Browser Pipeline

Date: 2026-07-10

This is the two-stage phone/browser pipeline using:

- stage 1: `tracker.tflite`, 960 px aircraft tracker/detector
- stage 2: `classifier.tflite`, robust/noise-trained 416 px aircraft-type classifier
- runtime smoothing: 10-frame majority vote for live camera scans
- upload smoothing: user can upload 1 to 10 images of the same aircraft, then
  the browser majority-votes across those uploaded images
- crop policy: tracker crop expanded by `2.5x` before classifier resize

The classifier is the robust follow-up `yolov8n-cls` model trained for phone/airport
domain shift. It uses the same class-label setup as the previous browser
classifier, plus realistic robustness augmentation:

- 6,000 FGVC real-box training crops
- 8,618 scraped Wikimedia full-image class-label samples
- 667 FGVC real-box validation crops
- 3,333 FGVC real-box official test crops
- motion/gaussian blur, noise, JPEG/compression, brightness/contrast, color shift,
  and reduced-resolution corruptions during training

Scraped images do not provide box supervision here. They are used only as
classifier class-label samples.

## Files

- `index.html` - browser UI
- `app.js` - tracker + classifier inference logic
- `style.css` - UI styling
- `tracker.tflite` - stage-1 aircraft tracker
- `classifier.tflite` - stage-2 aircraft-type classifier
- `labels.txt` - 100 FGVC aircraft variant labels

## Validation

The PyTorch pipeline audit on the 40-image Munich airport benchmark reported:

- top-1 aircraft-type accuracy: 19 / 40 = 47.5%
- top-5 aircraft-type accuracy: 28 / 40 = 70.0%
- tracker/crop detections: 40 / 40

The fixed browser batch benchmark, without temporal voting, reported:

- top-1 aircraft-type accuracy: 20 / 40 = 50.0%
- top-5 aircraft-type accuracy: 28 / 40 = 70.0%
- tracker/crop detections: 40 / 40

After the evening robustness sweep, the local shared GOATSOFAI browser app was
updated from crop expansion `1.8` to `2.5`. The PC/PyTorch benchmark with the
same robust v1 classifier and `crop_expand=2.5` reached:

- top-1 aircraft-type accuracy: 24 / 40 = 60.0%
- top-5 aircraft-type accuracy: 31 / 40 = 77.5%

The deployed browser implementation now exposes the deployment-time voting
behavior:

- camera path: 10-frame burst vote;
- upload path: 1-10 image vote over selected images of the same aircraft;
- report JSON fields: `vote_mode`, `vote_size`, `upload_vote_count`,
  `uploaded_images`, per-sample predictions, and runtime telemetry.

Public page:

```text
https://wickenhauserjakob-ops.github.io/GOATSOFAI-browser-demo/airport-pipeline-960-robust-classifier-416/
```

Temporal voting is a live-camera runtime layer. It is expected to help when
frame-to-frame predictions vary, but it does not fix systematic class collapse
such as every A320 frame being predicted as A321.

The exported classifier TFLite shape is:

- input shape: `[1, 416, 416, 3]`
- output shape: `[1, 100]`

The browser JavaScript decodes the classifier output as a 100-class probability
vector. It applies softmax only if the output is not already normalized.

## Local Run

From the repo root:

```powershell
python -m http.server 8000 -d browser_inference
```

Then open:

```text
http://localhost:8000/airport_pipeline_960_robust_classifier_416/
```

For phone camera testing, use the hosted HTTPS/GitHub Pages copy. Mobile camera
access is not reliable from a plain HTTP LAN URL.
