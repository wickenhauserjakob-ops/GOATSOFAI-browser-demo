# GOATSOFAI Browser Demo

Static browser/TFLite demo for the unified 416 baseline model.

Open:
https://wickenhauserjakob-ops.github.io/GOATSOFAI-browser-demo/

Fresh telemetry build:
https://wickenhauserjakob-ops.github.io/GOATSOFAI-browser-demo/telemetry-v6/

The page reports model load time, model size, per-scan latency, average latency,
estimated FPS, run count, camera resolution, browser/device hints, memory/tensor
state, network hints, and battery level/rate when the browser exposes those
APIs. Browser JavaScript does not expose real CPU/GPU wattage, so power is
reported only via available Battery Status API data.

The telemetry build also exposes raw class-score diagnostics for live failures,
including a direct A320 vs 737-200 comparison and the top raw classes before the
confidence threshold is applied.
