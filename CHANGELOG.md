# Changelog

## v2.0.0 — 2026-07-08

Splatline v2 is a complete rebuild of the video-to-3D pipeline with
state-of-the-art reconstruction, temporal human motion recovery, and a
modern FastAPI backend.

### Reconstruction

- **Pluggable backend registry** with first-run selector. Users choose
  between SHARP (non-commercial research) and TripoSplat (MIT, fully open).
  Selection persists across sessions.
- **TripoSplat** (MIT, SIGGRAPH 2026) added as fully open-source backend —
  single-image 3D Gaussian Splatting with learned density control.
- **DepthSplat** (MIT, CVPR 2025) tracked for multi-view keyframe
  reconstruction.
- **LongSplat** (NVlabs, ICCV 2025) tracked for video-native temporal
  coherence — produces a single coherent 3DGS scene from the whole video
  instead of independent per-frame splats.
- **VGGT** (CVPR 2025 Best Paper) tracked for geometry bootstrap.
- **gsplat** (Apache-2.0) integrated for Python-side rendering and export.

### Human Pipeline (tiered)

- **Tier 1 — Fast skeleton:** MotionBERT (MIT, ICCV 2023) for temporal 3D
  pose lifting. Replaces per-frame depth-lifting with a dual-stream
  spatio-temporal transformer that looks at up to 243 frames at once,
  producing smooth, temporally consistent 3D motion. Fused with Gaussian
  splat depth for metric scale recovery — the key advantage over pure
  monocular approaches.
- **Tier 2 — Full SMPL mesh:** HMR 2.0 / 4DHumans (MIT) for SMPL body mesh
  recovery + 3D tracking from video. Produces a textured 3D human body in
  the scene, not just a skeleton. Multi-person with identity tracking.
- **PHALP** (MIT) for 3D-aware identity tracking — maintains consistent
  person IDs through occlusion events.
- **YOLO26-pose** (Ultralytics, 2026) as default 2D pose detector — NMS-free,
  72% COCO AP, 1.8-12.2ms latency.
- **RTMPose** (Apache-2.0, MMPose) as AGPL-free alternative 2D pose detector.

### Backend

- **FastAPI + SSE** replaces stdlib HTTP server. Real-time progress
  streaming for long-running ML jobs, Pydantic validation, auto OpenAPI docs.
- **ffmpegcv** optional GPU-accelerated video I/O (drop-in OpenCV replacement).
- **ONNX Runtime** export utilities for INT8 pose model quantization
  (2-4x speedup, <1% accuracy loss).
- **Hugging Face Hub** for model download and caching (versioned, no
  re-downloads).

### Frontend

- **Backend selector** with license transparency on first run.
- **Human tier toggle** — skeleton (fast) vs SMPL mesh (detailed).
- **SSE progress streaming** — live log updates without polling.
- **Rerun web viewer** support via `@rerun-io/web-viewer-react`.

### Project

- **MIT license** for Splatline itself.
- **NOTICE.md** documents all dependency licenses and restrictions.
- Version bumped to 2.0.0 across package.json, setup.py.

### Backwards compatibility

v1 scripts (`video_to_3d_high_quality.py`, `video_to_3d_with_pose.py`,
`visualize_with_rerun.py`, etc.) continue to work unchanged. The v2
human pipeline is opt-in via `--human-tier skeleton|mesh|both`.

---

## v1.0.0 — 2026-04

Initial Splatline release.

- Apple SHARP monocular 3D Gaussian Splatting from video frames.
- Rerun visualization with 3D scene, depth maps, navigation.
- YOLO pose detection with depth-lifted 3D skeletons.
- Athlete Twin biomechanics: frame metrics, event detection, exports.
- Electron + React desktop app (Field Movement Lab).
