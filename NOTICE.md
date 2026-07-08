# Splatline — Third-Party Licenses

Splatline is MIT-licensed. Some optional backends and models have different
licenses. This file lists every dependency and its license so users can make
informed choices.

## Splatline

- **License:** MIT
- **Copyright:** 2026 Jaskirat Singh

## Reconstruction Backends

| Backend | License | Commercial use | Notes |
|---------|---------|---------------|-------|
| Apple SHARP | apple-amlr (non-commercial) | No | Research-only. Default for research use. |
| TripoSplat | MIT | Yes | Fully open. SIGGRAPH 2026. |
| DepthSplat | MIT | Yes | Multi-view. CVPR 2025. |
| LongSplat | NVlabs (check terms) | Check | Video-native. ICCV 2025. |
| VGGT | Custom (commercial checkpoint available) | Apply | CVPR 2025 Best Paper. |
| SplineGS | MIT | Yes | Dynamic monocular. CVPR 2025. |

## Human Pose / Mesh Models

| Model | License | Commercial use | Notes |
|-------|---------|---------------|-------|
| Ultralytics YOLO26-pose | AGPL-3.0 | Commercial license available | Default 2D pose. AGPL triggers for networked services. |
| RTMPose (MMPose) | Apache-2.0 | Yes | Alternative 2D pose. |
| MotionBERT | MIT | Yes | Temporal 3D lifting. ICCV 2023. |
| HMR 2.0 / 4DHumans | MIT | Yes | SMPL mesh + tracking. |
| PHALP | MIT | Yes | 3D-aware tracking. |
| SMPL body model | Max Planck Institute | Registration required | Needed by HMR 2.0. Register at smplify.is.tue.mpg.de. |
| Meta Sapiens2 | CC-BY-NC 4.0 | No | Optional. ICLR 2026. |
| SMPLer-X | Research | Check | Optional expressive bodies. |

## Core Dependencies

| Dependency | License |
|-----------|---------|
| PyTorch | BSD-style |
| Rerun SDK | Apache-2.0 |
| Rerun web viewer | MIT |
| gsplat | Apache-2.0 |
| Three.js | MIT |
| FastAPI | MIT |
| OpenCV | Apache-2.0 |
| NumPy | BSD-3 |
| Pillow | HPND |
| SciPy | BSD-3 |
| tqdm | MIT |
| ONNX Runtime | MIT |
| Hugging Face Hub | Apache-2.0 |
| ffmpegcv | MIT |
| Electron | MIT |
| React | MIT |
| Vite | MIT |

## Important Notes

1. **SHARP is non-commercial.** If you use Splatline commercially, select
   TripoSplat as your backend.
2. **Ultralytics is AGPL-3.0.** If you deploy Splatline as a networked
   service, you must open-source your service code or purchase an Ultralytics
   commercial license. For local desktop use, AGPL does not trigger.
3. **SMPL model requires registration.** Download from smplify.is.tue.mpg.de
   after registering. Place `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in
   the HMR 2.0 data directory.
4. **Sapiens2 is CC-BY-NC.** Optional backend for highest-accuracy human
   pose. Not for commercial use.
