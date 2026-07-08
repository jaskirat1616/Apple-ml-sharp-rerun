# Splat Models

Splatline v2 has a pluggable backend registry for splat reconstruction models. On first run, the backend selector prompts you to choose between **SHARP** (non-commercial research) and **TripoSplat** (MIT, fully open). The selection persists across sessions in `~/.splatline/config.json`.

The v1 default is Apple SHARP because it directly matches the Splatline contract: per-frame 3D Gaussian `.ply` output, OpenCV-style camera coordinates, and metric scale from a single image. For commercial use, select **TripoSplat** instead.

## Backend Registry

| Backend | License | Commercial OK | Status | Venue | Notes |
| --- | --- | --- | --- | --- | --- |
| `sharp` | apple-amlr (non-commercial) | No | default (research) | — | Per-frame 3DGS from a single image. Non-commercial research only. |
| `triposplat` | MIT | Yes | available | SIGGRAPH 2026 | Fully open single-image 3DGS with learned density control. |
| `longsplat` | NVlabs (check terms) | Check | tracked | ICCV 2025 | Video-native temporal coherence — one coherent 3DGS scene from the whole video instead of independent per-frame splats. |
| `splinegs` | MIT | Yes | tracked | CVPR 2025 | Dynamic monocular Gaussian splatting. |
| `depthsplat` | MIT | Yes | tracked | CVPR 2025 | Multi-view depth plus Gaussian splatting; can save `.ply`. |
| `vggt` | Custom (commercial checkpoint available) | Apply | tracked | CVPR 2025 Best Paper | Feed-forward geometry foundation for camera, depth, point maps, and tracks. |
| `noposplat` | — | — | research reference | — | Real-time Gaussian reconstruction from sparse unposed views. |
| `anysplat` | — | — | research reference | — | Unconstrained image collections with Gaussian, depth, and camera heads. |
| `volsplat` | — | — | research reference | — | Voxel-aligned Gaussian prediction for adaptive scene complexity. |

> **Commercial use:** pick a backend with `commercial_ok = Yes` (e.g. `triposplat`). SHARP is research-only. See [NOTICE.md](../NOTICE.md) for the full license breakdown.

## Current Default

`sharp` (research) / `triposplat` (commercial)

- Source: Apple `ml-sharp` / VAST-AI-Research `TripoSplat`
- Input: one image, or extracted video frames processed one at a time
- Output: 3DGS `.ply`
- Coordinates: OpenCV convention, X right, Y down, Z forward
- Device support: CPU, CUDA, and MPS for prediction
- SHARP default checkpoint: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`

```bash
# SHARP (non-commercial research, default)
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend sharp \
  --device mps \
  --sharp-internal-size 1536 \
  --pose-model yolo26x-pose.pt
```

Use the MIT-licensed TripoSplat backend (commercial-safe):

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend triposplat \
  --device mps
```

Use a local SHARP checkpoint:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend sharp \
  --sharp-checkpoint /path/to/sharp_2572gikvuh.pt
```

Use a different SHARP-compatible checkpoint URL:

```bash
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend sharp \
  --sharp-model-url https://example.com/custom_sharp.pt
```

## Tracked Research Backends

The registry also tracks research references that are not drop-in Splatline backends yet. The [Backend Registry](#backend-registry) table above lists them with their license and status; adapters can be added cleanly as each project stabilizes.

## Why SHARP Stays Default (Research)

For sports-science analysis, Splatline needs reliable per-frame splats that can be aligned with pose detections immediately. SHARP is still the best research default for that exact workflow because it works from ordinary monocular frames and writes standard PLY files. Multi-view methods may become better for longer clips, but they need additional work to preserve frame timing, camera alignment, and export compatibility.

For commercial deployments, use **TripoSplat** — it is MIT-licensed and produces the same per-frame 3DGS `.ply` output without the non-commercial restriction.

## Next Backend Adapter Target

The most useful next adapter is probably **DepthSplat** for multi-frame clips where several nearby frames can be grouped into a stronger reconstruction. It already exposes saved Gaussian `.ply` output in its project workflow, but it expects a separate CUDA research environment, so it should be integrated as an optional external backend rather than a required dependency.

