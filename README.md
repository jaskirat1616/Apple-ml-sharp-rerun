# Splatline

Convert 2D videos and photos into interactive 3D scenes using pluggable Gaussian-splatting backends, Rerun, and the SuperSplat editor. MIT-licensed. Runs on macOS — no CUDA required.

> **v2.0.0** adds three new reconstruction backends (VGGT, DepthSplat, LongSplat) alongside SHARP and TripoSplat, a tiered human motion pipeline, an Electron desktop player with true Gaussian splat rendering, and a FastAPI + SSE backend. See [CHANGELOG.md](CHANGELOG.md).

## Demo

![Demo Video Preview](docs/assets/demo_preview.gif)

**[Download the full demo video](docs/assets/demo_video.mov)** | [Thumbnail](docs/assets/demo_thumbnail.jpg)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install sharp
npm install  # for the Electron player

# Convert a video to 3D Gaussian splats
python run_video_3d.py

# View in Rerun (point cloud or solid mesh)
python run_video_3d.py          # point cloud mode
python run_video_3d.py solid    # solid mesh mode

# Or view in the Electron desktop player (true splat rendering)
python run_video_electron.py --output-dir output_grok_3d

# Or in the browser-based SuperSplat editor
python run_video_splat.py --output-dir output_grok_3d
```

The SHARP model (~2.5GB) downloads automatically on first run.

**3D viewer controls:** Left-drag to rotate, right-drag to pan, scroll to zoom, double-click to reset.

---

## Reconstruction Backends

Five backends, swappable with `--splat-backend <name>`:

| Backend | License | Type | Key Strength |
|---------|---------|------|-------------|
| **SHARP** (Apple) | Non-commercial | Per-frame | Fast single-image 3DGS (v1 default) |
| **TripoSplat** | MIT | Per-frame | Commercial-safe single-image 3DGS |
| **VGGT** (CVPR 2025 Best Paper) | MIT code / CC-BY-NC checkpoint | Video-native | Camera poses + dense depth in <1s. Replaces COLMAP |
| **DepthSplat** (CVPR 2025) | MIT | Video-native | Multi-view depth-conditioned splatting |
| **LongSplat** (ICCV 2025) | NVlabs | Video-native | Single coherent scene with temporal consistency — no flickering |

```bash
python run_video_3d.py --splat-backend longsplat
```

See [docs/SPLAT_MODELS.md](docs/SPLAT_MODELS.md) for details.

### What v2 fixes

| v1 problem | v2 solution |
|---|---|
| Per-frame splats → flickering | LongSplat: one coherent scene |
| No camera poses (COLMAP needed) | VGGT: feed-forward poses in <1s |
| Single-image only | DepthSplat: 2+ views for geometric consistency |
| No global geometry | VGGT: dense depth + point cloud from whole video |

---

## Human Pipeline

Tiered human reconstruction with `--human-tier skeleton|mesh|both`:

- **Skeleton:** MotionBERT (ICCV 2023) — temporal 3D pose lifting across 243 frames, fused with splat depth for metric scale
- **Mesh:** HMR 2.0 / 4DHumans — SMPL body mesh recovery + PHALP identity tracking
- **2D detection:** YOLO26-pose (default) or RTMPose (Apache-2.0)

See [docs/ATHLETE_TWIN.md](docs/ATHLETE_TWIN.md).

---

## Viewers

### Electron Desktop Player (`run_video_electron.py`)

Native desktop app with true Gaussian splat rendering via the [SuperSplat](https://github.com/playcanvas/supersplat) (PlayCanvas) engine:

```bash
python run_video_electron.py --output-dir output_grok_3d
```

- GPU-accelerated splat compositing (not point clouds)
- 2D source video synced side-by-side at native 24fps
- Smooth timeline playback with on-demand frame loading and preloading
- Flash-free frame swapping (overlapping entity destruction)
- Voxel-grid + opacity subsampling to 600K splats for smooth real-time playback

**Requirements:** Node.js, `npm install`, pre-existing 3D output.

### SuperSplat Web Viewer (`run_video_splat.py`)

Same rendering engine, runs in a browser:

```bash
python run_video_splat.py --output-dir output_grok_3d
```

### Rerun Viewers

| Script | Description |
|--------|-------------|
| `run_video_3d.py` | Video → 3D pipeline + Rerun viewer (point cloud or solid mesh) |
| `run_image_3d.py` | Image → high-res solid 3D mesh (up to 11M vertices) |
| `run_slam_3d.py` | Monocular SLAM 3D mapping with camera trajectory |
| `run_splat_viewer.py` | View any PLY as splats in Rerun |

```bash
# Image to 3D mesh
python run_image_3d.py photo.jpg --res 1024

# SLAM mapping
python run_slam_3d.py video.mp4

# View a PLY file
python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
```

---

## FastAPI Backend

```bash
python ui/server.py
```

FastAPI + SSE backend with real-time progress streaming, Pydantic validation, and OpenAPI docs at `/docs`. Endpoints: `/api/config`, `/api/backends`, `/api/tiers`, `/api/jobs/{id}/stream` (SSE).

---

## Project Structure

```
splatline/
├── run_video_3d.py             # Video → 3D (point cloud or solid mesh)
├── run_image_3d.py             # Image → high-res solid 3D mesh
├── run_slam_3d.py              # Monocular SLAM 3D mapping
├── run_splat_viewer.py         # View any PLY as splats in Rerun
├── run_video_splat.py          # SuperSplat editor video viewer (web)
├── run_video_electron.py       # Electron desktop video player
├── ui/server.py                # FastAPI + SSE backend
├── electron/                   # Electron app (main, video-main, preload)
├── src/                        # React + Vite frontend
├── utils/
│   ├── backends/               # VGGT, DepthSplat, LongSplat backends
│   ├── human/                  # MotionBERT, HMR 2.0, PHALP tracking
│   ├── splat_models.py         # Backend registry
│   └── ...                     # Navigation, pathfinding, visualization
├── scripts/                    # v1 converters, visualizers, creative tools
├── docs/                       # SPLAT_MODELS.md, ATHLETE_TWIN.md
├── package.json                # Node.js / Electron config
└── requirements.txt            # Python dependencies
```

---

## Setup

### Python

```bash
pip install -r requirements.txt
pip install sharp  # Apple's 3DGS model (auto-downloads ~2.5GB on first run)
```

Or install SHARP from source: [apple/ml-sharp](https://github.com/apple/ml-sharp)

### Electron (optional, for desktop player)

```bash
npm install
```

### Verify

```bash
python -c "import rerun; import numpy; import torch; import sharp; print('OK')"
```

### System Requirements

- Python 3.8+, Node.js (for Electron)
- macOS, Linux, or Windows
- 8GB RAM minimum (16GB recommended)
- GPU optional: CUDA, MPS (Apple Silicon), or CPU

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'sharp'`**
→ `pip install sharp` or install from [source](https://github.com/apple/ml-sharp)

**Rerun viewer is blank / version mismatch**
→ `pip install rerun-sdk==0.23.1` to match your viewer, or update the viewer to match the SDK.

**Out of memory during processing**
→ Use `--max-frames 10` or lower resolution videos.

**GPU not detected**
→ CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
→ Apple Silicon: MPS is automatic. Use `mps` device.
→ CPU: `python run_video_3d.py --device cpu` (slower but works)

---

## Credits

- [Rerun](https://github.com/rerun-io/rerun) — visualization SDK (Apache-2.0)
- [Apple ML-SHARP](https://github.com/apple/ml-sharp) — monocular 3DGS model
- [SuperSplat](https://github.com/playcanvas/supersplat) — Gaussian splat editor (PlayCanvas engine)
- [VGGT](https://github.com/facebookresearch/vggt) — geometry foundation model
- [DepthSplat](https://github.com/autonomousvision/depthsplat) — multi-view depth-conditioned splatting
- [LongSplat](https://github.com/nvlabs/LongSplat) — video-native coherent 3DGS
- [MotionBERT](https://github.com/Walter0807/MotionBERT) — temporal 3D pose lifting
- [HMR 2.0](https://github.com/shubham-goel/4D-Humans) — SMPL body mesh recovery

See [NOTICE.md](NOTICE.md) for the full third-party license breakdown.

---

## License

**Splatline** is MIT-licensed. Some backends have different licenses:

- **SHARP** — non-commercial research only. Use TripoSplat (MIT) for commercial work.
- **VGGT checkpoint** — CC-BY-NC (code is MIT)
- **LongSplat** — check NVlabs terms before commercial use
- **YOLO26-pose** — AGPL-3.0 (local desktop use is fine; networked services need a commercial license)

See [NOTICE.md](NOTICE.md) for details.
