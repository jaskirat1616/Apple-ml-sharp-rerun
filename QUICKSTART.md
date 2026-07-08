# Quick Start Guide

## 🚀 v2 in 60 Seconds

Splatline v2.0.0 adds a pluggable backend registry, a tiered human pipeline, and a FastAPI + SSE backend. v1 commands still work unchanged.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the v2 core: FastAPI + uvicorn + sse-starlette (backend server), Hugging Face Hub (model caching), ONNX Runtime (inference optimization), and Ultralytics (YOLO26-pose). Optional backends (TripoSplat, MotionBERT, HMR 2.0) install from source — see `requirements.txt` comments.

### 2. First-Run Backend Selection

On first run, Splatline prompts you to pick a reconstruction backend. The choice persists in `~/.splatline/config.json`.

- **`sharp`** — Apple SHARP, non-commercial research only (default for research).
- **`triposplat`** — MIT-licensed, fully open, commercial-safe.

You can skip the prompt by passing `--splat-backend` directly (see step 3).

### 3. Run an Analysis with the v2 Flags

```bash
# SHARP backend + MotionBERT skeleton tier (v2 defaults)
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend sharp \
  --human-tier skeleton \
  --device mps \
  --view
```

```bash
# TripoSplat (MIT, commercial-safe) + full SMPL mesh via HMR 2.0
python scripts/sports/analyze_athlete_twin.py training.mp4 \
  --splat-backend triposplat \
  --human-tier mesh \
  --device mps \
  --view
```

New v2 flags:

| Flag | Choices | Default | Notes |
| --- | --- | --- | --- |
| `--splat-backend` | `sharp`, `triposplat`, `depthsplat`, `longsplat` | `sharp` | `sharp` is non-commercial; `triposplat` is MIT. |
| `--human-tier` | `skeleton`, `mesh`, `both` | `skeleton` | `skeleton`=MotionBERT, `mesh`=HMR 2.0 SMPL, `both`=both. |

See [docs/ATHLETE_TWIN.md](docs/ATHLETE_TWIN.md) for the full tiered pipeline docs and [docs/SPLAT_MODELS.md](docs/SPLAT_MODELS.md) for the backend registry.

### 4. Start the FastAPI Backend (Optional)

```bash
python ui/server.py
```

The v2 backend replaces the v1 stdlib HTTP server with FastAPI + Server-Sent Events for real-time progress streaming on long ML jobs. Auto OpenAPI docs are at `http://localhost:8000/docs`.

Key endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/config` | Get current Splatline config (backend, tier) |
| `POST` | `/api/config` | Update config |
| `GET` | `/api/backends` | List available reconstruction backends |
| `GET` | `/api/tiers` | List available human pipeline tiers |
| `POST` | `/api/jobs` | Create a new analysis job (multipart upload) |
| `GET` | `/api/jobs/{id}` | Get job status + outputs |
| `GET` | `/api/jobs/{id}/stream` | SSE stream of live logs |

---

## 🚀 Getting Started in 30 Seconds (v1)

The v1 commands below still work unchanged.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Visualize a Single 3D Scene

```bash
python scripts/visualizers/visualize_with_rerun.py -i output_test/IMG_4707.ply --size 2.0
```

### 3. Complete Video Viewer (Everything at Once)

```bash
python scripts/visualizers/video_complete_viewer.py -i output_video/gaussians/ --max-frames 30
```

### 4. Build Navigation Map

```bash
python scripts/navigation/build_navigation_map.py -i output_test/IMG_4707.ply --resolution 0.5
```

## 📂 Project Structure

```
scripts/
├── converters/    # Video → 3D conversion
├── visualizers/   # 3D visualization
├── navigation/    # Navigation & SLAM tools
└── creative/      # Effects & composition

utils/             # Reusable utility modules
tests/             # Test scripts
configs/           # Configuration files
```

## 🎮 3D Viewer Controls

- **Rotate**: Left click + drag
- **Pan**: Right click + drag (primary)
- **Zoom**: Mouse wheel / Trackpad scroll
- **Reset**: Double click

## 💡 Common Workflows

### Video to 3D Pipeline

1. **Convert video to 3D:**
   ```bash
   python scripts/converters/video_to_3d_high_quality.py video.mp4 mps
   ```

2. **View results:**
   ```bash
   python scripts/visualizers/video_complete_viewer.py -i output_video/gaussians/
   ```

### Navigation Analysis Pipeline

1. **Build navigation map:**
   ```bash
   python scripts/navigation/build_navigation_map.py -i scene.ply --resolution 0.5
   ```

2. **Plan path (if needed):**
   ```bash
   python scripts/navigation/build_navigation_map.py -i scene.ply \
       --plan-path --start 0 0 --goal 50 50
   ```

### Creative Effects Pipeline

1. **Apply depth effects:**
   ```bash
   python scripts/creative/apply_depth_effects.py -i scene.ply --effect fog
   ```

2. **Create camera path:**
   ```bash
   python scripts/creative/create_camera_path.py -i scene.ply --path orbit
   ```

## 🔧 Using Utility Modules

```python
from utils import (
    load_gaussian_data,
    render_depth_map,
    extract_ground_plane,
    find_free_paths,
    setup_complete_viewer_blueprint
)

# Load PLY file
data = load_gaussian_data("scene.ply")

# Render depth map
depth_map, depth_colored = render_depth_map(
    data['positions'],
    data['colors'],
    resolution=(1280, 720)
)

# Extract ground plane
ground_points, ground_mask, _, _ = extract_ground_plane(data['positions'])
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore utility modules in `utils/`
- Check out example scripts in `scripts/`

## 🆘 Troubleshooting

**Import errors?**
- Make sure you're running scripts from the project root
- Scripts automatically add project root to Python path

**Missing dependencies?**
- Run: `pip install -r requirements.txt`
- Make sure ML-SHARP core library is installed separately

**Can't find output files?**
- Check `output_*/` directories
- Use `--help` flag to see all options

