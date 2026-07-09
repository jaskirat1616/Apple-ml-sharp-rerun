# Splat Reconstruction Backends

Splatline v2 supports five 3D reconstruction backends, swappable with `--splat-backend <name>`. Each has different requirements, licenses, and hardware needs.

```bash
python run_video_3d.py --video video.mp4 --splat-backend <name>
```

## Quick Reference

| Backend | License | Hardware | Feed-forward | Install Difficulty |
|---------|---------|----------|-------------|-------------------|
| **sharp** | Non-commercial | CPU / MPS / CUDA | Yes | `pip install sharp` |
| **triposplat** | MIT | CPU / MPS / CUDA | Yes | Clone + pip install |
| **vggt** | MIT code / CC-BY-NC checkpoint | MPS / CUDA | Yes | `pip install vggt` |
| **depthsplat** | MIT | MPS / CUDA | Yes | Clone + pip install |
| **longsplat** | NVlabs | CUDA only | No (training) | Complex (CUDA submodules) |

---

## SHARP (default)

Apple's monocular 3DGS model. Fast, feed-forward, works from a single image. Non-commercial research only.

- **Repo:** https://github.com/apple/ml-sharp
- **Model:** Auto-downloads (~2.5GB) from `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
- **Device:** CPU, MPS (Apple Silicon), CUDA
- **License:** Apple AMLR — non-commercial research only

### Install

```bash
pip install sharp
```

That's it. The model weights download automatically on first run.

### Run

```bash
python run_video_3d.py --video video.mp4 --splat-backend sharp
python run_video_3d.py --video video.mp4 --splat-backend sharp --device mps
```

### Use a local checkpoint

```bash
python run_video_3d.py --video video.mp4 --splat-backend sharp --device mps
# Or set the checkpoint in code via load_sharp_predictor(checkpoint_path=...)
```

---

## TripoSplat

MIT-licensed single-image 3DGS. Commercial-safe alternative to SHARP. SIGGRAPH 2026.

- **Repo:** https://github.com/VAST-AI-Research/TripoSplat
- **Model:** Auto-downloads from HuggingFace (`VAST-AI-Research/TripoSplat`)
- **Device:** CPU, MPS, CUDA
- **License:** MIT — commercial OK

### Install

```bash
git clone https://github.com/VAST-AI-Research/TripoSplat.git
cd TripoSplat
pip install -e .
```

### Run

```bash
python run_video_3d.py --video video.mp4 --splat-backend triposplat
```

---

## VGGT

CVPR 2025 Best Paper. Feed-forward geometry foundation model — camera poses + dense depth + point cloud in under 1 second. Replaces COLMAP entirely.

- **Repo:** https://github.com/facebookresearch/vggt
- **Model:** Auto-downloads from HuggingFace (`facebook/VGGT-1B`, ~1GB)
- **Device:** MPS (Apple Silicon), CUDA. CPU works but is slow.
- **License:** MIT code, CC-BY-NC model checkpoint (non-commercial)

### Install

```bash
pip install vggt
```

Or from source:

```bash
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -e .
```

### Run

```bash
python run_video_3d.py --video video.mp4 --splat-backend vggt
python run_video_3d.py --video video.mp4 --splat-backend vggt --device mps
```

### How it works in Splatline

VGGT processes frames in chunks of 50 (configurable). For each chunk:
1. Forward pass → camera poses, depth maps, 3D world points
2. Points filtered by confidence score (>1.5 default)
3. Global point cloud deduplicated via voxel grid
4. Per-frame PLY written by projecting global cloud into each camera view

The output is per-frame 3DGS PLY files with positions, colors, and default Gaussian parameters (scale, rotation, opacity).

---

## DepthSplat

CVPR 2025. Multi-view depth-conditioned Gaussian splatting. Uses 2+ context views for geometrically consistent 3DGS (not single-image like SHARP).

- **Repo:** https://github.com/cvg/depthsplat
- **Model:** Auto-downloads from HuggingFace (`haofeixu/depthsplat`)
  - `small` (37M params, fast)
  - `base` (117M params, recommended)
  - `large` (360M params, best quality)
- **Device:** MPS (Apple Silicon), CUDA
- **License:** MIT — commercial OK

### Install

```bash
git clone https://github.com/cvg/depthsplat.git
cd depthsplat
pip install -r requirements.txt
```

### Run

```bash
python run_video_3d.py --video video.mp4 --splat-backend depthsplat
python run_video_3d.py --video video.mp4 --splat-backend depthsplat --device mps
```

### How it works in Splatline

DepthSplat selects overlapping keyframe groups from the video (every 5th frame by default, with 2 context views per group). Each group is reconstructed independently, producing per-keyframe PLY files. The keyframe interval and number of context views are configurable in `utils/backends/depthsplat_backend.py`.

---

## LongSplat

ICCV 2025. Video-native coherent 3DGS — produces a **single coherent scene** from the entire video using MASt3R pose estimation and temporal consistency losses. This is the only backend that solves flickering between frames.

- **Repo:** https://github.com/NVlabs/LongSplat
- **Model:** Trains from scratch per video (no pre-trained checkpoint)
- **Device:** **CUDA only** (uses `diff-gaussian-rasterization` which is CUDA-only). Does NOT work on MPS or CPU.
- **License:** NVlabs — check terms before commercial use
- **Type:** Training-based (optimizes per video, slower but temporally coherent)

### Install

```bash
git clone --recursive https://github.com/NVlabs/LongSplat.git
cd LongSplat
conda create -n longsplat python=3.10.13 cmake=3.14.0 -y
conda activate longsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim
```

### Run

```bash
# Set the LongSplat directory
export LONGSPLAT_DIR=~/LongSplat

# Run (CUDA required)
python run_video_3d.py --video video.mp4 --splat-backend longsplat --device cuda
```

Or pass the directory in code via `LongSplatBackend(longsplat_dir=...)`.

### How it works in Splatline

LongSplat is a training pipeline, not a feed-forward model. Splatline runs it via subprocess:
1. Frames are subsampled to ~10fps and resized to 512px width
2. LongSplat trains for 3000 iterations (configurable) with temporal consistency losses
3. The custom format is converted to standard 3DGS PLY via `convert_3dgs.py`
4. Per-frame symlinks are created pointing to the single coherent scene PLY

The result is a single coherent 3DGS scene — no flickering between frames. All downstream viewers (Rerun, Electron, web) work unchanged because the output is per-frame PLY files.

### Requirements

- **NVIDIA GPU with CUDA** — LongSplat uses CUDA-only rasterization submodules. It will not work on Apple Silicon (MPS) or CPU.
- ~8GB+ GPU memory for 512px frames
- Training takes minutes to hours depending on video length and iteration count

---

## Comparison

| Feature | SHARP | TripoSplat | VGGT | DepthSplat | LongSplat |
|---------|-------|-----------|------|-----------|-----------|
| Temporal coherence | No | No | Partial | Partial | **Yes** |
| Camera poses | No | No | **Yes** | No | Yes (MASt3R) |
| Dense depth | No | No | **Yes** | Yes | No |
| Multi-view | No | No | Yes | **Yes** | Yes (full video) |
| Feed-forward | Yes | Yes | Yes | Yes | No (training) |
| MPS support | Yes | Yes | Yes | Yes | **No** |
| CUDA required | No | No | No | No | **Yes** |
| Commercial OK | No | **Yes** | No (checkpoint) | **Yes** | Check |

## Choosing a Backend

- **Quick results on Mac:** SHARP (default, fast, non-commercial) or TripoSplat (MIT, commercial-safe)
- **Better geometry:** VGGT (camera poses + depth, replaces COLMAP) or DepthSplat (multi-view consistency)
- **No flickering:** LongSplat (single coherent scene, but requires NVIDIA GPU and training time)
- **Commercial use:** TripoSplat or DepthSplat (both MIT)

## Future Backends

Tracked but not yet implemented:

- **SplineGS** (CVPR 2025) — dynamic monocular Gaussian splatting. https://github.com/KAIST-VICLab/SplineGS
- **NoPoSplat** (ICLR 2025) — pose-free sparse view reconstruction. https://noposplat.github.io/
- **AnySplat** (SIGGRAPH Asia 2025) — unconstrained image collections. https://github.com/InternRobotics/AnySplat
- **VolSplat** (ECCV 2026) — voxel-aligned Gaussian prediction. https://github.com/ziplab/VolSplat
