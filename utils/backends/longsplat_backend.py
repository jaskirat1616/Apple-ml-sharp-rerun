"""
LongSplat backend — video-native coherent 3DGS from full video.

LongSplat produces a SINGLE coherent 3D Gaussian Splat scene from an
entire video, not independent per-frame splats. This is the biggest
reconstruction quality upgrade: no more flickering between frames.

Unlike SHARP (feed-forward per frame) and DepthSplat (feed-forward per
view group), LongSplat optimizes Gaussians across the whole video using
MASt3R for pose estimation and temporal consistency losses. This means
it requires training (optimization) per video, not just inference.

Paper: "LongSplat: Long-Sequence Gaussian Splatting" (ICCV 2025)
Repo: https://github.com/NVlabs/LongSplat

Install:
  git clone --recursive https://github.com/NVlabs/LongSplat.git
  cd LongSplat
  conda create -n longsplat python=3.10.13 cmake=3.14.0 -y
  conda activate longsplat
  conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install -r requirements.txt
  pip install submodules/simple-knn
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/fused-ssim

The adapter runs LongSplat via subprocess (it's a training pipeline, not
a Python library), then converts the output to per-frame PLY files that
match Splatline's contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class LongSplatBackend:
    """LongSplat video-native coherent 3DGS backend.

    Produces a single coherent 3DGS scene from a full video by optimizing
    Gaussians with temporal consistency. Runs via subprocess since LongSplat
    is a training pipeline, not a feed-forward model.

    Usage:
        backend = LongSplatBackend(
            longsplat_dir="/path/to/LongSplat",
            device="cuda",
            iterations=3000,
        )
        backend.convert(frames_dir, output_dir)
    """

    def __init__(
        self,
        longsplat_dir: Optional[Path] = None,
        device: str = "default",
        iterations: int = 3000,
        prune_ratio: float = 0.6,
        resize_width: int = 512,
        subsample_fps: float = 10.0,
    ):
        """
        Args:
            longsplat_dir: Path to LongSplat repository. If None, tries to
                find it in common locations or LONGSPLAT_DIR env var.
            device: torch device ("cuda" recommended, LongSplat needs GPU)
            iterations: Training iterations (more = better quality, slower)
            prune_ratio: Pruning ratio for convert_3dgs.py (0.6 recommended)
            resize_width: Resize frames to this width (512 recommended)
            subsample_fps: Subsample video to this FPS before training
        """
        self.device = device
        self.iterations = iterations
        self.prune_ratio = prune_ratio
        self.resize_width = resize_width
        self.subsample_fps = subsample_fps
        self._longsplat_dir = self._find_longsplat_dir(longsplat_dir)

    def _find_longsplat_dir(self, explicit: Optional[Path] = None) -> Path:
        """Find the LongSplat installation directory."""
        if explicit:
            p = Path(explicit)
            if p.exists() and (p / "train.py").exists():
                return p
            raise FileNotFoundError(f"LongSplat not found at {explicit}")

        # Check env var
        env_dir = os.environ.get("LONGSPLAT_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists() and (p / "train.py").exists():
                return p

        # Check common locations
        candidates = [
            Path.home() / "LongSplat",
            Path.home() / "repos" / "LongSplat",
            Path("/opt/LongSplat"),
            Path.cwd().parent / "LongSplat",
        ]
        for p in candidates:
            if p.exists() and (p / "train.py").exists():
                return p

        raise FileNotFoundError(
            "LongSplat installation not found. Set LONGSPLAT_DIR env var or "
            "pass longsplat_dir= to LongSplatBackend().\n"
            "Install: git clone --recursive https://github.com/NVlabs/LongSplat.git"
        )

    def convert(
        self,
        frames_dir: Path,
        output_dir: Path,
        video_path: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> Path:
        """Convert video frames to a coherent 3DGS scene using LongSplat.

        This runs the full LongSplat training pipeline:
        1. Prepare data in LongSplat's expected directory structure
        2. Run training (optimizes Gaussians with temporal consistency)
        3. Convert custom format to standard 3DGS PLY
        4. Copy output to Splatline's gaussians directory

        Args:
            frames_dir: Directory containing extracted video frames.
            output_dir: Output directory.
            video_path: Original video path (for FPS extraction). Optional.
            skip_existing: Skip if output already exists.

        Returns:
            Path to the gaussians directory.
        """
        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done
        final_ply = gaussians_dir / "longsplat_scene.ply"
        if skip_existing and final_ply.exists():
            print(f"  LongSplat: output exists at {final_ply}, skipping")
            return gaussians_dir

        # Step 1: Prepare data directory
        scene_name = frames_dir.parent.name or "splatline_scene"
        data_dir = self._longsplat_dir / "data" / scene_name
        images_dir = data_dir / "images"

        print(f"  LongSplat: preparing data in {data_dir}")
        self._prepare_data(frames_dir, images_dir, video_path)

        # Step 2: Run training
        model_dir = self._longsplat_dir / "outputs" / scene_name / "baseline"
        print(f"  LongSplat: training ({self.iterations} iterations)...")
        self._run_training(data_dir, model_dir)

        # Step 3: Convert to standard 3DGS
        converted_dir = model_dir / "converted_3dgs"
        print(f"  LongSplat: converting to standard 3DGS...")
        self._convert_to_3dgs(model_dir, converted_dir)

        # Step 4: Copy output
        converted_ply = converted_dir / "point_cloud.ply"
        if not converted_ply.exists():
            # Try alternative location
            converted_ply = converted_dir / f"iteration_{self.iterations}" / "point_cloud.ply"

        if converted_ply.exists():
            shutil.copy2(converted_ply, final_ply)
            print(f"  LongSplat: copied to {final_ply}")

            # Also create per-frame symlinks for Splatline compatibility
            # (downstream code expects per-frame PLY files)
            self._create_per_frame_links(frames_dir, gaussians_dir, final_ply)
        else:
            raise RuntimeError(
                f"LongSplat conversion failed — no PLY found at {converted_ply}"
            )

        print(f"  LongSplat: done, coherent scene at {gaussians_dir}")
        return gaussians_dir

    def _prepare_data(
        self,
        frames_dir: Path,
        images_dir: Path,
        video_path: Optional[Path],
    ) -> None:
        """Prepare data in LongSplat's expected directory structure.

        LongSplat expects: data/$SCENE/images/*.jpg
        Frames should be subsampled to ~10fps and resized to 512px width.
        """
        import cv2

        images_dir.mkdir(parents=True, exist_ok=True)

        # Get original FPS if video provided
        source_fps = 30.0
        if video_path and video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

        # Calculate frame skip for target subsample FPS
        frame_skip = max(1, int(source_fps / self.subsample_fps))

        # Collect and sort frame paths
        frame_paths = sorted(
            p for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            for p in frames_dir.glob(f"*{ext}")
        )

        count = 0
        for idx, frame_path in enumerate(frame_paths):
            if idx % frame_skip != 0:
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # Resize to target width
            h, w = img.shape[:2]
            if w > self.resize_width:
                new_h = int(h * self.resize_width / w)
                img = cv2.resize(img, (self.resize_width, new_h))

            # Save as JPEG (LongSplat expects .jpg)
            out_path = images_dir / f"frame_{count:06d}.jpg"
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

        print(f"  LongSplat: prepared {count} frames (subsampled to ~{self.subsample_fps}fps)")

    def _run_training(self, data_dir: Path, model_dir: Path) -> None:
        """Run LongSplat training via subprocess."""
        cmd = [
            sys.executable,
            str(self._longsplat_dir / "train.py"),
            "--source_path", str(data_dir),
            "--model_path", str(model_dir),
            "--iterations", str(self.iterations),
        ]

        if self.device != "default":
            cmd.extend(["--device", self.device])

        print(f"  LongSplat: $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self._longsplat_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  LongSplat stdout: {result.stdout[-2000:]}")
            print(f"  LongSplat stderr: {result.stderr[-2000:]}")
            raise RuntimeError(f"LongSplat training failed (exit {result.returncode})")

    def _convert_to_3dgs(self, model_dir: Path, converted_dir: Path) -> None:
        """Convert LongSplat custom format to standard 3DGS using convert_3dgs.py."""
        cmd = [
            sys.executable,
            str(self._longsplat_dir / "convert_3dgs.py"),
            "-m", str(model_dir),
            "--prune_ratio", str(self.prune_ratio),
        ]

        print(f"  LongSplat: $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self._longsplat_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  LongSplat convert stdout: {result.stdout[-2000:]}")
            print(f"  LongSplat convert stderr: {result.stderr[-2000:]}")
            raise RuntimeError(f"LongSplat conversion failed (exit {result.returncode})")

    def _create_per_frame_links(
        self,
        frames_dir: Path,
        gaussians_dir: Path,
        scene_ply: Path,
    ) -> None:
        """Create per-frame PLY symlinks pointing to the coherent scene.

        Splatline's downstream code expects per-frame PLY files. Since
        LongSplat produces a single coherent scene, we create symlinks
        from each frame name to the scene PLY.
        """
        frame_paths = sorted(
            p for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            for p in frames_dir.glob(f"*{ext}")
        )

        for frame_path in frame_paths:
            link_path = gaussians_dir / f"{frame_path.stem}.ply"
            if link_path.exists():
                continue
            try:
                # Symlink (Unix) — on Windows, copy
                if os.name == "nt":
                    shutil.copy2(scene_ply, link_path)
                else:
                    os.symlink(scene_ply, link_path)
            except OSError:
                shutil.copy2(scene_ply, link_path)


def convert_video_with_longsplat(
    frames_dir: Path,
    output_dir: Path,
    video_path: Optional[Path] = None,
    device: str = "default",
    iterations: int = 3000,
    skip_existing: bool = True,
) -> Path:
    """Convenience function: convert video to coherent 3DGS using LongSplat.

    Args:
        frames_dir: Directory containing extracted video frames.
        output_dir: Output directory.
        video_path: Original video path (for FPS extraction).
        device: torch device (cuda recommended).
        iterations: Training iterations.
        skip_existing: Skip if output exists.

    Returns:
        Path to the gaussians directory.
    """
    backend = LongSplatBackend(
        device=device,
        iterations=iterations,
    )
    return backend.convert(
        frames_dir, output_dir, video_path=video_path, skip_existing=skip_existing
    )
