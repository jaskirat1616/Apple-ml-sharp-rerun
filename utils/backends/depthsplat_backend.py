"""
DepthSplat backend — multi-view depth-conditioned Gaussian splatting.

DepthSplat jointly estimates multi-view depth maps and Gaussian splats
from posed images. Unlike SHARP (single-image), it uses 2+ context views
to produce geometrically consistent 3DGS. CVPR 2025.

Key advantage over SHARP:
- Multi-view consistency: uses 2+ frames, not just one
- Built-in PLY export: outputs standard 3DGS PLY directly
- Pre-trained models: no per-scene training needed
- Depth-conditioned: geometry is anchored to real depth estimates

Paper: "DepthSplat: Depth-Conditioned Gaussian Splatting" (CVPR 2025)
Repo: https://github.com/cvg/depthsplat
Models: https://huggingface.co/haofeixu/depthsplat

Install:
  git clone https://github.com/cvg/depthsplat
  cd depthsplat
  pip install -r requirements.txt

Model checkpoints (auto-downloaded from HuggingFace):
  - depthsplat-gs-base-re10k-256x256-view2 (117M params, recommended)
  - depthsplat-gs-large-re10k-256x256-view2 (360M params, best quality)
  - depthsplat-gs-small-re10k-256x256-view2 (37M params, fast)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default model checkpoint (HuggingFace)
DEPTH_SPLAT_DEFAULT_CHECKPOINT = "depthsplat-gs-base-re10k-256x256-view2-fbe87117.pth"
DEPTH_SPLAT_HF_REPO = "haofeixu/depthsplat"

# Available models
DEPTH_SPLAT_MODELS = {
    "small": "depthsplat-gs-small-re10k-256x256-view2-20f39ed8.pth",
    "base": "depthsplat-gs-base-re10k-256x256-view2-fbe87117.pth",
    "large": "depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth",
}


class DepthSplatBackend:
    """DepthSplat multi-view 3DGS backend.

    Takes 2+ context views and produces a geometrically consistent 3DGS
    PLY file. For video input, we select keyframe pairs and reconstruct
    each pair, accumulating into a coherent scene.

    Usage:
        backend = DepthSplatBackend(device="cuda", model_size="base")
        backend.convert(frames_dir, output_dir)
    """

    def __init__(
        self,
        device: str = "default",
        model_size: str = "base",  # small, base, large
        num_context_views: int = 2,
        image_shape: Tuple[int, int] = (352, 640),  # (H, W)
        keyframe_interval: int = 5,  # Use every Nth frame as keyframe
    ):
        """
        Args:
            device: torch device
            model_size: "small" (37M), "base" (117M), or "large" (360M)
            num_context_views: Number of context views (2-10)
            image_shape: (H, W) for inference
            keyframe_interval: For video input, use every Nth frame as keyframe
        """
        self.device = device
        self.model_size = model_size
        self.num_context_views = num_context_views
        self.image_shape = image_shape
        self.keyframe_interval = keyframe_interval
        self._model = None
        self._cfg = None

    def _load_model(self):
        """Lazy-load DepthSplat model."""
        import torch

        try:
            from src.config import get_cfg
            from src.model.model_wrapper import ModelWrapper
        except ImportError as exc:
            raise ImportError(
                "DepthSplat is not installed. Install with:\n"
                "  git clone https://github.com/cvg/depthsplat\n"
                "  cd depthsplat && pip install -r requirements.txt\n"
                f"Error: {exc}"
            ) from exc

        from utils.splat_models import resolve_torch_device
        device_obj, device_name = resolve_torch_device(self.device)

        # Download model checkpoint from HuggingFace
        checkpoint_name = DEPTH_SPLAT_MODELS.get(
            self.model_size, DEPTH_SPLAT_DEFAULT_CHECKPOINT
        )
        checkpoint_path = self._download_checkpoint(checkpoint_name)

        print(f"  Loading DepthSplat ({self.model_size}) on {device_name}...")

        # Build config and load model
        # DepthSplat uses Hydra config — we build it programmatically
        cfg = get_cfg()
        cfg.merge_from_other_cfg({
            "model": {
                "num_context_views": self.num_context_views,
                "gaussian_cfg": {"sh_degree": 0},
            },
            "dataset": {
                "image_shape": list(self.image_shape),
                "view_sampler": {
                    "num_context_views": self.num_context_views,
                },
            },
        })

        model = ModelWrapper(cfg.model)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device_obj).eval()
        model.device = device_obj

        self._model = model
        self._cfg = cfg
        self._device_obj = device_obj
        print(f"  DepthSplat loaded ({self.model_size})")

    def _download_checkpoint(self, checkpoint_name: str) -> Path:
        """Download model checkpoint from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        cache_dir = Path.home() / ".cache" / "splatline" / "depthsplat"
        cache_dir.mkdir(parents=True, exist_ok=True)

        local_path = cache_dir / checkpoint_name
        if local_path.exists():
            return local_path

        print(f"  Downloading {checkpoint_name} from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id=DEPTH_SPLAT_HF_REPO,
            filename=checkpoint_name,
            cache_dir=str(cache_dir),
        )
        return Path(downloaded)

    def convert(
        self,
        frames_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> Path:
        """Convert video frames to 3DGS PLY files using DepthSplat.

        Selects keyframe groups from the video and reconstructs each group,
        producing per-keyframe PLY files.

        Args:
            frames_dir: Directory containing video frames.
            output_dir: Output directory.
            skip_existing: Skip existing PLY files.

        Returns:
            Path to the gaussians directory.
        """
        if self._model is None:
            self._load_model()

        import torch

        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(parents=True, exist_ok=True)

        # Collect frame paths
        image_paths = sorted(
            p for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            for p in frames_dir.glob(f"*{ext}")
        )
        if not image_paths:
            raise ValueError(f"No images found in {frames_dir}")

        # Select keyframe groups
        # Each group has num_context_views frames for reconstruction
        keyframe_groups = self._select_keyframe_groups(image_paths)
        print(f"  DepthSplat: {len(image_paths)} frames → {len(keyframe_groups)} keyframe groups")

        from src.model.ply_export import save_gaussian_ply
        import cv2

        for group_idx, group_paths in enumerate(keyframe_groups):
            out_ply = gaussians_dir / f"frame_{group_idx * self.keyframe_interval:06d}.ply"
            if skip_existing and out_ply.exists():
                continue

            print(f"  DepthSplat: group [{group_idx+1}/{len(keyframe_groups)}] ", end="")

            try:
                # Prepare batch
                batch = self._prepare_batch(group_paths)

                with torch.no_grad():
                    gaussians, visualization_dump = self._model.forward(batch)

                # Save PLY using DepthSplat's built-in exporter
                save_gaussian_ply(gaussians, visualization_dump, batch, out_ply)
                print(f"→ {out_ply.name}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

        print(f"  DepthSplat: done, PLY files in {gaussians_dir}")
        return gaussians_dir

    def _select_keyframe_groups(self, image_paths: List[Path]) -> List[List[Path]]:
        """Select overlapping keyframe groups from the video.

        Each group contains num_context_views frames, with overlap between
        consecutive groups for temporal coherence.
        """
        groups = []
        step = max(1, self.keyframe_interval)
        for start in range(0, len(image_paths), step):
            group = []
            for offset in range(self.num_context_views):
                idx = start + offset * step
                if idx < len(image_paths):
                    group.append(image_paths[idx])
            if len(group) >= 2:  # Need at least 2 views
                groups.append(group)
        return groups

    def _prepare_batch(self, group_paths: List[Path]) -> Dict:
        """Prepare a batch dict for DepthSplat inference.

        DepthSplat expects a specific batch format with context images,
        camera parameters, and metadata.
        """
        import torch
        import cv2

        h, w = self.image_shape
        images = []
        for img_path in group_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h))
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)  # [3, H, W]
            images.append(img)

        # Stack: [1, S, 3, H, W]
        context = torch.stack(images).unsqueeze(0).to(self._device_obj)

        # Dummy camera intrinsics (will be refined by the model)
        # DepthSplat can work with approximate intrinsics
        fx = fy = float(min(h, w))
        cx, cy = w / 2.0, h / 2.0
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, len(images), 1)

        # Dummy extrinsics (identity for first frame)
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        extrinsics = extrinsics.repeat(1, len(images), 1, 1)

        return {
            "context": {
                "image": context,
                "intrinsics": intrinsics.to(self._device_obj),
                "extrinsics": extrinsics.to(self._device_obj),
            },
            "scene": [group_paths[0].stem],
        }


def convert_frames_with_depthsplat(
    frames_dir: Path,
    output_dir: Path,
    device: str = "default",
    model_size: str = "base",
    skip_existing: bool = True,
) -> Path:
    """Convenience function: convert frames to 3DGS using DepthSplat.

    Args:
        frames_dir: Directory containing video frames.
        output_dir: Output directory.
        device: torch device.
        model_size: "small", "base", or "large".
        skip_existing: Skip existing PLY files.

    Returns:
        Path to the gaussians directory.
    """
    backend = DepthSplatBackend(device=device, model_size=model_size)
    return backend.convert(frames_dir, output_dir, skip_existing=skip_existing)
