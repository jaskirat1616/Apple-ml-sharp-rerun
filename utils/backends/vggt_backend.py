"""
VGGT backend — geometry foundation model for video-to-3D.

VGGT (Visual Geometry Grounded Transformer) is a feed-forward model that
infers cameras, depth maps, point maps, and 3D tracks from 1-100s of images
in a single forward pass. CVPR 2025 Best Paper.

This is the most impactful reconstruction upgrade for Splatline because:
1. It replaces COLMAP entirely — no more slow SfM for camera poses
2. It gives dense depth maps for every frame — metric scale geometry
3. It gives a dense 3D point cloud — can be converted to Gaussian splats
4. It's feed-forward — no per-scene training needed

Paper: "VGGT: Visual Geometry Grounded Transformer" (CVPR 2025 Best Paper)
Repo: https://github.com/facebookresearch/vggt
Model: https://huggingface.co/facebook/VGGT-1B

Install:
  pip install -e .  (from vggt repo)
  # or
  pip install vggt

The adapter converts VGGT's point cloud output to per-frame 3DGS PLY files
that match Splatline's existing contract, so downstream visualization and
analysis code works unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class VGGTBackend:
    """VGGT geometry foundation model backend.

    Feed-forward inference: video frames → camera poses + depth + point cloud.
    Converts the dense point cloud to per-frame 3DGS PLY files.

    Usage:
        backend = VGGTBackend(device="cuda")
        backend.convert(video_frames_dir, output_dir)
    """

    def __init__(
        self,
        device: str = "default",
        model_name: str = "facebook/VGGT-1B",
        max_frames_per_pass: int = 50,
        confidence_threshold: float = 1.5,
    ):
        """
        Args:
            device: torch device ("default", "cuda", "mps", "cpu")
            model_name: HuggingFace model name
            max_frames_per_pass: VGGT processes frames in chunks. More frames
                = better global consistency but more memory. 50 is a good
                balance for 8GB+ GPUs.
            confidence_threshold: Minimum confidence score for point cloud
                filtering. Higher = fewer but more reliable points.
        """
        self.device = device
        self.model_name = model_name
        self.max_frames_per_pass = max_frames_per_pass
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._device_obj = None

    def _load_model(self):
        """Lazy-load VGGT model from HuggingFace."""
        import torch
        from utils.splat_models import resolve_torch_device
        self._device_obj, device_name = resolve_torch_device(self.device)

        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "VGGT is not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/vggt\n"
                "  cd vggt && pip install -e .\n"
                "  # or: pip install vggt\n"
                f"Error: {exc}"
            ) from exc

        # Use bfloat16 on Ampere+ GPUs, float16 otherwise
        if self._device_obj.type == "cuda":
            capability = torch.cuda.get_device_capability()[0]
            self._dtype = torch.bfloat16 if capability >= 8 else torch.float16
        else:
            self._dtype = torch.float32

        print(f"  Loading VGGT model ({self.model_name}) on {device_name}...")
        self._model = VGGT.from_pretrained(self.model_name).to(self._device_obj)
        self._model.eval()
        print(f"  VGGT loaded (dtype={self._dtype})")

    def convert(
        self,
        frames_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> Path:
        """Convert video frames to 3D point cloud PLY files using VGGT.

        Processes frames in chunks, accumulates the global point cloud, then
        writes per-frame PLY files by projecting the global cloud into each
        camera view.

        Args:
            frames_dir: Directory containing extracted video frames.
            output_dir: Output directory (gaussians/ subdirectory created).
            skip_existing: Skip frames that already have PLY files.

        Returns:
            Path to the gaussians directory.
        """
        if self._model is None:
            self._load_model()

        import torch
        from vggt.utils.load_fn import load_and_preprocess_images

        gaussians_dir = output_dir / "gaussians"
        gaussians_dir.mkdir(parents=True, exist_ok=True)

        # Collect frame paths
        image_paths = sorted(
            p for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            for p in frames_dir.glob(f"*{ext}")
        )
        if not image_paths:
            raise ValueError(f"No images found in {frames_dir}")

        print(f"  VGGT: processing {len(image_paths)} frames in chunks of {self.max_frames_per_pass}")

        # Process in chunks and accumulate global point cloud
        all_points = []
        all_colors = []
        all_cameras = []  # (extrinsic, intrinsic) per frame

        for start in range(0, len(image_paths), self.max_frames_per_pass):
            end = min(start + self.max_frames_per_pass, len(image_paths))
            chunk_paths = image_paths[start:end]

            print(f"  VGGT: frames [{start+1}-{end}/{len(image_paths)}]", end="  ")

            # Load and preprocess images
            images = load_and_preprocess_images([str(p) for p in chunk_paths])
            images = images.to(self._device_obj)

            # Forward pass
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self._dtype):
                    predictions = self._model(images)

            # Extract outputs
            # pose_enc: [1, S, 9] — camera pose encoding
            # depth: [1, S, H, W, 1] — per-frame depth
            # world_points: [1, S, H, W, 3] — 3D world coordinates per pixel
            # world_points_conf: [1, S, H, W] — confidence per point

            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], images.shape[-2:]
            )

            world_points = predictions["world_points"].squeeze(0).cpu().numpy()  # [S, H, W, 3]
            world_points_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()  # [S, H, W]

            # Load original images for colors
            import cv2
            for idx, img_path in enumerate(chunk_paths):
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                # Resize world points to match original image if needed
                points = world_points[idx]  # [H, W, 3]
                conf = world_points_conf[idx]  # [H, W]

                # Filter by confidence
                mask = conf > self.confidence_threshold
                if mask.sum() < 100:
                    continue

                valid_points = points[mask]  # [N, 3]

                # Get colors for valid points
                # Resize mask to original image size for color sampling
                if points.shape[:2] != (h, w):
                    mask_full = cv2.resize(
                        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    mask_full = mask

                # Sample colors at valid pixel locations
                ys, xs = np.where(mask_full)
                # Downsample if too many points
                if len(ys) > 50000:
                    indices = np.random.choice(len(ys), 50000, replace=False)
                    ys, xs = ys[indices], xs[indices]
                    valid_points = valid_points[indices]

                colors = img_rgb[ys, xs].astype(np.float32) / 255.0

                all_points.append(valid_points)
                all_colors.append(colors)
                all_cameras.append((
                    extrinsic[0, idx].cpu().numpy(),
                    intrinsic[0, idx].cpu().numpy(),
                ))

            print(f"done ({len(chunk_paths)} frames)")

        if not all_points:
            raise RuntimeError("VGGT produced no valid 3D points")

        # Merge all points into global point cloud
        global_points = np.concatenate(all_points, axis=0)
        global_colors = np.concatenate(all_colors, axis=0)

        # Remove duplicates (points close together)
        print(f"  VGGT: {len(global_points)} raw points, deduplicating...")
        global_points, global_colors = self._deduplicate_points(global_points, global_colors)
        print(f"  VGGT: {len(global_points)} unique points after dedup")

        # Write per-frame PLY files
        # For each frame, project the global cloud into that camera's view
        # and write the visible subset as a PLY
        print(f"  VGGT: writing per-frame PLY files...")
        for idx, img_path in enumerate(image_paths):
            out_ply = gaussians_dir / f"{img_path.stem}.ply"
            if skip_existing and out_ply.exists():
                continue

            if idx < len(all_cameras):
                ext, int = all_cameras[idx]
                visible_points, visible_colors = self._filter_visible_points(
                    global_points, global_colors, ext, int
                )
            else:
                visible_points, visible_colors = global_points, global_colors

            self._write_ply(out_ply, visible_points, visible_colors)
            print(f"  [{idx+1}/{len(image_paths)}] {img_path.name}: {len(visible_points)} points", end="\r")

        print(f"\n  VGGT: done, {len(image_paths)} PLY files written to {gaussians_dir}")
        return gaussians_dir

    def _deduplicate_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicate points using voxel grid downsampling."""
        # Voxel grid downsampling: quantize points to voxel grid, keep one per voxel
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(
            voxel_indices, axis=0, return_index=True
        )
        return points[unique_indices], colors[unique_indices]

    def _filter_visible_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter points visible from a given camera view.

        Projects points into the camera and keeps those in front of the
        camera and within the image bounds.
        """
        # Transform world points to camera space
        # extrinsic is [4, 4] world-to-camera transform
        ones = np.ones((len(points), 1), dtype=np.float32)
        points_h = np.hstack([points, ones])  # [N, 4]
        points_cam = (extrinsic @ points_h.T).T[:, :3]  # [N, 3]

        # Keep points in front of camera (positive Z in camera space)
        in_front = points_cam[:, 2] > 0
        if in_front.sum() == 0:
            return points, colors

        # Project to image plane
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        points_front = points_cam[in_front]
        colors_front = colors[in_front]

        u = (fx * points_front[:, 0] / points_front[:, 2] + cx).astype(int)
        v = (fy * points_front[:, 1] / points_front[:, 2] + cy).astype(int)

        # Keep points within reasonable image bounds
        # (VGGT uses 518px internally, so allow some margin)
        in_bounds = (u >= -50) & (u < 600) & (v >= -50) & (v < 600)

        return points_front[in_bounds], colors_front[in_bounds]

    def _write_ply(self, path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
        """Write a point cloud as a 3DGS-compatible PLY file.

        Writes a minimal 3DGS PLY with positions, colors, and default
        Gaussian parameters (scale, rotation, opacity) that Splatline's
        existing PLY loader can read.
        """
        n = len(positions)
        if n == 0:
            # Write empty PLY
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            )
            return

        # Default Gaussian parameters
        scales = np.ones((n, 3), dtype=np.float32) * 0.005  # Small default scale
        opacities = np.ones((n, 1), dtype=np.float32)
        rotations = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))  # Identity quaternion

        with path.open("w") as f:
            f.write("ply\n")
            f.write("format binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("property float red\n")
            f.write("property float green\n")
            f.write("property float blue\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("property float opacity\n")
            f.write("end_header\n")

            # Build vertex array
            normals = np.zeros((n, 3), dtype=np.float32)
            vertex_data = np.hstack([
                positions.astype(np.float32),
                normals,
                colors.astype(np.float32),
                scales,
                rotations,
                opacities,
            ])
            vertex_data.tofile(f)


def convert_video_with_vggt(
    frames_dir: Path,
    output_dir: Path,
    device: str = "default",
    skip_existing: bool = True,
    max_frames_per_pass: int = 50,
) -> Path:
    """Convenience function: convert video frames to 3D using VGGT.

    Args:
        frames_dir: Directory containing extracted video frames.
        output_dir: Output directory.
        device: torch device.
        skip_existing: Skip existing PLY files.
        max_frames_per_pass: Frames per VGGT forward pass.

    Returns:
        Path to the gaussians directory.
    """
    backend = VGGTBackend(
        device=device,
        max_frames_per_pass=max_frames_per_pass,
    )
    return backend.convert(frames_dir, output_dir, skip_existing=skip_existing)
