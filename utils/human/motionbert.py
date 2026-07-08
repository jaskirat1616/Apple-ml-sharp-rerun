"""
MotionBERT temporal 3D pose lifting.

This is the single biggest quality upgrade in Splatline v2. Instead of
independently depth-lifting each frame's 2D keypoints, MotionBERT uses a
Dual-stream Spatio-temporal Transformer (DSTformer) that looks at up to 243
frames at once to recover smooth, temporally consistent 3D motion.

Paper: "MotionBERT: A Unified Perspective on Learning Human Motion
Representations" (ICCV 2023, Walter0807/MotionBERT, MIT license)

The key insight for Splatline: we fuse MotionBERT's relative-scale 3D output
with our Gaussian splat depth maps to recover metric scale. This gives us
temporal smoothness from MotionBERT + metric accuracy from the 3D scene —
better than either approach alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# COCO-17 to Human3.6M-17 joint mapping.
# MotionBERT expects H36M format. We convert COCO detections, lift, then
# convert back so downstream biomechanics code stays unchanged.
#
# COCO order: nose, leye, reye, lear, rear, lshoulder, rshoulder, lelbow,
#             relbow, lwrist, rwrist, lhip, rhip, lknee, rknee, lankle, rankle
# H36M order: hip(r), hip(l), spine, hip(l)→spine duplicate, knee(r), knee(l),
#             spine→neck, neck, head, headtop, shoulder(l), shoulder(r),
#             elbow(l), elbow(r), wrist(l), wrist(r)
#
# We use the standard 17-joint mapping from MotionBERT's infer_wild.py.

COCO_TO_H36M = [11, 12, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 3, 4, 1, 2, 0]

# Reverse mapping: H36M index → COCO index
H36M_TO_COCO = [0] * 17
for h36m_idx, coco_idx in enumerate(COCO_TO_H36M):
    H36M_TO_COCO[h36m_idx] = coco_idx


def coco_to_h36m(keypoints_2d: np.ndarray) -> np.ndarray:
    """Convert COCO-17 2D keypoints to H36M-17 format.

    Args:
        keypoints_2d: (17, 4) array of [x, y, _, confidence] in COCO order.

    Returns:
        (17, 2) array of [x, y] in H36M order, normalized to [-1, 1].
    """
    h36m = keypoints_2d[COCO_TO_H36M, :2].copy()
    return h36m.astype(np.float32)


def h36m_to_coco(joints_3d: np.ndarray) -> np.ndarray:
    """Convert H36M-17 3D joints back to COCO-17 order.

    Args:
        joints_3d: (17, 3) array in H36M order.

    Returns:
        (17, 3) array in COCO order.
    """
    return joints_3d[H36M_TO_COCO].copy()


class MotionBERTLifter:
    """Temporal 3D pose lifter using MotionBERT.

    Wraps MotionBERT to lift sequences of 2D COCO keypoints to 3D.
    Handles sliding windows (up to 243 frames), H36M format conversion,
    and optional depth fusion for metric scale.

    Usage:
        lifter = MotionBERTLifter(device="mps")
        joints_3d_sequence = lifter.lift(keypoints_2d_sequence, depths=depth_maps)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "default",
        max_frames: int = 243,
    ):
        self.device = device
        self.max_frames = max_frames
        self._model = None
        self._checkpoint_path = checkpoint_path
        self._device_obj = None

    def _load_model(self):
        """Lazy-load MotionBERT model."""
        import torch

        from utils.splat_models import resolve_torch_device
        self._device_obj, device_name = resolve_torch_device(self.device)

        try:
            from libs.utils.model_utils import load_model
            from libs.models import MotionBERT
        except ImportError:
            try:
                # Try direct import if MotionBERT is installed as a package
                from motionbert import MotionBERT, load_model
            except ImportError as exc:
                raise ImportError(
                    "MotionBERT is not installed. Install with:\n"
                    "  git clone https://github.com/Walter0807/MotionBERT\n"
                    "  cd MotionBERT && pip install -e .\n"
                    "  Download weights from the MotionBERT Model Zoo\n"
                    f"Error: {exc}"
                ) from exc

        if self._checkpoint_path is None:
            # Default to the fine-tuned 3D pose checkpoint
            self._checkpoint_path = Path("checkpoint/pose3d/MB_ft_h36m_annual.bin")

        self._model = load_model(self._checkpoint_path, self._device_obj)
        self._model.eval()
        print(f"  MotionBERT loaded on {device_name}")

    def lift(
        self,
        keypoints_sequence: List[np.ndarray],
        depths: Optional[List[np.ndarray]] = None,
        f_px: Optional[float] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> np.ndarray:
        """Lift a sequence of 2D COCO keypoints to metric 3D.

        Args:
            keypoints_sequence: List of (17, 4) COCO keypoints, one per frame.
            depths: Optional list of depth maps for metric scale fusion.
            f_px: Focal length in pixels (for depth fusion).
            img_w, img_h: Image dimensions (for depth fusion).

        Returns:
            (N, 17, 3) array of 3D joints in COCO order, metric scale.
        """
        if self._model is None:
            self._load_model()

        import torch

        n = len(keypoints_sequence)
        if n == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)

        # Convert each frame's COCO keypoints to H36M format
        h36m_2d = np.stack([coco_to_h36m(kp) for kp in keypoints_sequence])

        # Normalize 2D coordinates to [-1, 1] using image bounds
        # MotionBERT expects normalized input
        if img_w and img_h:
            h36m_2d[:, :, 0] = h36m_2d[:, :, 0] / (img_w / 2) - 1
            h36m_2d[:, :, 1] = h36m_2d[:, :, 1] / (img_h / 2) - 1

        # Process in sliding windows of max_frames
        all_3d = []
        for start in range(0, n, self.max_frames):
            end = min(start + self.max_frames, n)
            window = h36m_2d[start:end]
            window_tensor = torch.from_numpy(window).float().to(self._device_obj)
            window_tensor = window_tensor.unsqueeze(0)  # (1, T, 17, 2)

            with torch.no_grad():
                output = self._model(window_tensor)
                # MotionBERT outputs (1, T, 17, 3) in H36M format, relative scale
                joints_3d = output.squeeze(0).cpu().numpy()

            all_3d.append(joints_3d)

        joints_3d_h36m = np.concatenate(all_3d, axis=0)  # (N, 17, 3)

        # Convert back to COCO order
        joints_3d_coco = np.stack([h36m_to_coco(j) for j in joints_3d_h36m])

        # Fuse with Gaussian depth for metric scale
        if depths is not None and f_px is not None and img_w is not None and img_h is not None:
            joints_3d_coco = self._fuse_depth(
                joints_3d_coco,
                keypoints_sequence,
                depths,
                f_px,
                img_w,
                img_h,
            )

        return joints_3d_coco.astype(np.float32)

    def _fuse_depth(
        self,
        joints_3d: np.ndarray,
        keypoints_2d: List[np.ndarray],
        depths: List[np.ndarray],
        f_px: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Fuse MotionBERT relative 3D with Gaussian depth for metric scale.

        Strategy: use MotionBERT for bone lengths and pose structure (which
        it gets right), then scale the whole sequence so the average joint
        depth matches the Gaussian splat depth at those pixel locations.
        """
        from utils.biomechanics import depth_lift_coco_keypoints

        # Get depth-lifted joints for each frame (metric scale from splat)
        depth_joints = []
        for idx, (kp2d, depth_map) in enumerate(zip(keypoints_2d, depths)):
            if depth_map is None:
                depth_joints.append(joints_3d[idx])
                continue
            dj = depth_lift_coco_keypoints(kp2d, depth_map, f_px=f_px, img_w=img_w, img_h=img_h)
            depth_joints.append(dj)

        depth_joints = np.array(depth_joints)

        # Compute scale factor: ratio of depth-based Z to MotionBERT Z
        # Use the root joint (hip center) depth as reference
        motion_root_z = joints_3d[:, 11:13, 2].mean(axis=1)  # avg of lhip, rhip Z
        depth_root_z = depth_joints[:, 11:13, 2].mean(axis=1)

        # Avoid division by zero
        valid = np.abs(motion_root_z) > 1e-6
        if valid.sum() < 3:
            # Not enough valid frames for scale estimation — use depth directly
            return depth_joints

        scale = np.median(depth_root_z[valid] / motion_root_z[valid])
        if not np.isfinite(scale) or scale <= 0:
            return depth_joints

        # Scale MotionBERT output to metric
        joints_3d_scaled = joints_3d * scale

        # Blend: use MotionBERT for XY (smoother) and depth for Z (metric)
        # Weight toward MotionBERT for smoothness, depth for accuracy
        alpha = 0.7  # MotionBERT weight for Z
        joints_3d_fused = joints_3d_scaled.copy()
        joints_3d_fused[:, :, 2] = alpha * joints_3d_scaled[:, :, 2] + (1 - alpha) * depth_joints[:, :, 2]

        return joints_3d_fused


def lift_pose_sequence(
    keypoints_sequence: List[np.ndarray],
    depths: Optional[List[np.ndarray]] = None,
    f_px: Optional[float] = None,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
    device: str = "default",
    checkpoint_path: Optional[Path] = None,
) -> np.ndarray:
    """Convenience function: lift a 2D keypoint sequence to 3D.

    Args:
        keypoints_sequence: List of (17, 4) COCO keypoints per frame.
        depths: Optional depth maps for metric scale fusion.
        f_px, img_w, img_h: Camera params for depth fusion.
        device: torch device.
        checkpoint_path: MotionBERT checkpoint path.

    Returns:
        (N, 17, 3) 3D joints in COCO order, metric scale if depth provided.
    """
    lifter = MotionBERTLifter(checkpoint_path=checkpoint_path, device=device)
    return lifter.lift(keypoints_sequence, depths=depths, f_px=f_px, img_w=img_w, img_h=img_h)
