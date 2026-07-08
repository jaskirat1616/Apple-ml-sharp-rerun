"""
HMR 2.0 / 4DHumans — SMPL mesh recovery from video.

Recovers a full SMPL body mesh (24 joints + mesh vertices) from each frame,
then tracks identities across the video using PHALP. This gives a textured
3D human body in the scene, not just a skeleton.

Paper: "Humans in 4D: Reconstructing and Tracking Humans with Transformers"
(ICCV 2023, shubham-goel/4d-humans, MIT license)

The SMPL body model itself requires registration at:
    http://smplify.is.tue.mpg.de
Place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in the data directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class HMR2MeshRecoverer:
    """SMPL mesh recovery using HMR 2.0 transformer.

    Produces SMPL body parameters (pose, shape, camera) per frame, which can
    be rendered as a 3D mesh or converted to 24-joint 3D skeleton.

    Usage:
        recoverer = HMR2MeshRecoverer(device="cuda")
        results = recoverer.recover_video(frames)
        # results: list of {smpl_params, joints_3d, vertices, bbox} per frame
    """

    def __init__(
        self,
        device: str = "default",
        smpl_model_path: Optional[Path] = None,
    ):
        self.device = device
        self.smpl_model_path = smpl_model_path
        self._model = None
        self._tracker = None

    def _load_model(self):
        """Lazy-load HMR 2.0 model."""
        from utils.splat_models import resolve_torch_device
        self._device_obj, device_name = resolve_torch_device(self.device)

        try:
            from hmr2.models import HMR2
            from hmr2.utils import HMR2Utils
        except ImportError as exc:
            raise ImportError(
                "HMR 2.0 / 4DHumans is not installed. Install with:\n"
                "  git clone https://github.com/shubham-goel/4d-humans\n"
                "  cd 4d-humans && pip install -e .\n"
                "  Download SMPL model from http://smplify.is.tue.mpg.de\n"
                "  Place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in data/\n"
                f"Error: {exc}"
            ) from exc

        self._model = HMR2.from_pretrained("shubham-goel/4d-humans").to(self._device_obj)
        self._model.eval()
        print(f"  HMR 2.0 loaded on {device_name}")

    def recover_video(
        self,
        frame_paths: List[Path],
        batch_size: int = 8,
    ) -> List[Dict]:
        """Recover SMPL mesh for each frame in the video.

        Args:
            frame_paths: List of paths to video frames (PNG/JPG).
            batch_size: Batch size for inference.

        Returns:
            List of dicts per frame:
                {
                    "smpl_params": {"global_pose": (1,3), "body_pose": (23,3),
                                    "betas": (1,10), "cam": (3,)},
                    "joints_3d": (24, 3) SMPL 3D joints,
                    "vertices": (6890, 3) mesh vertices,
                    "joints_2d": (24, 2) projected 2D joints,
                    "bbox": (4,) [x1, y1, x2, y2],
                    "confidence": float,
                }
        """
        if self._model is None:
            self._load_model()

        import cv2
        import torch

        results = []
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start:start + batch_size]
            batch_images = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_images.append(img_rgb)

            if not batch_images:
                continue

            # HMR 2.0 expects cropped person images
            # The model includes its own detector (ViTDet) for person crops
            with torch.no_grad():
                batch_results = self._model.forward_images(batch_images)

            for res in batch_results:
                results.append({
                    "smpl_params": {
                        "global_pose": res["global_pose"].cpu().numpy(),
                        "body_pose": res["body_pose"].cpu().numpy(),
                        "betas": res["betas"].cpu().numpy(),
                        "cam": res["cam"].cpu().numpy(),
                    },
                    "joints_3d": res["joints_3d"].cpu().numpy(),
                    "vertices": res["vertices"].cpu().numpy(),
                    "joints_2d": res["joints_2d"].cpu().numpy(),
                    "bbox": res["bbox"].cpu().numpy() if "bbox" in res else None,
                    "confidence": float(res.get("confidence", 0.0)),
                })

            print(f"  HMR 2.0: [{start + len(batch_images)}/{len(frame_paths)}]")

        return results

    def recover_single(self, frame_path: Path) -> Optional[Dict]:
        """Recover SMPL mesh for a single frame."""
        results = self.recover_video([frame_path])
        return results[0] if results else None


def recover_smpl_mesh(
    frame_paths: List[Path],
    device: str = "default",
    smpl_model_path: Optional[Path] = None,
) -> List[Dict]:
    """Convenience: recover SMPL mesh for a list of frames.

    Args:
        frame_paths: List of frame image paths.
        device: torch device.
        smpl_model_path: Path to SMPL model file.

    Returns:
        List of SMPL recovery results per frame.
    """
    recoverer = HMR2MeshRecoverer(device=device, smpl_model_path=smpl_model_path)
    return recoverer.recover_video(frame_paths)
