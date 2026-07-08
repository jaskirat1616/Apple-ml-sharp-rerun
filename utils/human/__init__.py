"""
Splatline v2 human reconstruction pipeline — tiered architecture.

Tier 1 (skeleton): YOLO26-pose 2D → MotionBERT temporal 3D lifting + depth fusion
Tier 2 (mesh):     HMR 2.0 / 4DHumans SMPL mesh + PHALP 3D tracking

The key innovation: MotionBERT looks at the whole motion sequence (up to 243
frames) to lift 2D poses to 3D, producing smooth temporally consistent motion
instead of independent per-frame estimates. We fuse this with Gaussian splat
depth to recover metric scale — the advantage that pure monocular methods lack.
"""

from .motionbert import (
    MotionBERTLifter,
    lift_pose_sequence,
    coco_to_h36m,
    h36m_to_coco,
)
from .hmr2 import HMR2MeshRecoverer, recover_smpl_mesh
from .tracking import track_persons_3d, PersonTrack

__all__ = [
    "MotionBERTLifter",
    "lift_pose_sequence",
    "coco_to_h36m",
    "h36m_to_coco",
    "HMR2MeshRecoverer",
    "recover_smpl_mesh",
    "track_persons_3d",
    "PersonTrack",
]
