"""
Splatline v2 reconstruction backend adapters.

Each adapter wraps a state-of-the-art 2D-to-3D reconstruction method and
converts its output to Splatline's standard PLY contract.

Backends:
  - vggt:      VGGT geometry foundation model (CVPR 2025 Best Paper)
               Feed-forward camera poses + depth + point clouds from video frames
  - depthsplat: DepthSplat multi-view 3DGS (CVPR 2025)
               Built-in PLY export, pre-trained models
  - longsplat:  LongSplat video-native coherent 3DGS (ICCV 2025)
               Single coherent scene from full video via training
"""

from .vggt_backend import VGGTBackend, convert_video_with_vggt
from .depthsplat_backend import DepthSplatBackend, convert_frames_with_depthsplat
from .longsplat_backend import LongSplatBackend, convert_video_with_longsplat

__all__ = [
    "VGGTBackend",
    "convert_video_with_vggt",
    "DepthSplatBackend",
    "convert_frames_with_depthsplat",
    "LongSplatBackend",
    "convert_video_with_longsplat",
]
