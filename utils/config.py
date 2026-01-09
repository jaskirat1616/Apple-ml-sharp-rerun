"""
Configuration utilities and default settings.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ViewerConfig:
    """Configuration for 3D viewers."""
    point_size_multiplier: float = 1.0
    opacity_threshold: float = 0.1
    downsample: float = 1.0
    rotation_degrees: float = 0.0
    default_focal_length: float = 1000.0
    max_depth: float = 100.0


@dataclass
class NavigationConfig:
    """Configuration for navigation tools."""
    obstacle_height: float = 0.5
    grid_resolution: float = 0.5
    ground_threshold: float = 0.3
    ground_grid_size: float = 5.0
    cluster_distance: float = 2.0
    min_obstacle_height: float = 0.5


@dataclass
class DepthConfig:
    """Configuration for depth rendering."""
    default_resolution: Tuple[int, int] = (1280, 720)
    focal_length: float = 1000.0
    max_depth: float = 100.0
    sampling_step: int = 200  # Sample every Nth pixel for point cloud conversion


@dataclass
class ConversionConfig:
    """Configuration for video to 3D conversion."""
    output_fps: int = 30
    output_quality: str = "high"  # "low", "medium", "high"
    max_frames: Optional[int] = None
    skip_frames: int = 1


# Default configurations
DEFAULT_VIEWER_CONFIG = ViewerConfig()
DEFAULT_NAVIGATION_CONFIG = NavigationConfig()
DEFAULT_DEPTH_CONFIG = DepthConfig()
DEFAULT_CONVERSION_CONFIG = ConversionConfig()


def get_output_dir(base_dir: Path, video_name: Optional[str] = None) -> Path:
    """
    Get standardized output directory path.
    
    Args:
        base_dir: Base output directory
        video_name: Optional video name for subdirectory
    
    Returns:
        Path to output directory
    """
    if video_name:
        return base_dir / f"output_{video_name}"
    return base_dir / "output"


def get_frames_dir(output_dir: Path) -> Path:
    """Get frames directory path."""
    return output_dir / "frames"


def get_gaussians_dir(output_dir: Path) -> Path:
    """Get gaussians directory path."""
    return output_dir / "gaussians"


def get_json_dir(output_dir: Path) -> Path:
    """Get JSON data directory path."""
    return output_dir / "json"

