"""
Utility modules for 3D Gaussian Splatting visualization and navigation.
"""

from .depth_rendering import render_depth_map, depth_map_to_3d_points
from .frame_processing import process_frame_complete, load_gaussian_data, load_video_frame
from .navigation import extract_ground_plane, detect_obstacles, compute_occupancy_grid_2d
from .pathfinding import find_free_paths
from .visualization import (
    setup_complete_viewer_blueprint,
    setup_navigation_viewer_blueprint,
    log_camera_transform,
    create_occupancy_grid_image
)
from .config import (
    ViewerConfig,
    NavigationConfig,
    DepthConfig,
    ConversionConfig,
    DEFAULT_VIEWER_CONFIG,
    DEFAULT_NAVIGATION_CONFIG,
    DEFAULT_DEPTH_CONFIG,
    DEFAULT_CONVERSION_CONFIG,
    get_output_dir,
    get_frames_dir,
    get_gaussians_dir,
    get_json_dir
)
from .io_utils import (
    load_json,
    save_json,
    load_image,
    save_image,
    find_ply_files,
    find_frame_images,
    get_frame_number
)
from .geometry import (
    rotation_matrix_y,
    rotate_points,
    scale_points,
    translate_points,
    compute_bounding_box,
    compute_center,
    normalize_points,
    compute_distances
)

__all__ = [
    # Depth rendering
    'render_depth_map',
    'depth_map_to_3d_points',
    # Frame processing
    'process_frame_complete',
    'load_gaussian_data',
    'load_video_frame',
    # Navigation
    'extract_ground_plane',
    'detect_obstacles',
    'compute_occupancy_grid_2d',
    # Pathfinding
    'find_free_paths',
    # Visualization
    'setup_complete_viewer_blueprint',
    'setup_navigation_viewer_blueprint',
    'log_camera_transform',
    'create_occupancy_grid_image',
    # Config
    'ViewerConfig',
    'NavigationConfig',
    'DepthConfig',
    'ConversionConfig',
    'DEFAULT_VIEWER_CONFIG',
    'DEFAULT_NAVIGATION_CONFIG',
    'DEFAULT_DEPTH_CONFIG',
    'DEFAULT_CONVERSION_CONFIG',
    'get_output_dir',
    'get_frames_dir',
    'get_gaussians_dir',
    'get_json_dir',
    # IO utilities
    'load_json',
    'save_json',
    'load_image',
    'save_image',
    'find_ply_files',
    'find_frame_images',
    'get_frame_number',
    # Geometry
    'rotation_matrix_y',
    'rotate_points',
    'scale_points',
    'translate_points',
    'compute_bounding_box',
    'compute_center',
    'normalize_points',
    'compute_distances',
]

