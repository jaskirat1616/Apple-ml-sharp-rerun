"""
Frame processing utilities for Gaussian Splatting data.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils


def load_gaussian_data(ply_path: Path, opacity_threshold: float = 0.1):
    """
    Load and process Gaussian Splatting data from a PLY file.
    
    Args:
        ply_path: Path to PLY file
        opacity_threshold: Minimum opacity to include point
    
    Returns:
        Dictionary containing positions, colors, scales, opacities, metadata
        Returns None if loading fails
    """
    try:
        gaussians, metadata = load_ply(ply_path)
        positions = gaussians.mean_vectors.cpu().numpy().squeeze()
        colors = gaussians.colors.cpu().numpy().squeeze()
        scales = gaussians.singular_values.cpu().numpy().squeeze()
        opacities = gaussians.opacities.cpu().numpy().squeeze()
        
        # Filter by opacity
        mask = opacities > opacity_threshold
        positions = positions[mask]
        colors_filtered = colors[mask]
        scales = scales[mask]
        
        # Convert colors from linear RGB to sRGB if needed
        if metadata.color_space == "linearRGB":
            colors_torch = torch.from_numpy(colors_filtered).float()
            colors_filtered = cs_utils.linearRGB2sRGB(colors_torch).numpy()
        
        if len(positions) == 0:
            return None
        
        return {
            'positions': positions,
            'colors': colors_filtered,
            'scales': scales,
            'opacities': opacities[mask],
            'metadata': metadata,
        }
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        return None


def load_video_frame(frames_dir: Path, frame_name: str):
    """
    Load a video frame image if available.
    
    Args:
        frames_dir: Directory containing frame images
        frame_name: Base name of frame (without extension)
    
    Returns:
        Numpy array of image, or None if not found
    """
    if frames_dir is None or not frames_dir.exists():
        return None
    
    frame_path = frames_dir / f"{frame_name}.png"
    if frame_path.exists():
        try:
            return np.array(Image.open(frame_path))
        except Exception:
            return None
    return None


def process_frame_complete(ply_path: Path, frame_idx: int, frames_dir: Path = None,
                          obstacle_height: float = 0.5, resolution: float = 0.5):
    """
    Process a single frame with all data including depth maps and navigation info.
    
    Args:
        ply_path: Path to PLY file
        frame_idx: Frame index
        frames_dir: Directory containing video frames
        obstacle_height: Height threshold for obstacle detection
        resolution: Resolution for occupancy grid
    
    Returns:
        Dictionary with all processed frame data, or None if processing fails
    """
    from .depth_rendering import render_depth_map
    from .navigation import extract_ground_plane, detect_obstacles, compute_occupancy_grid_2d
    
    # Load Gaussian data
    data = load_gaussian_data(ply_path)
    if data is None:
        return None
    
    positions = data['positions']
    colors = data['colors']
    scales = data['scales']
    metadata = data['metadata']
    
    # Load video frame if available
    frame_name = ply_path.stem
    video_frame = load_video_frame(frames_dir, frame_name)
    
    # Render depth map
    focal_length = getattr(metadata, 'focal_length_px', 1000)
    depth_resolution = (video_frame.shape[1], video_frame.shape[0]) if video_frame is not None else (1280, 720)
    depth_map, depth_colored = render_depth_map(
        positions, colors, depth_resolution, focal_length
    )
    
    # Navigation: Detect ground
    ground_points, ground_mask, _, ground_info = extract_ground_plane(positions)
    ground_level = np.percentile(positions[:, 1], 10) if len(positions) > 0 else 0.0
    
    # Detect obstacles
    obstacle_points, _ = detect_obstacles(positions, ground_mask, obstacle_height)
    
    # Build occupancy grid
    occupancy_grid, _ = compute_occupancy_grid_2d(
        positions, ground_info, resolution, obstacle_height
    )
    
    return {
        'frame': frame_idx,
        'positions': positions,
        'colors': colors,
        'scales': scales,
        'ground_points': ground_points,
        'obstacle_points': obstacle_points,
        'ground_level': ground_level,
        'occupancy_grid': occupancy_grid,
        'depth_map': depth_map,
        'depth_colored': depth_colored,
        'video_frame': video_frame,
        'focal_length': focal_length,
        'depth_width': depth_resolution[0],
        'depth_height': depth_resolution[1],
    }

