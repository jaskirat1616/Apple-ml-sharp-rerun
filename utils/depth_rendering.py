"""
Depth map rendering utilities for 3D Gaussian Splatting.
"""

import numpy as np
import cv2


def render_depth_map(positions, colors, resolution=(1280, 720), focal_length=1000, max_depth=100):
    """
    Render depth map and colored depth visualization from 3D points.
    
    Args:
        positions: Array of 3D positions (N, 3)
        colors: Array of RGB colors (N, 3)
        resolution: Tuple of (width, height)
        focal_length: Camera focal length in pixels
        max_depth: Maximum depth value
    
    Returns:
        depth_map: Grayscale depth map (H, W)
        depth_colored: Colored depth visualization (H, W, 3)
    """
    width, height = resolution
    
    # Initialize buffers
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    color_map = np.zeros((height, width, 3), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)
    
    # Project points to image plane
    for pos, col in zip(positions, colors):
        x, y, z = pos
        
        # Skip points behind camera or too far
        if z <= 0 or z > max_depth:
            continue
        
        # Project to image coordinates - match standard image coordinate system (origin at top-left)
        # In image coords: u increases right, v increases down
        # In 3D coords: x increases right, y increases up
        # For correct orientation: positive y (up) should map to smaller v (top of image)
        u = int(focal_length * x / z + width / 2)
        v = int(height / 2 - focal_length * y / z)  # Negative because y up = v down in image
        
        # Check bounds
        if 0 <= u < width and 0 <= v < height:
            # Keep closest depth and accumulate color
            if z < depth_map[v, u]:
                depth_map[v, u] = z
            color_map[v, u] += col
            count_map[v, u] += 1
    
    # Average colors
    mask = count_map > 0
    color_map[mask] = color_map[mask] / count_map[mask, np.newaxis]
    
    # Create colored depth visualization
    depth_normalized = np.clip(depth_map / max_depth, 0, 1)
    depth_colored = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Mark invalid regions as black
    invalid_mask = depth_map >= max_depth
    depth_colored[invalid_mask] = [0, 0, 0]
    
    # No flip needed - depth map should match video frame orientation directly
    return depth_map, depth_colored


def depth_map_to_3d_points(depth_map, depth_colored, focal_length, width, height):
    """
    Convert depth map to 3D point cloud for visualization.
    
    Args:
        depth_map: Depth map (H, W)
        depth_colored: Colored depth visualization (H, W, 3)
        focal_length: Camera focal length in pixels
        width: Image width
        height: Image height
    
    Returns:
        depth_points: 3D points (N, 3)
        depth_colors: RGB colors (N, 3)
    """
    # Sample points from depth map (every Nth pixel for performance)
    step = max(1, min(width, height) // 200)  # Adjust for performance
    
    depth_points = []
    depth_colors = []
    
    # Get valid depth range
    depth_max = depth_map.max()
    depth_min = depth_map[depth_map < depth_max].min() if (depth_map < depth_max).any() else depth_max
    
    for v in range(0, height, step):
        for u in range(0, width, step):
            z = depth_map[v, u]
            
            # Skip invalid depth
            if z >= depth_max * 0.99 or z <= 0:
                continue
            
            # Convert pixel to 3D coordinates
            x = (u - width / 2) * z / focal_length
            y = (height / 2 - v) * z / focal_length  # Flip Y for correct orientation
            
            depth_points.append([x, y, z])
            # Get color from depth colored image
            color = depth_colored[v, u] / 255.0  # Normalize to [0,1]
            depth_colors.append(color)
    
    if len(depth_points) == 0:
        return np.array([]), np.array([])
    
    return np.array(depth_points), np.array(depth_colors)

