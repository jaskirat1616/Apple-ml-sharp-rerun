"""
Visualization setup utilities for Rerun.
"""

import numpy as np
import rerun as rr


def setup_complete_viewer_blueprint():
    """
    Set up the blueprint for complete video viewer with dual 3D windows.
    
    Returns:
        Blueprint configuration for Rerun viewer
    """
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            # Top: TWO 3D views side by side - BOTH have full camera controls
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial3DView(
                    name="🎨 3D Scene - Colored Point Cloud",
                    origin="world",
                    # Camera controls: rotate, pan, zoom are automatically enabled
                ),
                rr.blueprint.Spatial3DView(
                    name="🗺️ 3D Depth Map - Point Cloud", 
                    origin="world",
                    # Camera controls: rotate, pan, zoom are automatically enabled
                ),
                column_shares=[1, 1]
            ),
            # Bottom row: Video, Depth, and Occupancy grid
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial2DView(name="📹 Original Video", origin="video/frame"),
                rr.blueprint.Spatial2DView(name="🗺️ Depth Map (Colored)", origin="depth/colored"),
                rr.blueprint.Spatial2DView(name="🧭 Occupancy Grid (Top-Down)", origin="grid/occupancy"),
                column_shares=[1, 1, 1]
            ),
            row_shares=[3, 1.5]
        ),
        collapse_panels=False,
    )
    return blueprint


def setup_navigation_viewer_blueprint():
    """
    Set up the blueprint for navigation map viewer.
    
    Returns:
        Blueprint configuration for Rerun viewer
    """
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            rr.blueprint.Spatial3DView(name="3D Navigation Map", origin="world"),
            rr.blueprint.Spatial2DView(name="Occupancy Grid (Top-Down)", origin="grid"),
            row_shares=[2, 1]
        ),
        collapse_panels=False,
    )
    return blueprint


def log_camera_transform(entity_path, mean_pos, std_pos, distance_multiplier=3.0):
    """
    Log camera transform for proper pan/rotate/zoom controls.
    
    Args:
        entity_path: Entity path for the camera transform
        mean_pos: Mean position of points (3,)
        std_pos: Standard deviation of points (3,)
        distance_multiplier: Multiplier for camera distance
    """
    camera_distance = np.linalg.norm(std_pos) * distance_multiplier
    camera_pos = mean_pos + np.array([0, 0, camera_distance])
    
    rr.log(
        entity_path,
        rr.Transform3D(
            translation=camera_pos,
            mat3x3=np.eye(3)
        )
    )


def create_occupancy_grid_image(occupancy_grid):
    """
    Create colored visualization of occupancy grid.
    
    Args:
        occupancy_grid: 2D array where 0=free, 1=occupied
    
    Returns:
        Colored image array (H, W, 3) with uint8 dtype
    """
    grid_viz = np.zeros((*occupancy_grid.shape, 3), dtype=np.uint8)
    grid_viz[occupancy_grid == 0] = [0, 255, 0]   # Green = free
    grid_viz[occupancy_grid == 1] = [255, 0, 0]   # Red = occupied
    return grid_viz

