#!/usr/bin/env python3
"""
Example: Building a navigation map from a 3D scene.

This demonstrates how to use the navigation utilities to
detect ground, obstacles, and build an occupancy grid.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import (
    load_gaussian_data,
    extract_ground_plane,
    detect_obstacles,
    compute_occupancy_grid_2d,
    setup_navigation_viewer_blueprint
)
import rerun as rr
import numpy as np


def navigation_example(ply_path: Path, resolution: float = 0.5, obstacle_height: float = 0.5):
    """Example of building a navigation map."""
    print(f"Loading {ply_path}...")
    
    # Load Gaussian Splatting data
    data = load_gaussian_data(ply_path)
    if data is None:
        print("Failed to load PLY file")
        return
    
    positions = data['positions']
    print(f"Loaded {len(positions):,} points")
    
    # Extract ground plane
    print("Detecting ground plane...")
    ground_points, ground_mask, _, ground_info = extract_ground_plane(positions)
    print(f"  Found {len(ground_points):,} ground points")
    
    # Detect obstacles
    print("Detecting obstacles...")
    obstacle_points, clusters = detect_obstacles(positions, ground_mask, obstacle_height)
    print(f"  Found {len(obstacle_points):,} obstacle points")
    print(f"  Detected {len(clusters)} obstacle clusters")
    
    # Build occupancy grid
    print(f"Computing occupancy grid (resolution: {resolution}m)...")
    occupancy_grid, grid_info = compute_occupancy_grid_2d(
        positions, ground_info, resolution, obstacle_height
    )
    free_space = (occupancy_grid == 0).sum()
    occupied_space = (occupancy_grid == 1).sum()
    print(f"  Free cells: {free_space:,} ({100*free_space/occupancy_grid.size:.1f}%)")
    print(f"  Occupied cells: {occupied_space:,} ({100*occupied_space/occupancy_grid.size:.1f}%)")
    
    # Visualize
    print("Visualizing navigation map...")
    rr.init("Navigation Map Example", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    blueprint = setup_navigation_viewer_blueprint()
    rr.send_blueprint(blueprint)
    
    # Log full point cloud (gray)
    rr.log(
        "world/scene/full_cloud",
        rr.Points3D(
            positions=positions,
            colors=[[0.5, 0.5, 0.5]] * len(positions),
            radii=[0.1] * len(positions)
        )
    )
    
    # Log ground (green)
    if len(ground_points) > 0:
        rr.log(
            "world/scene/ground",
            rr.Points3D(
                positions=ground_points,
                colors=[[0.0, 1.0, 0.0]] * len(ground_points),
                radii=[0.3] * len(ground_points)
            )
        )
    
    # Log obstacles (red)
    if len(obstacle_points) > 0:
        rr.log(
            "world/scene/obstacles",
            rr.Points3D(
                positions=obstacle_points,
                colors=[[1.0, 0.0, 0.0]] * len(obstacle_points),
                radii=[0.2] * len(obstacle_points)
            )
        )
    
    # Log occupancy grid
    grid_img = (occupancy_grid * 255).astype(np.uint8)
    rr.log("grid/occupancy", rr.Image(grid_img))
    
    print("Navigation map ready! Close the window when done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Navigation map example")
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Path to PLY file")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Occupancy grid resolution (default: 0.5m)")
    parser.add_argument("--obstacle-height", type=float, default=0.5,
                       help="Obstacle height threshold (default: 0.5m)")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        sys.exit(1)
    
    navigation_example(args.input, args.resolution, args.obstacle_height)

