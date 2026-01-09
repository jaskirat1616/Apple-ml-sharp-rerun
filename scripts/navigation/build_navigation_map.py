#!/usr/bin/env python3
"""
Build 3D navigation maps from Gaussian Splat scenes.
Extract obstacles, free space, and navigable paths for autonomous systems.

Applications:
- Robot navigation
- Drone path planning
- Self-driving car simulation
- AR/VR navigation
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import rerun as rr
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.frame_processing import load_gaussian_data
from utils.navigation import extract_ground_plane, detect_obstacles, compute_occupancy_grid_2d
from utils.pathfinding import find_free_paths
from utils.visualization import setup_navigation_viewer_blueprint


def voxelize_scene(positions, voxel_size=1.0):
    """Convert point cloud to voxel grid."""
    # Calculate voxel indices
    voxel_indices = np.floor(positions / voxel_size).astype(int)
    
    # Get unique voxels
    unique_voxels = np.unique(voxel_indices, axis=0)
    
    # Convert back to world coordinates (center of voxel)
    voxel_centers = (unique_voxels + 0.5) * voxel_size
    
    return voxel_centers, unique_voxels


def visualize_navigation_map(positions, ground_points, obstacle_points, 
                             clusters, occupancy_grid, grid_info, 
                             path=None, output_json=None):
    """Visualize the navigation map in Rerun."""
    
    # Initialize Rerun
    # IMPORTANT: Version mismatch causes blank viewer - fix with: pip install rerun-sdk==0.23.1
    rr.init("Navigation Map", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Set time sequence to ensure data is logged (using old API for viewer v0.23.1)
    rr.set_time_sequence("frame", 0)
    
    # Blueprint for navigation view
    blueprint = setup_navigation_viewer_blueprint()
    rr.send_blueprint(blueprint)
    
    print("  Logging data to Rerun...")
    
    # Downsample large point clouds for better compatibility
    max_points = 500000  # Limit points to avoid version compatibility issues
    if len(positions) > max_points:
        print(f"  Downsampling {len(positions):,} points to {max_points:,} for compatibility...")
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
    
    # Log full point cloud (gray) - use numpy arrays for colors/radii
    gray_color = np.array([0.5, 0.5, 0.5])
    colors_full = np.tile(gray_color, (len(positions), 1))
    radii_full = np.full(len(positions), 0.1)
    
    rr.log(
        "world/scene/full_cloud",
        rr.Points3D(
            positions=positions,
            colors=colors_full,
            radii=radii_full
        )
    )
    
    # Log ground plane (green)
    if len(ground_points) > 0:
        green_color = np.array([0.0, 1.0, 0.0])
        colors_ground = np.tile(green_color, (len(ground_points), 1))
        radii_ground = np.full(len(ground_points), 0.3)
        
        rr.log(
            "world/scene/ground",
            rr.Points3D(
                positions=ground_points,
                colors=colors_ground,
                radii=radii_ground
            )
        )
    
    # Log obstacles (red)
    if len(obstacle_points) > 0:
        # Downsample obstacles if too many
        if len(obstacle_points) > max_points:
            indices = np.random.choice(len(obstacle_points), max_points, replace=False)
            obstacle_points = obstacle_points[indices]
        
        red_color = np.array([1.0, 0.0, 0.0])
        colors_obstacles = np.tile(red_color, (len(obstacle_points), 1))
        radii_obstacles = np.full(len(obstacle_points), 0.2)
        
        rr.log(
            "world/scene/obstacles",
            rr.Points3D(
                positions=obstacle_points,
                colors=colors_obstacles,
                radii=radii_obstacles
            )
        )
    
    # Log obstacle bounding boxes
    for i, cluster in enumerate(clusters):
        min_pt = np.array(cluster['min'])
        max_pt = np.array(cluster['max'])
        center = np.array(cluster['center'])
        size = np.array(cluster['size'])
        
        # Log as box - use numpy arrays
        orange_color = np.array([255, 100, 0]) / 255.0  # Normalize to [0,1]
        
        rr.log(
            f"world/obstacles/bbox_{i}",
            rr.Boxes3D(
                half_sizes=np.array([size / 2]),
                centers=np.array([center]),
                colors=np.array([orange_color])
            )
        )
    
    # Log occupancy grid as 2D image
    # Convert to RGB format for better compatibility
    grid_img = (occupancy_grid * 255).astype(np.uint8)
    # Expand to 3 channels (H, W) -> (H, W, 3)
    if len(grid_img.shape) == 2:
        grid_img = np.stack([grid_img, grid_img, grid_img], axis=-1)
    
    rr.log(
        "grid/occupancy",
        rr.Image(grid_img)
    )
    
    # Log path if provided
    if path is not None:
        min_x, max_x, min_z, max_z, resolution = grid_info
        path_3d = []
        for i, j in path:
            x = min_x + i * resolution
            z = min_z + j * resolution
            y = positions[:, 1].mean()  # Height at ground level
            path_3d.append([x, y, z])
        
        path_3d = np.array(path_3d)
        cyan_color = np.array([0.0, 1.0, 1.0])  # Cyan
        
        rr.log(
            "world/navigation/path",
            rr.LineStrips3D(
                [path_3d],
                colors=np.array([cyan_color]),
                radii=np.array([0.5])
            )
        )
        
        # Mark start and end
        start_color = np.array([0.0, 1.0, 0.0])  # Green
        goal_color = np.array([1.0, 0.0, 1.0])   # Magenta
        
        rr.log(
            "world/navigation/start",
            rr.Points3D(
                positions=np.array([path_3d[0]]),
                colors=np.array([start_color]),
                radii=np.array([2.0])
            )
        )
        rr.log(
            "world/navigation/goal",
            rr.Points3D(
                positions=np.array([path_3d[-1]]),
                colors=np.array([goal_color]),
                radii=np.array([2.0])
            )
        )
    
    # Export data if requested
    if output_json:
        export_data = {
            'num_obstacles': len(clusters),
            'obstacles': clusters,
            'occupancy_grid': {
                'shape': occupancy_grid.shape,
                'resolution': grid_info[4],
                'bounds': {
                    'min_x': grid_info[0],
                    'max_x': grid_info[1],
                    'min_z': grid_info[2],
                    'max_z': grid_info[3],
                }
            },
            'ground_level': float(ground_points[:, 1].mean()) if len(ground_points) > 0 else 0.0,
            'navigable_area': float((occupancy_grid == 0).sum() * grid_info[4] ** 2),
        }
        
        if path is not None:
            export_data['path'] = [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in path_3d]
        
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported navigation data to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Build navigation maps from 3D Gaussian Splats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic navigation map
  python build_navigation_map.py -i output_test/IMG_4707.ply
  
  # With path planning
  python build_navigation_map.py -i output_test/IMG_4707.ply --plan-path --start 0 0 --goal 50 50
  
  # High resolution occupancy grid
  python build_navigation_map.py -i output_test/IMG_4707.ply --resolution 0.5
  
  # Export for robot navigation system
  python build_navigation_map.py -i output_test/IMG_4707.ply -o navigation_map.json
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to PLY file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file for navigation data")
    parser.add_argument("--resolution", type=float, default=1.0, 
                       help="Occupancy grid resolution in meters (default: 1.0)")
    parser.add_argument("--obstacle-height", type=float, default=0.5,
                       help="Minimum obstacle height in meters (default: 0.5)")
    parser.add_argument("--plan-path", action="store_true", help="Plan path from start to goal")
    parser.add_argument("--start", nargs=2, type=float, help="Start position (x z)")
    parser.add_argument("--goal", nargs=2, type=float, help="Goal position (x z)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1
    
    print(f"Building navigation map from {args.input.name}...")
    print("=" * 60)
    
    # Load scene
    print("Loading scene...")
    data = load_gaussian_data(args.input, opacity_threshold=0.1)
    if data is None:
        print("Error: Failed to load PLY file")
        return 1
    
    positions = data['positions']
    print(f"  Loaded {len(positions):,} points")
    
    # Extract ground plane
    print("\nDetecting ground plane...")
    ground_points, ground_mask, height_map, ground_info = extract_ground_plane(positions)
    print(f"  Ground points: {len(ground_points):,}")
    print(f"  Ground level: {ground_points[:, 1].mean():.2f}m")
    
    # Detect obstacles
    print("\nDetecting obstacles...")
    obstacle_points, clusters = detect_obstacles(positions, ground_mask, args.obstacle_height)
    print(f"  Obstacle points: {len(obstacle_points):,}")
    print(f"  Detected {len(clusters)} obstacle clusters")
    
    # Compute occupancy grid
    print(f"\nComputing occupancy grid (resolution: {args.resolution}m)...")
    occupancy_grid, grid_info = compute_occupancy_grid_2d(
        positions, ground_info, args.resolution, args.obstacle_height
    )
    free_space = (occupancy_grid == 0).sum()
    occupied_space = (occupancy_grid == 1).sum()
    total_cells = occupancy_grid.size
    
    print(f"  Grid size: {occupancy_grid.shape}")
    print(f"  Free cells: {free_space:,} ({100*free_space/total_cells:.1f}%)")
    print(f"  Occupied cells: {occupied_space:,} ({100*occupied_space/total_cells:.1f}%)")
    print(f"  Navigable area: {free_space * args.resolution**2:.1f} m²")
    
    # Path planning
    path_3d = None
    if args.plan_path:
        if not args.start or not args.goal:
            print("\nError: --start and --goal required for path planning")
            return 1
        
        print("\nPlanning path...")
        min_x, max_x, min_z, max_z, resolution = grid_info
        
        # Convert world coords to grid coords
        start_i = int((args.start[0] - min_x) / resolution)
        start_j = int((args.start[1] - min_z) / resolution)
        goal_i = int((args.goal[0] - min_x) / resolution)
        goal_j = int((args.goal[1] - min_z) / resolution)
        
        # Ensure within bounds
        start_i = np.clip(start_i, 0, occupancy_grid.shape[0]-1)
        start_j = np.clip(start_j, 0, occupancy_grid.shape[1]-1)
        goal_i = np.clip(goal_i, 0, occupancy_grid.shape[0]-1)
        goal_j = np.clip(goal_j, 0, occupancy_grid.shape[1]-1)
        
        path = find_free_paths(occupancy_grid, (start_i, start_j), (goal_i, goal_j))
        
        if path:
            print(f"  ✓ Found path with {len(path)} waypoints")
            print(f"  Path length: ~{len(path) * resolution:.1f}m")
        else:
            print("  ✗ No path found (obstacles blocking)")
            path = None
    else:
        path = None
    
    # Visualize
    print("\nVisualizing navigation map...")
    visualize_navigation_map(
        positions, ground_points, obstacle_points, 
        clusters, occupancy_grid, grid_info,
        path, args.output
    )
    
    print("\n" + "=" * 60)
    print("✓ Navigation map ready!")
    print("\n🗺️  Map Information:")
    print(f"   Ground level: {ground_points[:, 1].mean():.2f}m")
    print(f"   Obstacles: {len(clusters)}")
    print(f"   Navigable area: {free_space * args.resolution**2:.1f}m²")
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT: Version Mismatch Detected!")
    print("=" * 60)
    print("\n   If the Rerun viewer is BLANK or shows nothing:")
    print("   • Your Viewer is v0.23.1 (older)")
    print("   • Your SDK is v0.27.0 (newer)")
    print("   • This causes decode errors and blank visualization")
    print("\n   SOLUTION - Downgrade SDK to match viewer:")
    print("   pip install rerun-sdk==0.23.1")
    print("\n   Then run this script again to see the visualization.")
    print("\n   Alternative: Update viewer to v0.27.0 (see Rerun docs)")
    print("=" * 60)
    print("\nPress Ctrl+C to exit.")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

