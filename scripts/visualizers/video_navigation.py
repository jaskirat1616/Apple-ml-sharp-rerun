#!/usr/bin/env python3
"""
Video Navigation - Process navigation data from video sequences.

Shows obstacles, ground plane, and navigation paths evolving through time.
Perfect for analyzing dynamic scenes, robot navigation through spaces, or drone flights.
"""

import argparse
import numpy as np
from pathlib import Path
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch
from tqdm import tqdm


def process_frame_navigation(ply_path: Path, frame_idx: int, 
                             obstacle_height: float = 0.5,
                             resolution: float = 0.5):
    """Process a single frame for navigation data."""
    # Load scene
    gaussians, metadata = load_ply(ply_path)
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    
    # Filter by opacity
    mask = opacities > 0.1
    positions = positions[mask]
    
    if len(positions) == 0:
        return None
    
    # Detect ground
    ground_level = np.percentile(positions[:, 1], 10)
    ground_mask = np.abs(positions[:, 1] - ground_level) < obstacle_height
    ground_points = positions[ground_mask]
    
    # Detect obstacles
    obstacle_mask = ~ground_mask & (positions[:, 1] > (ground_level + obstacle_height))
    obstacle_points = positions[obstacle_mask]
    
    # Build occupancy grid
    if len(positions) > 0:
        min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
        min_z, max_z = positions[:, 2].min(), positions[:, 2].max()
        
        nx = max(1, int((max_x - min_x) / resolution) + 1)
        nz = max(1, int((max_z - min_z) / resolution) + 1)
        
        occupancy_grid = np.zeros((nx, nz))
        
        for pos in positions:
            if pos[1] > (ground_level + obstacle_height):
                i = int((pos[0] - min_x) / resolution)
                j = int((pos[2] - min_z) / resolution)
                if 0 <= i < nx and 0 <= j < nz:
                    occupancy_grid[i, j] = 1
        
        free_cells = (occupancy_grid == 0).sum()
        occupied_cells = (occupancy_grid == 1).sum()
    else:
        occupancy_grid = np.zeros((1, 1))
        free_cells = 0
        occupied_cells = 0
    
    return {
        'frame': frame_idx,
        'positions': positions,
        'ground_points': ground_points,
        'obstacle_points': obstacle_points,
        'ground_level': ground_level,
        'occupancy_grid': occupancy_grid,
        'free_cells': free_cells,
        'occupied_cells': occupied_cells,
    }


def visualize_video_navigation(input_dir: Path, max_frames: int = None,
                               resolution: float = 0.5,
                               obstacle_height: float = 0.5,
                               skip_frames: int = 1):
    """
    Visualize navigation data through video sequence.
    """
    # Find all PLY files
    ply_files = sorted(list(input_dir.glob("*.ply")))
    
    if len(ply_files) == 0:
        print(f"❌ No PLY files found in {input_dir}")
        return 1
    
    if max_frames:
        ply_files = ply_files[:max_frames]
    
    # Skip frames if requested
    if skip_frames > 1:
        ply_files = ply_files[::skip_frames]
    
    print("=" * 70)
    print("🎬 VIDEO NAVIGATION ANALYSIS")
    print("=" * 70)
    print(f"\n📁 Input: {input_dir}")
    print(f"📊 Frames to process: {len(ply_files)}")
    print(f"⚙️  Grid resolution: {resolution}m")
    print(f"📏 Obstacle height threshold: {obstacle_height}m")
    if skip_frames > 1:
        print(f"⏭️  Processing every {skip_frames} frames")
    
    # Check for corresponding frame images
    frames_dir = input_dir.parent / "frames"
    has_frames = frames_dir.exists()
    
    if has_frames:
        print(f"🖼️  Found frame images: {frames_dir}")
    
    # Initialize Rerun
    print("\n🎨 Initializing Rerun viewer...")
    rr.init("🎬 Video Navigation Analysis", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Set up blueprint with timeline
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            rr.blueprint.Spatial3DView(
                name="🎯 3D Navigation (Time-series)", 
                origin="world"
            ),
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial2DView(name="🗺️  Occupancy Grid", origin="grid"),
                rr.blueprint.Spatial2DView(name="📹 Video Frame", origin="video") if has_frames else None,
                column_shares=[1, 1] if has_frames else [1]
            ),
            row_shares=[2, 1]
        ),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    
    print("\n🔄 Processing frames...")
    print("=" * 70)
    
    stats = {
        'total_points': [],
        'ground_points': [],
        'obstacle_points': [],
        'free_area': [],
        'occupied_area': [],
    }
    
    # Process each frame
    for idx, ply_path in enumerate(tqdm(ply_files, desc="Analyzing navigation")):
        frame_idx = int(ply_path.stem.split('_')[-1]) if 'frame_' in ply_path.stem else idx
        
        # Set timeline
        rr.set_time_sequence("frame", frame_idx)
        
        # Load corresponding video frame if available
        if has_frames:
            frame_name = ply_path.stem
            frame_path = frames_dir / f"{frame_name}.png"
            if frame_path.exists():
                try:
                    from PIL import Image
                    img = np.array(Image.open(frame_path))
                    rr.log("video/frame", rr.Image(img))
                except Exception as e:
                    pass
        
        # Process navigation data
        nav_data = process_frame_navigation(ply_path, frame_idx, obstacle_height, resolution)
        
        if nav_data is None:
            continue
        
        # Log full scene (light gray, small)
        if len(nav_data['positions']) > 0:
            rr.log(
                "world/scene/points",
                rr.Points3D(
                    positions=nav_data['positions'],
                    colors=[[0.7, 0.7, 0.7]] * len(nav_data['positions']),
                    radii=[0.05] * len(nav_data['positions'])
                )
            )
        
        # Log ground (green)
        if len(nav_data['ground_points']) > 0:
            rr.log(
                "world/navigation/ground",
                rr.Points3D(
                    positions=nav_data['ground_points'],
                    colors=[[0.0, 1.0, 0.0]] * len(nav_data['ground_points']),
                    radii=[0.15] * len(nav_data['ground_points'])
                )
            )
        
        # Log obstacles (red)
        if len(nav_data['obstacle_points']) > 0:
            rr.log(
                "world/navigation/obstacles",
                rr.Points3D(
                    positions=nav_data['obstacle_points'],
                    colors=[[1.0, 0.0, 0.0]] * len(nav_data['obstacle_points']),
                    radii=[0.12] * len(nav_data['obstacle_points'])
                )
            )
        
        # Log occupancy grid
        occupancy_grid = nav_data['occupancy_grid']
        grid_viz = np.zeros((*occupancy_grid.shape, 3), dtype=np.uint8)
        grid_viz[occupancy_grid == 0] = [0, 200, 0]  # Green = free
        grid_viz[occupancy_grid == 1] = [200, 0, 0]  # Red = occupied
        
        rr.log("grid/occupancy", rr.Image(grid_viz))
        
        # Track statistics
        stats['total_points'].append(len(nav_data['positions']))
        stats['ground_points'].append(len(nav_data['ground_points']))
        stats['obstacle_points'].append(len(nav_data['obstacle_points']))
        stats['free_area'].append(nav_data['free_cells'] * resolution ** 2)
        stats['occupied_area'].append(nav_data['occupied_cells'] * resolution ** 2)
        
        # Log ground level indicator
        if len(nav_data['positions']) > 0:
            center = nav_data['positions'].mean(axis=0)
            ground_indicator = center.copy()
            ground_indicator[1] = nav_data['ground_level']
            
            rr.log(
                "world/info/ground_level",
                rr.Points3D(
                    positions=[ground_indicator],
                    colors=[[0, 255, 255]],
                    radii=[0.5]
                )
            )
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 Statistics (across {len(ply_files)} frames):")
    print(f"\n  Points per frame:")
    print(f"    • Total: {np.mean(stats['total_points']):.0f} avg (min: {np.min(stats['total_points']):.0f}, max: {np.max(stats['total_points']):.0f})")
    print(f"    • Ground: {np.mean(stats['ground_points']):.0f} avg")
    print(f"    • Obstacles: {np.mean(stats['obstacle_points']):.0f} avg")
    
    print(f"\n  Navigable area:")
    print(f"    • Free space: {np.mean(stats['free_area']):.1f} m² avg")
    print(f"    • Occupied: {np.mean(stats['occupied_area']):.1f} m² avg")
    print(f"    • Free ratio: {100 * np.mean(stats['free_area']) / (np.mean(stats['free_area']) + np.mean(stats['occupied_area'])):.1f}%")
    
    print(f"\n🎮 Rerun Controls:")
    print(f"  ⏯️  Timeline: Use the slider at bottom to navigate frames")
    print(f"  🔄 Rotate: Left click + drag")
    print(f"  ↔️  Pan: Shift + left click + drag")
    print(f"  🔍 Zoom: Mouse wheel")
    print(f"  🎯 Reset: Double click")
    
    print(f"\n📈 What You're Seeing:")
    print(f"  🟢 Green points = Ground/navigable surface")
    print(f"  🔴 Red points = Obstacles to avoid")
    print(f"  📊 Top-down grid = Bird's eye occupancy map")
    if has_frames:
        print(f"  📹 Video frames = Original footage")
    
    print(f"\n💡 Use the timeline to:")
    print(f"  • See how obstacles change over time")
    print(f"  • Find best navigation routes")
    print(f"  • Analyze scene dynamics")
    print(f"  • Spot clearance areas")
    
    print("\nPress Ctrl+C to exit.")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze navigation data through video sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze full video
  python video_navigation.py -i output_13588904_3840_2160_30fps/gaussians/
  
  # First 30 frames only
  python video_navigation.py -i output_13588904_3840_2160_30fps/gaussians/ --max-frames 30
  
  # Higher resolution grid
  python video_navigation.py -i output_grok_video_full/gaussians/ --resolution 0.3
  
  # Skip frames for faster preview
  python video_navigation.py -i output_video/gaussians/ --skip 5
  
  # Adjust obstacle detection
  python video_navigation.py -i output_video/gaussians/ --obstacle-height 0.3
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Directory containing PLY files (gaussians)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum number of frames to process")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Occupancy grid resolution in meters (default: 0.5)")
    parser.add_argument("--obstacle-height", type=float, default=0.5,
                       help="Minimum obstacle height in meters (default: 0.5)")
    parser.add_argument("--skip", type=int, default=1,
                       help="Process every Nth frame (default: 1 = all frames)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: {args.input} does not exist")
        return 1
    
    if not args.input.is_dir():
        print(f"❌ Error: {args.input} is not a directory")
        print(f"Tip: Use 'build_navigation_map.py' for single files")
        return 1
    
    return visualize_video_navigation(
        args.input,
        args.max_frames,
        args.resolution,
        args.obstacle_height,
        args.skip
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())

