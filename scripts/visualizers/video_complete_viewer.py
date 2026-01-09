#!/usr/bin/env python3
"""
Complete Video Viewer - Shows EVERYTHING together in Rerun!

Displays simultaneously:
1. 🎬 Original video frames (2D)
2. 🗺️ Depth maps (2D colored visualization)
3. 📊 Occupancy grid (2D top-down)
4. 🎯 3D point cloud (full scene)
5. 🟢 Navigation data (ground + obstacles)

Perfect for comprehensive scene analysis!
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import rerun as rr
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.frame_processing import process_frame_complete
from utils.depth_rendering import depth_map_to_3d_points
from utils.visualization import (
    setup_complete_viewer_blueprint,
    log_camera_transform,
    create_occupancy_grid_image
)


def visualize_complete_video(input_dir: Path, max_frames: int = None,
                            resolution: float = 0.5,
                            obstacle_height: float = 0.5,
                            skip_frames: int = 1,
                            size_multiplier: float = 1.0):
    """
    Complete visualization with everything!
    """
    # Find all PLY files
    ply_files = sorted(list(input_dir.glob("*.ply")))
    
    if len(ply_files) == 0:
        print(f"❌ No PLY files found in {input_dir}")
        return 1
    
    if max_frames:
        ply_files = ply_files[:max_frames]
    
    if skip_frames > 1:
        ply_files = ply_files[::skip_frames]
    
    print("=" * 80)
    print("🎬 COMPLETE VIDEO VIEWER - Everything in One Place!")
    print("=" * 80)
    print(f"\n📁 Input: {input_dir}")
    print(f"📊 Frames: {len(ply_files)}")
    print(f"⚙️  Settings:")
    print(f"   • Grid resolution: {resolution}m")
    print(f"   • Obstacle height: {obstacle_height}m")
    print(f"   • Point size: {size_multiplier}x")
    if skip_frames > 1:
        print(f"   • Skip: Every {skip_frames} frames")
    
    # Check for frames
    frames_dir = input_dir.parent / "frames"
    has_frames = frames_dir.exists()
    
    if has_frames:
        print(f"✅ Found video frames: {frames_dir}")
    else:
        print(f"⚠️  No video frames found (will generate from 3D)")
    
    # Initialize Rerun
    # Note: Version mismatch warnings between Viewer and SDK can be ignored
    print("\n🎨 Launching Rerun viewer...")
    rr.init("🎬 Complete Video Viewer - All Data", spawn=True)
    
    # Set up coordinate system for proper camera controls
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Set up comprehensive blueprint with TWO 3D views side by side
    blueprint = setup_complete_viewer_blueprint()
    rr.send_blueprint(blueprint)
    
    print("\n🔄 Processing frames (this may take a moment)...")
    print("=" * 80)
    
    # Process each frame
    for idx, ply_path in enumerate(tqdm(ply_files, desc="Loading complete data")):
        frame_idx = int(ply_path.stem.split('_')[-1]) if 'frame_' in ply_path.stem else idx
        
        # Set timeline
        rr.set_time_sequence("frame", frame_idx)
        
        # Process all data
        data = process_frame_complete(
            ply_path, frame_idx, frames_dir, 
            obstacle_height, resolution
        )
        
        if data is None:
            continue
        
        # 1. LOG VIDEO FRAME
        if data['video_frame'] is not None:
            rr.log("video/frame", rr.Image(data['video_frame']))
        
        # 2. LOG COLORED DEPTH MAP
        # Depth map is already correctly oriented from render_depth_map (no additional flip needed)
        rr.log("depth/colored", rr.Image(data['depth_colored']))
        
        # 3. LEFT WINDOW: LOG FULL COLORED 3D POINT CLOUD (Scene)
        rr.log(
            "world/scene/points",
            rr.Points3D(
                positions=data['positions'],
                colors=data['colors'],
                radii=np.mean(data['scales'], axis=1) * 0.3 * size_multiplier
            )
        )
        
        # 4. RIGHT WINDOW: LOG DEPTH MAP AS 3D POINT CLOUD
        # Convert depth map to 3D points
        depth_3d_points, depth_3d_colors = depth_map_to_3d_points(
            data['depth_map'], 
            data['depth_colored'],
            data['focal_length'],
            data['depth_width'],
            data['depth_height']
        )
        
        if len(depth_3d_points) > 0:
            rr.log(
                "world/depth/points",
                rr.Points3D(
                    positions=depth_3d_points,
                    colors=depth_3d_colors,
                    radii=[1.0] * len(depth_3d_points)
                )
            )
        
        # 6. LOG OCCUPANCY GRID
        occupancy_grid = data['occupancy_grid']
        grid_viz = create_occupancy_grid_image(occupancy_grid)
        rr.log("grid/occupancy", rr.Image(grid_viz))
        
        # 7. SET UP CAMERA TRANSFORMS for proper pan/rotate/zoom controls
        if len(data['positions']) > 0:
            mean_pos = data['positions'].mean(axis=0)
            std_pos = data['positions'].std(axis=0)
            log_camera_transform("world/camera", mean_pos, std_pos)
            
            # Also set up camera for depth view if we have depth points
            if len(depth_3d_points) > 0:
                depth_mean = depth_3d_points.mean(axis=0)
                depth_std = depth_3d_points.std(axis=0)
                log_camera_transform("world/depth/camera", depth_mean, depth_std)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE VIEWER READY!")
    print("=" * 80)
    
    print("\n📺 What You're Seeing:")
    print("\n  🎨 3D Scene Window (Top Left):")
    print("     • FULL COLORED POINT CLOUD")
    print("     • Shows original colors from video")
    print("     • Your complete 3D reconstruction")
    print("     • Rotate/pan/zoom to explore")
    
    print("\n  🗺️  3D Depth Map Window (Top Right):")
    print("     • DEPTH MAP AS 3D POINT CLOUD")
    print("     • Colored by depth (blue=near, red=far)")
    print("     • Rotate/pan/zoom to explore")
    
    print("\n  📹 Original Video (Bottom Left):")
    print("     • Your source video frames")
    print("     • Compare with 3D reconstructions")
    
    print("\n  🗺️ Depth Map (Bottom Middle):")
    print("     • Blue/Purple = Near objects")
    print("     • Red/Yellow = Far objects")
    print("     • Generated from 3D points")
    
    print("\n  🧭 Occupancy Grid (Bottom Right):")
    print("     • Green = Free space (robot can go)")
    print("     • Red = Occupied (obstacles)")
    print("     • Bird's eye / top-down view")
    
    print("\n🎮 3D VIEW CONTROLS (BOTH WINDOWS - FULL CONTROLS ENABLED!):")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ⏯️  Timeline (bottom) → Scrub through frames")
    print("")
    print("  🎨 LEFT WINDOW (Colored Scene):")
    print("     🔄 ROTATE:  Left click + drag")
    print("     ↔️  PAN:     Right click + drag  ⭐ PRIMARY METHOD")
    print("               OR Middle mouse + drag")
    print("               OR Shift + Left click + drag")
    print("     🔍 ZOOM:    Mouse wheel / Trackpad scroll")
    print("     🎯 RESET:   Double click anywhere")
    print("     ✋ FREE MOVE: Right click + drag (hand tool)")
    print("")
    print("  🗺️  RIGHT WINDOW (3D Depth Map):")
    print("     🔄 ROTATE:  Left click + drag")
    print("     ↔️  PAN:     Right click + drag  ⭐ PRIMARY METHOD")
    print("               OR Middle mouse + drag")
    print("               OR Shift + Left click + drag")
    print("     🔍 ZOOM:    Mouse wheel / Trackpad scroll")
    print("     🎯 RESET:   Double click anywhere")
    print("     ✋ FREE MOVE: Right click + drag (hand tool)")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n  ✨ Both windows are INDEPENDENT!")
    print("     • Pan each window separately")
    print("     • Zoom each window separately")
    print("     • Rotate each window separately")
    print("\n  💡 Pan/Free Move Tips:")
    print("     • RIGHT CLICK + DRAG is the primary pan method")
    print("     • This gives you 'hand move' / 'free move' control")
    print("     • Click in middle of 3D view, then drag")
    print("     • Alternative: Middle mouse button + drag")
    print("     • Alternative: Shift + Left click + drag")
    
    print("\n💡 Pro Tips:")
    print("  • Use timeline to see how scene changes frame-by-frame")
    print("  • Compare video with colored 3D point cloud")
    print("  • Watch depth map to understand distance")
    print("  • Occupancy grid shows robot's bird's eye view")
    print("  • Green/red overlays in 3D show navigation data")
    print("\n🖱️  3D Camera Controls (BOTH WINDOWS WORK THE SAME!):")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  LEFT WINDOW           │  RIGHT WINDOW              │")
    print("  │  (Colored Scene)       │  (Navigation)              │")
    print("  ├────────────────────────┼────────────────────────────┤")
    print("  │  ✅ Rotate             │  ✅ Rotate                 │")
    print("  │  ✅ Pan (Shift+click)  │  ✅ Pan (Shift+click)      │")
    print("  │  ✅ Zoom (wheel)       │  ✅ Zoom (wheel)           │")
    print("  │  ✅ Reset (dbl-click)  │  ✅ Reset (dbl-click)      │")
    print("  └────────────────────────┴────────────────────────────┘")
    print("\n  🎯 Best Pan Method: Shift + Left click + drag")
    print("  📱 On trackpad: Two-finger click + drag")
    
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
        description="Complete video viewer - everything in one place!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View your high-quality video with everything
  python video_complete_viewer.py -i output_13588904_3840_2160_30fps/gaussians/ --max-frames 30
  
  # All frames
  python video_complete_viewer.py -i output_13588904_3840_2160_30fps/gaussians/
  
  # Faster preview (every 5th frame)
  python video_complete_viewer.py -i output_grok_video_full/gaussians/ --skip 5
  
  # Larger point sizes for better visibility
  python video_complete_viewer.py -i output_video/gaussians/ --size 2.0 --max-frames 20
  
  # Fine navigation grid
  python video_complete_viewer.py -i output_video/gaussians/ --resolution 0.3 --max-frames 30
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Directory containing PLY files (gaussians)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    parser.add_argument("--skip", type=int, default=1,
                       help="Process every Nth frame (default: 1)")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Occupancy grid resolution (default: 0.5m)")
    parser.add_argument("--obstacle-height", type=float, default=0.5,
                       help="Obstacle height threshold (default: 0.5m)")
    parser.add_argument("--size", type=float, default=1.0,
                       help="Point size multiplier (default: 1.0)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: {args.input} does not exist")
        return 1
    
    if not args.input.is_dir():
        print(f"❌ Error: {args.input} is not a directory")
        return 1
    
    return visualize_complete_video(
        args.input,
        args.max_frames,
        args.resolution,
        args.obstacle_height,
        args.skip,
        args.size
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())

