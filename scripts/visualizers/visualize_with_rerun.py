#!/usr/bin/env python3

"""
Visualize 3D Gaussian Splats using Rerun.

This script loads .ply files and visualizes them in Rerun for interactive 3D exploration.
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
from PIL import Image


def visualize_gaussian_ply(ply_path: Path, frame_idx: int = 0, downsample: float = 1.0, size_multiplier: float = 1.0, frames_dir: Optional[Path] = None, rotation_degrees: float = 0.0):
    """
    Load and visualize a Gaussian Splatting PLY file in Rerun.
    
    Args:
        ply_path: Path to the .ply file
        frame_idx: Frame index for timeline
        downsample: Fraction of points to keep (0.1 = 10%, 1.0 = 100%)
        size_multiplier: Multiplier for splat size (default: 1.0 = optimal size)
        frames_dir: Optional directory containing corresponding frame images
        rotation_degrees: Rotation angle in degrees around Y-axis (default: 0.0)
    """
    print(f"Loading {ply_path.name}...")
    
    # Set timeline first
    rr.set_time_sequence("frame", frame_idx)
    
    # Load and display the corresponding 2D frame if available
    if frames_dir and frames_dir.exists():
        frame_name = ply_path.stem  # e.g., "frame_000000"
        frame_path = frames_dir / f"{frame_name}.png"
        if frame_path.exists():
            try:
                img = np.array(Image.open(frame_path))
                rr.log("video/frame", rr.Image(img))
                print(f"  Loaded frame image: {frame_path.name}")
            except Exception as e:
                print(f"  Warning: Could not load frame image: {e}")
    
    # Load the PLY file
    gaussians, metadata = load_ply(ply_path)
    
    # Extract data - Gaussians3D uses mean_vectors, singular_values, quaternions
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()  # (N, 3)
    colors = gaussians.colors.cpu().numpy().squeeze()  # (N, 3)
    scales = gaussians.singular_values.cpu().numpy().squeeze()  # (N, 3)
    opacities = gaussians.opacities.cpu().numpy().squeeze()  # (N,)
    
    original_count = positions.shape[0]
    
    # Convert colors from linearRGB to sRGB for proper display
    # SHARP outputs linearRGB colors, but viewers expect sRGB
    if metadata.color_space == "linearRGB":
        import torch
        colors_torch = torch.from_numpy(colors).float()
        colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()
        print(f"  Converted colors from linearRGB to sRGB")
    
    # Filter by opacity - only show visible gaussians (opacity > 0.1)
    opacity_mask = opacities > 0.1
    positions = positions[opacity_mask]
    colors = colors[opacity_mask]
    scales = scales[opacity_mask]
    opacities = opacities[opacity_mask]
    
    if opacity_mask.sum() < original_count:
        print(f"  Filtered to {opacity_mask.sum():,} visible points (opacity > 0.1)")
    
    # Apply rotation if requested
    if rotation_degrees != 0.0:
        # Rotate around Y-axis (up axis)
        angle_rad = np.radians(rotation_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix around Y-axis
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Apply rotation to positions
        positions = positions @ rotation_matrix.T
        print(f"  Applied {rotation_degrees}° rotation around Y-axis")
    
    # Downsample if requested
    if downsample < 1.0:
        n_points = int(positions.shape[0] * downsample)
        indices = np.random.choice(positions.shape[0], n_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        scales = scales[indices]
        opacities = opacities[indices]
        print(f"  Downsampled to {positions.shape[0]:,} points ({downsample*100:.1f}%)")
    
    print(f"  Final points: {positions.shape[0]:,}")
    print(f"  Resolution: {metadata.resolution_px}")
    print(f"  Focal length: {metadata.focal_length_px:.2f}px")
    
    # Log the 3D points with colors
    # Use the same entity path for all frames so they don't stack
    # Use smaller default size (0.3) for better visual quality
    effective_size = size_multiplier * 0.3
    rr.log(
        "world/gaussians",
        rr.Points3D(
            positions=positions,
            colors=colors,
            radii=np.mean(scales, axis=1) * effective_size,
        )
    )
    
    # Log metadata as text
    metadata_text = f"""Frame: {ply_path.stem}
Points: {positions.shape[0]:,}
Resolution: {metadata.resolution_px[0]} x {metadata.resolution_px[1]}
Focal Length: {metadata.focal_length_px:.2f} px
Color Space: {metadata.color_space}

Statistics:
- Mean position: ({positions.mean(axis=0)[0]:.3f}, {positions.mean(axis=0)[1]:.3f}, {positions.mean(axis=0)[2]:.3f})
- Position std: ({positions.std(axis=0)[0]:.3f}, {positions.std(axis=0)[1]:.3f}, {positions.std(axis=0)[2]:.3f})
- Mean opacity: {opacities.mean():.3f}
- Mean scale: {scales.mean():.4f}
"""
    rr.log(
        "info/frame_metadata",
        rr.TextDocument(metadata_text, media_type=rr.MediaType.TEXT),
        static=False
    )
    
    # Set up camera transform for proper interaction
    # Calculate good camera position based on point cloud bounds
    mean_pos = positions.mean(axis=0)
    std_pos = positions.std(axis=0)
    
    # Position camera to view the entire scene
    # Move camera back in Z to see the whole point cloud
    camera_distance = np.linalg.norm(std_pos) * 3.0
    camera_pos = mean_pos + np.array([0, 0, camera_distance])
    
    # Log the camera transform
    rr.log(
        "world/camera",
        rr.Transform3D(
            translation=camera_pos,
            mat3x3=np.eye(3)
        )
    )
    
    print(f"  Camera position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
    print(f"  Scene center: ({mean_pos[0]:.2f}, {mean_pos[1]:.2f}, {mean_pos[2]:.2f})")

def visualize_directory(input_dir: Path, max_frames: int = 0, downsample: float = 1.0, size_multiplier: float = 1.0, rotation_degrees: float = 0.0):
    """
    Visualize all PLY files in a directory.

    Args:
        input_dir: Directory containing .ply files
        max_frames: Maximum number of frames to visualize
        downsample: Fraction of points to keep per frame
        size_multiplier: Multiplier for splat size (default: 1.0 = optimal size)
        rotation_degrees: Rotation angle in degrees around Y-axis (default: 0.0)
    """
    # Find all PLY files
    ply_files = sorted(list(input_dir.glob("*.ply")))
    
    if len(ply_files) == 0:
        print(f"No PLY files found in {input_dir}")
        return
    
    if max_frames:
        ply_files = ply_files[:max_frames]
    
    # Look for corresponding frames directory (sibling to gaussians directory)
    frames_dir_path = input_dir.parent / "frames"
    frames_dir: Optional[Path] = frames_dir_path if frames_dir_path.exists() else None
    
    if frames_dir:
        print(f"\nFound frame images directory: {frames_dir}")
    else:
        print(f"\nNo frame images directory found (looked for: {frames_dir_path})")
    
    print(f"\nFound {len(ply_files)} PLY files to visualize")
    if downsample < 1.0:
        print(f"Downsampling to {downsample*100:.1f}% of points per frame")
    print("="*60)
    
    # Initialize Rerun with proper 3D view settings
    rr.init("3D Gaussian Splats Video Viewer", spawn=True)
    
    # Set up the view coordinates - use standard OpenGL/3D coordinate system
    # This ensures proper camera controls
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Configure blueprint for optimal 3D viewing with full camera controls
    # This sets up the 3D view with orbit camera that supports rotate, pan, and zoom
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Vertical(
            rr.blueprint.Spatial3DView(
                name="3D View", 
                origin="world",
                # Enable all camera interaction modes
            ),
            rr.blueprint.Horizontal(
                rr.blueprint.Spatial2DView(name="Frame View", origin="video/frame"),
                rr.blueprint.TextDocumentView(name="Info", origin="info/frame_metadata"),
                column_shares=[2, 1]
            ),
            row_shares=[3, 1]
        ),
        collapse_panels=False,
    )
    
    rr.send_blueprint(blueprint)
    
    # Process each PLY file
    for idx, ply_path in enumerate(ply_files):
        visualize_gaussian_ply(ply_path, idx, downsample, size_multiplier, frames_dir, rotation_degrees)
    
    print("\n" + "="*60)
    print(f"✓ Loaded {len(ply_files)} frames into Rerun")
    print("="*60)
    print("\n🎬 Rerun viewer is now open!")
    print(f"\n📹 Video has {len(ply_files)} frames")
    print("\nTimeline Controls:")
    print("  - Use the timeline slider at the bottom to navigate frames")
    print("  - Click the play button to animate through frames")
    print("  - Each frame shows the 2D image (if available) and 3D gaussians")
    print("\n🎮 3D View Controls:")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  🔄 ROTATE:   Left click + drag")
    print("  ↔️  PAN:      Right click + drag")
    print("             OR Shift + Left click + drag")
    print("             OR Middle click + drag")
    print("  🔍 ZOOM:     Mouse wheel / Trackpad scroll")
    print("  🎯 RESET:    Double click anywhere")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n  💡 TIP: If pan isn't working, try:")
    print("     - Shift + Left click + drag (works on all systems)")
    print("     - Middle mouse button + drag")
    print("     - On trackpad: Two-finger click + drag")
    print("\nNote: You may see version warnings - these can be ignored.")
    print("Press Ctrl+C to exit.")
    
    # Keep the script running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize 3D Gaussian Splats in Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all PLY files in a directory
  python visualize_with_rerun.py -i output_grok_video/gaussians/

  # Visualize with 20%% of points (much faster)
  python visualize_with_rerun.py -i output_grok_video/gaussians/ --downsample 0.2

  # Visualize with larger splats
  python visualize_with_rerun.py -i output_grok_video/gaussians/ --size 2.0

  # Visualize with smaller splats
  python visualize_with_rerun.py -i output_grok_video/gaussians/ --size 0.5

  # Rotate by 180 degrees (flip)
  python visualize_with_rerun.py -i output_grok_video/gaussians/ --rotate 180

  # Visualize a single PLY file
  python visualize_with_rerun.py -i output_grok_video/gaussians/frame_000000.ply

  # Limit to first 5 frames with 10%% downsampling and rotation
  python visualize_with_rerun.py -i output_grok_video/gaussians/ --max-frames 5 --downsample 0.1 --rotate 180
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to PLY file or directory containing PLY files"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to load (default: all)"
    )
    
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Fraction of points to keep (0.1 = 10%%, 0.5 = 50%%, 1.0 = 100%%). Lower values = faster performance."
    )
    
    parser.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="Size multiplier for splats (default: 1.0 = optimal size). Use 2.0 for 2x larger, 0.5 for 2x smaller."
    )
    
    parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Rotation angle in degrees around Y-axis (e.g., 180 to flip, 90 for 90° rotation). Default: 0"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1
    
    # Handle single file or directory
    if args.input.is_file():
        if args.input.suffix != ".ply":
            print(f"Error: {args.input} is not a PLY file")
            return 1
        
        # Visualize single file
        rr.init("3D Gaussian Splats Viewer", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Configure blueprint for optimal 3D viewing with full camera controls
        blueprint = rr.blueprint.Blueprint(
            rr.blueprint.Spatial3DView(
                name="3D View", 
                origin="world",
                # Enable all camera interaction modes
            ),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)
        
        visualize_gaussian_ply(args.input, 0, args.downsample, args.size, None, args.rotate)
        
        print("\n🎬 Rerun viewer is now open!")
        if args.rotate != 0.0:
            print(f"   Rotated by {args.rotate}° around Y-axis")
        print("\nNote: You may see version warnings - these can be ignored.")
        print("Press Ctrl+C to exit.")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    
    elif args.input.is_dir():
        visualize_directory(args.input, args.max_frames, args.downsample, args.size, args.rotate)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
