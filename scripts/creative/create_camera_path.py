#!/usr/bin/env python3
"""
Create custom camera paths through 3D Gaussian Splat scenes.
Generate smooth orbiting, dolly, or custom camera movements.
"""

import argparse
import numpy as np
from pathlib import Path
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch


def generate_orbit_path(center, radius, num_frames=60, height_variation=0.0):
    """Generate circular orbit camera path."""
    angles = np.linspace(0, 2 * np.pi, num_frames)
    
    positions = []
    for i, angle in enumerate(angles):
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        y = center[1] + height_variation * np.sin(angle * 2)  # Optional height variation
        positions.append([x, y, z])
    
    return np.array(positions)


def generate_dolly_zoom_path(start_pos, end_pos, num_frames=30):
    """Generate dolly in/out path."""
    positions = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        pos = start_pos + t * (end_pos - start_pos)
        positions.append(pos)
    
    return np.array(positions)


def generate_figure_eight_path(center, radius, num_frames=120):
    """Generate figure-8 path."""
    angles = np.linspace(0, 4 * np.pi, num_frames)
    
    positions = []
    for angle in angles:
        x = center[0] + radius * np.sin(angle)
        z = center[2] + radius * np.sin(2 * angle) / 2
        y = center[1] + radius * 0.2 * np.cos(angle * 3)
        positions.append([x, y, z])
    
    return np.array(positions)


def visualize_camera_path(ply_path: Path, camera_positions: np.ndarray, output_name: str = "camera_path"):
    """Visualize the scene with animated camera path."""
    
    print(f"Loading {ply_path.name}...")
    gaussians, metadata = load_ply(ply_path)
    
    # Extract and prepare point cloud data
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    
    # Convert colors if needed
    if metadata.color_space == "linearRGB":
        colors_torch = torch.from_numpy(colors).float()
        colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()
    
    # Filter by opacity
    opacity_mask = opacities > 0.1
    positions = positions[opacity_mask]
    colors = colors[opacity_mask]
    scales = scales[opacity_mask]
    
    print(f"Loaded {positions.shape[0]:,} points")
    
    # Initialize Rerun
    rr.init(f"Camera Path: {output_name}", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Log the static point cloud
    rr.set_time_sequence("frame", 0)
    rr.log(
        "world/scene",
        rr.Points3D(
            positions=positions,
            colors=colors,
            radii=np.mean(scales, axis=1) * 0.3,
        )
    )
    
    # Calculate look-at point (center of scene)
    scene_center = positions.mean(axis=0)
    
    # Animate camera along path
    for frame_idx, cam_pos in enumerate(camera_positions):
        rr.set_time_sequence("frame", frame_idx)
        
        # Calculate camera orientation (look at scene center)
        forward = scene_center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Simple up vector
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create rotation matrix
        rotation = np.column_stack([right, up, -forward])
        
        # Log camera
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=cam_pos,
                mat3x3=rotation,
            )
        )
        
        # Log camera position as a point for visualization
        rr.log(
            "world/camera_path",
            rr.Points3D(
                positions=[cam_pos],
                colors=[[255, 0, 0]],
                radii=[1.0]
            )
        )
    
    print(f"\n✓ Created camera path with {len(camera_positions)} frames")
    print(f"  Scene center: ({scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f})")
    print("\nUse the timeline to animate the camera!")
    print("Press Ctrl+C to exit.")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(
        description="Create custom camera paths through 3D scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Orbit around a scene
  python create_camera_path.py -i output_test/IMG_4707.ply --orbit --radius 100 --frames 60
  
  # Figure-8 path
  python create_camera_path.py -i output_test/IMG_4707.ply --figure8 --radius 80 --frames 120
  
  # Custom dolly zoom
  python create_camera_path.py -i output_test/IMG_4707.ply --dolly --start-dist 50 --end-dist 200 --frames 60
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to PLY file")
    parser.add_argument("--orbit", action="store_true", help="Create circular orbit path")
    parser.add_argument("--figure8", action="store_true", help="Create figure-8 path")
    parser.add_argument("--dolly", action="store_true", help="Create dolly in/out path")
    parser.add_argument("--radius", type=float, default=100, help="Path radius (default: 100)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames (default: 60)")
    parser.add_argument("--height-var", type=float, default=0, help="Height variation for orbit (default: 0)")
    parser.add_argument("--start-dist", type=float, default=50, help="Start distance for dolly")
    parser.add_argument("--end-dist", type=float, default=200, help="End distance for dolly")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1
    
    # Load scene to get center point
    print("Analyzing scene...")
    gaussians, _ = load_ply(args.input)
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    scene_center = positions.mean(axis=0)
    
    # Generate camera path based on mode
    if args.orbit:
        print(f"Generating orbit path: {args.frames} frames, radius {args.radius}")
        camera_positions = generate_orbit_path(
            scene_center, 
            args.radius, 
            args.frames,
            args.height_var
        )
        path_name = "orbit"
    
    elif args.figure8:
        print(f"Generating figure-8 path: {args.frames} frames, radius {args.radius}")
        camera_positions = generate_figure_eight_path(
            scene_center,
            args.radius,
            args.frames
        )
        path_name = "figure8"
    
    elif args.dolly:
        print(f"Generating dolly path: {args.frames} frames")
        start_pos = scene_center + np.array([0, 0, args.start_dist])
        end_pos = scene_center + np.array([0, 0, args.end_dist])
        camera_positions = generate_dolly_zoom_path(start_pos, end_pos, args.frames)
        path_name = "dolly"
    
    else:
        print("Error: Please specify --orbit, --figure8, or --dolly")
        return 1
    
    # Visualize
    visualize_camera_path(args.input, camera_positions, path_name)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

