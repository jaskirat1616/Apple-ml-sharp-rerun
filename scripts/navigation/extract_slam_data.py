#!/usr/bin/env python3
"""
Extract SLAM data from 3D Gaussian Splat video sequences.

Outputs:
- Depth maps per frame
- Camera trajectory
- Point cloud registration
- Feature tracking data
- TUM/KITTI format exports

Use cases:
- Robot SLAM systems
- Visual odometry
- AR/VR tracking
- Drone navigation
"""

import argparse
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import json
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch
from tqdm import tqdm


def render_depth_map(positions, resolution=(640, 480), focal_length=500, max_depth=100):
    """
    Render depth map from 3D point cloud.
    Projects points to image plane and records depth values.
    """
    width, height = resolution
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    
    # Project points to image plane
    # Simple pinhole camera model
    for pos in positions:
        x, y, z = pos
        
        # Skip points behind camera
        if z <= 0:
            continue
        
        # Project to image coordinates
        u = int(focal_length * x / z + width / 2)
        v = int(focal_length * (-y) / z + height / 2)  # Flip Y
        
        # Check bounds
        if 0 <= u < width and 0 <= v < height:
            # Keep closest depth
            if z < depth_map[v, u]:
                depth_map[v, u] = z
    
    return depth_map


def estimate_camera_pose(positions_prev, positions_curr):
    """
    Estimate camera movement between frames using ICP-like approach.
    Returns rotation matrix and translation vector.
    """
    # Simple centroid-based alignment
    centroid_prev = positions_prev.mean(axis=0)
    centroid_curr = positions_curr.mean(axis=0)
    
    # Translation is difference in centroids
    translation = centroid_curr - centroid_prev
    
    # For rotation, use SVD (simplified ICP)
    H = (positions_prev - centroid_prev).T @ (positions_curr - centroid_curr)
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    
    return rotation, translation


def extract_features_from_depth(depth_map, min_disparity=0.1):
    """
    Extract feature points from depth map for tracking.
    """
    # Find edges in depth (depth discontinuities)
    sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Find strong edges
    threshold = np.percentile(edges, 95)
    feature_mask = edges > threshold
    
    # Get feature coordinates
    features = np.argwhere(feature_mask)
    
    return features


def export_tum_format(camera_trajectory, output_file):
    """
    Export camera trajectory in TUM RGB-D dataset format.
    Format: timestamp tx ty tz qx qy qz qw
    """
    from scipy.spatial.transform import Rotation
    
    with open(output_file, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i, pose in enumerate(camera_trajectory):
            timestamp = i * 0.033  # ~30fps
            tx, ty, tz = pose['translation']
            
            # Convert rotation matrix to quaternion
            R = np.array(pose['rotation'])
            quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
            qx, qy, qz, qw = quat
            
            f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"  Exported TUM format: {output_file}")


def export_kitti_format(camera_trajectory, output_file):
    """
    Export camera trajectory in KITTI odometry format.
    Format: 12 values representing 3x4 transformation matrix (row-major)
    """
    with open(output_file, 'w') as f:
        for pose in camera_trajectory:
            R = np.array(pose['rotation'])
            t = np.array(pose['translation']).reshape(3, 1)
            
            # Create 3x4 transformation matrix
            T = np.hstack([R, t])
            
            # Write as single line (row-major)
            values = T.flatten()
            line = ' '.join([f"{v:.6f}" for v in values])
            f.write(line + '\n')
    
    print(f"  Exported KITTI format: {output_file}")


def process_video_sequence(input_dir: Path, output_dir: Path, 
                          max_frames: int = None,
                          resolution: tuple = (640, 480),
                          export_format: str = 'all'):
    """
    Process entire video sequence and extract SLAM data.
    """
    # Find all PLY files
    ply_files = sorted(list(input_dir.glob("*.ply")))
    
    if max_frames:
        ply_files = ply_files[:max_frames]
    
    if len(ply_files) == 0:
        print(f"No PLY files found in {input_dir}")
        return 1
    
    print(f"Processing {len(ply_files)} frames...")
    print(f"Output directory: {output_dir}")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    (output_dir / "depth_maps").mkdir(exist_ok=True)
    (output_dir / "depth_colored").mkdir(exist_ok=True)
    
    # Storage for trajectory
    camera_trajectory = []
    depth_stats = []
    
    prev_positions = None
    
    # Process each frame
    for frame_idx, ply_path in enumerate(tqdm(ply_files, desc="Extracting SLAM data")):
        # Load Gaussian splat
        gaussians, metadata = load_ply(ply_path)
        positions = gaussians.mean_vectors.cpu().numpy().squeeze()
        colors = gaussians.colors.cpu().numpy().squeeze()
        opacities = gaussians.opacities.cpu().numpy().squeeze()
        
        # Filter by opacity
        mask = opacities > 0.1
        positions = positions[mask]
        colors = colors[mask]
        
        # Convert colors if needed
        if metadata.color_space == "linearRGB":
            colors_torch = torch.from_numpy(colors).float()
            colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()
        
        # Estimate camera pose (relative to previous frame)
        if prev_positions is not None and len(prev_positions) > 100 and len(positions) > 100:
            try:
                # Sample points for faster computation
                n_samples = min(1000, len(prev_positions), len(positions))
                prev_sample = prev_positions[np.random.choice(len(prev_positions), n_samples, replace=False)]
                curr_sample = positions[np.random.choice(len(positions), n_samples, replace=False)]
                
                rotation, translation = estimate_camera_pose(prev_sample, curr_sample)
            except:
                rotation = np.eye(3)
                translation = np.zeros(3)
        else:
            rotation = np.eye(3)
            translation = np.zeros(3)
        
        # Accumulate trajectory
        if frame_idx == 0:
            cumulative_rotation = np.eye(3)
            cumulative_translation = np.zeros(3)
        else:
            cumulative_translation = cumulative_translation + cumulative_rotation @ translation
            cumulative_rotation = cumulative_rotation @ rotation
        
        camera_trajectory.append({
            'frame': frame_idx,
            'rotation': cumulative_rotation.tolist(),
            'translation': cumulative_translation.tolist(),
            'relative_rotation': rotation.tolist(),
            'relative_translation': translation.tolist(),
        })
        
        # Render depth map
        focal_length = metadata.focal_length_px if hasattr(metadata, 'focal_length_px') else 500
        depth_map = render_depth_map(positions, resolution, focal_length)
        
        # Save raw depth map (16-bit PNG)
        depth_normalized = (depth_map / depth_map.max() * 65535).astype(np.uint16)
        depth_img = Image.fromarray(depth_normalized)
        depth_img.save(output_dir / "depth_maps" / f"depth_{frame_idx:06d}.png")
        
        # Save colored depth map for visualization
        depth_colored = cv2.applyColorMap(
            (depth_map / depth_map.max() * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )
        cv2.imwrite(
            str(output_dir / "depth_colored" / f"depth_{frame_idx:06d}.png"),
            depth_colored
        )
        
        # Statistics
        valid_depth = depth_map[depth_map < depth_map.max()]
        if len(valid_depth) > 0:
            depth_stats.append({
                'frame': frame_idx,
                'min_depth': float(valid_depth.min()),
                'max_depth': float(valid_depth.max()),
                'mean_depth': float(valid_depth.mean()),
                'median_depth': float(np.median(valid_depth)),
            })
        
        prev_positions = positions
    
    # Export trajectory in various formats
    print("\nExporting camera trajectory...")
    
    # JSON format (complete data)
    with open(output_dir / "trajectory.json", 'w') as f:
        json.dump({
            'frames': camera_trajectory,
            'depth_stats': depth_stats,
            'num_frames': len(ply_files),
            'resolution': resolution,
        }, f, indent=2)
    print(f"  Exported JSON: trajectory.json")
    
    # TUM format
    if export_format in ['tum', 'all']:
        try:
            export_tum_format(camera_trajectory, output_dir / "trajectory_tum.txt")
        except ImportError:
            print("  Warning: scipy not available, skipping TUM format")
    
    # KITTI format
    if export_format in ['kitti', 'all']:
        export_kitti_format(camera_trajectory, output_dir / "trajectory_kitti.txt")
    
    # Simple XYZ trajectory for plotting
    with open(output_dir / "trajectory_xyz.txt", 'w') as f:
        f.write("# frame x y z\n")
        for pose in camera_trajectory:
            frame = pose['frame']
            x, y, z = pose['translation']
            f.write(f"{frame} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"  Exported XYZ trajectory: trajectory_xyz.txt")
    
    # Create summary
    print("\n" + "=" * 60)
    print(f"✓ SLAM data extraction complete!")
    print(f"\nOutput files:")
    print(f"  - {len(ply_files)} depth maps (16-bit PNG)")
    print(f"  - {len(ply_files)} colored depth maps (visualization)")
    print(f"  - Camera trajectory (JSON, TUM, KITTI, XYZ formats)")
    print(f"\nCamera Motion:")
    if len(camera_trajectory) > 1:
        total_translation = np.linalg.norm(camera_trajectory[-1]['translation'])
        print(f"  Total displacement: {total_translation:.2f} m")
        print(f"  Final position: {camera_trajectory[-1]['translation']}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract SLAM data from 3D Gaussian Splat sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from video sequence
  python extract_slam_data.py -i output_grok_video/gaussians/ -o slam_data/
  
  # With specific resolution
  python extract_slam_data.py -i output_grok_video/gaussians/ -o slam_data/ --resolution 1280 720
  
  # Only TUM format
  python extract_slam_data.py -i output_grok_video/gaussians/ -o slam_data/ --format tum
  
  # First 30 frames only
  python extract_slam_data.py -i output_grok_video/gaussians/ -o slam_data/ --max-frames 30
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Input directory containing PLY files")
    parser.add_argument("-o", "--output", type=Path, required=True,
                       help="Output directory for SLAM data")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480],
                       metavar=("WIDTH", "HEIGHT"), help="Depth map resolution")
    parser.add_argument("--format", choices=['all', 'tum', 'kitti', 'json'],
                       default='all', help="Export format")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1
    
    resolution = tuple(args.resolution)
    
    return process_video_sequence(
        args.input, args.output, 
        args.max_frames, resolution, 
        args.format
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())

