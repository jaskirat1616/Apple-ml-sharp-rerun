#!/usr/bin/env python3
"""
Apply depth-based effects to 3D Gaussian Splats.
- Atmospheric fog
- Depth-based color grading
- Selective focus (bokeh)
- Distance-based filtering
"""

import argparse
import numpy as np
from pathlib import Path
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch


def apply_fog_effect(positions, colors, fog_start, fog_end, fog_color=np.array([0.8, 0.9, 1.0])):
    """Apply atmospheric fog based on distance from camera."""
    # Calculate distance from origin (camera position)
    distances = np.linalg.norm(positions, axis=1)
    
    # Calculate fog factor (0 = no fog, 1 = full fog)
    fog_factor = np.clip((distances - fog_start) / (fog_end - fog_start), 0, 1)
    
    # Blend colors with fog
    fog_factor = fog_factor[:, np.newaxis]  # Make it (N, 1) for broadcasting
    new_colors = colors * (1 - fog_factor) + fog_color * fog_factor
    
    return new_colors


def apply_depth_color_grading(positions, colors, near_color, far_color, depth_range):
    """Apply color grading that varies with depth."""
    distances = np.linalg.norm(positions, axis=1)
    
    # Normalize distances to [0, 1]
    t = np.clip((distances - depth_range[0]) / (depth_range[1] - depth_range[0]), 0, 1)
    t = t[:, np.newaxis]
    
    # Blend between near and far color tints
    color_tint = near_color * (1 - t) + far_color * t
    
    # Apply tint to colors
    new_colors = colors * color_tint
    new_colors = np.clip(new_colors, 0, 1)
    
    return new_colors


def apply_distance_filter(positions, colors, scales, opacities, min_dist, max_dist):
    """Keep only points within a distance range."""
    distances = np.linalg.norm(positions, axis=1)
    mask = (distances >= min_dist) & (distances <= max_dist)
    
    return (
        positions[mask],
        colors[mask],
        scales[mask],
        opacities[mask],
        mask.sum()
    )


def apply_selective_focus(positions, scales, opacities, focus_distance, focus_range):
    """Simulate depth of field - blur points far from focus distance."""
    distances = np.linalg.norm(positions, axis=1)
    
    # Calculate blur amount
    blur_amount = np.abs(distances - focus_distance) / focus_range
    blur_amount = np.clip(blur_amount, 0, 1)
    
    # Increase scale for blurred points (simulates bokeh)
    new_scales = scales * (1 + blur_amount[:, np.newaxis] * 3)
    
    # Reduce opacity for very blurred points
    new_opacities = opacities * (1 - blur_amount * 0.5)
    
    return new_scales, new_opacities


def visualize_with_effects(ply_path: Path, effect_type: str, **kwargs):
    """Load PLY and apply depth-based effects."""
    
    print(f"Loading {ply_path.name}...")
    gaussians, metadata = load_ply(ply_path)
    
    # Extract data
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    
    # Convert colors
    if metadata.color_space == "linearRGB":
        colors_torch = torch.from_numpy(colors).float()
        colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()
    
    # Filter by opacity
    opacity_mask = opacities > 0.1
    positions = positions[opacity_mask]
    colors = colors[opacity_mask]
    scales = scales[opacity_mask]
    opacities = opacities[opacity_mask]
    
    original_count = positions.shape[0]
    print(f"Loaded {original_count:,} points")
    
    # Calculate scene bounds
    distances = np.linalg.norm(positions, axis=1)
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    
    print(f"Depth range: {min_dist:.1f} to {max_dist:.1f} (mean: {mean_dist:.1f})")
    
    # Apply effects
    if effect_type == "fog":
        fog_start = kwargs.get('fog_start', mean_dist)
        fog_end = kwargs.get('fog_end', max_dist)
        fog_color = kwargs.get('fog_color', np.array([0.7, 0.8, 0.9]))
        
        print(f"Applying fog: start={fog_start:.1f}, end={fog_end:.1f}")
        colors = apply_fog_effect(positions, colors, fog_start, fog_end, fog_color)
        effect_name = "Fog"
    
    elif effect_type == "grading":
        near_color = kwargs.get('near_color', np.array([1.2, 1.0, 0.9]))  # Warm
        far_color = kwargs.get('far_color', np.array([0.8, 0.9, 1.2]))    # Cool
        depth_range = kwargs.get('depth_range', [min_dist, max_dist])
        
        print(f"Applying color grading: warm→cool")
        colors = apply_depth_color_grading(positions, colors, near_color, far_color, depth_range)
        effect_name = "Color Grading"
    
    elif effect_type == "filter":
        min_d = kwargs.get('min_dist', min_dist)
        max_d = kwargs.get('max_dist', max_dist)
        
        print(f"Filtering distance: {min_d:.1f} to {max_d:.1f}")
        positions, colors, scales, opacities, new_count = apply_distance_filter(
            positions, colors, scales, opacities, min_d, max_d
        )
        print(f"Kept {new_count:,} / {original_count:,} points ({100*new_count/original_count:.1f}%)")
        effect_name = "Distance Filter"
    
    elif effect_type == "focus":
        focus_distance = kwargs.get('focus_distance', mean_dist)
        focus_range = kwargs.get('focus_range', (max_dist - min_dist) * 0.2)
        
        print(f"Applying selective focus: distance={focus_distance:.1f}, range={focus_range:.1f}")
        scales, opacities = apply_selective_focus(positions, scales, opacities, focus_distance, focus_range)
        effect_name = "Selective Focus"
    
    else:
        print(f"Unknown effect: {effect_type}")
        return 1
    
    # Initialize Rerun
    rr.init(f"Depth Effect: {effect_name}", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Log the modified point cloud
    rr.log(
        "world/scene",
        rr.Points3D(
            positions=positions,
            colors=colors,
            radii=np.mean(scales, axis=1) * 0.3,
        )
    )
    
    print(f"\n✓ Applied {effect_name} effect")
    print("Press Ctrl+C to exit.")
    
    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Apply depth-based effects to 3D Gaussian Splats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Atmospheric fog
  python apply_depth_effects.py -i output_test/IMG_4707.ply --fog --fog-start 80 --fog-end 200
  
  # Depth color grading (warm foreground, cool background)
  python apply_depth_effects.py -i output_test/IMG_4707.ply --grading
  
  # Distance filter (slice the scene)
  python apply_depth_effects.py -i output_test/IMG_4707.ply --filter --min-dist 50 --max-dist 150
  
  # Selective focus (depth of field)
  python apply_depth_effects.py -i output_test/IMG_4707.ply --focus --focus-dist 100 --focus-range 30
        """
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to PLY file")
    
    # Effect types (mutually exclusive)
    effect_group = parser.add_mutually_exclusive_group(required=True)
    effect_group.add_argument("--fog", action="store_true", help="Apply atmospheric fog")
    effect_group.add_argument("--grading", action="store_true", help="Apply depth color grading")
    effect_group.add_argument("--filter", action="store_true", help="Filter by distance")
    effect_group.add_argument("--focus", action="store_true", help="Apply selective focus")
    
    # Fog parameters
    parser.add_argument("--fog-start", type=float, help="Fog start distance")
    parser.add_argument("--fog-end", type=float, help="Fog end distance")
    
    # Filter parameters
    parser.add_argument("--min-dist", type=float, help="Minimum distance")
    parser.add_argument("--max-dist", type=float, help="Maximum distance")
    
    # Focus parameters
    parser.add_argument("--focus-dist", type=float, help="Focus distance")
    parser.add_argument("--focus-range", type=float, help="Focus range (DOF)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1
    
    # Determine effect type
    if args.fog:
        effect_type = "fog"
        kwargs = {}
        if args.fog_start is not None:
            kwargs['fog_start'] = args.fog_start
        if args.fog_end is not None:
            kwargs['fog_end'] = args.fog_end
    
    elif args.grading:
        effect_type = "grading"
        kwargs = {}
    
    elif args.filter:
        effect_type = "filter"
        kwargs = {}
        if args.min_dist is not None:
            kwargs['min_dist'] = args.min_dist
        if args.max_dist is not None:
            kwargs['max_dist'] = args.max_dist
    
    elif args.focus:
        effect_type = "focus"
        kwargs = {}
        if args.focus_dist is not None:
            kwargs['focus_distance'] = args.focus_dist
        if args.focus_range is not None:
            kwargs['focus_range'] = args.focus_range
    
    return visualize_with_effects(args.input, effect_type, **kwargs)


if __name__ == "__main__":
    import sys
    sys.exit(main())

