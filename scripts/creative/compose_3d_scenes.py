#!/usr/bin/env python3
"""
Compose multiple 3D Gaussian Splat scenes into one.
Create collages, arrange multiple objects, or build complex scenes.
"""

import argparse
import numpy as np
from pathlib import Path
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
import torch
from typing import List, Tuple


def load_and_prepare_scene(ply_path: Path, downsample: float = 1.0):
    """Load a PLY file and return prepared numpy arrays."""
    print(f"  Loading {ply_path.name}...")
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
    
    # Downsample if needed
    if downsample < 1.0:
        n_points = int(positions.shape[0] * downsample)
        indices = np.random.choice(positions.shape[0], n_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        scales = scales[indices]
    
    print(f"    {positions.shape[0]:,} points")
    
    return positions, colors, scales


def transform_scene(positions: np.ndarray, 
                    translation: np.ndarray = np.zeros(3),
                    rotation_y: float = 0.0,
                    scale: float = 1.0) -> np.ndarray:
    """Apply 3D transformation to a scene."""
    
    # Scale
    if scale != 1.0:
        positions = positions * scale
    
    # Rotate around Y axis
    if rotation_y != 0.0:
        angle = np.radians(rotation_y)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        positions = positions @ rotation_matrix.T
    
    # Translate
    positions = positions + translation
    
    return positions


def create_grid_layout(scenes: List[Tuple], grid_size: Tuple[int, int], spacing: float = 200):
    """Arrange scenes in a grid layout."""
    grid_rows, grid_cols = grid_size
    composed_positions = []
    composed_colors = []
    composed_scales = []
    
    scene_idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if scene_idx >= len(scenes):
                break
            
            positions, colors, scales = scenes[scene_idx]
            
            # Calculate grid position
            x_offset = (col - grid_cols/2 + 0.5) * spacing
            z_offset = (row - grid_rows/2 + 0.5) * spacing
            
            # Transform scene
            transformed_pos = transform_scene(
                positions, 
                translation=np.array([x_offset, 0, z_offset])
            )
            
            composed_positions.append(transformed_pos)
            composed_colors.append(colors)
            composed_scales.append(scales)
            
            scene_idx += 1
    
    # Concatenate all scenes
    all_positions = np.vstack(composed_positions)
    all_colors = np.vstack(composed_colors)
    all_scales = np.vstack(composed_scales)
    
    return all_positions, all_colors, all_scales


def create_circular_layout(scenes: List[Tuple], radius: float = 200):
    """Arrange scenes in a circle."""
    num_scenes = len(scenes)
    composed_positions = []
    composed_colors = []
    composed_scales = []
    
    for i, (positions, colors, scales) in enumerate(scenes):
        # Calculate angle for this scene
        angle = (i / num_scenes) * 360
        
        # Position in circle
        angle_rad = np.radians(angle)
        x_offset = radius * np.cos(angle_rad)
        z_offset = radius * np.sin(angle_rad)
        
        # Transform scene (also rotate to face center)
        transformed_pos = transform_scene(
            positions,
            translation=np.array([x_offset, 0, z_offset]),
            rotation_y=angle + 180,  # Face inward
            scale=0.8
        )
        
        composed_positions.append(transformed_pos)
        composed_colors.append(colors)
        composed_scales.append(scales)
    
    # Concatenate all scenes
    all_positions = np.vstack(composed_positions)
    all_colors = np.vstack(composed_colors)
    all_scales = np.vstack(composed_scales)
    
    return all_positions, all_colors, all_scales


def create_custom_layout(scenes: List[Tuple], positions: List[np.ndarray], 
                        rotations: List[float], scales: List[float]):
    """Custom layout with specified transforms for each scene."""
    composed_positions = []
    composed_colors = []
    composed_scales = []
    
    for (scene_pos, colors, scene_scales), trans, rot, scale in zip(scenes, positions, rotations, scales):
        transformed_pos = transform_scene(scene_pos, trans, rot, scale)
        composed_positions.append(transformed_pos)
        composed_colors.append(colors)
        
        # Scale the gaussian scales too
        composed_scales.append(scene_scales * scale)
    
    all_positions = np.vstack(composed_positions)
    all_colors = np.vstack(composed_colors)
    all_scales = np.vstack(composed_scales)
    
    return all_positions, all_colors, all_scales


def main():
    parser = argparse.ArgumentParser(
        description="Compose multiple 3D scenes into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid layout (2x2)
  python compose_3d_scenes.py --grid 2 2 --spacing 250 \\
      output_test/IMG_4707.ply \\
      output_test/IMG_4753.ply \\
      output_test/IMG_6496.ply \\
      output_test/IMG_6497.ply
  
  # Circular arrangement
  python compose_3d_scenes.py --circular --radius 300 \\
      output_test/*.ply
  
  # With downsampling for performance
  python compose_3d_scenes.py --grid 3 3 --downsample 0.3 \\
      output_test/*.ply
        """
    )
    
    parser.add_argument("input_files", nargs="+", type=Path, help="PLY files to compose")
    
    # Layout options
    layout_group = parser.add_mutually_exclusive_group(required=True)
    layout_group.add_argument("--grid", nargs=2, type=int, metavar=("ROWS", "COLS"), 
                             help="Arrange in grid (rows cols)")
    layout_group.add_argument("--circular", action="store_true", 
                             help="Arrange in circle")
    
    parser.add_argument("--spacing", type=float, default=200, 
                       help="Grid spacing (default: 200)")
    parser.add_argument("--radius", type=float, default=200, 
                       help="Circle radius (default: 200)")
    parser.add_argument("--downsample", type=float, default=1.0,
                       help="Downsample each scene (0.3 = 30%% of points)")
    
    args = parser.parse_args()
    
    # Expand any wildcards and check files
    input_files = []
    for pattern in args.input_files:
        if pattern.exists():
            if pattern.is_file():
                input_files.append(pattern)
        else:
            # Try glob pattern
            parent = pattern.parent if pattern.parent.exists() else Path('.')
            matched = list(parent.glob(pattern.name))
            input_files.extend(matched)
    
    if not input_files:
        print("Error: No valid PLY files found")
        return 1
    
    print(f"\nComposing {len(input_files)} scenes...")
    print("=" * 60)
    
    # Load all scenes
    scenes = []
    for ply_path in input_files:
        try:
            scene_data = load_and_prepare_scene(ply_path, args.downsample)
            scenes.append(scene_data)
        except Exception as e:
            print(f"  Warning: Failed to load {ply_path.name}: {e}")
    
    if not scenes:
        print("Error: No scenes loaded successfully")
        return 1
    
    print("\nComposing scene...")
    
    # Create composition based on layout
    if args.grid:
        rows, cols = args.grid
        if len(scenes) > rows * cols:
            print(f"Warning: {len(scenes)} scenes but grid is {rows}x{cols}, using first {rows*cols}")
            scenes = scenes[:rows * cols]
        
        all_positions, all_colors, all_scales = create_grid_layout(
            scenes, (rows, cols), args.spacing
        )
        layout_name = f"{rows}x{cols} Grid"
    
    elif args.circular:
        all_positions, all_colors, all_scales = create_circular_layout(
            scenes, args.radius
        )
        layout_name = "Circular"
    
    total_points = all_positions.shape[0]
    print(f"\n✓ Composed scene: {len(scenes)} objects, {total_points:,} total points")
    print(f"  Layout: {layout_name}")
    
    # Initialize Rerun
    rr.init(f"3D Composition: {layout_name}", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Set up blueprint
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Spatial3DView(name="Composed Scene", origin="world"),
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)
    
    # Log the composed scene
    rr.log(
        "world/composition",
        rr.Points3D(
            positions=all_positions,
            colors=all_colors,
            radii=np.mean(all_scales, axis=1) * 0.3,
        )
    )
    
    print("\n🎬 Composed scene is ready!")
    print("\n3D View Controls:")
    print("  🔄 Rotate: Left click + drag")
    print("  ↔️  Pan:    Shift + Left click + drag")
    print("  🔍 Zoom:   Mouse wheel")
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

