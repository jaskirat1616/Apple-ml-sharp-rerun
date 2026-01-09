#!/usr/bin/env python3
"""
Example: Basic visualization of a 3D Gaussian Splatting scene.

This demonstrates how to use the utility modules to visualize
a PLY file with minimal code.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_gaussian_data
from utils.visualization import setup_complete_viewer_blueprint
import rerun as rr
import numpy as np


def visualize_example(ply_path: Path):
    """Simple example of visualizing a PLY file."""
    print(f"Loading {ply_path}...")
    
    # Load Gaussian Splatting data
    data = load_gaussian_data(ply_path)
    if data is None:
        print("Failed to load PLY file")
        return
    
    positions = data['positions']
    colors = data['colors']
    scales = data['scales']
    
    print(f"Loaded {len(positions):,} points")
    
    # Initialize Rerun
    rr.init("Example Visualization", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Setup blueprint
    blueprint = setup_complete_viewer_blueprint()
    rr.send_blueprint(blueprint)
    
    # Log point cloud
    rr.log(
        "world/scene/points",
        rr.Points3D(
            positions=positions,
            colors=colors,
            radii=np.mean(scales, axis=1) * 0.3
        )
    )
    
    print("Visualization ready! Close the window when done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example visualization")
    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Path to PLY file")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        sys.exit(1)
    
    visualize_example(args.input)

