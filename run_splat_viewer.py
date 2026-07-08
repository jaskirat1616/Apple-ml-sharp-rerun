#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats in Rerun.

Renders PLY files as Gaussian splats with proper colors (sRGB converted from
SHARP's linearRGB), opacity filtering, and scale-based radii.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
  python run_splat_viewer.py output_frame_000000_3d/frame_000000_points.ply
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_ply_splat(ply_path):
    """Load a PLY file and return splat data with sRGB colors."""
    from sharp.utils.gaussians import load_ply
    from sharp.utils import color_space as cs_utils

    gaussians, metadata = load_ply(Path(ply_path))

    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()

    # SHARP stores colors in linearRGB — convert to sRGB for correct display
    colors = cs_utils.linearRGB2sRGB(torch.from_numpy(colors).float()).numpy()

    # Filter low-opacity points
    if opacities.ndim > 0:
        mask = opacities > 0.1
        positions = positions[mask]
        colors = colors[mask]
        if scales.ndim > 1 and scales.shape[0] == len(opacities):
            scales = scales[mask]

    return positions, colors, scales


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_splat_viewer.py <ply_file>")
        print()
        print("Example:")
        print("  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply")
        print("  python run_splat_viewer.py output_frame_000000_3d/frame_000000_points.ply")
        sys.exit(1)

    ply_path = Path(sys.argv[1]).resolve()
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER (Rerun)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"File size: {ply_path.stat().st_size / 1e6:.1f} MB")
    print()

    # Load splat data
    print("Loading PLY...")
    t0 = time.time()
    positions, colors, scales = load_ply_splat(ply_path)
    print(f"  Loaded {len(positions):,} splats in {time.time()-t0:.1f}s")
    print(f"  Color range: {colors.min():.3f} - {colors.max():.3f} (mean {colors.mean():.3f})")

    # Compute radii from scales
    if scales.ndim > 1:
        radii = np.mean(scales, axis=1) * 2.0
    else:
        radii = scales * 2.0

    # Launch Rerun
    import rerun as rr

    rr.init("Splatline Splat Viewer")
    rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

    # Log the splats
    print("\nLogging to Rerun...")
    rr.log("splats",
        rr.Points3D(
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            colors=np.ascontiguousarray(colors, dtype=np.float32),
            radii=np.ascontiguousarray(radii, dtype=np.float32),
        )
    )

    print(f"\nDone! {len(positions):,} splats in Rerun.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
