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
    """Load a PLY file and return splat data with sRGB colors.

    Supports both SHARP 3DGS format (f_dc_0/1/2, opacity, scale, rot) and
    simple point cloud format (red/green/blue).
    """
    from plyfile import PlyData
    import numpy as np

    ply = PlyData.read(str(ply_path))
    vertex = ply['vertex']
    props = [p.name for p in vertex.properties]

    # Positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)

    # Colors — handle both SHARP (f_dc) and simple (red/green/blue) formats
    if 'f_dc_0' in props:
        # SHARP 3DGS format — SH coefficients, convert to sRGB
        SH_C0 = 0.28209479177387814
        colors = np.stack([
            0.5 + SH_C0 * vertex['f_dc_0'],
            0.5 + SH_C0 * vertex['f_dc_1'],
            0.5 + SH_C0 * vertex['f_dc_2'],
        ], axis=-1).astype(np.float32)
    elif 'red' in props:
        # Simple point cloud format — already sRGB
        colors = np.stack([
            vertex['red'].astype(np.float32) / 255.0,
            vertex['green'].astype(np.float32) / 255.0,
            vertex['blue'].astype(np.float32) / 255.0,
        ], axis=-1)
    else:
        colors = np.full((len(positions), 3), 0.8, dtype=np.float32)

    # Scales
    if 'scale_0' in props:
        scales = np.stack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2']),
        ], axis=-1).astype(np.float32)
    else:
        scales = np.full((len(positions), 3), 0.01, dtype=np.float32)

    # Opacity filter
    if 'opacity' in props:
        opacities = 1.0 / (1.0 + np.exp(-vertex['opacity'].astype(np.float32)))
        mask = opacities > 0.1
        positions = positions[mask]
        colors = colors[mask]
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
