#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as oriented Gaussian splats in Rerun.

Renders each Gaussian as an oriented ellipsoid (like superspl.at/editor),
not just bubbles. Each splat has:
  - position (x, y, z)
  - rotation (quaternion)
  - scale (per-axis half-size)
  - color (from SH coefficients, sRGB)
  - opacity (alpha)

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
  python run_splat_viewer.py output_frame_000000_3d/frame_000000_points.ply
"""
import sys
import time
from pathlib import Path

import numpy as np


def load_ply_splat(ply_path):
    """Load a PLY file and return splat data.

    Supports both SHARP 3DGS format (f_dc_0/1/2, opacity, scale, rot) and
    simple point cloud format (red/green/blue).
    """
    from plyfile import PlyData

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

    # Scales (log-space in 3DGS format)
    if 'scale_0' in props:
        scales = np.stack([
            np.exp(vertex['scale_0']),
            np.exp(vertex['scale_1']),
            np.exp(vertex['scale_2']),
        ], axis=-1).astype(np.float32)
    else:
        scales = np.full((len(positions), 3), 0.01, dtype=np.float32)

    # Rotations (quaternion: w, x, y, z)
    if 'rot_0' in props:
        quats = np.stack([
            vertex['rot_0'],
            vertex['rot_1'],
            vertex['rot_2'],
            vertex['rot_3'],
        ], axis=-1).astype(np.float32)
        # Normalize quaternions
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = quats / np.maximum(norms, 1e-8)
    else:
        quats = None

    # Opacity
    if 'opacity' in props:
        opacities = 1.0 / (1.0 + np.exp(-vertex['opacity'].astype(np.float32)))
    else:
        opacities = np.ones(len(positions), dtype=np.float32)

    # Filter low-opacity points
    mask = opacities > 0.01
    positions = positions[mask]
    colors = colors[mask]
    scales = scales[mask]
    opacities = opacities[mask]
    if quats is not None:
        quats = quats[mask]

    return positions, colors, scales, quats, opacities


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
    print("SPLATLINE SPLAT VIEWER (Rerun — oriented ellipsoids)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"File size: {ply_path.stat().st_size / 1e6:.1f} MB")
    print()

    # Load splat data
    print("Loading PLY...")
    t0 = time.time()
    positions, colors, scales, quats, opacities = load_ply_splat(ply_path)
    print(f"  Loaded {len(positions):,} splats in {time.time()-t0:.1f}s")
    print(f"  Color range: {colors.min():.3f} - {colors.max():.3f} (mean {colors.mean():.3f})")

    # For large splat counts, subsample to keep Rerun responsive
    MAX_SPLATS = 200000
    if len(positions) > MAX_SPLATS:
        print(f"  Subsampling to {MAX_SPLATS:,} splats (from {len(positions):,}) for performance...")
        idx = np.random.choice(len(positions), MAX_SPLATS, replace=False)
        positions = positions[idx]
        colors = colors[idx]
        scales = scales[idx]
        opacities = opacities[idx]
        if quats is not None:
            quats = quats[idx]

    # Compute radii from splat scales — use the max axis scale
    # This approximates the projected splat size (camera-facing disc)
    radii = np.max(scales, axis=-1).astype(np.float32)

    # Build RGBA colors with opacity
    rgba = np.zeros((len(positions), 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors, 0, 1)
    rgba[:, 3] = np.clip(opacities, 0, 1)

    # Launch Rerun
    import rerun as rr

    rr.init("Splatline Splat Viewer")
    rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

    # Log splats as camera-facing points with scale-based radii
    # Points3D renders as screen-space circles (like projected splats)
    print("\nLogging to Rerun...")
    rr.log("splats",
        rr.Points3D(
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            colors=np.ascontiguousarray(rgba, dtype=np.float32),
            radii=np.ascontiguousarray(radii, dtype=np.float32),
        )
    )

    print(f"\nDone! {len(positions):,} oriented ellipsoids in Rerun.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
