#!/usr/bin/env python3
"""
Splatline Image-to-3D — Convert a single image to a high-resolution solid 3D mesh.

Pipeline:
  1. Load image and convert to 3D Gaussians using SHARP
  2. Reconstruct a solid triangle mesh via voxel-based surface reconstruction
     (marching cubes at 1024-voxel resolution for maximum detail)
  3. Project the source image onto the mesh using bilinear color interpolation
  4. Visualize in Rerun and export to PLY/OBJ

Usage:
  python run_image_3d.py <image_file>
  python run_image_3d.py /path/to/photo.jpg
  python run_image_3d.py /path/to/photo.png --res 1024
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def convert_image_to_gaussians(image_path, output_dir, device="mps", internal_size=1536):
    """Convert a single image to 3D Gaussians PLY using SHARP."""
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply, unproject_gaussians

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    print(f"Device: {device}")

    # Load model
    print("Loading SHARP model...")
    MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device_obj)
    print("Model loaded")

    img_path = Path(image_path)
    out_ply = gaussians_dir / f"{img_path.stem}.ply"

    print(f"Converting {img_path.name}...", end="  ")

    image, _, f_px = io.load_rgb(img_path)
    height, width = image.shape[:2]

    # Preprocess
    image_pt = torch.from_numpy(image.copy()).float().to(device_obj).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device_obj)

    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_size, internal_size),
        mode="bilinear",
        align_corners=True,
    )

    with torch.no_grad():
        gaussians_ndc = predictor(image_resized, disparity_factor)

    # Build intrinsics for unprojection
    intrinsics = (
        torch.tensor([
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]).float().to(device_obj)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_size / width
    intrinsics_resized[1] *= internal_size / height

    # Unproject from NDC to metric 3D space
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device_obj), intrinsics_resized,
        (internal_size, internal_size),
    )

    save_ply(gaussians, f_px, (height, width), out_ply)
    print("done")

    # Copy image to frames dir for color projection
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"{img_path.stem}.png"
    if not frame_path.exists():
        cv2.imwrite(str(frame_path), cv2.imread(str(img_path)))

    return out_ply, frame_path


def load_gaussian_cloud(ply_path):
    """Load a PLY and return positions, colors, scales, opacities as numpy arrays."""
    from sharp.utils.gaussians import load_ply
    gaussians, metadata = load_ply(Path(ply_path))
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    return positions, colors, scales, opacities


def reconstruct_solid_mesh(positions, colors, resolution=1024, dilation_iter=3):
    """Reconstruct a solid triangle mesh from a point cloud.

    Uses voxel-based surface reconstruction:
      1. Voxelize the point cloud into a boolean 3D grid
      2. Dilate to close small gaps
      3. Fill interior holes to make it solid
      4. Marching cubes to extract a triangle mesh surface

    Args:
        positions: Nx3 array of 3D point positions
        colors: Nx3 array of point colors (0-1 range)
        resolution: Number of voxels along the longest axis (higher = more detail)
        dilation_iter: Number of dilation iterations for gap closing

    Returns:
        verts: Mx3 array of mesh vertices (in original coordinate space)
        faces: Fx3 array of triangle indices
        normals: Mx3 array of vertex normals
    """
    from scipy.ndimage import binary_fill_holes, binary_dilation
    from skimage import measure

    print(f"  Voxel grid: {resolution} resolution", end="  ")

    extent = positions.max(axis=0) - positions.min(axis=0)
    voxel_size = extent.max() / resolution
    origin = positions.min(axis=0)
    shifted = positions - origin

    voxel_indices = np.floor(shifted / voxel_size).astype(int)
    shape = tuple(voxel_indices.max(axis=0) + 1)
    matrix = np.zeros(shape, dtype=bool)
    flat = np.ravel_multi_index(
        [voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]], shape
    )
    matrix.ravel()[flat] = True

    # Dilate to close small gaps, then fill holes to make it solid
    dilated = binary_dilation(matrix, iterations=dilation_iter)
    filled = binary_fill_holes(dilated)

    # Marching cubes via scikit-image
    verts, faces, normals, _ = measure.marching_cubes(filled, level=0.5)

    # Transform vertices back to original coordinate space
    verts = verts * voxel_size + origin

    grid_mb = np.prod(shape) / 1e6
    print(f"grid={shape} ({grid_mb:.1f}M), mesh={len(verts):,} verts, {len(faces):,} faces")

    return verts, faces, normals


def project_colors_bilinear(verts, source_img, f_px, positions, colors):
    """Project source image colors onto mesh vertices using bilinear interpolation.

    Blends 60% bilinear-interpolated image colors with 40% point cloud colors
    for maximum detail and smoothness.

    Args:
        verts: Mx3 array of mesh vertices
        source_img: HxWx3 float32 array (RGB, 0-255)
        f_px: Focal length in pixels
        positions: Nx3 array of original point cloud positions
        colors: Nx3 array of point cloud colors (0-1 range)

    Returns:
        Mx4 uint8 array of RGBA vertex colors
    """
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    img_h, img_w = source_img.shape[:2]

    # Project 3D vertices to 2D image coordinates
    z = verts[:, 2].copy()
    z[np.abs(z) < 1e-6] = 1e-6
    u_f = f_px * verts[:, 0] / z + img_w / 2
    v_f = f_px * verts[:, 1] / z + img_h / 2

    # Bilinear interpolation for smooth colors
    img_colors = np.zeros((len(verts), 3), dtype=np.float32)
    for c in range(3):
        img_colors[:, c] = map_coordinates(
            source_img[:, :, c], [v_f, u_f], order=1, mode='reflect'
        )

    # Also get colors from nearest point cloud point (full detail)
    color_tree = cKDTree(positions)
    _, nearest = color_tree.query(verts, k=1)
    cloud_colors = (colors[nearest] * 255).astype(np.float32)

    # Blend: 60% bilinear image + 40% point cloud for maximum detail
    vertex_colors = (img_colors * 0.6 + cloud_colors * 0.4).clip(0, 255).astype(np.uint8)

    # Add alpha
    vertex_colors_rgba = np.hstack([
        vertex_colors,
        np.full((len(vertex_colors), 1), 255, dtype=np.uint8),
    ])

    return vertex_colors_rgba


def export_ply(verts, faces, colors, output_path):
    """Export mesh to PLY format with vertex colors."""
    print(f"  Exporting PLY: {output_path}")
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            v = verts[i]
            c = colors[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]} {c[3]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"  PLY saved ({len(verts):,} verts, {len(faces):,} faces)")


def export_obj(verts, faces, colors, output_path):
    """Export mesh to OBJ format with vertex colors (as MTL-like inline)."""
    print(f"  Exporting OBJ: {output_path}")
    with open(output_path, 'w') as f:
        for i in range(len(verts)):
            v = verts[i]
            c = colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]/255:.6f} {c[1]/255:.6f} {c[2]/255:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  OBJ saved ({len(verts):,} verts, {len(faces):,} faces)")


def view_points_in_rerun(positions, colors, scales, source_img_rgb, f_px):
    """Visualize the SHARP Gaussian point cloud in Rerun."""
    import rerun as rr

    rr.init("Splatline Image-to-3D Point Cloud (SHARP)")
    rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

    h, w = source_img_rgb.shape[:2]

    # Compute radii from scales
    if scales.ndim > 1:
        radii = np.mean(scales, axis=1) * 2.0
    else:
        radii = scales * 2.0

    # Log the point cloud
    rr.log("image/points",
        rr.Points3D(
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            colors=np.ascontiguousarray(colors, dtype=np.float32),
            radii=np.ascontiguousarray(radii, dtype=np.float32),
        )
    )

    # Log source image with pinhole camera
    rr.log("image/camera",
        rr.Pinhole(width=w, height=h, focal_length=float(f_px)),
    )
    rr.log("image/camera/image", rr.Image(source_img_rgb.astype(np.uint8)))

    print("\nRerun viewer is open. Explore the SHARP point cloud.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def export_point_ply(positions, colors, output_path):
    """Export point cloud to PLY format with colors."""
    print(f"  Exporting PLY: {output_path}")
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(positions)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(positions)):
            p = positions[i]
            c = colors[i]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")
    print(f"  PLY saved ({len(positions):,} points)")


def view_in_rerun(verts, faces, normals, colors_rgba, source_img_rgb, f_px):
    """Visualize the solid mesh in Rerun with the source image."""
    import rerun as rr

    rr.init("Splatline Image-to-3D Solid Mesh")
    rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

    h, w = source_img_rgb.shape[:2]

    # Log the solid mesh
    rr.log("image/solid_mesh",
        rr.Mesh3D(
            vertex_positions=np.ascontiguousarray(verts, dtype=np.float32),
            triangle_indices=np.ascontiguousarray(faces, dtype=np.uint32),
            vertex_colors=colors_rgba,
            vertex_normals=np.ascontiguousarray(normals, dtype=np.float32),
        )
    )

    # Log source image with pinhole camera
    rr.log("image/camera",
        rr.Pinhole(width=w, height=h, focal_length=float(f_px)),
    )
    rr.log("image/camera/image", rr.Image(source_img_rgb.astype(np.uint8)))

    print("\nRerun viewer is open. Explore the 3D mesh.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a single image to a 3D model using Apple SHARP."
    )
    parser.add_argument("image", type=str, help="Path to the input image (JPG, PNG)")
    parser.add_argument("--mode", type=str, default="solid", choices=["points", "solid"],
                        help="Output mode: 'points' (SHARP point cloud) or 'solid' (mesh) (default: solid)")
    parser.add_argument("--res", type=int, default=1024,
                        help="Voxel resolution for solid mode (default: 1024, max: 1536)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Torch device: mps, cuda, or cpu")
    parser.add_argument("--internal-size", type=int, default=1536,
                        help="SHARP inference size (default: 1536)")
    parser.add_argument("--dilation", type=int, default=3,
                        help="Dilation iterations for gap closing in solid mode (default: 3)")
    parser.add_argument("--no-view", action="store_true",
                        help="Skip Rerun visualization, just export files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output_<image_name>)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"output_{image_path.stem}_3d")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    if args.mode == "points":
        print("SPLATLINE IMAGE-TO-3D — SHARP POINT CLOUD")
    else:
        print("SPLATLINE IMAGE-TO-3D — HIGH RESOLUTION SOLID MESH")
    print("=" * 60)
    print(f"Image: {image_path.name}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    if args.mode == "solid":
        print(f"Voxel resolution: {args.res}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Convert image to 3D Gaussians with SHARP
    ply_path, frame_path = convert_image_to_gaussians(
        image_path, output_dir, device=args.device, internal_size=args.internal_size
    )

    # Step 2: Load the Gaussian point cloud
    print("\nLoading Gaussian point cloud...")
    positions, colors, scales, opacities = load_gaussian_cloud(ply_path)

    # Filter low-opacity points
    if opacities.ndim > 0:
        mask = opacities > 0.1
        positions = positions[mask]
        colors = colors[mask]
        if scales.ndim > 1 and scales.shape[0] == len(opacities):
            scales = scales[mask]

    print(f"  Points: {len(positions):,}")

    if len(positions) < 100:
        print("Error: Not enough points")
        sys.exit(1)

    # Load source image for color projection and visualization
    source_img = cv2.imread(str(frame_path))
    source_img_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    source_img_f = source_img_rgb.astype(np.float32)

    from sharp.utils import io as sharp_io
    _, _, f_px = sharp_io.load_rgb(Path(frame_path))

    if args.mode == "points":
        # --- Point cloud mode (Apple SHARP raw output) ---
        print("\nPoint cloud mode — exporting raw SHARP Gaussians...")

        # Export point cloud PLY
        ply_out = output_dir / f"{image_path.stem}_points.ply"
        export_point_ply(positions, colors, ply_out)

        if not args.no_view:
            print("\nLaunching Rerun viewer...")
            view_points_in_rerun(positions, colors, scales, source_img_rgb, f_px)
        else:
            print("\nSkipping Rerun viewer (--no-view flag)")

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"Points: {len(positions):,}")
        print(f"PLY:   {ply_out}")

    else:
        # --- Solid mesh mode ---
        if len(positions) < 100:
            print("Error: Not enough points to reconstruct mesh")
            sys.exit(1)

        # Step 3: Reconstruct solid mesh
        print(f"\nReconstructing solid mesh at {args.res} resolution...")
        t0 = time.time()
        verts, faces, normals = reconstruct_solid_mesh(
            positions, colors, resolution=args.res, dilation_iter=args.dilation
        )
        print(f"  Reconstruction took {time.time() - t0:.1f}s")

        # Step 4: Project colors from source image
        print("\nProjecting colors from source image (bilinear interpolation)...")
        t0 = time.time()
        colors_rgba = project_colors_bilinear(verts, source_img_f, f_px, positions, colors)
        unique_colors = len(np.unique(colors_rgba[:, :3], axis=0))
        print(f"  Color projection took {time.time() - t0:.1f}s")
        print(f"  Unique colors: {unique_colors:,}")
        print(f"  Color stats: R={colors_rgba[:,0].mean():.0f} G={colors_rgba[:,1].mean():.0f} B={colors_rgba[:,2].mean():.0f}")

        # Step 5: Export mesh files
        print("\nExporting mesh files...")
        ply_out = output_dir / f"{image_path.stem}_solid_mesh.ply"
        obj_out = output_dir / f"{image_path.stem}_solid_mesh.obj"
        export_ply(verts, faces, colors_rgba, ply_out)
        export_obj(verts, faces, colors_rgba, obj_out)

        # Step 6: Visualize in Rerun
        if not args.no_view:
            print("\nLaunching Rerun viewer...")
            view_in_rerun(verts, faces, normals, colors_rgba, source_img_rgb, f_px)
        else:
            print("\nSkipping Rerun viewer (--no-view flag)")

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)
        print(f"Mesh: {len(verts):,} vertices, {len(faces):,} faces")
        print(f"PLY:  {ply_out}")
        print(f"OBJ:  {obj_out}")


if __name__ == "__main__":
    main()
