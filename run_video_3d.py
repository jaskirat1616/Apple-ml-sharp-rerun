#!/usr/bin/env python3
"""Run the full video-to-3D pipeline on a video and view in Rerun."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

VIDEO_PATH = Path("/Users/jaskiratsingh/Downloads/grok-video-a1a6d6a4-6f94-41c2-82b5-83ec305487ae.mp4")
OUTPUT_DIR = Path("output_grok_3d")
FRAME_SKIP = 3  # Extract every 3rd frame (24fps -> 8fps, 80 frames)
INTERNAL_SIZE = 1536  # SHARP inference size (default)


def extract_frames(video_path, output_dir, frame_skip=1):
    """Extract frames from video."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {total} frames @ {fps:.1f} fps")
    print(f"Extracting every {frame_skip}th frame...")

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            path = frames_dir / f"frame_{saved:06d}.png"
            cv2.imwrite(str(path), frame)
            saved += 1
            if saved % 10 == 0:
                print(f"  Extracted {saved}/{total // frame_skip}", end="\r")
        count += 1

    cap.release()
    print(f"\nExtracted {saved} frames")
    return frames_dir, fps / frame_skip


def convert_with_sharp(frames_dir, output_dir, device="mps"):
    """Convert frames to 3DGS PLY using SHARP."""
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

    # Get frames
    image_paths = sorted(frames_dir.glob("*.png"))
    print(f"Converting {len(image_paths)} frames...")

    for idx, img_path in enumerate(image_paths):
        out_ply = gaussians_dir / f"{img_path.stem}.ply"
        if out_ply.exists():
            print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} (skip, exists)")
            continue

        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name}", end="  ")

        image, _, f_px = io.load_rgb(img_path)
        height, width = image.shape[:2]

        # Preprocess
        image_pt = torch.from_numpy(image.copy()).float().to(device_obj).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        disparity_factor = torch.tensor([f_px / width]).float().to(device_obj)

        image_resized = F.interpolate(
            image_pt[None],
            size=(INTERNAL_SIZE, INTERNAL_SIZE),
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
        intrinsics_resized[0] *= INTERNAL_SIZE / width
        intrinsics_resized[1] *= INTERNAL_SIZE / height

        # Unproject from NDC to metric 3D space
        gaussians = unproject_gaussians(
            gaussians_ndc, torch.eye(4).to(device_obj), intrinsics_resized,
            (INTERNAL_SIZE, INTERNAL_SIZE),
        )

        save_ply(gaussians, f_px, (height, width), out_ply)
        print("done")

    return gaussians_dir


def view_in_rerun(gaussians_dir, frames_dir, fps):
    """View the 3D Gaussian sequence in Rerun (original point-cloud style)."""
    import rerun as rr
    from sharp.utils.gaussians import load_ply
    from sharp.utils import color_space as cs_utils

    ply_files = sorted(gaussians_dir.glob("*.ply"))
    print(f"\nLoading {len(ply_files)} PLY files into Rerun...")

    rr.init("Splatline 3D Video Viewer")
    rr.spawn()

    for i, ply_path in enumerate(ply_files):
        rr.set_time("frame", sequence=i)
        rr.set_time("time", duration=i / fps)

        gaussians, metadata = load_ply(Path(ply_path))

        positions = gaussians.mean_vectors.cpu().numpy().squeeze()
        colors = gaussians.colors.cpu().numpy().squeeze()
        scales = gaussians.singular_values.cpu().numpy().squeeze()

        # SHARP stores colors in linearRGB — convert to sRGB for correct display
        colors_t = torch.from_numpy(colors).float()
        colors = cs_utils.linearRGB2sRGB(colors_t).numpy()

        # Filter low-opacity points
        opacities = gaussians.opacities.cpu().numpy().squeeze()
        if opacities.ndim > 0:
            mask = opacities > 0.1
            positions = positions[mask]
            colors = colors[mask]
            if scales.ndim > 1 and scales.shape[0] == len(opacities):
                scales = scales[mask]

        radii = np.mean(scales, axis=1) * 2.0 if scales.ndim > 1 else scales * 2.0

        rr.log("video/gaussians", rr.Points3D(
            positions=positions,
            colors=colors,
            radii=radii,
        ))

        # Also log the original frame as a 2D image (needs pinhole camera for 3D)
        frame_path = frames_dir / f"{ply_path.stem}.png"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            rr.log("video/camera",
                rr.Pinhole(width=w, height=h, focal_length=w * 0.75),
            )
            rr.log("video/camera/image", rr.Image(img_rgb))

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(ply_files)}")

    print(f"\nDone! {len(ply_files)} frames loaded in Rerun.")
    print("Use the timeline to play through the 3D sequence.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def _load_gaussian_cloud(ply_path):
    """Load a PLY and return positions, colors, scales, opacities as numpy arrays."""
    from sharp.utils.gaussians import load_ply
    from sharp.utils import color_space as cs_utils
    gaussians, metadata = load_ply(Path(ply_path))
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
    # SHARP stores colors in linearRGB — convert to sRGB for correct display
    colors = cs_utils.linearRGB2sRGB(torch.from_numpy(colors).float()).numpy()
    return positions, colors, scales, opacities


def _voxel_downsample(positions, colors, voxel_size):
    """Downsample point cloud to one point per voxel, averaging colors."""
    voxel_indices = np.floor(positions / voxel_size).astype(np.int32)
    # Use a dict to group by voxel
    voxel_map = {}
    for idx in range(len(positions)):
        key = tuple(voxel_indices[idx])
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(idx)

    out_pos = np.zeros((len(voxel_map), 3), dtype=np.float32)
    out_col = np.zeros((len(voxel_map), 3), dtype=np.float32)
    for i, (key, indices) in enumerate(voxel_map.items()):
        out_pos[i] = positions[indices].mean(axis=0)
        out_col[i] = colors[indices].mean(axis=0)

    return out_pos, out_col


def view_solid_in_rerun(gaussians_dir, frames_dir, fps):
    """View the 3D Gaussian sequence as a solid textured mesh (no gaps).

    Uses scikit-image's marching cubes for fast surface reconstruction,
    then projects the source video frame onto the mesh for accurate colors.
    Blends image-projection colors (70%) with point-cloud colors (30%).
    """
    import rerun as rr
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_fill_holes
    from skimage import measure

    ply_files = sorted(gaussians_dir.glob("*.ply"))
    print(f"\nLoading {len(ply_files)} PLY files into Rerun (solid mesh mode)...")

    rr.init("Splatline Solid Mesh Viewer")
    rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

    for i, ply_path in enumerate(ply_files):
        rr.set_time("frame", sequence=i)
        rr.set_time("time", duration=i / fps)

        positions, colors, scales, opacities = _load_gaussian_cloud(ply_path)

        # Filter low-opacity
        if opacities.ndim > 0:
            mask = opacities > 0.1
            positions = positions[mask]
            colors = colors[mask]

        if len(positions) < 100:
            continue

        # Use ALL points — no subsampling, full resolution
        print(f"  [{i+1}/{len(ply_files)}] {len(positions)} pts -> ", end="")

        # Voxel grid: 768 voxels along longest axis — highest practical detail
        # With all 1.1M points, this gives ~2.9M vertices, ~70K unique colors
        extent = positions.max(axis=0) - positions.min(axis=0)
        voxel_size = extent.max() / 768.0
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
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(matrix, iterations=3)
        filled = binary_fill_holes(dilated)

        # Marching cubes via scikit-image (fast, returns normals too)
        verts, faces, normals, _ = measure.marching_cubes(filled, level=0.5)

        # Transform vertices back to original coordinate space
        verts = verts * voxel_size + origin

        print(f"{len(verts)} verts, {len(faces)} faces", end="  ")

        # Use original SHARP point cloud colors — no modification, no blending
        frame_path = frames_dir / f"{ply_path.stem}.png"
        color_tree = cKDTree(positions)
        _, nearest = color_tree.query(verts, k=1)
        vertex_colors = (colors[nearest] * 255).clip(0, 255).astype(np.uint8)

        # Add alpha
        vertex_colors_rgba = np.hstack([
            vertex_colors,
            np.full((len(vertex_colors), 1), 255, dtype=np.uint8),
        ])

        print("colored")

        # Log the solid mesh to Rerun
        rr.log("video/solid_mesh",
            rr.Mesh3D(
                vertex_positions=np.ascontiguousarray(verts, dtype=np.float32),
                triangle_indices=np.ascontiguousarray(faces, dtype=np.uint32),
                vertex_colors=vertex_colors_rgba,
                vertex_normals=np.ascontiguousarray(normals, dtype=np.float32),
            )
        )

        # Log source frame with pinhole camera
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            rr.log("video/camera",
                rr.Pinhole(width=w, height=h, focal_length=w * 0.75),
            )
            rr.log("video/camera/image", rr.Image(img_rgb))

        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(ply_files)}")

    print(f"\nDone! {len(ply_files)} frames loaded in Rerun (solid mesh mode).")
    print("Use the timeline to play through the 3D sequence.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


def _densify_point_cloud(positions, colors, tree, nn_dist):
    """Add midpoint between each point and its nearest neighbor to fill gaps."""
    dists, indices = tree.query(positions, k=2)  # k=2 because k=1 is self

    # Only interpolate between close neighbors (within 2x median distance)
    close_mask = dists[:, 1] < nn_dist * 2.0
    close_indices = np.where(close_mask)[0]
    neighbor_indices = indices[close_mask, 1]

    if len(close_indices) == 0:
        return positions, colors

    # Create midpoints
    mid_positions = (positions[close_indices] + positions[neighbor_indices]) / 2.0
    mid_colors = (colors[close_indices] + colors[neighbor_indices]) / 2.0

    # Combine original + midpoints
    all_positions = np.vstack([positions, mid_positions])
    all_colors = np.vstack([colors, mid_colors])

    return all_positions, all_colors


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Splatline: Convert a video to 3D and view in Rerun.",
        usage="python run_video_3d.py [mode] [--video PATH] [--output-dir DIR] [options]",
    )
    parser.add_argument("mode", nargs="?", default="points",
                        choices=["points", "solid"],
                        help="Viewer mode: 'points' (point cloud) or 'solid' (dense mesh). Default: points")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to input video file. Required for first run.")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory. Default: output_<video_name>")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Extract every Nth frame (default: 3, i.e. 24fps -> 8fps)")
    parser.add_argument("--splat-backend", "-b", type=str, default="sharp",
                        choices=["sharp", "triposplat", "vggt", "depthsplat", "longsplat"],
                        help="3D reconstruction backend (default: sharp)")
    parser.add_argument("--device", type=str, default="default",
                        choices=["default", "cuda", "mps", "cpu"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--internal-size", type=int, default=1536,
                        help="SHARP inference resolution (default: 1536)")
    args = parser.parse_args()

    # Resolve video path
    video_path = Path(args.video) if args.video else VIDEO_PATH
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        print("Usage: python run_video_3d.py --video /path/to/video.mp4")
        sys.exit(1)

    # Resolve output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"output_{video_path.stem}")

    print("=" * 60)
    print("SPLATLINE VIDEO-TO-3D PIPELINE")
    print("=" * 60)
    print(f"Input:   {video_path}")
    print(f"Output:  {output_dir}")
    print(f"Backend: {args.splat_backend}")
    print(f"Device:  {args.device}")

    frames_dir = output_dir / "frames"
    gaussians_dir = output_dir / "gaussians"

    # Step 1: Extract frames (skip if already done)
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        frames_dir, fps = extract_frames(video_path, output_dir, args.frame_skip)
    else:
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = source_fps / args.frame_skip
        print(f"Frames already extracted ({len(list(frames_dir.glob('*.png')))} frames)")

    # Save video path for the Electron player to find
    video_source_file = output_dir / "video_source.txt"
    if not video_source_file.exists():
        video_source_file.write_text(str(video_path))

    # Step 2: Convert to 3D (skip if already done)
    if not gaussians_dir.exists() or not list(gaussians_dir.glob("*.ply")):
        from utils.splat_models import convert_frames_to_splats
        gaussians_dir = convert_frames_to_splats(
            frames_dir=frames_dir,
            output_dir=output_dir,
            backend=args.splat_backend,
            device=args.device,
            internal_size=args.internal_size,
            video_path=video_path,
        )
    else:
        print(f"PLY files already exist ({len(list(gaussians_dir.glob('*.ply')))} files)")

    # Step 3: View in Rerun
    if args.mode == "solid":
        print("\nMode: SOLID SPLAT (dense, gap-filled)")
        view_solid_in_rerun(gaussians_dir, frames_dir, fps)
    else:
        print("\nMode: POINTS (original)")
        view_in_rerun(gaussians_dir, frames_dir, fps)


if __name__ == "__main__":
    main()
