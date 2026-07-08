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
    gaussians, metadata = load_ply(Path(ply_path))
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()
    colors = gaussians.colors.cpu().numpy().squeeze()
    scales = gaussians.singular_values.cpu().numpy().squeeze()
    opacities = gaussians.opacities.cpu().numpy().squeeze()
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
    from sharp.utils import io as sharp_io

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

        # Project colors from source image using BILINEAR interpolation
        # for smooth, high-quality color sampling (not blocky nearest-neighbor)
        frame_path = frames_dir / f"{ply_path.stem}.png"
        if frame_path.exists():
            source_img = cv2.imread(str(frame_path))
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_h, img_w = source_img.shape[:2]
            _, _, f_px = sharp_io.load_rgb(Path(frame_path))

            z = verts[:, 2].copy()
            z[np.abs(z) < 1e-6] = 1e-6
            u_f = f_px * verts[:, 0] / z + img_w / 2
            v_f = f_px * verts[:, 1] / z + img_h / 2

            # Bilinear interpolation for smooth colors
            from scipy.ndimage import map_coordinates
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
        else:
            # Fallback: use point cloud colors only
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
    import sys

    print("=" * 60)
    print("SPLATLINE VIDEO-TO-3D PIPELINE")
    print("=" * 60)
    print(f"Input: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    frames_dir = OUTPUT_DIR / "frames"
    gaussians_dir = OUTPUT_DIR / "gaussians"

    # Step 1: Extract frames (skip if already done)
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        frames_dir, fps = extract_frames(VIDEO_PATH, OUTPUT_DIR, FRAME_SKIP)
    else:
        import cv2
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = source_fps / FRAME_SKIP
        print(f"Frames already extracted ({len(list(frames_dir.glob('*.png')))} frames)")

    # Step 2: Convert to 3D with SHARP (skip if already done)
    if not gaussians_dir.exists() or not list(gaussians_dir.glob("*.ply")):
        gaussians_dir = convert_with_sharp(frames_dir, OUTPUT_DIR, device="mps")
    else:
        print(f"PLY files already exist ({len(list(gaussians_dir.glob('*.ply')))} files)")

    # Step 3: View in Rerun
    # Choose viewer mode: "points" (original) or "solid" (dense, gap-filled)
    mode = sys.argv[1] if len(sys.argv) > 1 else "points"

    if mode == "solid":
        print("\nMode: SOLID SPLAT (dense, gap-filled)")
        view_solid_in_rerun(gaussians_dir, frames_dir, fps)
    else:
        print("\nMode: POINTS (original)")
        view_in_rerun(gaussians_dir, frames_dir, fps)


if __name__ == "__main__":
    main()
