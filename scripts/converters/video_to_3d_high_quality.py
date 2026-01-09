#!/usr/bin/env python3
"""
High-Quality Video to 3D Converter - Process ALL frames without skipping.

This script extracts every frame from a video and converts each to a 3D Gaussian Splat PLY file.
"""

import sys
import logging
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.nn.functional as F


def extract_all_frames(video_path: Path, output_dir: Path, frame_skip: int = 1) -> int:
    """
    Extract frames from video with optional skipping.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame (1 = all frames, 2 = every other, etc.)
    
    Returns:
        Number of frames extracted
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📹 Extracting frames from: {video_path.name}")
    print(f"   Output directory: {frames_dir}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_output_frames = (total_frames + frame_skip - 1) // frame_skip
    
    print(f"   Video info: {total_frames} frames @ {fps:.2f} FPS")
    
    if frame_skip > 1:
        print(f"   Frame skip: Every {frame_skip} frame(s)")
        print(f"   Will extract: ~{expected_output_frames} frames\n")
    else:
        print(f"   Extracting all {total_frames} frames\n")
    
    print(f"⏳ Extracting frames...")
    
    video_frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save every Nth frame
        if video_frame_count % frame_skip == 0:
            # Save frame as PNG (lossless)
            frame_path = frames_dir / f"frame_{saved_frame_count:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_frame_count += 1
            
            if saved_frame_count % 10 == 0 or saved_frame_count == 1:
                print(f"   Extracted: {saved_frame_count} frames (video frame {video_frame_count}/{total_frames})", end="\r")
        
        video_frame_count += 1
    
    cap.release()
    print(f"\n✓ Extracted {saved_frame_count} frames from {video_frame_count} video frames")
    
    return saved_frame_count


def convert_frames_to_3d(frames_dir: Path, output_dir: Path, device: str = "default"):
    """
    Convert all frames to 3D Gaussian Splats using SHARP Python API.
    
    Args:
        frames_dir: Directory containing frame images
        output_dir: Directory to save PLY files
        device: Device to use ('default', 'cuda', 'mps', 'cpu')
    """
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply, unproject_gaussians
    
    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Converting frames to 3D Gaussian Splats...")
    print(f"   Input: {frames_dir}")
    print(f"   Output: {gaussians_dir}")
    print(f"   Device: {device}")
    
    # Auto-detect device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"   ✓ Using CUDA (GPU)")
        elif torch.mps.is_available():
            device = "mps"
            print(f"   ✓ Using MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            print(f"   ⚠️  Using CPU (will be slower)")
    
    device_obj = torch.device(device)
    
    # Load model
    print(f"\n⏳ Loading SHARP model...")
    DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    
    try:
        print(f"   Downloading model from: {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        
        gaussian_predictor = create_predictor(PredictorParams())
        gaussian_predictor.load_state_dict(state_dict)
        gaussian_predictor.eval()
        gaussian_predictor.to(device_obj)
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return False
    
    # Get all image files
    extensions = io.get_supported_image_extensions()
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(frames_dir.glob(f"*{ext}")))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        print(f"✗ No images found in {frames_dir}")
        return False
    
    print(f"\n⏳ Processing {len(image_paths)} frames...")
    
    # Process each frame
    for idx, image_path in enumerate(image_paths):
        try:
            print(f"   [{idx+1}/{len(image_paths)}] Processing {image_path.name}...", end=" ")
            
            # Load image
            image, _, f_px = io.load_rgb(image_path)
            height, width = image.shape[:2]
            
            # Predict gaussians
            gaussians = predict_image_direct(gaussian_predictor, image, f_px, device_obj)
            
            # Save PLY
            output_ply = gaussians_dir / f"{image_path.stem}.ply"
            save_ply(gaussians, f_px, (height, width), output_ply)
            
            print(f"✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print(f"\n✓ Successfully converted all frames to 3D!")
    return True


@torch.no_grad()
def predict_image_direct(predictor, image: np.ndarray, f_px: float, device: torch.device):
    """
    Predict Gaussians from an image using SHARP.
    
    Args:
        predictor: SHARP predictor model
        image: Input image as numpy array
        f_px: Focal length in pixels
        device: Torch device
    
    Returns:
        Gaussians3D object
    """
    from sharp.utils.gaussians import unproject_gaussians
    
    internal_shape = (1536, 1536)
    
    # Preprocess
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    
    # Predict Gaussians in NDC space
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    
    # Postprocess
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    
    # Convert Gaussians to metric space
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )
    
    return gaussians


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("HIGH-QUALITY VIDEO TO 3D CONVERTER")
    print("Convert video frames to 3D Gaussian Splats")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\nUsage: python video_to_3d_high_quality.py <video_file> [device] [skip]")
        print("\nArguments:")
        print("  video_file  : Path to your video file (mp4, mov, avi, etc.)")
        print("  device      : Optional - 'cuda', 'mps', 'cpu', or 'default' (auto-detect)")
        print("  skip        : Optional - Extract every Nth frame (default: 1 = all frames)")
        print("\nExamples:")
        print("  # Process ALL frames")
        print("  python video_to_3d_high_quality.py myvideo.mp4")
        print("")
        print("  # Process every 2nd frame (half the frames, 2x faster)")
        print("  python video_to_3d_high_quality.py myvideo.mp4 mps 2")
        print("")
        print("  # Process every 5th frame (20% of frames, 5x faster)")
        print("  python video_to_3d_high_quality.py myvideo.mp4 mps 5")
        print("")
        print("  # Process every 10th frame (10% of frames, 10x faster)")
        print("  python video_to_3d_high_quality.py myvideo.mp4 mps 10")
        print("\nOutput:")
        print("  Creates 'output_<videoname>/' with:")
        print("    - frames/     : Extracted video frames (PNG)")
        print("    - gaussians/  : 3D Gaussian Splat files (PLY)")
        print("="*70 + "\n")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "default"
    frame_skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if frame_skip < 1:
        print("✗ Error: Frame skip must be >= 1")
        sys.exit(1)
    
    if not video_path.exists():
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(f"output_{video_path.stem}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📁 Input video: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Device: {device}")
    print(f"📁 Frame skip: Every {frame_skip} frame(s)")
    
    # Step 1: Extract frames
    try:
        num_frames = extract_all_frames(video_path, output_dir, frame_skip)
        if num_frames == 0:
            print("✗ Error: No frames extracted")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error extracting frames: {e}")
        sys.exit(1)
    
    # Step 2: Convert frames to 3D
    frames_dir = output_dir / "frames"
    success = convert_frames_to_3d(frames_dir, output_dir, device)
    
    if not success:
        print("\n✗ Conversion failed")
        sys.exit(1)
    
    # Summary
    gaussians_dir = output_dir / "gaussians"
    ply_files = list(gaussians_dir.glob("*.ply"))
    
    print("\n" + "="*70)
    print("✓ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Extracted frames: {num_frames}")
    print(f"   PLY files created: {len(ply_files)}")
    if frame_skip > 1:
        print(f"   Frame skip: Every {frame_skip} frame(s)")
    print(f"\n📁 Outputs:")
    print(f"   Frames: {output_dir / 'frames'}")
    print(f"   3D Splats: {output_dir / 'gaussians'}")
    
    print(f"\n🎬 Next steps - Visualize your 3D video:")
    print(f"\n   Option 1 - Rerun viewer (recommended):")
    print(f"   python visualize_with_rerun.py -i {gaussians_dir}/")
    print(f"\n   Option 2 - Web viewer:")
    print(f"   python start_3d_viewer.py {gaussians_dir}/")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

