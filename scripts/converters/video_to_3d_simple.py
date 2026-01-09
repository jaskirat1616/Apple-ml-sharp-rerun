#!/usr/bin/env python3
"""
Simple Video to 3D Converter - Quick Start Version

Just run: python video_to_3d_simple.py your_video.mp4

This creates a "video_output" folder with all the 3D Gaussian Splats.
"""

import sys
from pathlib import Path
from video_to_3d import video_to_3d

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_to_3d_simple.py <video_file>")
        print("\nExample:")
        print("  python video_to_3d_simple.py myvideo.mp4")
        print("\nThis will create a 'video_output' folder with your 3D results.")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path("video_output") / video_path.stem
    
    print("\n" + "="*60)
    print("SIMPLE VIDEO TO 3D CONVERTER")
    print("="*60)
    print(f"Input: {video_path}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Process with sensible defaults:
    # - Process every 10th frame (faster, good for testing)
    # - Limit to 20 frames max (you can remove this limit)
    # - Auto-detect best device
    video_to_3d(
        video_path=video_path,
        output_dir=output_dir,
        checkpoint_path=None,  # Auto-download
        device="default",  # Auto-detect
        max_frames=20,  # Limit for quick testing
        frame_interval=10,  # Every 10th frame
        render_3d=False,  # Set to True if you have CUDA
        keep_frames=False,
    )
    
    print("\n" + "="*60)
    print("DONE! Check your results in:", output_dir / "gaussians")
    print("="*60)

