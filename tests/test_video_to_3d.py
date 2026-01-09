#!/usr/bin/env python3
"""
Test script for video-to-3D conversion.

This script demonstrates the full workflow and can be used to verify
that everything is working correctly.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import imageio.v2 as iio

def create_test_video(output_path: Path, duration: float = 1.0, fps: int = 30):
    """
    Create a simple test video with moving shapes.
    
    Args:
        output_path: Where to save the test video
        duration: Duration in seconds
        fps: Frames per second
    """
    print(f"Creating test video at {output_path}...")
    
    width, height = 640, 480
    num_frames = int(duration * fps)
    
    writer = iio.get_writer(output_path, fps=fps)
    
    for i in range(num_frames):
        # Create a frame with a moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :, 0] = int(255 * y / height)  # Red gradient
            frame[y, :, 2] = int(255 * (1 - y / height))  # Blue gradient
        
        # Moving circle
        center_x = int(width * (0.2 + 0.6 * i / num_frames))
        center_y = height // 2
        radius = 50
        
        # Draw circle
        y_grid, x_grid = np.ogrid[:height, :width]
        mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
        frame[mask] = [255, 255, 0]  # Yellow circle
        
        # Add text
        frame[10:30, 10:200] = [255, 255, 255]  # White background for text
        
        writer.append_data(frame)
    
    writer.close()
    print(f"Test video created with {num_frames} frames at {fps} FPS")


def run_test():
    """Run a complete test of the video-to-3D pipeline."""
    print("\n" + "="*70)
    print("VIDEO TO 3D CONVERSION - TEST SCRIPT")
    print("="*70 + "\n")
    
    # Create temporary test video
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_video = temp_path / "test_video.mp4"
        
        # Create test video
        create_test_video(test_video, duration=1.0, fps=30)
        
        # Import the video_to_3d function
        try:
            from video_to_3d import video_to_3d
        except ImportError:
            print("ERROR: Could not import video_to_3d module.")
            print("Make sure video_to_3d.py is in the same directory.")
            return False
        
        # Set up output directory
        output_dir = Path("test_output")
        
        print("\n" + "="*70)
        print("Running conversion...")
        print("="*70 + "\n")
        
        try:
            # Run conversion with limited frames for testing
            video_to_3d(
                video_path=test_video,
                output_dir=output_dir,
                checkpoint_path=None,  # Will download
                device="default",  # Auto-detect
                max_frames=3,  # Only process 3 frames for testing
                frame_interval=10,  # Every 10th frame
                render_3d=False,  # No rendering for test
                keep_frames=True,  # Keep frames for inspection
            )
            
            print("\n" + "="*70)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("="*70 + "\n")
            
            # Check outputs
            gaussians_dir = output_dir / "gaussians"
            frames_dir = output_dir / "frames"
            
            if gaussians_dir.exists():
                ply_files = list(gaussians_dir.glob("*.ply"))
                print(f"✓ Created {len(ply_files)} 3D Gaussian files")
                print(f"  Location: {gaussians_dir}")
            
            if frames_dir.exists():
                frame_files = list(frames_dir.glob("*.png"))
                print(f"✓ Extracted {len(frame_files)} frames")
                print(f"  Location: {frames_dir}")
            
            print("\nYou can now:")
            print("  1. View the .ply files in a 3D Gaussian Splatting viewer")
            print("  2. Inspect the extracted frames")
            print("  3. Try processing your own videos!")
            
            print("\n" + "="*70)
            print("Next steps:")
            print("  - Try: python video_to_3d_simple.py your_video.mp4")
            print("  - Read: VIDEO_TO_3D_GUIDE.md for full documentation")
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print("\n" + "="*70)
            print("TEST FAILED!")
            print("="*70 + "\n")
            print(f"Error: {e}")
            print("\nTroubleshooting:")
            print("  1. Make sure all requirements are installed:")
            print("     pip install -r requirements.txt")
            print("  2. Check that you have internet connection (to download model)")
            print("  3. Try with CPU device if GPU issues:")
            print("     (edit the script to set device='cpu')")
            return False


if __name__ == "__main__":
    print("\nThis script will:")
    print("  1. Create a test video")
    print("  2. Convert it to 3D using ml-sharp")
    print("  3. Verify the outputs")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    success = run_test()
    sys.exit(0 if success else 1)

