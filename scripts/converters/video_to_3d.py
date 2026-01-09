import rerun as rr
import glob
import os
import numpy as np
from pathlib import Path
from sharp.utils.gaussians import load_ply

# Change these to match your setup
splat_dir = "output_grok_video/gaussians"  # Folder with your .ply files (e.g., frame_000001.ply)
fps = 30  # Match your video's FPS (or lower for smoother feel)

rr.init("RoamVideo Sequence Viewer")
rr.spawn()  # Opens interactive window (or rr.connect() for web)

ply_files = sorted(glob.glob(os.path.join(splat_dir, "*.ply")))

print(f"Loading {len(ply_files)} frames...")

for i, ply_path in enumerate(ply_files):
    time_sec = i / fps  # Timestamp for timeline
    
    print(f"Loading frame {i+1}/{len(ply_files)}: {Path(ply_path).name}")
    
    # Load Gaussians using sharp's loader
    gaussians, metadata = load_ply(Path(ply_path))
    
    # Extract data - convert from torch to numpy
    positions = gaussians.mean_vectors.cpu().numpy().squeeze()  # (N, 3)
    colors = gaussians.colors.cpu().numpy().squeeze()  # (N, 3)
    scales = gaussians.singular_values.cpu().numpy().squeeze()  # (N, 3)
    
    # Log to Rerun with timeline
    rr.set_time_sequence("frame", i)
    rr.log("video/gaussians", rr.Points3D(
        positions=positions,
        colors=colors,
        radii=np.mean(scales, axis=1) * 0.5  # Use average scale as radius
    ))

print("\n✅ Sequence loaded! Play timeline, orbit freely 🚀")
print("\nControls:")
print("  - Left click + drag: Rotate view")
print("  - Right click + drag: Pan view")
print("  - Scroll: Zoom in/out")
print("  - Timeline slider: Navigate frames")
print("\nPress Ctrl+C to exit.")

# Keep script running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")