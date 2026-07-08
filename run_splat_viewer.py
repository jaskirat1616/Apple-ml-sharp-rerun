#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as proper Gaussian splats in the browser.

Uses the official @playcanvas/supersplat-viewer (same engine as superspl.at).
Renders PLY files as actual Gaussian splats with alpha blending, spherical
harmonics, and proper ellipsoid projection — not as point clouds or meshes.

Starts a local web server and opens the viewer in your default browser.
Works on macOS — no CUDA required, uses WebGL/WebGPU.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
  python run_splat_viewer.py output_frame_000000_3d/frame_000000_points.ply
"""
import sys
import os
import shutil
import json
import http.server
import socketserver
import threading
import webbrowser
import time
import subprocess
from pathlib import Path


# Default settings for the SuperSplat viewer
DEFAULT_SETTINGS = {
    "version": 2,
    "tonemapping": "aces",
    "highPrecisionRendering": True,
    "background": {
        "color": [0.1, 0.1, 0.18]
    },
    "postEffectSettings": {
        "sharpness": {"enabled": True, "amount": 0.5},
        "bloom": {"enabled": False, "intensity": 0.3, "blurLevel": 1},
        "grading": {
            "enabled": True,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 1.0,
            "tint": [1.0, 1.0, 1.0]
        },
        "vignette": {"enabled": False, "intensity": 0.3, "inner": 0.3, "outer": 0.8, "curvature": 1.0},
        "fringing": {"enabled": False, "intensity": 0.2}
    },
    "animTracks": [],
    "cameras": [],
    "annotations": [],
    "startMode": "default"
}


def convert_to_standard_ply(input_path, output_path):
    """Convert SHARP PLY (with extra elements) to standard 3DGS PLY.

    SHARP PLY files contain extra elements (extrinsic, intrinsic, image_size,
    frame, disparity, color_space, version) that standard 3DGS viewers like
    SuperSplat can't parse. This strips those extra elements and keeps only
    the vertex data with standard 3DGS properties.
    """
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(str(input_path))
    vertex = ply['vertex']

    # Check if it's already standard format (only vertex element)
    if len(ply.elements) == 1:
        # Already standard — just copy
        shutil.copy2(input_path, output_path)
        print(f"  PLY already standard format ({len(vertex.data):,} verts)")
        return

    # Rebuild with only vertex element
    # Keep all vertex properties as-is
    new_vertex = PlyElement.describe(vertex.data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))

    print(f"  Converted: {len(vertex.data):,} verts, stripped {len(ply.elements)-1} extra elements")
    print(f"  Output: {output_path.stat().st_size / 1e6:.1f} MB")


def find_viewer_files():
    """Find the supersplat-viewer files from the npm package."""
    # Check common npm locations
    candidates = [
        Path.home() / ".local/share/devin" / "splatline-viewer",
        Path("/tmp/splatline-web-viewer/node_modules/@playcanvas/supersplat-viewer/public"),
        Path("node_modules/@playcanvas/supersplat-viewer/public"),
    ]

    # Also check if npm is available and install if needed
    for c in candidates:
        if (c / "index.html").exists() and (c / "index.js").exists():
            return c

    # Try to install the package
    print("Installing @playcanvas/supersplat-viewer...")
    install_dir = Path("/tmp/splatline-web-viewer")
    if not install_dir.exists():
        result = subprocess.run(
            ["npm", "init", "-y"],
            cwd=str(install_dir.parent) if install_dir.parent.exists() else None,
            capture_output=True, text=True
        )
        install_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["npm", "init", "-y"],
            cwd=str(install_dir),
            capture_output=True, text=True
        )
        result = subprocess.run(
            ["npm", "install", "@playcanvas/supersplat-viewer"],
            cwd=str(install_dir),
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"Failed to install: {result.stderr}")
            return None

    viewer_src = install_dir / "node_modules/@playcanvas/supersplat-viewer/public"
    if viewer_src.exists():
        return viewer_src

    return None


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

    # Find or install the viewer files
    viewer_src = find_viewer_files()
    if not viewer_src:
        print("Error: Could not find or install @playcanvas/supersplat-viewer")
        print("Try: npm install @playcanvas/supersplat-viewer")
        sys.exit(1)

    # Create viewer directory
    viewer_dir = Path("/tmp/splatline_splat_viewer")
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Copy viewer files
    for fname in ["index.html", "index.css", "index.js"]:
        src = viewer_src / fname
        dst = viewer_dir / fname
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(f"  Copying {fname}...")
            shutil.copy2(src, dst)

    # Convert SHARP PLY to standard 3DGS PLY (strip extra elements)
    ply_copy = viewer_dir / "scene.ply"
    print(f"  Converting SHARP PLY to standard 3DGS format...")
    convert_to_standard_ply(ply_path, ply_copy)

    # Write settings.json
    settings_path = viewer_dir / "settings.json"
    settings_path.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))

    # Start local web server — find an available port
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def end_headers(self):
            # Ensure correct MIME types for ES modules
            if self.path.endswith('.js'):
                self.send_header('Content-Type', 'text/javascript')
            elif self.path.endswith('.css'):
                self.send_header('Content-Type', 'text/css')
            elif self.path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            elif self.path.endswith('.ply'):
                self.send_header('Content-Type', 'application/octet-stream')
            super().end_headers()

        def log_message(self, format, *args):
            pass

    PORT = 8765
    server = None
    for port in range(8765, 8780):
        try:
            server = socketserver.TCPServer(("localhost", port), Handler)
            PORT = port
            break
        except OSError:
            continue

    if server is None:
        print("Error: Could not find an available port (8765-8779 all in use)")
        sys.exit(1)

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER (powered by SuperSplat)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"File size: {ply_path.stat().st_size / 1e6:.1f} MB")
    print(f"Viewer: http://localhost:{PORT}")
    print()
    print("Controls:")
    print("  Left drag:  Orbit")
    print("  Right drag: Pan")
    print("  Scroll:     Zoom")
    print("  F:          Frame scene")
    print("  R:          Reset camera")
    print()
    print("Opening browser... (Ctrl+C to stop)")

    # Start server
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Open browser — use webgl mode for broader compatibility
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{PORT}/index.html?content=scene.ply&settings=settings.json&webgl")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
