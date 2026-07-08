#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats.

Renders PLY files as actual Gaussian splats in a web browser using the
official SuperSplat viewer (same engine as superspl.at). Each splat is
rendered as a camera-facing alpha-blended projected ellipsoid — not points
or meshes.

Automatically converts SHARP PLY files (which have extra elements that
SuperSplat rejects) to standard 3DGS format.

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
    """Convert SHARP PLY to standard 3DGS PLY (strip extra elements).

    SHARP PLY files contain extra elements (extrinsic, intrinsic, image_size,
    frame, disparity, color_space, version) that SuperSplat can't parse.
    """
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(input_path))

    # Already standard (only vertex element)?
    if len(ply.elements) == 1:
        shutil.copy2(input_path, output_path)
        print(f"  PLY already standard format ({ply.elements[0].count:,} verts)")
        return

    # Rebuild with only vertex element
    vertex = ply['vertex']
    new_vertex = PlyElement.describe(vertex.data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))

    print(f"  Converted: {vertex.count:,} verts, stripped {len(ply.elements)-1} extra elements")


def find_viewer_files():
    """Find the supersplat-viewer files from the npm package."""
    candidates = [
        Path("/tmp/splatline-web-viewer/node_modules/@playcanvas/supersplat-viewer/public"),
        Path("node_modules/@playcanvas/supersplat-viewer/public"),
        Path.home() / ".local/share/devin" / "splatline-viewer",
    ]

    for c in candidates:
        if (c / "index.html").exists() and (c / "index.js").exists():
            return c

    # Install the package
    print("Installing @playcanvas/supersplat-viewer...")
    install_dir = Path("/tmp/splatline-web-viewer")
    install_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["npm", "init", "-y"], cwd=str(install_dir),
                   capture_output=True, text=True)
    result = subprocess.run(
        ["npm", "install", "@playcanvas/supersplat-viewer"],
        cwd=str(install_dir), capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"npm install failed: {result.stderr}")
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
        sys.exit(1)

    ply_path = Path(sys.argv[1]).resolve()
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)

    # Find viewer files
    viewer_src = find_viewer_files()
    if not viewer_src:
        print("Error: Could not find or install @playcanvas/supersplat-viewer")
        print("Run: npm install @playcanvas/supersplat-viewer")
        sys.exit(1)

    # Always use a fresh viewer directory — clear old cached files
    viewer_dir = Path("/tmp/splatline_splat_viewer")
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Copy viewer files
    print("Setting up SuperSplat viewer...")
    for fname in ["index.html", "index.css", "index.js"]:
        shutil.copy2(viewer_src / fname, viewer_dir / fname)

    # Convert PLY to standard format
    print(f"Converting PLY ({ply_path.stat().st_size / 1e6:.1f} MB)...")
    ply_copy = viewer_dir / "scene.ply"
    convert_to_standard_ply(ply_path, ply_copy)

    # Write settings.json
    (viewer_dir / "settings.json").write_text(json.dumps(DEFAULT_SETTINGS, indent=2))

    # Start local web server with correct MIME types
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)

        def end_headers(self):
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

    # Find available port
    server = None
    PORT = 8765
    for port in range(8765, 8780):
        try:
            server = socketserver.TCPServer(("localhost", port), Handler)
            PORT = port
            break
        except OSError:
            continue

    if server is None:
        print("Error: No available port (8765-8779 all in use)")
        sys.exit(1)

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER (powered by SuperSplat)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"Viewer:   http://localhost:{PORT}")
    print()
    print("Controls:")
    print("  Left drag:   Orbit")
    print("  Right drag:  Pan")
    print("  Scroll:      Zoom")
    print("  F:           Frame scene")
    print("  R:           Reset camera")
    print()
    print("Opening browser... (Ctrl+C to stop)")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

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
