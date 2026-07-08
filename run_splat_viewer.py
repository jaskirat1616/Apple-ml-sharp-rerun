#!/usr/bin/env python3
"""
Splatline Splat Viewer — Full SuperSplat editor running locally.

Serves the official SuperSplat editor (https://github.com/playcanvas/supersplat)
from a local web server with your PLY file auto-loaded. You get every feature:
selection, cutting planes, splat deletion, transform tools, export, etc.

Automatically converts SHARP PLY files to standard 3DGS format.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
"""
import sys
import shutil
import http.server
import socketserver
import threading
import webbrowser
import time
import subprocess
from pathlib import Path


def convert_to_standard_ply(input_path, output_path):
    """Convert SHARP PLY to standard 3DGS PLY (strip extra elements)."""
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(input_path))

    if len(ply.elements) == 1:
        shutil.copy2(input_path, output_path)
        print(f"  PLY already standard ({ply.elements[0].count:,} verts)")
        return

    vertex = ply['vertex']
    new_vertex = PlyElement.describe(vertex.data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))
    print(f"  Converted: {vertex.count:,} verts, stripped {len(ply.elements)-1} extra elements")


def find_or_build_editor():
    """Find or build the SuperSplat editor dist files."""
    # Already built?
    dist_dir = Path("/tmp/supersplat/dist")
    if (dist_dir / "index.html").exists() and (dist_dir / "index.js").exists():
        return dist_dir

    # Clone and build
    print("Cloning SuperSplat editor...")
    repo_dir = Path("/tmp/supersplat")
    if not repo_dir.exists():
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/playcanvas/supersplat.git", str(repo_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"Clone failed: {result.stderr}")
            return None

    print("Installing dependencies...")
    subprocess.run(["npm", "install"], cwd=str(repo_dir),
                   capture_output=True, text=True, timeout=120)

    print("Building editor...")
    result = subprocess.run(["npm", "run", "build"], cwd=str(repo_dir),
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr[-500:]}")
        return None

    dist_dir = repo_dir / "dist"
    if (dist_dir / "index.html").exists():
        return dist_dir
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_splat_viewer.py <ply_file>")
        print("  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply")
        sys.exit(1)

    ply_path = Path(sys.argv[1]).resolve()
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)

    # Find or build the SuperSplat editor
    editor_dist = find_or_build_editor()
    if not editor_dist:
        print("Error: Could not build SuperSplat editor")
        print("Try: cd /tmp/supersplat && npm install && npm run build")
        sys.exit(1)

    # Fresh viewer directory — copy entire editor dist
    viewer_dir = Path("/tmp/splatline_splat_viewer")
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    print("Copying SuperSplat editor...")
    shutil.copytree(editor_dist, viewer_dir, dirs_exist_ok=True)

    # Convert PLY to standard format
    print(f"Converting PLY ({ply_path.stat().st_size / 1e6:.1f} MB)...")
    ply_copy = viewer_dir / "scene.ply"
    convert_to_standard_ply(ply_path, ply_copy)

    # Web server with correct MIME types
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
            elif self.path.endswith('.wasm'):
                self.send_header('Content-Type', 'application/wasm')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
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
    print("SPLATLINE SPLAT VIEWER (SuperSplat Editor)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"Editor:   http://localhost:{PORT}")
    print()
    print("Features: selection, cutting planes, splat deletion,")
    print("          transform tools, export, multi-splat, etc.")
    print()
    print("Controls:")
    print("  Left drag:    Orbit")
    print("  Right drag:   Pan")
    print("  Scroll:       Zoom")
    print("  F:            Frame scene")
    print("  Shift+click:  Select splats")
    print("  Drag .ply:    Load another file")
    print()
    print("Opening editor... (Ctrl+C to stop)")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(1)
    # Auto-load the PLY via URL param
    webbrowser.open(f"http://localhost:{PORT}/index.html?load=scene.ply&filename={ply_path.name}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
