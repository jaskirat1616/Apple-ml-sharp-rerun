#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats in the Splatline editor.

Serves the Splatline editor (powered by the PlayCanvas engine from
github.com/playcanvas/supersplat) from a local web server with your PLY file
auto-loaded. You get every feature: selection, cutting planes, splat deletion,
transform tools, export, etc.

Automatically converts SHARP PLY files to standard 3DGS format and subsamples
for fast browser loading.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
"""
import sys
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path

import numpy as np


def convert_and_subsample_ply(input_path, output_path, max_splats=500000):
    """Convert SHARP PLY to standard 3DGS PLY, stripping extra elements
    and subsampling to max_splats for fast browser loading."""
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(input_path))
    vertex = ply['vertex']
    data = vertex.data

    if len(data) > max_splats and 'opacity' in data.dtype.names:
        opacities = data['opacity']
        ops = 1.0 / (1.0 + np.exp(-opacities))
        top_idx = np.argsort(ops)[-max_splats:]
        data = data[top_idx]
        print(f"  Subsampled: {vertex.count:,} -> {len(data):,} splats (top opacity)")
    else:
        print(f"  Kept all {len(data):,} splats")

    if len(ply.elements) > 1:
        print(f"  Stripped {len(ply.elements)-1} extra elements")

    new_vertex = PlyElement.describe(data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))


def find_or_build_editor():
    """Find or build the Splatline editor dist files."""
    dist_dir = Path("/tmp/supersplat/dist")
    if (dist_dir / "index.html").exists() and (dist_dir / "index.js").exists():
        return dist_dir

    print("Cloning Splatline editor source...")
    repo_dir = Path("/tmp/supersplat")
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/playcanvas/supersplat.git", str(repo_dir)],
            capture_output=True, text=True, timeout=120
        )

    print("Installing dependencies...")
    subprocess.run(["npm", "install"], cwd=str(repo_dir),
                   capture_output=True, text=True, timeout=120)

    print("Building Splatline editor...")
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

    # Find or build the Splatline editor
    editor_dist = find_or_build_editor()
    if not editor_dist:
        print("Error: Could not build Splatline editor")
        sys.exit(1)

    # Patch the editor HTML: disable service worker, rename title
    index_html = (editor_dist / "index.html").read_text()
    patched = index_html.replace("navigator.serviceWorker", "null && navigator.serviceWorker")
    patched = patched.replace("<title>SuperSplat</title>", "<title>Splatline</title>")
    (editor_dist / "index.html").write_text(patched)

    # Convert and subsample PLY into the dist folder
    print(f"Converting PLY ({ply_path.stat().st_size / 1e6:.1f} MB)...")
    ply_copy = editor_dist / "scene.ply"
    convert_and_subsample_ply(ply_path, ply_copy)
    print(f"  Output: {ply_copy.stat().st_size / 1e6:.1f} MB")

    # Kill any existing serve process on port 3000
    subprocess.run(["lsof", "-ti:3000"], capture_output=True, text=True)
    subprocess.run("lsof -ti:3000 | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)

    # Start the serve dev server (proper MIME types, CORS, no redirects on root)
    print("Starting Splatline editor server...")
    serve_proc = subprocess.Popen(
        ["npx", "serve", str(editor_dist), "-C", "-l", "3000"],
        cwd=str(editor_dist.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to start
    time.sleep(3)

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"Editor:   http://localhost:3000")
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

    # Open browser — use root URL (not /index.html which redirects and loses query params)
    webbrowser.open(f"http://localhost:3000/?load=scene.ply&filename={ply_path.name}")

    try:
        serve_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve_proc.terminate()
        serve_proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
