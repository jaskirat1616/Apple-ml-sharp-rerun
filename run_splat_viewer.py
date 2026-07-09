#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats in the Splatline editor.

Serves the Splatline editor (powered by the PlayCanvas engine from
github.com/playcanvas/supersplat) from a local web server with your PLY file
auto-loaded. You get every feature: selection, cutting planes, splat deletion,
transform tools, export, etc.

Automatically converts SHARP PLY files to standard 3DGS format.
Shows full resolution — no subsampling.

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


def convert_to_standard_ply(input_path, output_path):
    """Convert SHARP PLY to standard 3DGS PLY (strip extra elements).
    Keeps ALL splats — full resolution."""
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
    """Find or build the Splatline editor dist files."""
    dist_dir = Path("/tmp/supersplat/dist")
    if (dist_dir / "index.html").exists() and (dist_dir / "index.js").exists():
        # Verify the events patch is present
        js = (dist_dir / "index.js").read_text()
        if "__splatlineEvents" not in js:
            print("Rebuilding editor (events patch missing)...")
            result = subprocess.run(["npm", "run", "build"], cwd=str(dist_dir.parent),
                                    capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"Build failed: {result.stderr[-500:]}")
                return None
        return dist_dir

    print("Cloning Splatline editor source...")
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

    # Patch the source to expose events on window for sequence loading
    main_ts = repo_dir / "src" / "main.ts"
    if main_ts.exists():
        src = main_ts.read_text()
        if "__splatlineEvents" not in src:
            src = src.replace(
                "const events = new Events();",
                "const events = new Events();\n    (window as any).__splatlineEvents = events;"
            )
            main_ts.write_text(src)
            print("  Patched main.ts: exposed events on window")

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

    # Convert PLY to standard format — FULL RESOLUTION, no subsampling
    print(f"Converting PLY ({ply_path.stat().st_size / 1e6:.1f} MB) — full resolution...")
    ply_copy = editor_dist / "scene.ply"
    convert_to_standard_ply(ply_path, ply_copy)
    print(f"  Output: {ply_copy.stat().st_size / 1e6:.1f} MB")

    # Kill any existing serve process on port 3000
    subprocess.run("lsof -ti:3000 | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)

    # Start the serve dev server (proper MIME types, CORS)
    print("Starting Splatline editor server...")
    serve_proc = subprocess.Popen(
        ["npx", "serve", str(editor_dist), "-C", "-l", "3000"],
        cwd=str(editor_dist.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER — FULL RESOLUTION")
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

    # Open browser — use root URL (serve redirects /index.html and loses query params)
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
