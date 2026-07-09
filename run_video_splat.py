#!/usr/bin/env python3
"""
Splatline Video Splat Viewer — Play 3D video frames in the Splatline editor.

Uses the Splatline editor (powered by PlayCanvas) with its built-in timeline
panel for PLY sequence playback. All editor features work: selection, cutting
planes, splat deletion, transform tools, export, etc.

Shows full resolution — no subsampling.

Usage:
  python run_video_splat.py [output_dir] [max_frames]

  python run_video_splat.py                          # default: output_grok_3d
  python run_video_splat.py output_grok_3d           # all frames
  python run_video_splat.py output_grok_3d 10        # first 10 frames
"""
import sys
import shutil
import json
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
        return

    vertex = ply['vertex']
    new_vertex = PlyElement.describe(vertex.data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))


def find_or_build_editor():
    """Find or build the Splatline editor dist files."""
    dist_dir = Path("/tmp/supersplat/dist")
    if (dist_dir / "index.html").exists() and (dist_dir / "index.js").exists():
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
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/playcanvas/supersplat.git", str(repo_dir)],
            capture_output=True, text=True, timeout=120
        )

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
    subprocess.run(["npm", "run", "build"], cwd=str(repo_dir),
                   capture_output=True, text=True, timeout=120)

    dist_dir = repo_dir / "dist"
    if (dist_dir / "index.html").exists():
        return dist_dir
    return None


# JavaScript injected into index.html to load PLY frames as a sequence
SEQUENCE_LOADER_JS = """
<script>
// Splatline video sequence loader — full resolution
(async function() {
    const manifest = await fetch('manifest.json').then(r => r.json());
    console.log('Splatline: Loading ' + manifest.frames + ' frames as PLY sequence (full resolution)');

    const files = [];
    for (let i = 0; i < manifest.frames; i++) {
        const filename = 'frame_' + String(i).padStart(4, '0') + '.ply';
        const url = 'frames/' + filename;
        console.log('Splatline: Fetching ' + filename + '...');
        try {
            const response = await fetch(url);
            if (!response.ok) { console.error('Splatline: Failed ' + filename); continue; }
            const blob = await response.blob();
            const file = new File([blob], filename, { type: 'application/octet-stream' });
            files.push({ filename: filename, contents: file });
        } catch(e) { console.error('Splatline: Error fetching ' + filename, e); }
    }

    console.log('Splatline: Importing ' + files.length + ' frames as sequence');

    function tryImport(retries) {
        if (window.__splatlineEvents) {
            window.__splatlineEvents.invoke('import', files).then(() => {
                console.log('Splatline: Sequence loaded successfully');
            }).catch(err => {
                console.error('Splatline: Import failed:', err);
            });
        } else if (retries > 0) {
            setTimeout(() => tryImport(retries - 1), 500);
        } else {
            console.error('Splatline: Editor events not available after 30 retries');
        }
    }
    tryImport(30);
})();
</script>
"""


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_grok_3d")
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

    gaussians_dir = output_dir / "gaussians"
    if not gaussians_dir.exists():
        print(f"Error: No gaussians directory at {gaussians_dir}")
        print("Run run_video_3d.py first to generate PLY files.")
        sys.exit(1)

    ply_files = sorted(gaussians_dir.glob("*.ply"))
    if not ply_files:
        print(f"Error: No PLY files in {gaussians_dir}")
        sys.exit(1)

    if max_frames:
        ply_files = ply_files[:max_frames]

    # Get FPS
    import cv2
    fps = 8.0
    for p in [Path("/Users/jaskiratsingh/Downloads/grok-video-a1a6d6a4-6f94-41c2-82b5-83ec305487ae.mp4")]:
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps = source_fps / 3
            break

    print("=" * 60)
    print("SPLATLINE VIDEO SPLAT VIEWER — FULL RESOLUTION")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print(f"PLY files:  {len(ply_files)}")
    print(f"FPS:        {fps:.1f}")
    print()

    # Find or build editor
    editor_dist = find_or_build_editor()
    if not editor_dist:
        print("Error: Could not build Splatline editor")
        sys.exit(1)

    # Patch the editor HTML
    index_html = (editor_dist / "index.html").read_text()
    patched = index_html.replace("navigator.serviceWorker", "null && navigator.serviceWorker")
    patched = patched.replace("<title>SuperSplat</title>", "<title>Splatline Video Viewer</title>")
    # Inject our sequence loader before the closing body tag
    patched = patched.replace("</body>", SEQUENCE_LOADER_JS + "\n</body>")
    (editor_dist / "index.html").write_text(patched)

    # Create frames directory in dist
    frames_dir = editor_dist / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Convert PLYs — FULL RESOLUTION, no subsampling
    print(f"Converting {len(ply_files)} PLY files — full resolution...")
    for i, ply_path in enumerate(ply_files):
        std_path = frames_dir / f"frame_{i:04d}.ply"
        convert_to_standard_ply(ply_path, std_path)
        size_mb = std_path.stat().st_size / 1e6
        print(f"  [{i+1}/{len(ply_files)}] {ply_path.name} -> {size_mb:.1f} MB")

    # Write manifest
    manifest = {
        "frames": len(ply_files),
        "fps": round(fps, 1),
        "source": str(output_dir),
    }
    (editor_dist / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Kill any existing serve process on port 3000
    subprocess.run("lsof -ti:3000 | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)

    # Start the serve dev server
    print("Starting Splatline editor server...")
    serve_proc = subprocess.Popen(
        ["npx", "serve", str(editor_dist), "-C", "-l", "3000"],
        cwd=str(editor_dist.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(3)

    print()
    print(f"Editor: http://localhost:3000")
    print()
    print("The editor will load all frames as a PLY sequence.")
    print("Use the timeline panel at the bottom to play/scrub through frames.")
    print()
    print("Editor features:")
    print("  Selection, cutting planes, splat deletion")
    print("  Transform tools, export, multi-splat")
    print("  Timeline: play/pause, scrub, frame-rate control")
    print()
    print("Opening editor... (Ctrl+C to stop)")

    # Open browser at root URL
    webbrowser.open(f"http://localhost:3000/")

    try:
        serve_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve_proc.terminate()
        serve_proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
