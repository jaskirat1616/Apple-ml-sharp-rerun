#!/usr/bin/env python3
"""
Splatline Video Splat Viewer — Play 3D video frames in the Splatline editor.

Uses the Splatline editor (powered by PlayCanvas) with its built-in timeline
panel for PLY sequence playback. All editor features work: selection, cutting
planes, splat deletion, transform tools, export, etc.

Frames are pre-subsampled for fast loading. The editor detects them as a
PLY sequence and shows a timeline scrubber at the bottom.

Usage:
  python run_video_splat.py [output_dir] [max_frames]

  python run_video_splat.py                          # default: output_grok_3d
  python run_video_splat.py output_grok_3d           # all frames
  python run_video_splat.py output_grok_3d 10        # first 10 frames
"""
import sys
import shutil
import json
import http.server
import socketserver
import threading
import webbrowser
import time
import subprocess
from pathlib import Path

import numpy as np


def convert_and_subsample_ply(input_path, output_path, max_splats=200000):
    """Convert SHARP PLY to standard 3DGS PLY, stripping extra elements
    and subsampling to max_splats for fast browser loading."""
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(input_path))
    vertex = ply['vertex']
    data = vertex.data

    # Subsample by opacity — keep the most visible splats
    if len(data) > max_splats and 'opacity' in data.dtype.names:
        opacities = data['opacity']
        ops = 1.0 / (1.0 + np.exp(-opacities))
        top_idx = np.argsort(ops)[-max_splats:]
        data = data[top_idx]

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
    subprocess.run(["npm", "run", "build"], cwd=str(repo_dir),
                   capture_output=True, text=True, timeout=120)

    dist_dir = repo_dir / "dist"
    if (dist_dir / "index.html").exists():
        return dist_dir
    return None


# JavaScript injected into index.html to load PLY frames as a sequence
SEQUENCE_LOADER_JS = """
<script>
// Splatline video sequence loader
// Fetches all PLY frames and imports them as a sequence into the editor
(async function() {
    const manifest = await fetch('manifest.json').then(r => r.json());
    console.log('Splatline: Loading ' + manifest.frames + ' frames as PLY sequence');

    // Fetch all PLY files as File objects
    const files = [];
    for (let i = 0; i < manifest.frames; i++) {
        const filename = 'frame_' + String(i).padStart(4, '0') + '.ply';
        const url = 'frames/' + filename;
        console.log('Splatline: Fetching ' + filename + '...');
        const response = await fetch(url);
        if (!response.ok) {
            console.error('Splatline: Failed to load ' + filename + ': ' + response.status);
            continue;
        }
        const blob = await response.blob();
        const file = new File([blob], filename, { type: 'application/octet-stream' });
        files.push({ filename: filename, contents: file });
    }

    console.log('Splatline: Importing ' + files.length + ' frames as sequence');

    // Wait for the editor to be ready, then import as a sequence
    // The editor's 'import' event detects PLY sequences by filename pattern
    // (frame_0000.ply, frame_0001.ply, etc.) and shows the timeline
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
            console.error('Splatline: Editor events not available');
        }
    }
    tryImport(20);
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
    video_path = None
    for p in [Path("/Users/jaskiratsingh/Downloads/grok-video-a1a6d6a4-6f94-41c2-82b5-83ec305487ae.mp4")]:
        if p.exists():
            video_path = p
            break

    fps = 8.0
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = source_fps / 3  # frame_skip=3

    print("=" * 60)
    print("SPLATLINE VIDEO SPLAT VIEWER")
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

    # Fresh viewer directory
    viewer_dir = Path("/tmp/splatline_video_viewer")
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir, ignore_errors=True)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    print("Copying Splatline editor...")
    shutil.copytree(editor_dist, viewer_dir, dirs_exist_ok=True)

    # Modify index.html: disable service worker, rename title, inject sequence loader
    index_html = (viewer_dir / "index.html").read_text()
    index_html = index_html.replace("navigator.serviceWorker", "null && navigator.serviceWorker")
    index_html = index_html.replace("<title>SuperSplat</title>", "<title>Splatline Video Viewer</title>")
    # Inject our sequence loader before the closing body tag
    index_html = index_html.replace("</body>", SEQUENCE_LOADER_JS + "\n</body>")
    (viewer_dir / "index.html").write_text(index_html)

    # Also patch the editor JS to expose events globally so our injected
    # script can call editor.import() to load the PLY sequence
    editor_js = (viewer_dir / "index.js").read_text()
    # The minified JS creates events like: const events=new RR
    # Expose it on window so our sequence loader can use it
    import re
    # Match: const events=new <Something>
    match = re.search(r'const events=new (\w+)', editor_js)
    if match:
        old = match.group(0)
        new = f"window.__splatlineEvents={old};const events=window.__splatlineEvents"
        editor_js = editor_js.replace(old, new, 1)
        print(f"  Patched editor JS: exposed events object")
    else:
        # Fallback: try without 'const'
        match2 = re.search(r'events=new (\w+)', editor_js)
        if match2:
            old = match2.group(0)
            editor_js = editor_js.replace(old, f"window.__splatlineEvents={old};events=window.__splatlineEvents", 1)
            print(f"  Patched editor JS (fallback): exposed events object")
        else:
            print("  WARNING: Could not find events object in editor JS")
    (viewer_dir / "index.js").write_text(editor_js)

    # Convert and subsample PLYs
    frames_dir = viewer_dir / "frames"
    frames_dir.mkdir()
    print(f"Converting {len(ply_files)} PLY files (subsampled to 200K splats each)...")
    for i, ply_path in enumerate(ply_files):
        std_path = frames_dir / f"frame_{i:04d}.ply"
        convert_and_subsample_ply(ply_path, std_path)
        size_mb = std_path.stat().st_size / 1e6
        print(f"  [{i+1}/{len(ply_files)}] {ply_path.name} -> {size_mb:.1f} MB")

    # Write manifest
    manifest = {
        "frames": len(ply_files),
        "fps": round(fps, 1),
        "source": str(output_dir),
    }
    (viewer_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Web server
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

    # Use port 9000+ to avoid stale service worker from previous editor
    server = None
    PORT = 9000
    for port in range(9000, 9020):
        try:
            socketserver.TCPServer.allow_reuse_address = True
            server = socketserver.ThreadingTCPServer(("localhost", port), Handler)
            PORT = port
            break
        except OSError:
            continue

    if server is None:
        print("Error: No available port")
        sys.exit(1)

    print()
    print(f"Editor: http://localhost:{PORT}")
    print()
    print("The editor will load all frames as a PLY sequence.")
    print("Use the timeline panel at the bottom to play/scrub through frames.")
    print()
    print("Editor features:")
    print("  Selection, cutting planes, splat deletion")
    print("  Transform tools, export, multi-splat")
    print("  Timeline: play/pause, scrub, frame-rate control")
    print()
    print("Controls:")
    print("  Left drag:    Orbit")
    print("  Right drag:   Pan")
    print("  Scroll:       Zoom")
    print("  F:            Frame scene")
    print()
    print("Opening editor... (Ctrl+C to stop)")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}/index.html")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
