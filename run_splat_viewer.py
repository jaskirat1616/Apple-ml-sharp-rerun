#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats in the browser.

Uses the official @playcanvas/supersplat-viewer (same engine as superspl.at).
Renders actual Gaussian splats — camera-facing alpha-blended projected
ellipsoids with spherical harmonics.

Automatically converts SHARP PLY files to standard 3DGS format.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
"""
import sys
import shutil
import json
import http.server
import socketserver
import threading
import webbrowser
import time
from pathlib import Path


VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>Splatline Splat Viewer</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<link rel="stylesheet" href="https://unpkg.com/@playcanvas/supersplat-viewer@1.27.1/public/index.css">
<style>
body { margin: 0; padding: 0; overflow: hidden; background: #1a1a2e; }
canvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; }
#loading {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  color: #fff; font-family: -apple-system, sans-serif; font-size: 18px;
  text-align: center; z-index: 9999;
}
#loading .bar {
  width: 300px; height: 4px; background: #333; border-radius: 2px;
  margin-top: 12px; overflow: hidden;
}
#loading .fill {
  height: 100%; background: #F26722; width: 0%; transition: width 0.3s;
}
#error {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  color: #ff6b6b; font-family: -apple-system, sans-serif; font-size: 16px;
  text-align: center; z-index: 9999; max-width: 600px; display: none;
  background: #1a1a2e; padding: 20px; border-radius: 12px;
}
</style>
</head>
<body>
<div id="loading">
  <div>Loading Gaussian Splats...</div>
  <div class="bar"><div class="fill" id="progress"></div></div>
</div>
<div id="error"></div>
<canvas id="application-canvas"></canvas>
<script type="module">
import { main } from 'https://unpkg.com/@playcanvas/supersplat-viewer@1.27.1/public/index.js';

const settings = {
  version: 2,
  tonemapping: 'aces',
  highPrecisionRendering: true,
  background: { color: [0.06, 0.06, 0.12] },
  postEffectSettings: {
    sharpness: { enabled: false, amount: 0.5 },
    bloom: { enabled: false, intensity: 0.3, blurLevel: 1 },
    grading: { enabled: false, brightness: 0, contrast: 0, saturation: 1, tint: [1,1,1] },
    vignette: { enabled: false, intensity: 0.3, inner: 0.3, outer: 0.8, curvature: 1 },
    fringing: { enabled: false, intensity: 0.2 }
  },
  animTracks: [],
  cameras: [],
  annotations: [],
  startMode: 'default'
};

const config = {
  contentUrl: './scene.ply',
  contents: fetch('./scene.ply'),
  renderer: 'webgl',
  noui: false,
  noanim: false,
  nofx: false,
};

const canvas = document.getElementById('application-canvas');

// Track loading progress
const origFetch = window.fetch;
window.fetch = function(url, opts) {
  if (url === './scene.ply' || url === 'scene.ply') {
    return origFetch(url, opts).then(response => {
      const total = parseInt(response.headers.get('content-length') || 0);
      const reader = response.body.getReader();
      let received = 0;
      return new Response(new ReadableStream({
        start(controller) {
          function pump() {
            reader.read().then(({ done, value }) => {
              if (done) {
                controller.close();
                document.getElementById('loading').style.display = 'none';
                return;
              }
              received += value.length;
              if (total > 0) {
                const pct = Math.min(100, Math.trunc(received / total * 100));
                document.getElementById('progress').style.width = pct + '%';
              }
              controller.enqueue(value);
              pump();
            }).catch(err => {
              console.error('Stream error:', err);
              controller.error(err);
            });
          }
          pump();
        }
      }), {
        headers: response.headers,
        status: response.status,
        statusText: response.statusText
      });
    });
  }
  return origFetch(url, opts);
};

main(canvas, settings, config).then(viewer => {
  console.log('SuperSplat viewer loaded successfully');
  document.getElementById('loading').style.display = 'none';
}).catch(err => {
  console.error('Failed to load viewer:', err);
  document.getElementById('loading').style.display = 'none';
  const errEl = document.getElementById('error');
  errEl.style.display = 'block';
  errEl.textContent = 'Error: ' + err.message;
});
</script>
</body>
</html>
"""


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_splat_viewer.py <ply_file>")
        print("  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply")
        sys.exit(1)

    ply_path = Path(sys.argv[1]).resolve()
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)

    # Fresh viewer directory
    viewer_dir = Path("/tmp/splatline_splat_viewer")
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Write the HTML viewer (loads supersplat engine from CDN)
    (viewer_dir / "index.html").write_text(VIEWER_HTML)

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
            elif self.path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            elif self.path.endswith('.ply'):
                self.send_header('Content-Type', 'application/octet-stream')
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
    print("Note: Large PLY files may take 10-30 seconds to load.")

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
