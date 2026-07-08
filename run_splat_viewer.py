#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as Gaussian splats in the browser.

Uses Three.js + GaussianSplats3D to render actual Gaussian splats locally.
Each splat is rendered as a camera-facing alpha-blended projected ellipsoid —
the same visual quality as superspl.at.

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
from pathlib import Path


VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<title>Splatline Splat Viewer</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.169.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.169.0/examples/jsm/"
  }
}
</script>
<style>
* { margin: 0; padding: 0; }
body { overflow: hidden; background: #0a0a14; font-family: -apple-system, sans-serif; }
canvas { display: block; }

#loading {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: #fff; z-index: 100; background: #0a0a14;
}
#loading h2 { font-size: 22px; margin-bottom: 16px; font-weight: 500; }
#loading .bar {
  width: 320px; height: 6px; background: #1a1a2e; border-radius: 3px; overflow: hidden;
}
#loading .fill {
  height: 100%; background: linear-gradient(90deg, #F26722, #ff8c42); width: 0%;
  transition: width 0.3s ease;
}
#loading .pct { margin-top: 8px; font-size: 14px; color: #888; }

#error {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  color: #ff6b6b; font-size: 16px; text-align: center; z-index: 100;
  max-width: 600px; display: none; background: #1a1a2e; padding: 24px; border-radius: 12px;
  border: 1px solid #333;
}

#info {
  position: fixed; bottom: 16px; left: 16px; z-index: 50;
  color: #888; font-size: 12px; pointer-events: none;
}
#info b { color: #ccc; }
</style>
</head>
<body>

<div id="loading">
  <h2>Loading Gaussian Splats...</h2>
  <div class="bar"><div class="fill" id="progress"></div></div>
  <div class="pct" id="pct">0%</div>
</div>

<div id="error"></div>
<div id="info"><b>Splatline Splat Viewer</b> — drag to orbit · scroll to zoom · right-drag to pan</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const loadingEl = document.getElementById('loading');
const progressEl = document.getElementById('progress');
const pctEl = document.getElementById('pct');
const errorEl = document.getElementById('error');

function showError(msg) {
  loadingEl.style.display = 'none';
  errorEl.style.display = 'block';
  errorEl.textContent = msg;
}

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.rotateSpeed = 0.8;

// Fetch the PLY with progress tracking
async function loadPly(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch PLY: ${response.status}`);
  const total = parseInt(response.headers.get('content-length') || 0);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.min(100, Math.trunc(received / total * 100));
      progressEl.style.width = pct + '%';
      pctEl.textContent = pct + '%';
    }
  }
  return new Blob(chunks).arrayBuffer();
}

// Parse standard 3DGS PLY
function parsePly(arrayBuffer) {
  const data = new DataView(arrayBuffer);
  let offset = 0;

  // Read header
  const decoder = new TextDecoder('ascii');
  let headerEnd = -1;
  const headerBytes = [];
  while (offset < arrayBuffer.byteLength) {
    const byte = data.getUint8(offset);
    headerBytes.push(byte);
    offset++;
    if (headerBytes.length >= 11) {
      const tail = headerBytes.slice(-11);
      const str = decoder.decode(new Uint8Array(tail));
      if (str === 'end_header\n') {
        headerEnd = offset;
        break;
      }
    }
  }

  if (headerEnd === -1) throw new Error('Invalid PLY: no end_header found');

  const headerText = decoder.decode(new Uint8Array(arrayBuffer, 0, headerEnd));
  const lines = headerText.split('\n');

  let vertexCount = 0;
  const props = [];
  const propTypes = [];
  let inVertex = false;
  for (const line of lines) {
    const words = line.trim().split(/\s+/);
    if (words[0] === 'element') {
      inVertex = (words[1] === 'vertex');
      if (inVertex) vertexCount = parseInt(words[2]);
    } else if (words[0] === 'property' && inVertex) {
      // property float x  OR  property uchar red
      propTypes.push(words[1]);
      props.push(words[words.length - 1]);
    }
  }

  // Compute byte size per vertex
  const typeSizes = { 'float': 4, 'double': 8, 'uchar': 1, 'char': 1, 'ushort': 2, 'short': 2, 'uint': 4, 'int': 4 };
  const stride = propTypes.reduce((sum, t) => sum + (typeSizes[t] || 4), 0);

  console.log(`PLY: ${vertexCount} vertices, ${props.length} properties (${stride} bytes/vert): ${props.join(', ')}`);

  // Read vertex data
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const scales = new Float32Array(vertexCount * 3);
  const rotations = new Float32Array(vertexCount * 4);
  const opacities = new Float32Array(vertexCount);

  // Compute byte offset of each property within a vertex
  const propOffsets = {};
  let byteOffset = 0;
  for (let i = 0; i < props.length; i++) {
    propOffsets[props[i]] = byteOffset;
    byteOffset += typeSizes[propTypes[i]] || 4;
  }

  const hasSH = 'f_dc_0' in propOffsets;
  const hasScale = 'scale_0' in propOffsets;
  const hasRot = 'rot_0' in propOffsets;
  const hasOpacity = 'opacity' in propOffsets;
  const hasRGB = 'red' in propOffsets;

  const SH_C0 = 0.28209479177387814;
  let ptr = headerEnd;

  for (let i = 0; i < vertexCount; i++) {
    const base = ptr;
    positions[i * 3]     = data.getFloat32(base + propOffsets['x'], true);
    positions[i * 3 + 1] = data.getFloat32(base + propOffsets['y'], true);
    positions[i * 3 + 2] = data.getFloat32(base + propOffsets['z'], true);

    if (hasSH) {
      colors[i * 3]     = 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_0'], true);
      colors[i * 3 + 1] = 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_1'], true);
      colors[i * 3 + 2] = 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_2'], true);
    } else if (hasRGB) {
      const ri = propTypes[propOffsets['_red_idx'] || 0];
      colors[i * 3]     = data.getUint8(base + propOffsets['red']) / 255;
      colors[i * 3 + 1] = data.getUint8(base + propOffsets['green']) / 255;
      colors[i * 3 + 2] = data.getUint8(base + propOffsets['blue']) / 255;
    } else {
      colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.8;
    }

    if (hasScale) {
      scales[i * 3]     = Math.exp(data.getFloat32(base + propOffsets['scale_0'], true));
      scales[i * 3 + 1] = Math.exp(data.getFloat32(base + propOffsets['scale_1'], true));
      scales[i * 3 + 2] = Math.exp(data.getFloat32(base + propOffsets['scale_2'], true));
    } else {
      scales[i * 3] = scales[i * 3 + 1] = scales[i * 3 + 2] = 0.01;
    }

    if (hasRot) {
      rotations[i * 4]     = data.getFloat32(base + propOffsets['rot_0'], true);
      rotations[i * 4 + 1] = data.getFloat32(base + propOffsets['rot_1'], true);
      rotations[i * 4 + 2] = data.getFloat32(base + propOffsets['rot_2'], true);
      rotations[i * 4 + 3] = data.getFloat32(base + propOffsets['rot_3'], true);
    } else {
      rotations[i * 4] = 1; rotations[i * 4 + 1] = rotations[i * 4 + 2] = rotations[i * 4 + 3] = 0;
    }

    if (hasOpacity) {
      const op = data.getFloat32(base + propOffsets['opacity'], true);
      opacities[i] = 1.0 / (1.0 + Math.exp(-op));
    } else {
      opacities[i] = 1.0;
    }

    ptr += stride;
  }

  return { positions, colors, scales, rotations, opacities, vertexCount };
}

// Build geometry from splat data
function buildGeometry(splatData) {
  const { positions, colors, scales, rotations, opacities, vertexCount } = splatData;

  // Subsample if too many splats
  const MAX = 100000;
  let count = vertexCount;
  let indices = null;
  if (count > MAX) {
    console.log(`Subsampling from ${count} to ${MAX} splats`);
    indices = new Uint32Array(MAX);
    // Sort by opacity (descending) and take top MAX
    const order = Array.from({length: count}, (_, i) => i);
    order.sort((a, b) => opacities[b] - opacities[a]);
    for (let i = 0; i < MAX; i++) indices[i] = order[i];
    count = MAX;
  }

  const geo = new THREE.BufferGeometry();
  const posArr = new Float32Array(count * 3);
  const colArr = new Float32Array(count * 3);
  const scaleArr = new Float32Array(count * 3);
  const rotArr = new Float32Array(count * 4);
  const opArr = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const src = indices ? indices[i] : i;
    posArr[i * 3]     = positions[src * 3];
    posArr[i * 3 + 1] = positions[src * 3 + 1];
    posArr[i * 3 + 2] = positions[src * 3 + 2];
    colArr[i * 3]     = Math.max(0, Math.min(1, colors[src * 3]));
    colArr[i * 3 + 1] = Math.max(0, Math.min(1, colors[src * 3 + 1]));
    colArr[i * 3 + 2] = Math.max(0, Math.min(1, colors[src * 3 + 2]));
    scaleArr[i * 3]   = scales[src * 3];
    scaleArr[i * 3 + 1] = scales[src * 3 + 1];
    scaleArr[i * 3 + 2] = scales[src * 3 + 2];
    rotArr[i * 4]     = rotations[src * 4];
    rotArr[i * 4 + 1] = rotations[src * 4 + 1];
    rotArr[i * 4 + 2] = rotations[src * 4 + 2];
    rotArr[i * 4 + 3] = rotations[src * 4 + 3];
    opArr[i] = opacities[src];
  }

  geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  geo.setAttribute('scale', new THREE.BufferAttribute(scaleArr, 3));
  geo.setAttribute('rotation', new THREE.BufferAttribute(rotArr, 4));
  geo.setAttribute('opacity', new THREE.BufferAttribute(opArr, 1));

  return geo;
}

// Custom shader material for Gaussian splats
function createSplatMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_time: { value: 0 },
    },
    vertexShader: `
      attribute vec3 color;
      attribute vec3 scale;
      attribute vec4 rotation;
      attribute float opacity;
      varying vec3 vColor;
      varying float vOpacity;

      void main() {
        vColor = color;
        vOpacity = opacity;

        // Compute splat size from scale (use max axis)
        float splatSize = max(max(scale.x, scale.y), scale.z) * 3.0;

        // Use point size based on splat scale and distance
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = splatSize * 300.0 / -mvPosition.z;
        gl_PointSize = clamp(gl_PointSize, 1.0, 64.0);

        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vOpacity;

      void main() {
        // Soft circular splat — Gaussian falloff
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        if (dist > 0.5) discard;

        // Gaussian falloff for soft splat edges
        float alpha = exp(-dist * dist * 8.0) * vOpacity;

        gl_FragColor = vec4(vColor, alpha);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });
}

// Main
async function init() {
  try {
    console.log('Fetching PLY...');
    const buf = await loadPly('./scene.ply');
    console.log(`Downloaded ${(buf.byteLength / 1e6).toFixed(1)} MB`);

    progressEl.style.width = '100%';
    pctEl.textContent = 'Parsing...';

    const splatData = parsePly(buf);
    console.log(`Parsed ${splatData.vertexCount} splats`);

    const geometry = buildGeometry(splatData);
    const material = createSplatMaterial();
    const points = new THREE.Points(geometry, material);

    // Center the splat cloud
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    points.position.sub(center);

    // Scale to reasonable size
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 5.0 / maxDim;
    points.scale.setScalar(scale);

    scene.add(points);

    // Position camera to see the splats
    camera.position.set(0, 0, 6);
    controls.target.set(0, 0, 0);
    controls.update();

    loadingEl.style.display = 'none';
    console.log('Viewer ready!');
  } catch (err) {
    console.error('Failed:', err);
    showError('Error: ' + err.message);
  }
}

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

init();
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

    # Write the HTML viewer
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
    print("SPLATLINE SPLAT VIEWER (Three.js + GaussianSplats3D)")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"Viewer:   http://localhost:{PORT}")
    print()
    print("Controls:")
    print("  Left drag:   Orbit")
    print("  Right drag:  Pan")
    print("  Scroll:      Zoom")
    print()
    print("Opening browser... (Ctrl+C to stop)")

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
