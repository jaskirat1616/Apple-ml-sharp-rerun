#!/usr/bin/env python3
"""
Splatline Video Splat Viewer — Play 3D video frames in the browser.

Renders each PLY frame as Gaussian splats with a timeline scrubber,
play/pause, and frame-by-frame navigation — like Rerun but in the browser.

Frames are loaded on-demand as you scrub through the timeline, so it
handles large PLY files without loading everything at once.

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
from pathlib import Path


def convert_to_standard_ply(input_path, output_path):
    """Convert SHARP PLY to standard 3DGS PLY (strip extra elements)."""
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(input_path))

    if len(ply.elements) == 1:
        shutil.copy2(input_path, output_path)
        return

    vertex = ply['vertex']
    new_vertex = PlyElement.describe(vertex.data, 'vertex')
    new_ply = PlyData([new_vertex], text=False, byte_order='<')
    new_ply.write(str(output_path))


VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<title>Splatline Video Viewer</title>
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
* { margin: 0; padding: 0; box-sizing: border-box; }
body { overflow: hidden; background: #0a0a14; font-family: -apple-system, sans-serif; color: #fff; }
canvas { display: block; }

#loading {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  z-index: 100; background: #0a0a14;
}
#loading h2 { font-size: 22px; margin-bottom: 16px; font-weight: 500; }
#loading .bar { width: 320px; height: 6px; background: #1a1a2e; border-radius: 3px; overflow: hidden; }
#loading .fill { height: 100%; background: linear-gradient(90deg, #F26722, #ff8c42); width: 0%; transition: width 0.3s; }
#loading .pct { margin-top: 8px; font-size: 14px; color: #888; }

#controls {
  position: fixed; bottom: 0; left: 0; right: 0;
  background: rgba(10,10,20,0.95); border-top: 1px solid #222;
  padding: 12px 20px; z-index: 50;
  display: flex; align-items: center; gap: 16px;
}
#playBtn {
  background: #F26722; border: none; color: #fff; width: 40px; height: 40px;
  border-radius: 50%; cursor: pointer; font-size: 18px; display: flex;
  align-items: center; justify-content: center; flex-shrink: 0;
}
#playBtn:hover { background: #ff8c42; }
#timeline {
  flex: 1; height: 8px; background: #1a1a2e; border-radius: 4px;
  position: relative; cursor: pointer;
}
#timelineFill {
  height: 100%; background: linear-gradient(90deg, #F26722, #ff8c42);
  border-radius: 4px; width: 0%; pointer-events: none;
}
#timelineHandle {
  position: absolute; top: 50%; left: 0%; transform: translate(-50%, -50%);
  width: 16px; height: 16px; background: #fff; border-radius: 50%;
  pointer-events: none; box-shadow: 0 0 8px rgba(0,0,0,0.5);
}
#frameInfo {
  font-size: 13px; color: #888; min-width: 120px; text-align: right;
  font-variant-numeric: tabular-nums;
}
#frameInfo b { color: #fff; }

#info {
  position: fixed; top: 16px; left: 16px; z-index: 50;
  font-size: 12px; color: #888; pointer-events: none;
}
#info b { color: #ccc; }

#error {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  color: #ff6b6b; font-size: 16px; text-align: center; z-index: 100;
  max-width: 600px; display: none; background: #1a1a2e; padding: 24px; border-radius: 12px;
  border: 1px solid #333;
}
</style>
</head>
<body>

<div id="loading">
  <h2>Loading Splatline Video Viewer...</h2>
  <div class="bar"><div class="fill" id="progress"></div></div>
  <div class="pct" id="pct">0%</div>
</div>

<div id="error"></div>
<div id="info"><b>Splatline Video Viewer</b> — drag to orbit · scroll to zoom · right-drag to pan</div>

<div id="controls">
  <button id="playBtn">&#9654;</button>
  <div id="timeline">
    <div id="timelineFill"></div>
    <div id="timelineHandle"></div>
  </div>
  <div id="frameInfo">Frame <b id="curFrame">0</b> / <b id="totFrame">0</b></div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const loadingEl = document.getElementById('loading');
const progressEl = document.getElementById('progress');
const pctEl = document.getElementById('pct');
const errorEl = document.getElementById('error');
const playBtn = document.getElementById('playBtn');
const timeline = document.getElementById('timeline');
const timelineFill = document.getElementById('timelineFill');
const timelineHandle = document.getElementById('timelineHandle');
const curFrameEl = document.getElementById('curFrame');
const totFrameEl = document.getElementById('totFrame');

function showError(msg) {
  loadingEl.style.display = 'none';
  errorEl.style.display = 'block';
  errorEl.textContent = msg;
}

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 1000);
camera.position.set(0, 1, 4);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.rotateSpeed = 0.8;

// Load manifest
let manifest = null;
let currentFrame = 0;
let isPlaying = false;
let playTimer = null;
let splatMesh = null;
const frameCache = new Map();
const MAX_CACHE = 5;

// Parse PLY binary
function parsePly(arrayBuffer) {
  const data = new DataView(arrayBuffer);
  const decoder = new TextDecoder('ascii');

  // Find end_header
  let offset = 0;
  let headerEnd = -1;
  while (offset < arrayBuffer.byteLength - 10) {
    if (data.getUint8(offset) === 0x65 && // 'e'
        data.getUint8(offset+1) === 0x6e && // 'n'
        data.getUint8(offset+2) === 0x64 && // 'd'
        data.getUint8(offset+3) === 0x5f && // '_'
        data.getUint8(offset+4) === 0x68 && // 'h'
        data.getUint8(offset+5) === 0x65 && // 'e'
        data.getUint8(offset+6) === 0x61 && // 'a'
        data.getUint8(offset+7) === 0x64 && // 'd'
        data.getUint8(offset+8) === 0x65 && // 'e'
        data.getUint8(offset+9) === 0x72) {
      // skip to after the newline
      let nl = offset + 10;
      while (nl < arrayBuffer.byteLength && data.getUint8(nl) !== 0x0a) nl++;
      headerEnd = nl + 1;
      break;
    }
    offset++;
  }

  if (headerEnd === -1) throw new Error('Invalid PLY: no end_header');

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
      propTypes.push(words[1]);
      props.push(words[words.length - 1]);
    }
  }

  const typeSizes = { 'float': 4, 'double': 8, 'uchar': 1, 'char': 1, 'ushort': 2, 'short': 2, 'uint': 4, 'int': 4 };
  const propOffsets = {};
  let byteOffset = 0;
  for (let i = 0; i < props.length; i++) {
    propOffsets[props[i]] = byteOffset;
    byteOffset += typeSizes[propTypes[i]] || 4;
  }
  const stride = byteOffset;

  const hasSH = 'f_dc_0' in propOffsets;
  const hasScale = 'scale_0' in propOffsets;
  const hasRot = 'rot_0' in propOffsets;
  const hasOpacity = 'opacity' in propOffsets;
  const hasRGB = 'red' in propOffsets;
  const SH_C0 = 0.28209479177387814;

  // Subsample if too many splats
  const MAX_SPLATS = 200000;
  let indices = null;
  let count = vertexCount;
  if (count > MAX_SPLATS) {
    // Sample every Nth vertex (fast, no sorting needed)
    const step = Math.ceil(count / MAX_SPLATS);
    indices = [];
    for (let i = 0; i < count; i += step) indices.push(i);
    count = indices.length;
  }

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const opacities = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const vi = indices ? indices[i] : i;
    const base = headerEnd + vi * stride;

    positions[i * 3]     = data.getFloat32(base + propOffsets['x'], true);
    positions[i * 3 + 1] = data.getFloat32(base + propOffsets['y'], true);
    positions[i * 3 + 2] = data.getFloat32(base + propOffsets['z'], true);

    if (hasSH) {
      colors[i * 3]     = Math.max(0, Math.min(1, 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_0'], true)));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_1'], true)));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * data.getFloat32(base + propOffsets['f_dc_2'], true)));
    } else if (hasRGB) {
      colors[i * 3]     = data.getUint8(base + propOffsets['red']) / 255;
      colors[i * 3 + 1] = data.getUint8(base + propOffsets['green']) / 255;
      colors[i * 3 + 2] = data.getUint8(base + propOffsets['blue']) / 255;
    } else {
      colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.8;
    }

    if (hasScale) {
      const sx = Math.exp(data.getFloat32(base + propOffsets['scale_0'], true));
      const sy = Math.exp(data.getFloat32(base + propOffsets['scale_1'], true));
      const sz = Math.exp(data.getFloat32(base + propOffsets['scale_2'], true));
      sizes[i] = Math.max(sx, sy, sz);
    } else {
      sizes[i] = 0.01;
    }

    if (hasOpacity) {
      const op = data.getFloat32(base + propOffsets['opacity'], true);
      opacities[i] = 1.0 / (1.0 + Math.exp(-op));
    } else {
      opacities[i] = 1.0;
    }
  }

  return { positions, colors, sizes, opacities, count };
}

// Splat shader material
const splatMaterial = new THREE.ShaderMaterial({
  uniforms: { u_time: { value: 0 } },
  vertexShader: `
    attribute vec3 color;
    attribute float size;
    attribute float opacity;
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vColor = color;
      vOpacity = opacity;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * 250.0 / -mvPosition.z;
      gl_PointSize = clamp(gl_PointSize, 1.0, 64.0);
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    varying float vOpacity;
    void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = length(coord);
      if (dist > 0.5) discard;
      float alpha = exp(-dist * dist * 8.0) * vOpacity;
      gl_FragColor = vec4(vColor, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.NormalBlending,
});

// First frame bounding box for consistent camera
let sceneCenter = new THREE.Vector3(0, 0, 0);
let sceneScale = 1.0;
let cameraInitialized = false;

function showFrame(splatData) {
  if (splatMesh) {
    scene.remove(splatMesh);
    splatMesh.geometry.dispose();
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(splatData.positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(splatData.colors, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(splatData.sizes, 1));
  geo.setAttribute('opacity', new THREE.BufferAttribute(splatData.opacities, 1));

  splatMesh = new THREE.Points(geo, splatMaterial);

  if (!cameraInitialized) {
    geo.computeBoundingBox();
    const box = geo.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    sceneCenter.copy(center);

    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    sceneScale = 4.0 / maxDim;

    splatMesh.position.sub(center);
    splatMesh.scale.setScalar(sceneScale);

    camera.position.set(0, 0, 5);
    controls.target.set(0, 0, 0);
    controls.update();
    cameraInitialized = true;
  } else {
    // Keep consistent camera across frames
    splatMesh.position.sub(sceneCenter);
    splatMesh.scale.setScalar(sceneScale);
  }

  scene.add(splatMesh);
}

async function loadFrame(idx) {
  if (frameCache.has(idx)) {
    showFrame(frameCache.get(idx));
    return;
  }

  const url = `frames/frame_${String(idx).padStart(4, '0')}.ply`;
  console.log(`Loading frame ${idx}: ${url}`);

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load frame ${idx}: ${response.status}`);
  const buf = await response.arrayBuffer();
  const splatData = parsePly(buf);

  // Cache management
  frameCache.set(idx, splatData);
  if (frameCache.size > MAX_CACHE) {
    // Remove oldest entry (lowest frame number that isn't current)
    const keys = Array.from(frameCache.keys()).sort((a, b) => a - b);
    for (const k of keys) {
      if (k !== idx && frameCache.size > MAX_CACHE) {
        frameCache.delete(k);
      }
    }
  }

  showFrame(splatData);
}

function updateTimeline() {
  const pct = manifest.frames > 1 ? (currentFrame / (manifest.frames - 1)) * 100 : 0;
  timelineFill.style.width = pct + '%';
  timelineHandle.style.left = pct + '%';
  curFrameEl.textContent = currentFrame;
}

async function setFrame(idx) {
  if (idx < 0 || idx >= manifest.frames) return;
  currentFrame = idx;
  updateTimeline();
  try {
    await loadFrame(idx);
  } catch (err) {
    console.error('Frame load error:', err);
    showError('Error loading frame ' + idx + ': ' + err.message);
    isPlaying = false;
    playBtn.innerHTML = '&#9654;';
  }
}

// Playback
function play() {
  isPlaying = true;
  playBtn.innerHTML = '&#10074;&#10074;';
  const fps = manifest.fps || 8;
  const interval = 1000 / fps;

  playTimer = setInterval(async () => {
    let next = currentFrame + 1;
    if (next >= manifest.frames) next = 0; // loop
    await setFrame(next);
  }, interval);
}

function pause() {
  isPlaying = false;
  playBtn.innerHTML = '&#9654;';
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

playBtn.addEventListener('click', () => {
  if (isPlaying) pause();
  else play();
});

// Timeline scrubbing
let isScrubbing = false;
async function scrubTo(clientX) {
  const rect = timeline.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  const frame = Math.round(pct * (manifest.frames - 1));
  if (frame !== currentFrame) {
    await setFrame(frame);
  }
}

timeline.addEventListener('mousedown', (e) => {
  isScrubbing = true;
  if (isPlaying) pause();
  scrubTo(e.clientX);
});

document.addEventListener('mousemove', (e) => {
  if (isScrubbing) scrubTo(e.clientX);
});

document.addEventListener('mouseup', () => {
  isScrubbing = false;
});

// Keyboard
document.addEventListener('keydown', (e) => {
  if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
  else if (e.key === 'ArrowLeft') setFrame(currentFrame - 1);
  else if (e.key === 'ArrowRight') setFrame(currentFrame + 1);
});

// Resize
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

// Init
async function init() {
  try {
    const resp = await fetch('manifest.json');
    manifest = await resp.json();
    console.log('Manifest:', manifest);
    totFrameEl.textContent = manifest.frames;

    progressEl.style.width = '100%';
    pctEl.textContent = 'Loading frame 0...';

    await setFrame(0);

    loadingEl.style.display = 'none';
    console.log('Viewer ready!');
  } catch (err) {
    console.error('Init failed:', err);
    showError('Error: ' + err.message);
  }
}

init();
</script>
</body>
</html>
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

    # Get FPS from the video
    import cv2
    video_path = None
    for p in [Path("/Users/jaskiratsingh/Downloads/grok-video-a1a6d6a4-6f94-41c2-82b5-83ec305487ae.mp4")]:
        if p.exists():
            video_path = p
            break

    fps = 8.0  # default
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frame_skip = 3
        fps = source_fps / frame_skip

    print("=" * 60)
    print("SPLATLINE VIDEO SPLAT VIEWER")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print(f"PLY files:  {len(ply_files)}")
    print(f"FPS:        {fps:.1f}")
    print()

    # Fresh viewer directory
    viewer_dir = Path("/tmp/splatline_video_viewer")
    if viewer_dir.exists():
        shutil.rmtree(viewer_dir)
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Write HTML
    (viewer_dir / "index.html").write_text(VIEWER_HTML)

    # Convert PLYs and copy to frames/ dir
    frames_dir = viewer_dir / "frames"
    frames_dir.mkdir()
    print(f"Converting {len(ply_files)} PLY files...")
    for i, ply_path in enumerate(ply_files):
        std_path = frames_dir / f"frame_{i:04d}.ply"
        convert_to_standard_ply(ply_path, std_path)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ply_files)}] converted")

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
    print(f"Viewer: http://localhost:{PORT}")
    print()
    print("Controls:")
    print("  Play button / Spacebar:  Play/pause")
    print("  Timeline scrubber:       Drag to seek")
    print("  Left/Right arrows:       Step frames")
    print("  Left drag:               Orbit camera")
    print("  Right drag:              Pan camera")
    print("  Scroll:                  Zoom")
    print()
    print("Opening viewer... (Ctrl+C to stop)")

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
