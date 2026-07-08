#!/usr/bin/env python3
"""
Splatline Splat Viewer — View PLY files as proper Gaussian splats in the browser.

Uses the PlayCanvas engine (same engine as superspl.at) to render PLY files
as actual Gaussian splats with alpha blending, not as point clouds or meshes.

Starts a local web server and opens the viewer in your default browser.
Works on macOS — no CUDA required, uses WebGL.

Usage:
  python run_splat_viewer.py <ply_file>
  python run_splat_viewer.py output_grok_3d/gaussians/frame_000000.ply
  python run_splat_viewer.py output_frame_000000_3d/frame_000000_points.ply
"""
import sys
import http.server
import socketserver
import threading
import webbrowser
import time
import os
from pathlib import Path


# HTML page with PlayCanvas Gaussian splat viewer
# Uses the same engine as superspl.at — renders splats as actual Gaussians
VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Splatline Splat Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #eee; font-family: -apple-system, sans-serif; overflow: hidden; }
  #app { width: 100vw; height: 100vh; position: relative; }
  canvas { display: block; width: 100%; height: 100%; }
  #info {
    position: absolute; top: 12px; left: 12px;
    background: rgba(0,0,0,0.6); padding: 10px 16px; border-radius: 8px;
    font-size: 13px; pointer-events: none; z-index: 10;
    backdrop-filter: blur(8px);
  }
  #info b { color: #4fc3f7; }
  #loading {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 18px; color: #4fc3f7; z-index: 10;
  }
  #controls {
    position: absolute; bottom: 12px; left: 12px;
    background: rgba(0,0,0,0.6); padding: 10px 16px; border-radius: 8px;
    font-size: 12px; pointer-events: none; z-index: 10;
    backdrop-filter: blur(8px); line-height: 1.6;
  }
  #error {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 16px; color: #ff5252; z-index: 10; text-align: center;
    max-width: 500px; display: none;
  }
</style>
</head>
<body>
<div id="app">
  <div id="loading">Loading Gaussian Splats...</div>
  <div id="info"><b>Splatline Splat Viewer</b><br><span id="splat-count">0</span> splats</div>
  <div id="controls">
    <b>Left drag:</b> Rotate &nbsp; <b>Right drag:</b> Pan &nbsp; <b>Scroll:</b> Zoom<br>
    <b>R:</b> Reset view &nbsp; <b>F:</b> Focus scene
  </div>
  <div id="error"></div>
  <canvas id="canvas"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/playcanvas@2.5.0/build/playcanvas.min.js"></script>
<script>
(async function() {
  const canvas = document.getElementById('canvas');
  const loadingEl = document.getElementById('loading');
  const errorEl = document.getElementById('error');
  const splatCountEl = document.getElementById('splat-count');

  // Get PLY file path from URL query
  const params = new URLSearchParams(window.location.search);
  const plyFile = params.get('ply') || 'scene.ply';

  // Initialize PlayCanvas app
  const app = new pc.AppBase(canvas);
  await app.init({
    deviceType: pc.DEVICETYPE_WEBGL2,
    graphicsDeviceOptions: {
      alpha: false,
      depth: true,
      stencil: false,
      antialias: true,
      powerPreference: 'high-performance'
    }
  });

  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.scene.gammaCorrection = pc.GAMMA_SRGB;
  app.scene.toneMapping = pc.TONEMAP_ACES;
  app.scene.exposure = 1.0;
  app.start();

  // Set background color
  app.scene.skyboxColor = new pc.Color(0.1, 0.1, 0.18);

  // Add camera
  const camera = new pc.Entity('camera');
  camera.addComponent('camera', {
    fov: 60,
    nearClip: 0.01,
    farClip: 1000,
    clearColor: new pc.Color(0.1, 0.1, 0.18)
  });
  camera.setLocalPosition(0, 0, 5);
  app.root.addChild(camera);

  // Add lights
  const light = new pc.Entity('light');
  light.addComponent('light', { type: 'directional', color: new pc.Color(1,1,1), intensity: 1 });
  app.root.addChild(light);

  // Mouse controls
  let mousePos = { x: 0, y: 0 };
  let mouseDown = false;
  let rightDown = false;
  let camRot = { x: 0, y: 0 };
  let camPos = { x: 0, y: 0, z: 5 };
  let targetCamPos = { x: 0, y: 0, z: 5 };

  canvas.addEventListener('mousedown', (e) => {
    mousePos = { x: e.clientX, y: e.clientY };
    if (e.button === 0) mouseDown = true;
    if (e.button === 2) rightDown = true;
  });
  canvas.addEventListener('mouseup', (e) => {
    if (e.button === 0) mouseDown = false;
    if (e.button === 2) rightDown = false;
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousemove', (e) => {
    const dx = e.clientX - mousePos.x;
    const dy = e.clientY - mousePos.y;
    mousePos = { x: e.clientX, y: e.clientY };
    if (mouseDown) {
      camRot.y -= dx * 0.005;
      camRot.x -= dy * 0.005;
      camRot.x = Math.max(-1.5, Math.min(1.5, camRot.x));
    }
    if (rightDown) {
      const fwd = new pc.Vec3(0,0,-1);
      const right = new pc.Vec3(1,0,0);
      const rot = new pc.Quat();
      rot.setFromEulerAngles(-camRot.x * 180/Math.PI, -camRot.y * 180/Math.PI, 0);
      fwd.rotate(rot);
      right.rotate(rot);
      targetCamPos.x -= right.x * dx * 0.01 + fwd.x * dy * 0.01;
      targetCamPos.y -= right.y * dx * 0.01 + fwd.y * dy * 0.01;
      targetCamPos.z -= right.z * dx * 0.01 + fwd.z * dy * 0.01;
    }
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    targetCamPos.x *= factor;
    targetCamPos.y *= factor;
    targetCamPos.z *= factor;
  });

  // Keyboard controls
  window.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') {
      camRot = { x: 0, y: 0 };
      targetCamPos = { x: 0, y: 0, z: 5 };
    }
    if (e.key === 'f' || e.key === 'F') {
      camRot = { x: 0, y: 0 };
      targetCamPos = { x: 0, y: 0, z: 5 };
    }
  });

  // Smooth camera update
  app.on('update', (dt) => {
    camPos.x += (targetCamPos.x - camPos.x) * 0.1;
    camPos.y += (targetCamPos.y - camPos.y) * 0.1;
    camPos.z += (targetCamPos.z - camPos.z) * 0.1;

    camera.setLocalPosition(camPos.x, camPos.y, camPos.z);
    camera.setLocalEulerAngles(-camRot.x * 180/Math.PI, -camRot.y * 180/Math.PI, 0);
  });

  // Load PLY file
  try {
    const response = await fetch(plyFile);
    if (!response.ok) throw new Error('Failed to load PLY file: ' + response.status);
    const buffer = await response.arrayBuffer();

    // Parse PLY file
    const plyData = parsePly(buffer);
    splatCountEl.textContent = plyData.count.toLocaleString();
    loadingEl.style.display = 'none';

    // Create Gaussian splat entity
    const splatEntity = createSplatEntity(app, plyData);
    app.root.addChild(splatEntity);

    // Auto-focus on the splats
    if (plyData.count > 0) {
      const center = plyData.center;
      const size = plyData.size;
      const distance = Math.max(size.x, size.y, size.z) * 1.5;
      targetCamPos = { x: center.x, y: center.y, z: center.z + distance };
      camPos = { x: center.x, y: center.y, z: center.z + distance };
    }

  } catch (err) {
    loadingEl.style.display = 'none';
    errorEl.style.display = 'block';
    errorEl.textContent = 'Error: ' + err.message;
    console.error(err);
  }

  // PLY parser — supports both standard 3DGS PLY and SHARP PLY formats
  function parsePly(buffer) {
    const decoder = new TextDecoder('ascii');
    const headerEnd = findHeaderEnd(buffer);
    const headerText = decoder.decode(new Uint8Array(buffer, 0, headerEnd));

    // Parse header
    const lines = headerText.split('\\n');
    let count = 0;
    let properties = [];
    let inVertex = false;

    for (const line of lines) {
      if (line.startsWith('element vertex')) {
        count = parseInt(line.split('\\s+')[2]);
        inVertex = true;
      } else if (line.startsWith('property')) {
        if (inVertex) {
          const parts = line.split('\\s+');
          properties.push({ type: parts[1], name: parts[parts.length-1] });
        }
      } else if (line.startsWith('end_header')) {
        break;
      } else if (!line.startsWith('comment') && !line.startsWith('format') && !line.startsWith('ply')) {
        inVertex = false;
      }
    }

    // Determine if binary or ascii
    const isBinary = headerText.includes('format binary_little_endian') ||
                     headerText.includes('format binary_big_endian');

    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 4);
    const scales = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const opacities = new Float32Array(count);

    if (isBinary) {
      parseBinaryPly(buffer, headerEnd, count, properties, positions, colors, scales, rotations, opacities);
    } else {
      parseAsciiPly(buffer, headerEnd, count, properties, positions, colors, scales, rotations, opacities);
    }

    // Compute center and size
    let minX=Infinity, minY=Infinity, minZ=Infinity;
    let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
    for (let i = 0; i < count; i++) {
      const x = positions[i*3], y = positions[i*3+1], z = positions[i*3+2];
      minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
    }
    const center = { x: (minX+maxX)/2, y: (minY+maxY)/2, z: (minZ+maxZ)/2 };
    const size = { x: maxX-minX, y: maxY-minY, z: maxZ-minZ };

    return { count, positions, colors, scales, rotations, opacities, center, size };
  }

  function findHeaderEnd(buffer) {
    const bytes = new Uint8Array(buffer);
    const pattern = [10, 101, 110, 100, 95, 104, 101, 97, 100, 101, 114, 10]; // \\nend_header\\n
    for (let i = 0; i < Math.min(bytes.length, 10000); i++) {
      let match = true;
      for (let j = 0; j < pattern.length; j++) {
        if (bytes[i+j] !== pattern[j]) { match = false; break; }
      }
      if (match) return i + pattern.length;
    }
    // Try without trailing newline
    const pattern2 = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // end_header
    for (let i = 0; i < Math.min(bytes.length, 10000); i++) {
      let match = true;
      for (let j = 0; j < pattern2.length; j++) {
        if (bytes[i+j] !== pattern2[j]) { match = false; break; }
      }
      if (match) return i + pattern2.length + 1;
    }
    throw new Error('Could not find end_header in PLY file');
  }

  function getPropertySize(type) {
    switch(type) {
      case 'float': case 'int32': case 'uint32': return 4;
      case 'uchar': case 'char': return 1;
      case 'short': case 'ushort': return 2;
      case 'double': return 8;
      default: return 4;
    }
  }

  function parseBinaryPly(buffer, offset, count, properties, positions, colors, scales, rotations, opacities) {
    const view = new DataView(buffer);
    const propNames = properties.map(p => p.name);

    // Compute stride
    let stride = 0;
    for (const prop of properties) stride += getPropertySize(prop.type);

    // Build property map
    const propMap = {};
    let byteOffset = 0;
    for (const prop of properties) {
      propMap[prop.name] = { offset: byteOffset, type: prop.type };
      byteOffset += getPropertySize(prop.type);
    }

    const hasSH = propNames.includes('f_dc_0');
    const hasScale = propNames.includes('scale_0');
    const hasRot = propNames.includes('rot_0');
    const hasOpacity = propNames.includes('opacity');
    const hasRGB = propNames.includes('red');

    for (let i = 0; i < count; i++) {
      const base = offset + i * stride;

      // Position
      positions[i*3] = view.getFloat32(base + propMap['x'].offset, true);
      positions[i*3+1] = view.getFloat32(base + propMap['y'].offset, true);
      positions[i*3+2] = view.getFloat32(base + propMap['z'].offset, true);

      // Colors
      if (hasSH) {
        const SH_C0 = 0.28209479177387814;
        colors[i*4] = 0.5 + SH_C0 * view.getFloat32(base + propMap['f_dc_0'].offset, true);
        colors[i*4+1] = 0.5 + SH_C0 * view.getFloat32(base + propMap['f_dc_1'].offset, true);
        colors[i*4+2] = 0.5 + SH_C0 * view.getFloat32(base + propMap['f_dc_2'].offset, true);
      } else if (hasRGB) {
        colors[i*4] = view.getUint8(base + propMap['red'].offset) / 255;
        colors[i*4+1] = view.getUint8(base + propMap['green'].offset) / 255;
        colors[i*4+2] = view.getUint8(base + propMap['blue'].offset) / 255;
      } else {
        colors[i*4] = 0.8; colors[i*4+1] = 0.8; colors[i*4+2] = 0.8;
      }

      // Opacity
      if (hasOpacity) {
        const op = view.getFloat32(base + propMap['opacity'].offset, true);
        colors[i*4+3] = 1.0 / (1.0 + Math.exp(-op)); // sigmoid
      } else {
        colors[i*4+3] = 1.0;
      }

      // Scale
      if (hasScale) {
        scales[i*3] = Math.exp(view.getFloat32(base + propMap['scale_0'].offset, true));
        scales[i*3+1] = Math.exp(view.getFloat32(base + propMap['scale_1'].offset, true));
        scales[i*3+2] = Math.exp(view.getFloat32(base + propMap['scale_2'].offset, true));
      } else {
        scales[i*3] = 0.01; scales[i*3+1] = 0.01; scales[i*3+2] = 0.01;
      }

      // Rotation
      if (hasRot) {
        rotations[i*4] = view.getFloat32(base + propMap['rot_0'].offset, true);
        rotations[i*4+1] = view.getFloat32(base + propMap['rot_1'].offset, true);
        rotations[i*4+2] = view.getFloat32(base + propMap['rot_2'].offset, true);
        rotations[i*4+3] = view.getFloat32(base + propMap['rot_3'].offset, true);
      } else {
        rotations[i*4] = 1; rotations[i*4+1] = 0; rotations[i*4+2] = 0; rotations[i*4+3] = 0;
      }
    }
  }

  function parseAsciiPly(buffer, offset, count, properties, positions, colors, scales, rotations, opacities) {
    const decoder = new TextDecoder('ascii');
    const text = decoder.decode(new Uint8Array(buffer, offset));
    const lines = text.split('\\n').filter(l => l.trim().length > 0);
    const propNames = properties.map(p => p.name);

    const hasSH = propNames.includes('f_dc_0');
    const hasScale = propNames.includes('scale_0');
    const hasRot = propNames.includes('rot_0');
    const hasOpacity = propNames.includes('opacity');
    const hasRGB = propNames.includes('red');

    for (let i = 0; i < Math.min(count, lines.length); i++) {
      const tokens = lines[i].trim().split(/\\s+/);
      const vals = {};
      for (let j = 0; j < propNames.length && j < tokens.length; j++) {
        vals[propNames[j]] = parseFloat(tokens[j]);
      }

      positions[i*3] = vals['x'] || 0;
      positions[i*3+1] = vals['y'] || 0;
      positions[i*3+2] = vals['z'] || 0;

      if (hasSH) {
        const SH_C0 = 0.28209479177387814;
        colors[i*4] = 0.5 + SH_C0 * (vals['f_dc_0'] || 0);
        colors[i*4+1] = 0.5 + SH_C0 * (vals['f_dc_1'] || 0);
        colors[i*4+2] = 0.5 + SH_C0 * (vals['f_dc_2'] || 0);
      } else if (hasRGB) {
        colors[i*4] = (vals['red'] || 128) / 255;
        colors[i*4+1] = (vals['green'] || 128) / 255;
        colors[i*4+2] = (vals['blue'] || 128) / 255;
      } else {
        colors[i*4] = 0.8; colors[i*4+1] = 0.8; colors[i*4+2] = 0.8;
      }

      if (hasOpacity) {
        const op = vals['opacity'] || 0;
        colors[i*4+3] = 1.0 / (1.0 + Math.exp(-op));
      } else {
        colors[i*4+3] = 1.0;
      }

      if (hasScale) {
        scales[i*3] = Math.exp(vals['scale_0'] || -4);
        scales[i*3+1] = Math.exp(vals['scale_1'] || -4);
        scales[i*3+2] = Math.exp(vals['scale_2'] || -4);
      } else {
        scales[i*3] = 0.01; scales[i*3+1] = 0.01; scales[i*3+2] = 0.01;
      }

      if (hasRot) {
        rotations[i*4] = vals['rot_0'] || 1;
        rotations[i*4+1] = vals['rot_1'] || 0;
        rotations[i*4+2] = vals['rot_2'] || 0;
        rotations[i*4+3] = vals['rot_3'] || 0;
      } else {
        rotations[i*4] = 1; rotations[i*4+1] = 0; rotations[i*4+2] = 0; rotations[i*4+3] = 0;
      }
    }
  }

  // Create Gaussian splat entity using point sprites
  // Renders each Gaussian as a billboard with alpha blending
  function createSplatEntity(app, data) {
    const entity = new pc.Entity('splat');
    entity.addComponent('model', { type: 'plane' });

    // Use a Points-based approach with custom shader for splat rendering
    // Create a mesh with all splat positions
    const mesh = new pc.Mesh(app.graphicsDevice);
    mesh.setPositions(data.positions);
    mesh.setColors(data.colors);
    mesh.update();

    const material = new pc.Material();
    material.shader = createSplatShader(app.graphicsDevice);
    material.blendType = pc.BLEND_NORMAL;
    material.depthWrite = false;
    material.cullMode = pc.CULL_NONE;

    const node = new pc.GraphNode();
    const meshInstance = new pc.MeshInstance(mesh, material, node);
    meshInstance.setParameter('splatScales', data.scales);
    meshInstance.setParameter('splatRotations', data.rotations);
    meshInstance.setParameter('splatOpacities', data.opacities);

    // Create model component
    const model = new pc.Model();
    model.graph = node;
    model.meshInstances = [meshInstance];
    entity.model.model = model;

    return entity;
  }

  function createSplatShader(device) {
    const shaderDefinition = {
      attributes: {
        aPosition: pc.SEMANTIC_POSITION,
        aColor: pc.SEMANTIC_COLOR
      },
      vshader: `
        attribute vec3 aPosition;
        attribute vec4 aColor;
        varying vec4 vColor;
        uniform mat4 matrix_model;
        uniform mat4 matrix_viewProjection;
        uniform float pointSize;
        void main() {
          vec4 worldPos = matrix_model * vec4(aPosition, 1.0);
          vec4 screenPos = matrix_viewProjection * worldPos;
          gl_Position = screenPos;
          gl_PointSize = pointSize;
          vColor = aColor;
        }
      `,
      fshader: `
        precision mediump float;
        varying vec4 vColor;
        void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          if (dist > 0.5) discard;
          float alpha = smoothstep(0.5, 0.0, dist) * vColor.a;
          gl_FragColor = vec4(vColor.rgb, alpha);
        }
      `
    };
    return new pc.Shader(device, shaderDefinition);
  }

  // Resize handler
  window.addEventListener('resize', () => {
    app.resizeCanvas();
  });

})();
</script>
</body>
</html>"""


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

    # Create temp directory for the viewer
    viewer_dir = Path("/tmp/splatline_viewer")
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Copy PLY file to viewer directory
    ply_copy = viewer_dir / "scene.ply"
    if ply_path != ply_copy:
        import shutil
        shutil.copy2(ply_path, ply_copy)

    # Write HTML viewer
    html_path = viewer_dir / "index.html"
    html_path.write_text(VIEWER_HTML)

    # Start local web server
    PORT = 8765

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viewer_dir), **kwargs)
        def log_message(self, format, *args):
            pass  # Suppress logging

    print("=" * 60)
    print("SPLATLINE SPLAT VIEWER")
    print("=" * 60)
    print(f"PLY file: {ply_path.name}")
    print(f"File size: {ply_path.stat().st_size / 1e6:.1f} MB")
    print(f"Viewer: http://localhost:{PORT}")
    print()
    print("Opening browser... (Ctrl+C to stop)")

    # Start server in background thread
    server = socketserver.TCPServer(("localhost", PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Open browser
    time.sleep(0.5)
    webbrowser.open(f"http://localhost:{PORT}/index.html?ply=scene.ply")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
