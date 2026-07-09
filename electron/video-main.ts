import { BrowserWindow, shell } from "electron";
import { existsSync, mkdirSync, rmSync, writeFileSync, readdirSync, copyFileSync, readFileSync, statSync, createReadStream } from "node:fs";
import path from "node:path";
import http from "node:http";
import os from "node:os";
import { spawn, type ChildProcess } from "node:child_process";

let mainWindow: BrowserWindow | null = null;
let staticServer: http.Server | null = null;
let pythonProcess: ChildProcess | null = null;

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".ply": "application/octet-stream",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const startStaticServer = (rootDir: string, port: number): Promise<void> => {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      let urlPath = req.url?.split("?")[0] || "/";
      if (urlPath === "/") urlPath = "/index.html";

      // Decode URL path
      urlPath = decodeURIComponent(urlPath);

      const filePath = path.join(rootDir, urlPath);
      if (!filePath.startsWith(rootDir)) {
        res.writeHead(403);
        res.end("Forbidden");
        return;
      }

      if (!existsSync(filePath)) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }

      const ext = path.extname(filePath);
      const mimeType = MIME_TYPES[ext] || "application/octet-stream";

      // Use streaming for large files (PLY files can be 60MB+)
      // readFileSync blocks the event loop and crashes Electron's network service
      try {
        const stat = statSync(filePath);
        res.writeHead(200, {
          "Content-Type": mimeType,
          "Content-Length": stat.size,
          "Cache-Control": "no-store, no-cache, must-revalidate",
          "Access-Control-Allow-Origin": "*",
        });
        const stream = createReadStream(filePath);
        stream.on("error", (err) => {
          console.error(`Error streaming ${filePath}:`, err);
          if (!res.writableEnded) res.end();
        });
        stream.pipe(res);
      } catch (err) {
        console.error(`Error serving ${filePath}:`, err);
        res.writeHead(500);
        res.end("Internal error");
      }
    });

    server.on("error", reject);
    server.listen(port, "127.0.0.1", () => {
      staticServer = server;
      resolve();
    });
  });
};

const SEQUENCE_LOADER_JS = `
<style>
#splatline-2d-panel {
  position: fixed; bottom: 80px; left: 10px; width: 240px; height: 180px;
  background: rgba(17,17,17,0.9); border: 1px solid #333; border-radius: 6px;
  z-index: 5000; display: flex; align-items: center; justify-content: center;
  overflow: hidden; pointer-events: none;
}
#splatline-2d-panel video { width: 100%; height: 100%; object-fit: contain; }
#splatline-2d-panel .label {
  position: absolute; top: 4px; left: 8px; color: #888; font-size: 10px;
  font-family: sans-serif; text-shadow: 0 1px 2px #000; z-index: 1;
}
#splatline-2d-panel.hidden { display: none; }
#splatline-loading {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  z-index: 10001; color: #fff; font-family: sans-serif; text-align: center;
  background: rgba(0,0,0,0.8); padding: 20px 40px; border-radius: 8px;
}
#splatline-loading .bar { width: 200px; height: 4px; background: #333; border-radius: 2px; margin: 10px auto 0; }
#splatline-loading .fill { height: 100%; background: #4a9eff; border-radius: 2px; width: 0%; transition: width 0.3s; }
</style>
<div id="splatline-2d-panel" class="hidden">
  <div class="label">2D Source</div>
  <video id="splatline-2d-video" muted loop playsinline></video>
</div>
<div id="splatline-loading">
  <div>Loading 3D frames...</div>
  <div class="bar"><div class="fill" id="splatline-load-fill"></div></div>
</div>
<script>
(async function() {
    const manifest = await fetch('manifest.json').then(r => r.json());
    console.log('Splatline: Loading ' + manifest.frames + ' frames, fps=' + manifest.fps + ', has2d=' + manifest.has2d);

    const panel2d = document.getElementById('splatline-2d-panel');
    const video2d = document.getElementById('splatline-2d-video');
    const loadingEl = document.getElementById('splatline-loading');
    const loadFill = document.getElementById('splatline-load-fill');
    if (manifest.has2d && manifest.videoSrc) {
        video2d.src = manifest.videoSrc;
        panel2d.classList.remove('hidden');
    }

    // Build URLs and names for on-demand fetching (avoids OOM — no need to
    // load all 81 PLY files as File objects in JS heap at once)
    const urls = [];
    const names = [];
    for (let i = 0; i < manifest.frames; i++) {
        const filename = 'frame_' + String(i).padStart(4, '0') + '.ply';
        urls.push('frames/' + filename);
        names.push(filename);
    }
    loadFill.style.width = '10%';

    console.log('Splatline: Setting URL sequence with ' + urls.length + ' frames');

    function tryImport(retries) {
        if (window.__splatlineEvents) {
            const ev = window.__splatlineEvents;
            // Use sequence.setPlyUrls — fetches PLYs on-demand from server
            ev.fire('sequence.setPlyUrls', { urls: urls, names: names });
            ev.fire('timeline.frame', 0);
            ev.fire('timeline.setFrameRate', manifest.fps);
            ev.fire('timeline.setLoop', true);

            // Sync 2D video to 3D timeline.
            // The 3D timeline advances at fps=8 (24/3). Each 3D frame corresponds
            // to a specific time in the 2D video: t = frame * frameSkip / sourceFps.
            // The 2D video plays continuously at native 24fps for smoothness.
            // We re-sync only on large drift (>0.3s) or when 3D playback pauses/scrubs.
            if (manifest.has2d && video2d) {
                video2d.play().catch(() => {});
                let is3dPlaying = false;

                ev.on('timeline.playing', (playing) => {
                    is3dPlaying = playing;
                    if (playing) {
                        video2d.play().catch(() => {});
                    } else {
                        video2d.pause();
                    }
                });

                ev.on('timeline.frame', (frame) => {
                    if (!video2d) return;
                    const targetT = frame * (manifest.frameSkip || 3) / (manifest.sourceFps || 24);
                    if (!is3dPlaying) {
                        // Paused/scrubbing: seek exactly
                        video2d.currentTime = targetT;
                    } else {
                        // Playing: only re-sync if drift > 0.3s
                        if (Math.abs(video2d.currentTime - targetT) > 0.3) {
                            video2d.currentTime = targetT;
                        }
                    }
                });
            }

            // Track real preload progress from the editor
            let started = false;
            ev.on('sequence.preloadProgress', (data) => {
                const pct = Math.round((data.loaded / data.total) * 100);
                loadFill.style.width = pct + '%';
                loadingEl.querySelector('div').textContent =
                    'Preloading 3D frames... ' + data.loaded + '/' + data.total;
            });

            // Start playing as soon as first 3 frames are ready
            ev.on('sequence.preloadReady', () => {
                if (!started) {
                    started = true;
                    loadingEl.style.display = 'none';
                    console.log('Splatline: First frames ready, starting playback');
                    ev.fire('timeline.setPlaying', true);
                }
            });

            // Safety: if preloadReady doesn't fire within 30s, start anyway
            setTimeout(() => {
                if (!started) {
                    started = true;
                    loadingEl.style.display = 'none';
                    console.warn('Splatline: Preload timeout, starting playback anyway');
                    ev.fire('timeline.setPlaying', true);
                }
            }, 30000);
        } else if (retries > 0) {
            setTimeout(() => tryImport(retries - 1), 500);
        } else {
            console.error('Splatline: Editor events not available after 30 retries');
            loadingEl.style.display = 'none';
        }
    }
    tryImport(30);
})();
</script>
`;

const getPythonPath = () => {
  if (process.env.SPLATLINE_PYTHON && existsSync(process.env.SPLATLINE_PYTHON)) {
    return process.env.SPLATLINE_PYTHON;
  }
  const homebrewPython = "/opt/homebrew/bin/python3.11";
  if (existsSync(homebrewPython)) return homebrewPython;
  return process.platform === "win32" ? "python" : "python3";
};

const prepareViewer = async (outputDir: string, maxFrames: number | null) => {
  const gaussiansDir = path.join(outputDir, "gaussians");
  const frames2dDir = path.join(outputDir, "frames");

  if (!existsSync(gaussiansDir)) {
    throw new Error(`No gaussians directory at ${gaussiansDir}`);
  }

  let plyFiles = readdirSync(gaussiansDir).filter((f) => f.endsWith(".ply")).sort();
  if (plyFiles.length === 0) {
    throw new Error(`No PLY files in ${gaussiansDir}`);
  }
  if (maxFrames && maxFrames > 0) {
    plyFiles = plyFiles.slice(0, maxFrames);
  }

  const has2d = existsSync(frames2dDir) && readdirSync(frames2dDir).some((f) => f.endsWith(".png"));

  const viewerDir = path.join("/tmp", "splatline_electron_video");
  if (existsSync(viewerDir)) {
    rmSync(viewerDir, { recursive: true });
  }
  mkdirSync(viewerDir, { recursive: true });

  // Find the editor dist — check /tmp/supersplat/dist (where the web viewer builds it)
  // and also os.tmpdir() as fallback
  const editorDistCandidates = [
    "/tmp/supersplat/dist",
    path.join(os.tmpdir(), "supersplat", "dist"),
  ];
  const editorDist = editorDistCandidates.find((p) => existsSync(p));
  if (!editorDist) {
    throw new Error("Splatline editor not built. Run the web viewer first to build it.");
  }

  // Copy editor files (except index.html)
  for (const file of readdirSync(editorDist)) {
    if (file === "index.html") continue;
    const src = path.join(editorDist, file);
    const dst = path.join(viewerDir, file);
    if (existsSync(src)) {
      try { copyFileSync(src, dst); } catch { /* skip dirs */ }
    }
  }

  // Subsample PLY frames for smooth real-time playback.
  // Full-res PLYs are 63MB/frame with 1.18M splats — too slow for 8fps swapping.
  // 600K splats (~32MB) gives a good balance: ~2x faster than full-res while
  // preserving visual quality. Uses voxel-grid downsampling to maintain spatial
  // coverage (not just highest-opacity, which creates holes).
  const framesDir = path.join(viewerDir, "frames");
  mkdirSync(framesDir, { recursive: true });

  const MAX_SPLATS = 600000;
  const pythonPath = getPythonPath();
  const subsampleScript = path.join(viewerDir, "_subsample.py");
  writeFileSync(subsampleScript, `
import sys, numpy as np

def subsample_ply(in_path, out_path, max_splats):
    with open(in_path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        # Parse header: only count properties under 'element vertex'
        vertex_count = 0
        props = []
        current_element = None
        for line in header.split(b'\\n'):
            line = line.strip()
            if line.startswith(b'element '):
                current_element = line.split()[1]
                if current_element == b'vertex':
                    vertex_count = int(line.split()[-1])
            elif line.startswith(b'property') and current_element == b'vertex':
                props.append(line)
        # Determine property sizes
        fmt_map = {'float': 4, 'uchar': 1, 'uint': 4, 'ushort': 2, 'int': 4, 'short': 2}
        prop_sizes = []
        for p in props:
            parts = p.split()
            ptype = parts[1].decode()
            prop_sizes.append(fmt_map.get(ptype, 4))
        vertex_size = sum(prop_sizes)
        data = f.read(vertex_count * vertex_size)
        # Read remaining data (extrinsic, intrinsic, etc.)
        extra_data = f.read()

    if vertex_count <= max_splats:
        with open(out_path, 'wb') as f:
            f.write(header)
            f.write(data)
            f.write(extra_data)
        return vertex_count

    # Voxel-grid + opacity hybrid subsampling: preserves spatial coverage
    # (no holes) while keeping the most opaque splats within each voxel.
    prop_names = []
    for p in props:
        parts = p.split()
        prop_names.append(parts[-1].decode() if len(parts) >= 3 else '')

    x_idx = next((i for i, n in enumerate(prop_names) if n in ('x', 'pos_x')), 0)
    y_idx = next((i for i, n in enumerate(prop_names) if n in ('y', 'pos_y')), 1)
    z_idx = next((i for i, n in enumerate(prop_names) if n in ('z', 'pos_z')), 2)
    opacity_idx = next((i for i, n in enumerate(prop_names) if 'opacity' in n), None)

    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, vertex_size)

    x_off = sum(prop_sizes[:x_idx])
    y_off = sum(prop_sizes[:y_idx])
    z_off = sum(prop_sizes[:z_idx])
    xs = arr[:, x_off:x_off+4].copy().view(np.float32).flatten()
    ys = arr[:, y_off:y_off+4].copy().view(np.float32).flatten()
    zs = arr[:, z_off:z_off+4].copy().view(np.float32).flatten()

    if opacity_idx is not None:
        o_off = sum(prop_sizes[:opacity_idx])
        opacities = arr[:, o_off:o_off+4].copy().view(np.float32).flatten()
        opacities = 1.0 / (1.0 + np.exp(-opacities))
    else:
        opacities = np.ones(vertex_count, dtype=np.float32)

    coords = np.column_stack([xs, ys, zs])
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_size[bbox_size == 0] = 1.0

    volume = np.prod(bbox_size)
    voxel_size = (volume / max_splats) ** (1.0 / 3.0)
    voxel_size = max(voxel_size, 1e-6)

    voxel_indices = np.floor((coords - bbox_min) / voxel_size).astype(np.int64)
    voxel_hash = voxel_indices[:, 0] * 73856093 ^ voxel_indices[:, 1] * 19349663 ^ voxel_indices[:, 2] * 83492791
    sort_order = np.lexsort((-opacities, voxel_hash))
    sorted_voxel = voxel_hash[sort_order]

    unique_mask = np.empty(len(sort_order), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = sorted_voxel[1:] != sorted_voxel[:-1]
    selected = sort_order[unique_mask]

    if len(selected) > max_splats:
        selected = selected[np.argsort(-opacities[selected])[:max_splats]]
    elif len(selected) < max_splats:
        remaining = sort_order[~unique_mask]
        needed = max_splats - len(selected)
        remaining_top = remaining[np.argsort(-opacities[remaining])[:needed]]
        selected = np.concatenate([selected, remaining_top])

    indices = np.sort(selected)
    subsampled = arr[indices]

    new_header = header.replace(
        b'element vertex ' + str(vertex_count).encode(),
        b'element vertex ' + str(max_splats).encode()
    )

    with open(out_path, 'wb') as f:
        f.write(new_header)
        f.write(subsampled.tobytes())
        f.write(extra_data)

    return max_splats

in_dir = sys.argv[1]
out_dir = sys.argv[2]
files = sys.argv[3:]
for i, fname in enumerate(files):
    # Write with 4-digit zero-padded name (matches JS expectations)
    out_name = 'frame_' + str(i).zfill(4) + '.ply'
    n = subsample_ply(in_dir + '/' + fname, out_dir + '/' + out_name, ${MAX_SPLATS})
    if (i+1) % 10 == 0:
        print(f'  Subsampled {i+1}/{len(files)} ({n} splats)', flush=True)
print(f'Done: {len(files)} frames subsampled to <= ${MAX_SPLATS} splats', flush=True)
`);

  console.log(`Subsampling ${plyFiles.length} PLY files to ${MAX_SPLATS} splats each...`);
  try {
    const { execSync } = await import("node:child_process");
    execSync(
      `"${pythonPath}" "${subsampleScript}" "${gaussiansDir}" "${framesDir}" ${plyFiles.join(" ")}`,
      { stdio: "inherit", timeout: 120000 }
    );
  } catch (err) {
    console.error("Subsampling failed, copying full-res PLYs instead:", err);
    for (let i = 0; i < plyFiles.length; i++) {
      copyFileSync(
        path.join(gaussiansDir, plyFiles[i]),
        path.join(framesDir, `frame_${String(i).padStart(4, "0")}.ply`)
      );
    }
  }

  // Find and copy the source video for native 24fps playback
  let videoSrc: string | null = null;
  const sourceFps = 24.0;
  const frameSkip = 3;

  let sourceVideoPath: string | null = null;

  // Priority 1: video_source.txt (written by run_video_3d.py with --video)
  const metaPath = path.join(outputDir, "video_source.txt");
  if (existsSync(metaPath)) {
    sourceVideoPath = readFileSync(metaPath, "utf8").trim();
  }

  // Priority 2: VIDEO_PATH from run_video_3d.py (legacy fallback)
  if (!sourceVideoPath) {
    const video3dPath = path.join(outputDir, "..", "run_video_3d.py");
    if (existsSync(video3dPath)) {
      const content = readFileSync(video3dPath, "utf8");
      const match = content.match(/VIDEO_PATH\s*=\s*Path\(["']([^"']+)["']\)/);
      if (match) {
        sourceVideoPath = match[1];
      }
    }
  }

  // Priority 3: most recent video in Downloads (last resort)
  if (!sourceVideoPath) {
    const downloadsDir = path.join(os.homedir(), "Downloads");
    if (existsSync(downloadsDir)) {
      const videos = readdirSync(downloadsDir)
        .filter((f) => f.endsWith(".mp4") || f.endsWith(".mov") || f.endsWith(".avi"))
        .map((f) => ({ name: f, path: path.join(downloadsDir, f), mtime: statSync(path.join(downloadsDir, f)).mtimeMs }))
        .sort((a, b) => b.mtime - a.mtime);
      if (videos.length > 0) {
        sourceVideoPath = videos[0].path;
      }
    }
  }

  if (sourceVideoPath && existsSync(sourceVideoPath)) {
    const videoDst = path.join(viewerDir, "source.mp4");
    try {
      copyFileSync(sourceVideoPath, videoDst);
      videoSrc = "source.mp4";
      console.log(`Copied source video: ${path.basename(sourceVideoPath)}`);
    } catch (e) {
      console.error("Failed to copy source video:", e);
    }
  } else {
    console.warn("No source video found");
  }

  // Patch editor HTML
  let html = readFileSync(path.join(editorDist, "index.html"), "utf8");
  html = html.replace("navigator.serviceWorker", "null && navigator.serviceWorker");
  html = html.replace("<title>SuperSplat</title>", "<title>Splatline Video Player</title>");
  html = html.replace("</body>", SEQUENCE_LOADER_JS + "\n</body>");
  writeFileSync(path.join(viewerDir, "index.html"), html);

  // Write manifest with video metadata for native fps playback
  writeFileSync(
    path.join(viewerDir, "manifest.json"),
    JSON.stringify({
      frames: plyFiles.length,
      fps: sourceFps / frameSkip,
      has2d: !!videoSrc,
      videoSrc,
      sourceFps,
      frameSkip,
      source: outputDir,
    }, null, 2)
  );

  return { viewerDir, frameCount: plyFiles.length };
};

export const startVideoPlayer = async () => {
  const args = process.argv.slice(2);
  // When launched via `SPLATLINE_VIDEO=1 electron .`, args are electron's own
  // Find the output dir from env or default
  const outputDir = process.env.SPLATLINE_OUTPUT_DIR || path.resolve(process.cwd(), "output_grok_3d");
  const maxFrames = process.env.SPLATLINE_MAX_FRAMES ? parseInt(process.env.SPLATLINE_MAX_FRAMES, 10) : null;

  console.log("Splatline Video Player — Electron");
  console.log(`Output dir: ${outputDir}`);
  console.log(`Max frames: ${maxFrames ?? "all"}`);

  const { viewerDir, frameCount } = await prepareViewer(outputDir, maxFrames);
  console.log(`Prepared ${frameCount} frames in ${viewerDir}`);

  const port = 9123;
  await startStaticServer(viewerDir, port);
  console.log(`Static server: http://127.0.0.1:${port}`);

  mainWindow = new BrowserWindow({
    width: 1380,
    height: 900,
    minWidth: 1040,
    minHeight: 720,
    backgroundColor: "#050505",
    title: "Splatline Video Player",
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  // Retry loadURL — the network service can fail on first attempt
  const url = `http://127.0.0.1:${port}/`;
  let loaded = false;
  for (let attempt = 0; attempt < 3 && !loaded; attempt++) {
    try {
      await mainWindow.loadURL(url);
      loaded = true;
    } catch (err) {
      console.error(`Load attempt ${attempt + 1} failed:`, err);
      if (attempt < 2) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }
  if (!loaded) {
    console.error("Failed to load after 3 attempts");
  }
};

export const stopVideoPlayer = () => {
  if (staticServer) {
    staticServer.close();
    staticServer = null;
  }
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
};
