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
  position: fixed; top: 10px; left: 10px; width: 200px; height: 280px;
  background: rgba(17,17,17,0.9); border: 1px solid #333; border-radius: 6px;
  z-index: 5000; display: flex; align-items: center; justify-content: center;
  overflow: hidden; pointer-events: none;
}
#splatline-2d-panel img { max-width: 100%; max-height: 100%; object-fit: contain; }
#splatline-2d-panel .label {
  position: absolute; top: 4px; left: 8px; color: #888; font-size: 10px;
  font-family: sans-serif; text-shadow: 0 1px 2px #000;
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
  <img id="splatline-2d-img" />
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
    const img2d = document.getElementById('splatline-2d-img');
    const loadingEl = document.getElementById('splatline-loading');
    const loadFill = document.getElementById('splatline-load-fill');
    if (manifest.has2d) panel2d.classList.remove('hidden');

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
            ev.on('timeline.frame', (frame) => {
                if (manifest.has2d) {
                    img2d.src = 'video2d/frame_' + String(frame).padStart(4, '0') + '.jpg';
                }
            });

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

const prepareViewer = (outputDir: string, maxFrames: number | null) => {
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

  // Copy PLY frames
  const framesDir = path.join(viewerDir, "frames");
  mkdirSync(framesDir, { recursive: true });
  for (let i = 0; i < plyFiles.length; i++) {
    copyFileSync(
      path.join(gaussiansDir, plyFiles[i]),
      path.join(framesDir, `frame_${String(i).padStart(4, "0")}.ply`)
    );
  }

  // Convert 2D frames to JPEG
  if (has2d) {
    const video2dDir = path.join(viewerDir, "video2d");
    mkdirSync(video2dDir, { recursive: true });
    pythonProcess = spawn(getPythonPath(), [
      "-c",
      `import cv2, sys, os
src_dir = sys.argv[1]
dst_dir = sys.argv[2]
count = int(sys.argv[3])
for i in range(count):
    src = os.path.join(src_dir, f"frame_{i:06d}.png")
    if not os.path.exists(src):
        continue
    dst = os.path.join(dst_dir, f"frame_{i:04d}.jpg")
    img = cv2.imread(src)
    if img is not None:
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
`,
      frames2dDir, video2dDir, String(plyFiles.length),
    ], { stdio: ["ignore", "pipe", "pipe"] });
  }

  // Patch editor HTML
  let html = readFileSync(path.join(editorDist, "index.html"), "utf8");
  html = html.replace("navigator.serviceWorker", "null && navigator.serviceWorker");
  html = html.replace("<title>SuperSplat</title>", "<title>Splatline Video Player</title>");
  html = html.replace("</body>", SEQUENCE_LOADER_JS + "\n</body>");
  writeFileSync(path.join(viewerDir, "index.html"), html);

  // Write manifest
  writeFileSync(
    path.join(viewerDir, "manifest.json"),
    JSON.stringify({ frames: plyFiles.length, fps: 8.0, has2d, source: outputDir }, null, 2)
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

  const { viewerDir, frameCount } = prepareViewer(outputDir, maxFrames);
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
