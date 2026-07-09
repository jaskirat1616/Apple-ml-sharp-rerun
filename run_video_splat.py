#!/usr/bin/env python3
"""
Splatline Video Splat Viewer — Smooth 3D video playback in the Splatline editor.

Uses the Splatline editor (PlayCanvas engine) with its built-in:
  - High-quality Gaussian splat renderer (same as the editor)
  - PLY sequence timeline with play/pause/loop/scrub
  - In-place splat data swapping for smooth frame transitions
  - Selection, cutting planes, transform tools, export, etc.

Adds:
  - 2D source video side-by-side with 3D
  - Auto-play with loop on load
  - FPS matching to source video
  - Pre-converted PLYs (stripped of extra elements) for fast loading

Usage:
  python run_video_splat.py [output_dir] [max_frames]

  python run_video_splat.py                          # default: output_grok_3d
  python run_video_splat.py output_grok_3d           # all frames
  python run_video_splat.py output_grok_3d 10        # first 10 frames
"""
import sys
import re
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


def find_source_video():
    """Find the source video file."""
    downloads = Path("/Users/jaskiratsingh/Downloads")
    videos = sorted(downloads.glob("grok-video-*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


# JavaScript injected into index.html:
# 1. Loads all PLY frames as a sequence (triggers editor's timeline)
# 2. Sets FPS and enables auto-play with loop
# 3. Adds 2D video panel side-by-side
# 4. Syncs 2D frame with 3D timeline frame
SEQUENCE_LOADER_JS = """
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

    // Build URLs for on-demand fetching — avoids OOM with large sequences
    // (81 frames x 66MB = 5.3GB if loaded as File objects at once)
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

            // Sync 2D video to 3D timeline — let the video play continuously
            // at native 24fps for smooth playback. Only re-sync if it drifts.
            if (manifest.has2d && video2d) {
                video2d.play().catch(() => {});
                ev.on('timeline.frame', (frame) => {
                    if (!video2d) return;
                    const targetT = frame * (manifest.frameSkip || 3) / (manifest.sourceFps || 24);
                    if (Math.abs(video2d.currentTime - targetT) > 0.3) {
                        video2d.currentTime = targetT;
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
"""


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_grok_3d")
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) > 0 else None

    gaussians_dir = output_dir / "gaussians"
    frames_2d_dir = output_dir / "frames"

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

    # Check for 2D frames
    has_2d = frames_2d_dir.exists() and any(frames_2d_dir.glob("*.png"))

    # Get FPS from source video
    import cv2
    source_video = find_source_video()
    fps = 8.0
    if source_video:
        cap = cv2.VideoCapture(str(source_video))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        fps = source_fps / 3  # frame_skip=3

    print("=" * 60)
    print("SPLATLINE VIDEO PLAYER — EDITOR + 2D")
    print("=" * 60)
    print(f"Output dir:  {output_dir}")
    print(f"PLY files:   {len(ply_files)}")
    print(f"FPS:         {fps:.1f}")
    print(f"2D video:    {'yes' if has_2d else 'no'}")
    print(f"Duration:    {len(ply_files)/fps:.1f}s")
    print()

    # Find or build editor
    editor_dist = find_or_build_editor()
    if not editor_dist:
        print("Error: Could not build Splatline editor")
        sys.exit(1)

    # Patch the editor HTML
    index_html = (editor_dist / "index.html").read_text()
    patched = index_html.replace("navigator.serviceWorker", "null && navigator.serviceWorker")
    patched = patched.replace("<title>SuperSplat</title>", "<title>Splatline Video Player</title>")
    # Inject our sequence loader + 2D panel before closing body
    patched = patched.replace("</body>", SEQUENCE_LOADER_JS + "\n</body>")
    (editor_dist / "index.html").write_text(patched)

    # Create frames directory in dist and convert PLYs
    frames_dir = editor_dist / "frames"
    frames_dir.mkdir(exist_ok=True)
    # Clean old frames
    for old in frames_dir.glob("*.ply"):
        old.unlink()

    print(f"Converting {len(ply_files)} PLY files — full resolution...")
    for i, ply_path in enumerate(ply_files):
        std_path = frames_dir / f"frame_{i:04d}.ply"
        convert_to_standard_ply(ply_path, std_path)
        size_mb = std_path.stat().st_size / 1e6
        print(f"  [{i+1}/{len(ply_files)}] {ply_path.name} -> {size_mb:.1f} MB")

    # Copy source video for native 24fps playback
    # Read the correct video path from run_video_3d.py
    video_src = None
    source_fps = 24.0
    frame_skip = 3

    source_video_path = None
    # Try to read VIDEO_PATH from run_video_3d.py
    video_3d_script = output_dir.parent / "run_video_3d.py"
    if video_3d_script.exists():
        content = video_3d_script.read_text()
        match = re.search(r'VIDEO_PATH\s*=\s*Path\(["\']([^"\']+)["\']\)', content)
        if match:
            source_video_path = Path(match.group(1))

    # Fallback: check for a metadata file in the output dir
    if not source_video_path:
        meta_path = output_dir / "video_source.txt"
        if meta_path.exists():
            source_video_path = Path(meta_path.read_text().strip())

    # Fallback: use the most recently modified grok-video in Downloads
    if not source_video_path or not source_video_path.exists():
        downloads_dir = Path("/Users/jaskiratsingh/Downloads")
        if downloads_dir.exists():
            grok_videos = sorted(downloads_dir.glob("grok-video-*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if grok_videos:
                source_video_path = grok_videos[0]

    if source_video_path and source_video_path.exists():
        video_dst = editor_dist / "source.mp4"
        print(f"Copying source video: {source_video_path.name}")
        shutil.copy2(source_video_path, video_dst)
        video_src = "source.mp4"
    else:
        print("Warning: No source video found")

    # Write manifest
    manifest = {
        "frames": len(ply_files),
        "fps": round(source_fps / frame_skip, 1),
        "has2d": bool(video_src),
        "videoSrc": video_src,
        "sourceFps": source_fps,
        "frameSkip": frame_skip,
        "source": str(output_dir),
    }
    (editor_dist / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Kill any existing serve process on port 3000
    subprocess.run("lsof -ti:3000 | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)

    # Start the serve dev server (proper MIME types for ES modules)
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
    print("The editor loads all frames as a PLY sequence with:")
    print("  - Built-in timeline (play/pause/scrub at bottom)")
    print("  - Auto-play with loop enabled")
    print("  - 2D source video panel on the right")
    print("  - FPS matched to source video")
    print()
    print("Editor features (all work during playback):")
    print("  Selection, cutting planes, splat deletion")
    print("  Transform tools, export, multi-splat")
    print()
    print("Controls:")
    print("  Space:     Play/Pause")
    print("  ←/→:       Prev/Next frame")
    print("  L:         Toggle loop (in editor settings)")
    print("  Left drag:  Orbit camera")
    print("  Right drag: Pan camera")
    print("  Scroll:    Zoom")
    print("  F:         Frame scene")
    print()
    print("Opening player... (Ctrl+C to stop)")

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
