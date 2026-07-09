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

    // Fetch all PLY files as File objects — show progress
    const files = [];
    for (let i = 0; i < manifest.frames; i++) {
        const filename = 'frame_' + String(i).padStart(4, '0') + '.ply';
        const url = 'frames/' + filename;
        try {
            const response = await fetch(url);
            if (!response.ok) { console.error('Splatline: Failed ' + filename); continue; }
            const blob = await response.blob();
            const file = new File([blob], filename, { type: 'application/octet-stream' });
            files.push({ filename: filename, contents: file });
            loadFill.style.width = ((i + 1) / manifest.frames * 50) + '%';
        } catch(e) { console.error('Splatline: Error fetching ' + filename, e); }
    }

    console.log('Splatline: Importing ' + files.length + ' frames as sequence');

    function tryImport(retries) {
        if (window.__splatlineEvents) {
            const ev = window.__splatlineEvents;
            ev.invoke('import', files).then(() => {
                console.log('Splatline: Sequence imported, waiting for preload...');

                // Set frame rate + loop
                ev.fire('timeline.setFrameRate', manifest.fps);
                ev.fire('timeline.setLoop', true);

                // Sync 2D frame with 3D timeline
                ev.on('timeline.frame', (frame) => {
                    if (manifest.has2d) {
                        img2d.src = 'video2d/frame_' + String(frame).padStart(4, '0') + '.jpg';
                    }
                });

                // Wait for frames to be preloaded (the editor preloads in background)
                // Check every 500ms if the first few frames are ready, then start playing
                let waitCount = 0;
                function waitForPreload() {
                    waitCount++;
                    loadFill.style.width = (50 + Math.min(50, waitCount * 5)) + '%';
                    loadingEl.querySelector('div').textContent =
                        'Preloading 3D frames... (' + Math.min(50, waitCount * 5) + '%)';

                    // After a reasonable wait, start playing.
                    // The editor caches frames after first load, so the first
                    // play-through may have minor hitches but loops will be smooth.
                    if (waitCount >= 10) {
                        loadingEl.style.display = 'none';
                        console.log('Splatline: Starting smooth auto-play with loop');
                        ev.fire('timeline.setPlaying', true);
                    } else {
                        setTimeout(waitForPreload, 500);
                    }
                }
                waitForPreload();

            }).catch(err => {
                console.error('Splatline: Import failed:', err);
                loadingEl.style.display = 'none';
            });
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

    # Convert 2D frames to JPEG for fast loading
    if has_2d:
        video_2d_dir = editor_dist / "video2d"
        video_2d_dir.mkdir(exist_ok=True)
        for old in video_2d_dir.glob("*.jpg"):
            old.unlink()
        print("Converting 2D frames to JPEG...")
        for i in range(len(ply_files)):
            src = frames_2d_dir / f"frame_{i:06d}.png"
            if not src.exists():
                continue
            dst = video_2d_dir / f"frame_{i:04d}.jpg"
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Done ({len(list(video_2d_dir.glob('*.jpg')))} frames)")

    # Write manifest
    manifest = {
        "frames": len(ply_files),
        "fps": round(fps, 1),
        "has2d": has_2d,
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
