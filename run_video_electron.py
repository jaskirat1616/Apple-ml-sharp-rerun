#!/usr/bin/env python3
"""
Splatline Video Player — Electron desktop launcher.

Launches the Splatline editor in an Electron desktop window with:
  - High-quality Gaussian splat renderer (PlayCanvas engine)
  - PLY sequence timeline with play/pause/loop/scrub
  - 2D source video overlay (bottom-left)
  - Auto-play with loop on load
  - FPS matching to source video
  - Frame caching + preloading for smooth playback

Usage:
  python run_video_electron.py [--output-dir DIR] [--max-frames N]

  python run_video_electron.py                                    # default output dir
  python run_video_electron.py --output-dir output_grok_3d        # specify dir
  python run_video_electron.py --output-dir output_grok_3d --max-frames 10
"""
import argparse
import sys
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Splatline Electron Video Player")
    parser.add_argument("--output-dir", "-o", type=str, default="output_grok_3d",
                        help="Output directory with gaussians/ subdirectory (default: output_grok_3d)")
    parser.add_argument("--max-frames", "-n", type=int, default=None,
                        help="Maximum number of frames to load")
    args = parser.parse_args()

    output_path = Path(args.output_dir).resolve()
    if not (output_path / "gaussians").exists():
        print(f"Error: No gaussians directory at {output_path / 'gaussians'}")
        print("Run run_video_3d.py first to generate PLY files.")
        sys.exit(1)

    # Check editor is built
    editor_dist = Path("/tmp/supersplat/dist")
    if not (editor_dist / "index.html").exists():
        print("Splatline editor not built. Building now...")
        repo = Path("/tmp/supersplat")
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1",
                           "https://github.com/playcanvas/supersplat.git", str(repo)],
                           capture_output=True, timeout=120)
        subprocess.run(["npm", "install"], cwd=str(repo), capture_output=True, timeout=120)

        # Patch main.ts to expose events
        main_ts = repo / "src" / "main.ts"
        if main_ts.exists():
            src = main_ts.read_text()
            if "__splatlineEvents" not in src:
                src = src.replace(
                    "const events = new Events();",
                    "const events = new Events();\n    (window as any).__splatlineEvents = events;"
                )
                main_ts.write_text(src)

        subprocess.run(["npm", "run", "build"], cwd=str(repo), capture_output=True, timeout=120)

    # Get FPS from video_source.txt or source video
    import cv2
    source_video = None
    video_source_file = output_path / "video_source.txt"
    if video_source_file.exists():
        source_video = Path(video_source_file.read_text().strip())
    if source_video and source_video.exists():
        cap = cv2.VideoCapture(str(source_video))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        fps = source_fps / 3
    else:
        fps = 8.0

    ply_count = len(list((output_path / "gaussians").glob("*.ply")))

    print("=" * 60)
    print("SPLATLINE VIDEO PLAYER — ELECTRON DESKTOP")
    print("=" * 60)
    print(f"Output dir:  {output_path}")
    print(f"PLY files:   {ply_count}")
    print(f"FPS:         {fps:.1f}")
    print(f"Duration:    {ply_count/fps:.1f}s")
    print()

    # Set env vars for Electron
    env = os.environ.copy()
    env["SPLATLINE_VIDEO"] = "1"
    env["SPLATLINE_OUTPUT_DIR"] = str(output_path)
    if args.max_frames:
        env["SPLATLINE_MAX_FRAMES"] = str(args.max_frames)

    # Compile electron TS and launch
    project_root = Path(__file__).parent.resolve()
    print("Compiling Electron...")
    result = subprocess.run(["npx", "tsc", "-p", "tsconfig.electron.json"],
                           cwd=str(project_root), capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Compile error: {result.stderr}")
        sys.exit(1)

    print("Launching Electron desktop player...")
    proc = subprocess.Popen(["npx", "electron", "."], cwd=str(project_root), env=env)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
