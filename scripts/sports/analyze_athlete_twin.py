#!/usr/bin/env python3
"""
Splatline Athlete Twin
======================

Turn ordinary sport video into a scene-aware biomechanical replay:
3D athlete skeleton, 3D scene context, frame metrics, movement events, and
coach/scientist exports.

Examples
--------
Analyze a new video end to end:
  python scripts/sports/analyze_athlete_twin.py training.mp4 --device mps --view

Analyze an existing Splatline conversion:
  python scripts/sports/analyze_athlete_twin.py \
      --frames-dir output_training/frames \
      --gaussians-dir output_training/gaussians \
      --fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.converters.video_to_3d_with_pose import (  # noqa: E402
    MAX_POSES,
    POSE_CONNECTIONS,
    _PERSON_BONE_COLORS,
    _enhance_frame_for_detection,
    _get_f_px,
    _smooth_joints,
    build_bone_segments,
    detect_poses,
    draw_pose_overlay,
    extract_frames,
    run_sharp_conversion,
    setup_pose_detector,
)
from utils.athlete_report import (  # noqa: E402
    write_evidence_summary,
    write_athlete_json,
    write_events_csv,
    write_markdown_report,
    write_metrics_csv,
)
from utils.biomechanics import (  # noqa: E402
    AthleteFrame,
    depth_lift_coco_keypoints,
    recompute_sequence_metrics,
    rescale_sequence_to_height,
    summarize_metrics,
)
from utils.frame_processing import load_gaussian_data  # noqa: E402
from utils.movement_evidence import build_evidence_summary, enrich_events_with_quality  # noqa: E402
from utils.sports_events import detect_sport_events, summarize_events  # noqa: E402


def _athlete_frame_from_dict(payload: dict) -> AthleteFrame:
    """Rehydrate an AthleteFrame from athlete_twin.json."""
    joints = np.zeros((17, 3), dtype=np.float32)
    confidence = np.zeros(17, dtype=np.float32)
    keypoints = np.zeros((17, 4), dtype=np.float32)

    for idx, joint in enumerate(payload.get("joints_3d", [])[:17]):
        joints[idx] = [
            float(joint.get("x") or 0.0),
            float(joint.get("y") or 0.0),
            float(joint.get("z") or 0.0),
        ]
        confidence[idx] = float(joint.get("confidence") or 0.0)

    for idx, point in enumerate(payload.get("keypoints_2d", [])[:17]):
        keypoints[idx] = [
            float(point.get("x_norm") or 0.0),
            float(point.get("y_norm") or 0.0),
            0.0,
            float(point.get("confidence") or confidence[idx]),
        ]

    if not np.any(confidence):
        confidence = keypoints[:, 3].astype(np.float32)

    return AthleteFrame(
        frame=int(payload.get("frame") or 0),
        time_s=float(payload.get("time_s") or 0.0),
        person_id=int(payload.get("person_id") or 0),
        joints_3d=joints,
        keypoints_2d=keypoints,
        confidence=confidence,
        metrics=payload.get("metrics", {}),
        flags=payload.get("flags", []),
    )


def _load_frames_from_json(path: Path) -> List[AthleteFrame]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_athlete_frame_from_dict(frame) for frame in data.get("frames", [])]


def _select_primary_pose(all_poses: list, person_index: Optional[int] = None) -> Optional[Tuple[int, np.ndarray]]:
    """Pick one athlete from YOLO pose detections."""
    if not all_poses:
        return None
    if person_index is not None:
        if 0 <= person_index < len(all_poses):
            return person_index, all_poses[person_index][0]
        return None

    best_idx = None
    best_score = -1.0
    for idx, (lm2d, _) in enumerate(all_poses):
        visible = lm2d[:, 3] > 0.15
        if visible.sum() < 6:
            continue
        xs = lm2d[visible, 0]
        ys = lm2d[visible, 1]
        area = float(max(xs.max() - xs.min(), 1e-6) * max(ys.max() - ys.min(), 1e-6))
        score = area * float(np.mean(lm2d[visible, 3]))
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None
    return best_idx, all_poses[best_idx][0]


def _match_frame_files(frames_dir: Path, gaussians_dir: Path, max_frames: Optional[int], skip_frames: int):
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    ply_files = sorted(gaussians_dir.glob("*.ply"))
    n = min(len(frame_files), len(ply_files))
    pairs = list(zip(frame_files[:n], ply_files[:n]))
    if skip_frames > 1:
        pairs = pairs[::skip_frames]
    if max_frames is not None:
        pairs = pairs[:max_frames]
    return pairs


def _frame_number(path: Path, fallback: int) -> int:
    parts = path.stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return fallback


def _analyze_pairs(
    pairs,
    fps: float,
    person_index: Optional[int],
    min_confidence: float,
    pose_model: str,
    pose_imgsz: int,
    pose_conf: float,
    pose_iou: float,
) -> List[AthleteFrame]:
    print("Setting up high-accuracy pose detector...")
    try:
        model = setup_pose_detector(pose_model)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pose model '{pose_model}'. "
            "Install/upgrade with `pip install -U ultralytics>=8.4.0` "
            "or pass --pose-model yolov8x-pose-p6.pt for legacy fallback."
        ) from exc
    smooth_state = None
    previous = None
    frames: List[AthleteFrame] = []

    for idx, (frame_path, ply_path) in enumerate(pairs):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            print(f"  [{idx + 1}/{len(pairs)}] skipped {frame_path.name}: image load failed")
            continue

        data = load_gaussian_data(ply_path, opacity_threshold=0.05)
        if data is None:
            print(f"  [{idx + 1}/{len(pairs)}] skipped {ply_path.name}: Gaussian load failed")
            continue

        img_h, img_w = frame_bgr.shape[:2]
        f_px = _get_f_px(data["metadata"], img_w)
        enhanced = _enhance_frame_for_detection(frame_bgr)
        all_poses = detect_poses(
            enhanced,
            model,
            img_w,
            img_h,
            conf=pose_conf,
            iou=pose_iou,
            imgsz=pose_imgsz,
        )
        selection = _select_primary_pose(all_poses, person_index)
        if selection is None:
            print(f"  [{idx + 1}/{len(pairs)}] {frame_path.name}: no athlete detected")
            continue

        detected_person_id, keypoints_2d = selection
        keypoints_2d = _smooth_joints(keypoints_2d, smooth_state)
        smooth_state = keypoints_2d

        if float(np.mean(keypoints_2d[:, 3])) < min_confidence:
            print(f"  [{idx + 1}/{len(pairs)}] {frame_path.name}: low confidence")
            continue

        joints_3d = depth_lift_coco_keypoints(
            keypoints_2d,
            data["positions"],
            f_px=f_px,
            img_w=img_w,
            img_h=img_h,
        )
        frame_number = _frame_number(frame_path, idx)
        time_s = frame_number / max(fps, 1e-6)

        placeholder = AthleteFrame(
            frame=frame_number,
            time_s=time_s,
            person_id=detected_person_id,
            joints_3d=joints_3d,
            keypoints_2d=keypoints_2d,
            confidence=keypoints_2d[:, 3].astype(np.float32),
            metrics={},
            flags=[],
        )
        current = recompute_sequence_metrics([previous, placeholder] if previous else [placeholder])[-1]
        frames.append(current)
        previous = current

        flag_text = ",".join(current.flags) if current.flags else "clean"
        print(f"  [{idx + 1}/{len(pairs)}] {frame_path.name}: athlete {detected_person_id}, {flag_text}")

    return frames


def _log_rerun_dashboard(
    frames: List[AthleteFrame],
    pairs,
    fps: float,
    size: float = 1.0,
) -> None:
    """Open a Rerun review dashboard with scene, skeleton, video, and metrics."""
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("Splatline Athlete Twin", spawn=True)
    try:
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial3DView(name="3D Scene + Athlete Twin", origin="/world"),
                        rrb.Spatial2DView(name="Video + Pose", origin="/camera/frame"),
                        column_shares=[2, 1],
                    ),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(name="Knee Flexion", origin="/metrics/knee_flexion"),
                        rrb.TimeSeriesView(name="Speed + Trunk", origin="/metrics/movement"),
                        column_shares=[1, 1],
                    ),
                    row_shares=[3, 1],
                ),
                collapse_panels=False,
            )
        )
    except Exception:
        pass

    frame_by_number = {frame.frame: frame for frame in frames}
    for idx, (frame_path, ply_path) in enumerate(pairs):
        frame_number = _frame_number(frame_path, idx)
        athlete_frame = frame_by_number.get(frame_number)
        if athlete_frame is None:
            continue

        rr.set_time_sequence("frame", athlete_frame.frame)
        rr.set_time_seconds("time", athlete_frame.time_s)

        data = load_gaussian_data(ply_path, opacity_threshold=0.08)
        if data is not None:
            rr.log(
                "world/scene",
                rr.Points3D(
                    positions=data["positions"],
                    colors=data["colors"],
                    radii=np.mean(data["scales"], axis=1) * 0.35 * size,
                ),
            )

        visible = athlete_frame.confidence > 0.15
        colors = np.array([[0, 220, 200] for _ in range(visible.sum())], dtype=np.uint8)
        rr.log(
            "world/athlete/joints",
            rr.Points3D(
                positions=athlete_frame.joints_3d[visible],
                colors=colors,
                radii=0.035 * size,
            ),
        )
        segments = build_bone_segments(athlete_frame.joints_3d, athlete_frame.keypoints_2d)
        rr.log(
            "world/athlete/skeleton",
            rr.LineStrips3D(
                strips=[[segment[0], segment[1]] for segment in segments],
                colors=_PERSON_BONE_COLORS[0],
                radii=0.01 * size,
            ),
        )

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is not None:
            overlay = draw_pose_overlay(frame_bgr, [(athlete_frame.keypoints_2d, None)])
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            rr.log("camera/frame", rr.Image(overlay_rgb))

        metrics = athlete_frame.metrics
        for key in ("left_knee_flexion_deg", "right_knee_flexion_deg"):
            value = metrics.get(key)
            if value is not None:
                rr.log(f"metrics/knee_flexion/{key}", rr.Scalar(value))
        for key in ("pelvis_speed_mps", "trunk_lean_deg", "vertical_velocity_mps"):
            value = metrics.get(key)
            if value is not None:
                rr.log(f"metrics/movement/{key}", rr.Scalar(value))

        for pose_idx in range(1, MAX_POSES):
            rr.log(f"world/athlete_extra_{pose_idx}/joints", rr.Points3D(positions=np.zeros((0, 3))))

    print("Rerun Athlete Twin dashboard is ready. Press Ctrl+C to close.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return


def _prepare_inputs(args) -> Tuple[Path, Path, Path, float, str]:
    """Return output_dir, frames_dir, gaussians_dir, fps, source label."""
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"output_{video_path.stem}_athlete_twin")
        fps, _ = extract_frames(video_path, output_dir, frame_skip=args.extract_skip)
        gaussians_dir = output_dir / "gaussians"
        if not args.skip_sharp:
            _run_splat_conversion(
                output_dir / "frames",
                output_dir,
                backend=args.splat_backend,
                device=args.device,
                model_url=args.sharp_model_url,
                checkpoint_path=args.sharp_checkpoint,
                internal_size=args.sharp_internal_size,
                video_path=video_path,
            )
        elif not gaussians_dir.exists():
            raise FileNotFoundError("--skip-sharp was set, but no gaussians directory exists")
        return output_dir, output_dir / "frames", gaussians_dir, fps, str(video_path)

    if not args.frames_dir or not args.gaussians_dir:
        raise ValueError("Provide either a video path or both --frames-dir and --gaussians-dir")

    frames_dir = Path(args.frames_dir)
    gaussians_dir = Path(args.gaussians_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not gaussians_dir.exists():
        raise FileNotFoundError(f"Gaussians directory not found: {gaussians_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else gaussians_dir.parent / "athlete_twin"
    return output_dir, frames_dir, gaussians_dir, args.fps, str(frames_dir)


def _run_splat_conversion(
    frames_dir: Path,
    output_dir: Path,
    backend: str,
    device: str,
    model_url,
    checkpoint_path,
    internal_size: int,
    video_path: Optional[Path] = None,
) -> None:
    """Run splat conversion with the selected backend."""
    if backend == "sharp":
        run_sharp_conversion(
            frames_dir,
            output_dir,
            device=device,
            model_url=model_url,
            checkpoint_path=checkpoint_path,
            internal_size=internal_size,
        )
    else:
        from utils.splat_models import convert_frames_to_splats
        convert_frames_to_splats(
            frames_dir=frames_dir,
            output_dir=output_dir,
            backend=backend,
            device=device,
            skip_existing=True,
            video_path=video_path,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze sport video as a scene-aware 3D Athlete Twin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", nargs="?", help="Input video to extract, convert, and analyze")
    parser.add_argument("--frames-dir", type=Path, help="Existing extracted frames directory")
    parser.add_argument("--gaussians-dir", type=Path, help="Existing Splatline Gaussian PLY directory")
    parser.add_argument("--output-dir", type=Path, help="Directory for Athlete Twin exports")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS when analyzing existing frames")
    parser.add_argument("--device", default="default", help="Compute device: default, mps, cuda, or cpu")
    parser.add_argument("--extract-skip", type=int, default=1, help="Extract every Nth source-video frame")
    parser.add_argument("--analysis-skip", type=int, default=1, help="Analyze every Nth extracted frame")
    parser.add_argument("--max-frames", type=int, help="Limit frames for quick analysis")
    parser.add_argument("--skip-sharp", action="store_true", help="Skip splat conversion for a video input")
    # v2: reconstruction backend selection
    parser.add_argument(
        "--splat-backend",
        default="sharp",
        choices=["sharp", "triposplat", "vggt", "depthsplat", "longsplat"],
        help="3D reconstruction backend: sharp (per-frame, non-commercial), triposplat (per-frame, MIT), "
             "vggt (geometry bootstrap, feed-forward), depthsplat (multi-view, MIT), longsplat (video-native coherent)",
    )
    # v2: human pipeline tier
    parser.add_argument(
        "--human-tier",
        default="skeleton",
        choices=["skeleton", "mesh", "both"],
        help="Human reconstruction tier: skeleton (MotionBERT, fast) or mesh (HMR 2.0 SMPL, detailed)",
    )
    # Legacy SHARP-specific args (kept for backwards compat, used when --splat-backend=sharp)
    parser.add_argument("--sharp-model-url", default=None, help="SHARP checkpoint URL")
    parser.add_argument("--sharp-checkpoint", type=Path, help="Local SHARP checkpoint path")
    parser.add_argument("--sharp-internal-size", type=int, default=1536, help="Square internal SHARP inference size")
    parser.add_argument("--person-index", type=int, help="Analyze a fixed detection index instead of largest athlete")
    parser.add_argument("--min-confidence", type=float, default=0.12, help="Minimum mean pose confidence")
    parser.add_argument("--pose-model", default="yolo26x-pose.pt", help="Ultralytics pose model or local .pt path")
    parser.add_argument("--pose-imgsz", type=int, default=960, help="Pose inference image size")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="YOLO pose detection confidence threshold")
    parser.add_argument("--pose-iou", type=float, default=0.45, help="YOLO pose NMS IoU threshold")
    parser.add_argument("--athlete-height-m", type=float, default=1.75, help="Height used to scale skeleton metrics")
    parser.add_argument("--no-height-scale", action="store_true", help="Leave scene units unscaled")
    parser.add_argument("--view", action="store_true", help="Open Rerun Athlete Twin dashboard after exports")
    parser.add_argument("--view-only", action="store_true", help="Open Rerun from existing exports without processing")
    parser.add_argument("--analysis-json", type=Path, help="Existing athlete_twin.json for --view-only")
    parser.add_argument("--size", type=float, default=1.0, help="Rerun skeleton/point size multiplier")
    return parser


def _apply_motionbert_lifting(frames: List[AthleteFrame], pairs, fps: float, args) -> List[AthleteFrame]:
    """Apply MotionBERT temporal 3D lifting with depth fusion.

    This replaces per-frame depth lifting with a dual-stream spatio-temporal
    transformer that looks at the whole motion sequence at once, producing
    smooth temporally consistent 3D motion. Fused with Gaussian splat depth
    for metric scale.
    """
    print("\n--- MotionBERT temporal 3D lifting ---")
    try:
        from utils.human.motionbert import MotionBERTLifter
    except ImportError as exc:
        print(f"  MotionBERT not available ({exc}), keeping depth-lifted joints")
        return frames

    if len(frames) < 2:
        print("  Not enough frames for temporal lifting, keeping depth-lifted joints")
        return frames

    # Collect 2D keypoints and depth maps
    keypoints_seq = [f.keypoints_2d for f in frames]
    depths = []
    for frame_path, ply_path in pairs:
        data = load_gaussian_data(ply_path, opacity_threshold=0.05)
        if data is not None:
            depths.append(data["positions"])
        else:
            depths.append(None)

    # Get image dimensions from first frame
    import cv2
    first_frame = cv2.imread(str(pairs[0][0]))
    img_h, img_w = first_frame.shape[:2] if first_frame is not None else (720, 1280)
    f_px = _get_f_px(load_gaussian_data(pairs[0][1])["metadata"], img_w) if depths and depths[0] is not None else 1000.0

    try:
        lifter = MotionBERTLifter(device=args.device)
        joints_3d_sequence = lifter.lift(
            keypoints_seq,
            depths=depths,
            f_px=f_px,
            img_w=img_w,
            img_h=img_h,
        )

        # Update frames with MotionBERT-lifted joints
        for idx, frame in enumerate(frames):
            if idx < len(joints_3d_sequence):
                frame.joints_3d = joints_3d_sequence[idx]

        print(f"  MotionBERT: lifted {len(frames)} frames with temporal coherence")
    except Exception as exc:
        print(f"  MotionBERT lifting failed ({exc}), keeping depth-lifted joints")

    return frames


def _apply_smpl_mesh_recovery(pairs, args) -> Optional[list]:
    """Apply HMR 2.0 SMPL mesh recovery to the video frames.

    Produces a full SMPL body mesh per frame — a textured 3D human body in
    the scene, not just a skeleton. Multi-person with identity tracking.
    """
    print("\n--- HMR 2.0 SMPL mesh recovery ---")
    try:
        from utils.human.hmr2 import recover_smpl_mesh
    except ImportError as exc:
        print(f"  HMR 2.0 not available ({exc}), skipping mesh recovery")
        return None

    frame_paths = [p[0] for p in pairs]
    try:
        results = recover_smpl_mesh(frame_paths, device=args.device)
        print(f"  HMR 2.0: recovered SMPL mesh for {len(results)} frames")
        return results
    except Exception as exc:
        print(f"  HMR 2.0 mesh recovery failed ({exc})")
        return None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.view_only:
            if not args.analysis_json:
                parser.error("--analysis-json is required with --view-only")
            if not args.frames_dir or not args.gaussians_dir:
                parser.error("--frames-dir and --gaussians-dir are required with --view-only")
            frames = _load_frames_from_json(args.analysis_json)
            pairs = _match_frame_files(
                Path(args.frames_dir),
                Path(args.gaussians_dir),
                args.max_frames,
                max(args.analysis_skip, 1),
            )
            if not frames:
                print(f"No frames found in {args.analysis_json}")
                return 1
            if not pairs:
                print("No frame/gaussian pairs found.")
                return 1
            _log_rerun_dashboard(frames, pairs, args.fps, size=args.size)
            return 0

        output_dir, frames_dir, gaussians_dir, fps, source = _prepare_inputs(args)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = _match_frame_files(frames_dir, gaussians_dir, args.max_frames, max(args.analysis_skip, 1))
        if not pairs:
            print("No frame/gaussian pairs found.")
            return 1

        print("=" * 80)
        print("SPLATLINE ATHLETE TWIN v2")
        print("=" * 80)
        print(f"Source: {source}")
        print(f"Backend: {args.splat_backend}")
        print(f"Human tier: {args.human_tier}")
        print(f"Frames: {frames_dir}")
        print(f"Gaussians: {gaussians_dir}")
        print(f"Output: {output_dir}")
        print(f"Analyzing {len(pairs)} frame pair(s) at {fps:.2f} FPS")

        frames = _analyze_pairs(
            pairs,
            fps=fps,
            person_index=args.person_index,
            min_confidence=args.min_confidence,
            pose_model=args.pose_model,
            pose_imgsz=args.pose_imgsz,
            pose_conf=args.pose_conf,
            pose_iou=args.pose_iou,
        )
        if not frames:
            print("No athlete frames were analyzed.")
            return 1

        # v2: Temporal 3D lifting with MotionBERT (if skeleton or both tier)
        if args.human_tier in ("skeleton", "both"):
            frames = _apply_motionbert_lifting(frames, pairs, fps, args)

        # v2: SMPL mesh recovery (if mesh or both tier)
        smpl_results = None
        if args.human_tier in ("mesh", "both"):
            smpl_results = _apply_smpl_mesh_recovery(pairs, args)

        scale_factor = None
        if not args.no_height_scale and args.athlete_height_m > 0:
            frames, scale_factor = rescale_sequence_to_height(frames, args.athlete_height_m)
            frames = recompute_sequence_metrics(frames)
        else:
            frames = recompute_sequence_metrics(frames)

        events = enrich_events_with_quality(frames, detect_sport_events(frames))
        summary = summarize_metrics(frames)
        summary.update(summarize_events(events))

        metadata = {
            "source": source,
            "frames_dir": str(frames_dir),
            "gaussians_dir": str(gaussians_dir),
            "fps": fps,
            "athlete_height_m": None if args.no_height_scale else args.athlete_height_m,
            "scale_factor": scale_factor,
            "analysis_skip": args.analysis_skip,
            "pose_model": args.pose_model,
            "pose_imgsz": args.pose_imgsz,
            "pose_conf": args.pose_conf,
            "pose_iou": args.pose_iou,
            "splat_backend": args.splat_backend,
            "human_tier": args.human_tier,
            "sharp_model_url": args.sharp_model_url,
            "sharp_checkpoint": str(args.sharp_checkpoint) if args.sharp_checkpoint else None,
            "sharp_internal_size": args.sharp_internal_size,
            "note": "Markerless video-derived metrics for coaching review; not clinical diagnosis.",
        }
        evidence = build_evidence_summary(frames, events, summary, metadata)
        metadata["evidence_quality_label"] = evidence.get("quality", {}).get("label")

        json_path = output_dir / "athlete_twin.json"
        metrics_path = output_dir / "metrics.csv"
        events_path = output_dir / "events.csv"
        report_path = output_dir / "report.md"
        evidence_path = output_dir / "evidence_summary.json"

        write_athlete_json(json_path, frames, events, summary, metadata)
        write_metrics_csv(metrics_path, frames)
        write_events_csv(events_path, events)
        write_evidence_summary(evidence_path, evidence)
        write_markdown_report(report_path, frames, events, summary, metadata)

        print("\nExports written:")
        print(f"  JSON:    {json_path}")
        print(f"  Metrics: {metrics_path}")
        print(f"  Events:  {events_path}")
        print(f"  Evidence:{evidence_path}")
        print(f"  Report:  {report_path}")
        print(f"\nDetected {len(events)} event(s).")

        if args.view:
            _log_rerun_dashboard(frames, pairs, fps, size=args.size)

        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
