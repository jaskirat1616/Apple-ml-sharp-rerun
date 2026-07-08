"""
Quality and evidence helpers for Splatline Field Movement Lab.

These helpers keep the analysis transparent: every event gets a compact
confidence/coverage summary and the exact frames a coach should inspect.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from .biomechanics import AthleteFrame, COCO17_JOINTS
from .sports_events import SportEvent


CRITICAL_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def quality_label(score: float) -> str:
    """Return a coach-facing quality label from a 0-1 score."""
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "low"
    return "review_only"


def compute_frame_quality(frame: AthleteFrame, min_confidence: float = 0.2) -> Dict[str, object]:
    """Summarize whether a frame is reliable enough for movement review."""
    confidence = np.asarray(frame.confidence, dtype=float)
    visible = confidence >= min_confidence
    critical_ids = [COCO17_JOINTS[name] for name in CRITICAL_JOINTS]
    critical_visible = confidence[critical_ids] >= min_confidence

    mean_confidence = float(np.nanmean(confidence)) if confidence.size else 0.0
    visible_fraction = float(np.mean(visible)) if visible.size else 0.0
    critical_fraction = float(np.mean(critical_visible)) if critical_visible.size else 0.0
    score = float(np.clip((0.45 * mean_confidence) + (0.25 * visible_fraction) + (0.30 * critical_fraction), 0.0, 1.0))

    missing_critical = [
        name
        for name in CRITICAL_JOINTS
        if confidence[COCO17_JOINTS[name]] < min_confidence
    ]
    warnings: List[str] = []
    if visible_fraction < 0.7:
        warnings.append("low_joint_coverage")
    if critical_fraction < 0.75:
        warnings.append("missing_key_lower_body_joints")
    if mean_confidence < 0.35:
        warnings.append("low_pose_confidence")

    return {
        "score": score,
        "label": quality_label(score),
        "mean_confidence": mean_confidence,
        "visible_joint_fraction": visible_fraction,
        "critical_joint_fraction": critical_fraction,
        "missing_critical_joints": missing_critical,
        "warnings": warnings,
    }


def _event_frames(frames: List[AthleteFrame], event: SportEvent) -> List[AthleteFrame]:
    return [frame for frame in frames if event.start_frame <= frame.frame <= event.end_frame]


def enrich_events_with_quality(frames: Iterable[AthleteFrame], events: Iterable[SportEvent]) -> List[SportEvent]:
    """Attach review-frame pointers and quality summaries to events."""
    frame_list = list(frames)
    event_list = list(events)
    by_number = {frame.frame: frame for frame in frame_list}

    for event in event_list:
        review_frames = [
            event.start_frame,
            event.peak_frame,
            event.end_frame,
        ]
        event.review_frames = sorted({frame for frame in review_frames if frame in by_number})

        region = _event_frames(frame_list, event)
        qualities = [compute_frame_quality(frame) for frame in region]
        if not qualities:
            event.quality = {
                "score": 0.0,
                "label": "review_only",
                "warnings": ["event_frames_missing"],
            }
            continue

        scores = np.array([float(item["score"]) for item in qualities], dtype=float)
        warnings = sorted({warning for item in qualities for warning in item.get("warnings", [])})
        event.quality = {
            "score": float(np.mean(scores)),
            "min_frame_score": float(np.min(scores)),
            "label": quality_label(float(np.mean(scores))),
            "frames_evaluated": len(qualities),
            "warnings": warnings,
        }

    return event_list


def build_evidence_summary(
    frames: Iterable[AthleteFrame],
    events: Iterable[SportEvent],
    summary: Dict[str, Optional[float]],
    metadata: Dict[str, object],
) -> Dict[str, object]:
    """Create the compact JSON shape consumed by the desktop app."""
    frame_list = list(frames)
    event_list = list(events)
    frame_quality = [compute_frame_quality(frame) for frame in frame_list]
    scores = np.array([float(item["score"]) for item in frame_quality], dtype=float) if frame_quality else np.array([])

    warnings = sorted({warning for item in frame_quality for warning in item.get("warnings", [])})
    if metadata.get("scale_factor") is None and metadata.get("athlete_height_m") is not None:
        warnings.append("height_scale_not_estimated")

    return {
        "schema": "splatline.movement_evidence.v1",
        "session": {
            "source": metadata.get("source"),
            "fps": metadata.get("fps"),
            "frames_analyzed": int(summary.get("frames_analyzed", len(frame_list)) or 0),
            "athlete_height_m": metadata.get("athlete_height_m"),
            "scale_factor": metadata.get("scale_factor"),
            "pose_model": metadata.get("pose_model"),
            "splat_backend": metadata.get("splat_backend"),
        },
        "quality": {
            "score": float(np.mean(scores)) if scores.size else 0.0,
            "min_frame_score": float(np.min(scores)) if scores.size else 0.0,
            "label": quality_label(float(np.mean(scores))) if scores.size else "review_only",
            "warnings": warnings,
        },
        "key_metrics": {
            "peak_pelvis_speed_mps": summary.get("pelvis_speed_mps_max"),
            "knee_flexion_asymmetry_pct": summary.get("knee_flexion_asymmetry_pct"),
            "max_trunk_lean_deg": summary.get("trunk_lean_deg_max"),
            "change_of_direction_count": summary.get("change_of_direction_count", 0.0),
            "hard_deceleration_count": summary.get("hard_deceleration_count", 0.0),
            "jump_landing_count": summary.get("jump_landing_count", 0.0),
            "high_speed_window_count": summary.get("high_speed_window_count", 0.0),
        },
        "events": [event.to_dict() for event in event_list],
    }
