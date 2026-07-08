"""
Export helpers for Splatline Athlete Twin analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .biomechanics import AthleteFrame, JOINT_NAMES
from .movement_evidence import compute_frame_quality
from .sports_events import SportEvent


def _clean_number(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def frame_to_dict(frame: AthleteFrame) -> Dict[str, object]:
    """Convert an AthleteFrame to JSON-serializable data."""
    joints = []
    for idx, point in enumerate(frame.joints_3d):
        joints.append(
            {
                "name": JOINT_NAMES.get(idx, str(idx)),
                "x": _clean_number(point[0]),
                "y": _clean_number(point[1]),
                "z": _clean_number(point[2]),
                "confidence": _clean_number(frame.confidence[idx]),
            }
        )

    keypoints_2d = []
    for idx, point in enumerate(frame.keypoints_2d):
        keypoints_2d.append(
            {
                "name": JOINT_NAMES.get(idx, str(idx)),
                "x_norm": _clean_number(point[0]),
                "y_norm": _clean_number(point[1]),
                "confidence": _clean_number(frame.confidence[idx]),
            }
        )

    return {
        "frame": frame.frame,
        "time_s": _clean_number(frame.time_s),
        "person_id": frame.person_id,
        "joints_3d": joints,
        "keypoints_2d": keypoints_2d,
        "metrics": {key: _clean_number(value) for key, value in frame.metrics.items()},
        "flags": frame.flags,
        "quality": compute_frame_quality(frame),
    }


def write_athlete_json(
    output_path: Path,
    frames: Iterable[AthleteFrame],
    events: Iterable[SportEvent],
    summary: Dict[str, Optional[float]],
    metadata: Dict[str, object],
) -> None:
    """Write full analysis data for downstream systems."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "splatline.athlete_twin.v1",
        "metadata": metadata,
        "summary": {key: _clean_number(value) for key, value in summary.items()},
        "events": [event.to_dict() for event in events],
        "frames": [frame_to_dict(frame) for frame in frames],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metrics_csv(output_path: Path, frames: Iterable[AthleteFrame]) -> None:
    """Write one row per analyzed frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_list = list(frames)
    metric_keys = sorted({key for frame in frame_list for key in frame.metrics})
    fieldnames = ["frame", "time_s", "person_id", "flags"] + metric_keys

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frame_list:
            row = {
                "frame": frame.frame,
                "time_s": f"{frame.time_s:.6f}",
                "person_id": frame.person_id,
                "flags": "|".join(frame.flags),
            }
            for key in metric_keys:
                value = frame.metrics.get(key)
                row[key] = "" if value is None else f"{float(value):.6f}"
            writer.writerow(row)


def write_events_csv(output_path: Path, events: Iterable[SportEvent]) -> None:
    """Write detected movement events."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_list = list(events)
    metric_keys = sorted({key for event in event_list for key in event.metrics})
    fieldnames = [
        "event_type",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "peak_frame",
        "notes",
    ] + metric_keys

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in event_list:
            row = {
                "event_type": event.event_type,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "start_time_s": f"{event.start_time_s:.6f}",
                "end_time_s": f"{event.end_time_s:.6f}",
                "peak_frame": event.peak_frame,
                "notes": "|".join(event.notes),
            }
            for key in metric_keys:
                value = event.metrics.get(key)
                row[key] = "" if value is None else f"{float(value):.6f}"
            writer.writerow(row)


def write_evidence_summary(output_path: Path, evidence: Dict[str, object]) -> None:
    """Write compact movement evidence for the desktop app and integrations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def _fmt(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def write_markdown_report(
    output_path: Path,
    frames: List[AthleteFrame],
    events: List[SportEvent],
    summary: Dict[str, Optional[float]],
    metadata: Dict[str, object],
) -> None:
    """Write a concise coach/scientist review report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_flags = []
    for key, value in sorted(summary.items()):
        if key.startswith("flag_") and key.endswith("_count") and value:
            flag = key[len("flag_") : -len("_count")]
            top_flags.append((flag, int(value)))

    lines = [
        "# Splatline Athlete Twin Report",
        "",
        "## Session",
        f"- Source: `{metadata.get('source', 'unknown')}`",
        f"- Frames analyzed: {int(summary.get('frames_analyzed', 0) or 0)}",
        f"- FPS: {_fmt(metadata.get('fps'), digits=2)}",
        f"- Scale factor: {_fmt(metadata.get('scale_factor'), digits=4)}",
        "",
        "## Key Metrics",
        f"- Evidence quality: {metadata.get('evidence_quality_label', 'n/a')}",
        f"- Peak pelvis speed: {_fmt(summary.get('pelvis_speed_mps_max'), ' m/s')}",
        f"- Mean left knee flexion: {_fmt(summary.get('left_knee_flexion_deg_mean'), ' deg')}",
        f"- Mean right knee flexion: {_fmt(summary.get('right_knee_flexion_deg_mean'), ' deg')}",
        f"- Knee flexion asymmetry: {_fmt(summary.get('knee_flexion_asymmetry_pct'), '%')}",
        f"- Max trunk lean: {_fmt(summary.get('trunk_lean_deg_max'), ' deg')}",
        "",
        "## Detected Events",
    ]

    if events:
        for event in events[:20]:
            notes = ", ".join(event.notes) if event.notes else "none"
            lines.append(
                f"- `{event.event_type}` frames {event.start_frame}-{event.end_frame} "
                f"(peak {event.peak_frame}, {event.start_time_s:.2f}s-{event.end_time_s:.2f}s, "
                f"quality {event.quality.get('label', 'n/a')}), notes: {notes}"
            )
    else:
        lines.append("- No movement events detected with the current thresholds.")

    lines.extend(["", "## Technique Flags"])
    if top_flags:
        for flag, count in top_flags:
            lines.append(f"- `{flag}`: {count} frame(s)")
    else:
        lines.append("- No frame-level technique flags crossed default thresholds.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- Metrics are generated from markerless video and Gaussian-depth lifting.",
            "- Use this report to target review frames and compare sessions, not as a standalone medical diagnosis.",
            "- Better calibration, stable camera placement, and full-body visibility improve measurement quality.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
