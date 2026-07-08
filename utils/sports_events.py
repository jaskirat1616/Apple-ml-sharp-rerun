"""
Event detection for sport-science movement review.

The detectors are intentionally transparent and threshold-based.  They create
review targets for coaches and analysts; they do not claim medical diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .biomechanics import AthleteFrame


@dataclass
class SportEvent:
    """Detected movement event."""

    event_type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    peak_frame: int
    metrics: Dict[str, Optional[float]]
    notes: List[str]
    quality: Dict[str, object] = field(default_factory=dict)
    review_frames: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "event_type": self.event_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "peak_frame": self.peak_frame,
            "metrics": self.metrics,
            "notes": self.notes,
        }
        if self.quality:
            payload["quality"] = self.quality
        if self.review_frames:
            payload["review_frames"] = self.review_frames
        return payload


def _valid_series(frames: List[AthleteFrame], key: str) -> np.ndarray:
    values = []
    for frame in frames:
        value = frame.metrics.get(key)
        values.append(np.nan if value is None else float(value))
    return np.array(values, dtype=float)


def _contiguous_regions(mask: np.ndarray, min_len: int = 1) -> List[tuple]:
    regions = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_len:
                regions.append((start, idx - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        regions.append((start, len(mask) - 1))
    return regions


def estimate_ground_y(frames: Iterable[AthleteFrame]) -> Optional[float]:
    """
    Estimate ground in camera Y coordinates from visible ankle positions.

    Camera-space Y grows downward, so the 90th percentile approximates a low
    point near the floor without being dominated by single-frame noise.
    """
    foot_values = [
        frame.metrics.get("foot_y")
        for frame in frames
        if frame.metrics.get("foot_y") is not None
    ]
    if not foot_values:
        return None
    return float(np.percentile(np.array(foot_values, dtype=float), 90))


def detect_jump_landings(
    frames: List[AthleteFrame],
    ground_y: Optional[float] = None,
    airborne_tolerance_m: float = 0.08,
    min_airborne_frames: int = 2,
) -> List[SportEvent]:
    """Detect airborne periods and landing frames from ankle height."""
    if len(frames) < 3:
        return []

    if ground_y is None:
        ground_y = estimate_ground_y(frames)
    if ground_y is None:
        return []

    foot_y = _valid_series(frames, "foot_y")
    vertical_velocity = _valid_series(frames, "vertical_velocity_mps")
    pelvis_y = _valid_series(frames, "pelvis_y")

    airborne = np.isfinite(foot_y) & (foot_y < ground_y - airborne_tolerance_m)
    events: List[SportEvent] = []

    for start_idx, end_idx in _contiguous_regions(airborne, min_airborne_frames):
        landing_idx = min(end_idx + 1, len(frames) - 1)
        takeoff_idx = max(start_idx - 1, 0)

        region = slice(takeoff_idx, landing_idx + 1)
        height_series = -pelvis_y[region]
        valid_height = height_series[np.isfinite(height_series)]
        jump_height_proxy = None
        if len(valid_height):
            jump_height_proxy = float(np.max(valid_height) - np.min(valid_height))

        landing_velocity = None
        if np.isfinite(vertical_velocity[landing_idx]):
            landing_velocity = float(vertical_velocity[landing_idx])

        notes = []
        landing_frame = frames[landing_idx]
        if "stiff_knee_position" in landing_frame.flags:
            notes.append("landing_review_stiff_knee")
        if "knee_alignment_deviation" in landing_frame.flags:
            notes.append("landing_review_knee_alignment")
        if landing_velocity is not None and landing_velocity < -1.5:
            notes.append("high_downward_velocity")

        events.append(
            SportEvent(
                event_type="jump_landing",
                start_frame=frames[takeoff_idx].frame,
                end_frame=frames[landing_idx].frame,
                start_time_s=frames[takeoff_idx].time_s,
                end_time_s=frames[landing_idx].time_s,
                peak_frame=frames[start_idx + int((end_idx - start_idx) / 2)].frame,
                metrics={
                    "airborne_frames": float(end_idx - start_idx + 1),
                    "jump_height_proxy_m": jump_height_proxy,
                    "landing_vertical_velocity_mps": landing_velocity,
                    "landing_left_knee_flexion_deg": landing_frame.metrics.get("left_knee_flexion_deg"),
                    "landing_right_knee_flexion_deg": landing_frame.metrics.get("right_knee_flexion_deg"),
                },
                notes=notes,
            )
        )

    return events


def detect_cuts_and_decelerations(
    frames: List[AthleteFrame],
    min_speed_mps: float = 1.2,
    cut_angle_deg: float = 35.0,
    decel_mps2: float = -2.5,
) -> List[SportEvent]:
    """Detect sharp heading changes and hard deceleration from pelvis path."""
    if len(frames) < 4:
        return []

    positions = []
    times = []
    for frame in frames:
        x = frame.metrics.get("pelvis_x")
        z = frame.metrics.get("pelvis_z")
        if x is None or z is None:
            positions.append([np.nan, np.nan])
        else:
            positions.append([float(x), float(z)])
        times.append(frame.time_s)

    positions_np = np.array(positions, dtype=float)
    times_np = np.array(times, dtype=float)

    events: List[SportEvent] = []
    previous_speed = None

    for idx in range(2, len(frames) - 1):
        p0, p1, p2 = positions_np[idx - 1], positions_np[idx], positions_np[idx + 1]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue

        dt_prev = max(times_np[idx] - times_np[idx - 1], 1e-6)
        dt_next = max(times_np[idx + 1] - times_np[idx], 1e-6)
        v_prev = (p1 - p0) / dt_prev
        v_next = (p2 - p1) / dt_next
        speed_prev = float(np.linalg.norm(v_prev))
        speed_next = float(np.linalg.norm(v_next))

        notes = []
        event_type = None
        peak_metric = None

        denom = np.linalg.norm(v_prev) * np.linalg.norm(v_next)
        if denom > 1e-8 and max(speed_prev, speed_next) >= min_speed_mps:
            angle = float(np.degrees(np.arccos(np.clip(np.dot(v_prev, v_next) / denom, -1.0, 1.0))))
            if angle >= cut_angle_deg:
                event_type = "change_of_direction"
                peak_metric = angle
                notes.append("review_plant_step_and_trunk_control")

        if previous_speed is not None:
            dt = max(times_np[idx] - times_np[idx - 1], 1e-6)
            acceleration = (speed_prev - previous_speed) / dt
            if acceleration <= decel_mps2:
                if event_type is None:
                    event_type = "hard_deceleration"
                    peak_metric = acceleration
                notes.append("review_braking_strategy")
        previous_speed = speed_prev

        if event_type is None:
            continue

        frame = frames[idx]
        if "knee_alignment_deviation" in frame.flags:
            notes.append("knee_alignment_flag_at_event")
        if "large_trunk_lean" in frame.flags:
            notes.append("trunk_lean_flag_at_event")

        events.append(
            SportEvent(
                event_type=event_type,
                start_frame=frames[idx - 1].frame,
                end_frame=frames[idx + 1].frame,
                start_time_s=frames[idx - 1].time_s,
                end_time_s=frames[idx + 1].time_s,
                peak_frame=frame.frame,
                metrics={
                    "entry_speed_mps": speed_prev,
                    "exit_speed_mps": speed_next,
                    "peak_metric": peak_metric,
                    "left_knee_flexion_deg": frame.metrics.get("left_knee_flexion_deg"),
                    "right_knee_flexion_deg": frame.metrics.get("right_knee_flexion_deg"),
                    "trunk_lean_deg": frame.metrics.get("trunk_lean_deg"),
                },
                notes=sorted(set(notes)),
            )
        )

    return events


def detect_sprint_windows(
    frames: List[AthleteFrame],
    percentile: float = 85.0,
    min_frames: int = 3,
) -> List[SportEvent]:
    """Detect high-speed windows relative to the analyzed clip."""
    if len(frames) < min_frames:
        return []

    speeds = _valid_series(frames, "pelvis_speed_mps")
    valid = speeds[np.isfinite(speeds)]
    if len(valid) < min_frames:
        return []

    threshold = max(float(np.percentile(valid, percentile)), 1.0)
    mask = np.isfinite(speeds) & (speeds >= threshold)
    events: List[SportEvent] = []

    for start_idx, end_idx in _contiguous_regions(mask, min_frames):
        region = speeds[start_idx : end_idx + 1]
        peak_local = int(np.nanargmax(region))
        peak_idx = start_idx + peak_local
        events.append(
            SportEvent(
                event_type="high_speed_window",
                start_frame=frames[start_idx].frame,
                end_frame=frames[end_idx].frame,
                start_time_s=frames[start_idx].time_s,
                end_time_s=frames[end_idx].time_s,
                peak_frame=frames[peak_idx].frame,
                metrics={
                    "threshold_mps": threshold,
                    "peak_speed_mps": float(speeds[peak_idx]),
                    "duration_s": float(frames[end_idx].time_s - frames[start_idx].time_s),
                },
                notes=["relative_to_clip_speed_peak"],
            )
        )

    return events


def detect_sport_events(frames: List[AthleteFrame]) -> List[SportEvent]:
    """Run all built-in movement event detectors."""
    events: List[SportEvent] = []
    events.extend(detect_jump_landings(frames))
    events.extend(detect_cuts_and_decelerations(frames))
    events.extend(detect_sprint_windows(frames))
    return sorted(events, key=lambda event: (event.start_frame, event.event_type))


def summarize_events(events: Iterable[SportEvent]) -> Dict[str, float]:
    """Count events by type for reporting."""
    summary: Dict[str, float] = {}
    for event in events:
        key = f"{event.event_type}_count"
        summary[key] = summary.get(key, 0.0) + 1.0
    return summary
