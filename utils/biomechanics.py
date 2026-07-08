"""
Biomechanics utilities for markerless sport analysis.

The functions in this module operate on COCO-17 keypoints in camera space:
X points right, Y points down, and Z points forward.  Derived metrics are
therefore useful for coaching review and longitudinal comparison, but they are
not a replacement for validated clinical measurement without calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


COCO17_JOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

JOINT_NAMES = {idx: name for name, idx in COCO17_JOINTS.items()}
NUM_COCO17_JOINTS = 17

LEFT_SIDE = ("left_shoulder", "left_hip", "left_knee", "left_ankle")
RIGHT_SIDE = ("right_shoulder", "right_hip", "right_knee", "right_ankle")


@dataclass
class AthleteFrame:
    """Frame-level athlete analysis result."""

    frame: int
    time_s: float
    person_id: int
    joints_3d: np.ndarray
    keypoints_2d: np.ndarray
    confidence: np.ndarray
    metrics: Dict[str, Optional[float]]
    flags: List[str]


def angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Return the angle between vectors in degrees."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def joint_angle(
    joints: np.ndarray,
    a: int,
    b: int,
    c: int,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.15,
) -> Optional[float]:
    """Return the included angle ABC in degrees."""
    if confidence is not None:
        if min(confidence[a], confidence[b], confidence[c]) < min_confidence:
            return None
    return angle_between(joints[a] - joints[b], joints[c] - joints[b])


def flexion_from_angle(angle_deg: Optional[float]) -> Optional[float]:
    """Convert an included limb angle to a flexion-style value."""
    if angle_deg is None:
        return None
    return float(max(0.0, 180.0 - angle_deg))


def midpoint(
    joints: np.ndarray,
    joint_a: int,
    joint_b: int,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.15,
) -> Optional[np.ndarray]:
    """Return the midpoint between two joints when both are visible enough."""
    if confidence is not None:
        if min(confidence[joint_a], confidence[joint_b]) < min_confidence:
            return None
    return (joints[joint_a] + joints[joint_b]) * 0.5


def pelvis_center(joints: np.ndarray, confidence: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Return the hip midpoint."""
    return midpoint(
        joints,
        COCO17_JOINTS["left_hip"],
        COCO17_JOINTS["right_hip"],
        confidence,
    )


def shoulder_center(joints: np.ndarray, confidence: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Return the shoulder midpoint."""
    return midpoint(
        joints,
        COCO17_JOINTS["left_shoulder"],
        COCO17_JOINTS["right_shoulder"],
        confidence,
    )


def estimate_body_height(joints: np.ndarray, confidence: np.ndarray, min_confidence: float = 0.2) -> Optional[float]:
    """
    Estimate body height from nose-to-ankle distance.

    COCO-17 lacks top-of-head and toe landmarks, so this underestimates true
    height.  It is still useful as a stable scale anchor across a session.
    """
    nose = COCO17_JOINTS["nose"]
    ankle_ids = [COCO17_JOINTS["left_ankle"], COCO17_JOINTS["right_ankle"]]
    visible_ankles = [idx for idx in ankle_ids if confidence[idx] >= min_confidence]
    if confidence[nose] < min_confidence or not visible_ankles:
        return None
    ankle_mid = np.mean(joints[visible_ankles], axis=0)
    height = float(np.linalg.norm(joints[nose] - ankle_mid))
    return height if height > 1e-6 else None


def rescale_sequence_to_height(
    frames: Iterable[AthleteFrame],
    target_height_m: float,
) -> Tuple[List[AthleteFrame], Optional[float]]:
    """Scale all frame skeletons by a session-wide height factor.

    A single session anchor is used so global pelvis displacement scales along
    with limb lengths.  That keeps speed and distance metrics in the same units.
    """
    frame_list = list(frames)
    heights = [
        estimate_body_height(frame.joints_3d, frame.confidence)
        for frame in frame_list
    ]
    valid_heights = [h for h in heights if h is not None]
    if not valid_heights:
        return frame_list, None

    observed = float(np.median(valid_heights))
    if observed <= 1e-6:
        return frame_list, None
    scale = float(target_height_m / observed)

    anchors = [
        pelvis_center(frame.joints_3d, frame.confidence)
        for frame in frame_list
    ]
    valid_anchors = [anchor for anchor in anchors if anchor is not None]
    session_anchor = (
        np.median(np.vstack(valid_anchors), axis=0)
        if valid_anchors
        else np.zeros(3, dtype=np.float32)
    )

    scaled = []
    for frame in frame_list:
        joints = (frame.joints_3d - session_anchor) * scale + session_anchor
        scaled.append(
            AthleteFrame(
                frame=frame.frame,
                time_s=frame.time_s,
                person_id=frame.person_id,
                joints_3d=joints,
                keypoints_2d=frame.keypoints_2d,
                confidence=frame.confidence,
                metrics=dict(frame.metrics),
                flags=list(frame.flags),
            )
        )
    return scaled, scale


def depth_lift_coco_keypoints(
    keypoints_2d: np.ndarray,
    gaussian_positions: np.ndarray,
    f_px: float,
    img_w: int,
    img_h: int,
    k: int = 20,
    min_depth: float = 0.05,
) -> np.ndarray:
    """
    Lift normalized COCO-17 keypoints into camera-space 3D using Gaussian depth.

    keypoints_2d must be shaped (17, 4) with normalized x/y and confidence in
    column 3.  The result keeps the camera-space origin instead of torso-centering
    so global athlete translation can be measured.
    """
    if keypoints_2d.shape[0] != NUM_COCO17_JOINTS:
        raise ValueError(f"Expected {NUM_COCO17_JOINTS} keypoints, got {keypoints_2d.shape[0]}")

    cx, cy = img_w / 2.0, img_h / 2.0
    kp_px = keypoints_2d[:, :2] * np.array([img_w, img_h], dtype=np.float32)
    joints = np.zeros((NUM_COCO17_JOINTS, 3), dtype=np.float32)

    valid = gaussian_positions[:, 2] > min_depth if len(gaussian_positions) else np.array([], dtype=bool)
    pts = gaussian_positions[valid] if len(gaussian_positions) else np.empty((0, 3), dtype=np.float32)

    if len(pts) >= 5 and f_px > 0:
        g_u = pts[:, 0] / pts[:, 2] * f_px + cx
        g_v = pts[:, 1] / pts[:, 2] * f_px + cy
        g_d = pts[:, 2]
        nearest_count = min(max(3, k), len(g_d))

        for idx, (px, py) in enumerate(kp_px):
            dist2 = (g_u - px) ** 2 + (g_v - py) ** 2
            nearest = np.argpartition(dist2, nearest_count - 1)[:nearest_count]
            depth = float(np.median(g_d[nearest]))
            joints[idx] = [
                (px - cx) / f_px * depth,
                (py - cy) / f_px * depth,
                depth,
            ]
        return joints

    # Fallback: project onto a flat plane using a neutral athlete-scale depth.
    fallback_depth = 2.0
    focal = max(float(f_px), float(img_w) * 0.85)
    for idx, (px, py) in enumerate(kp_px):
        joints[idx] = [
            (px - cx) / focal * fallback_depth,
            (py - cy) / focal * fallback_depth,
            fallback_depth,
        ]
    return joints


def _point_line_distance_2d(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> Optional[float]:
    """Signed 2D distance from point to line AB in the X/Y image-frontal plane."""
    a = line_a[:2]
    b = line_b[:2]
    p = point[:2]
    ab = b - a
    denom = float(np.linalg.norm(ab))
    if denom < 1e-8:
        return None
    cross = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
    return float(cross / denom)


def knee_valgus_proxy(
    joints: np.ndarray,
    side: str,
    confidence: Optional[np.ndarray] = None,
    min_confidence: float = 0.15,
) -> Optional[float]:
    """
    Return frontal-plane knee deviation normalized by hip-ankle distance.

    This is a video-derived proxy, not a diagnostic valgus measurement.
    Higher absolute values indicate the knee is farther from the hip-ankle line.
    """
    hip = COCO17_JOINTS[f"{side}_hip"]
    knee = COCO17_JOINTS[f"{side}_knee"]
    ankle = COCO17_JOINTS[f"{side}_ankle"]
    if confidence is not None:
        if min(confidence[hip], confidence[knee], confidence[ankle]) < min_confidence:
            return None

    distance = _point_line_distance_2d(joints[knee], joints[hip], joints[ankle])
    if distance is None:
        return None
    leg_len = float(np.linalg.norm(joints[ankle] - joints[hip]))
    if leg_len < 1e-8:
        return None
    return float(distance / leg_len)


def compute_frame_metrics(
    joints: np.ndarray,
    confidence: np.ndarray,
    time_s: float,
    previous: Optional[AthleteFrame] = None,
) -> Dict[str, Optional[float]]:
    """Compute sport-science metrics for one frame."""
    ids = COCO17_JOINTS

    left_knee_angle = joint_angle(joints, ids["left_hip"], ids["left_knee"], ids["left_ankle"], confidence)
    right_knee_angle = joint_angle(joints, ids["right_hip"], ids["right_knee"], ids["right_ankle"], confidence)
    left_hip_angle = joint_angle(joints, ids["left_shoulder"], ids["left_hip"], ids["left_knee"], confidence)
    right_hip_angle = joint_angle(joints, ids["right_shoulder"], ids["right_hip"], ids["right_knee"], confidence)
    left_elbow_angle = joint_angle(joints, ids["left_shoulder"], ids["left_elbow"], ids["left_wrist"], confidence)
    right_elbow_angle = joint_angle(joints, ids["right_shoulder"], ids["right_elbow"], ids["right_wrist"], confidence)

    pelvis = pelvis_center(joints, confidence)
    shoulders = shoulder_center(joints, confidence)

    trunk_lean = None
    if pelvis is not None and shoulders is not None:
        trunk = shoulders - pelvis
        trunk_lean = angle_between(trunk, np.array([0.0, -1.0, 0.0]))

    left_ankle = joints[ids["left_ankle"]]
    right_ankle = joints[ids["right_ankle"]]
    foot_y_values = []
    if confidence[ids["left_ankle"]] >= 0.15:
        foot_y_values.append(float(left_ankle[1]))
    if confidence[ids["right_ankle"]] >= 0.15:
        foot_y_values.append(float(right_ankle[1]))
    foot_y = float(max(foot_y_values)) if foot_y_values else None

    metrics: Dict[str, Optional[float]] = {
        "left_knee_angle_deg": left_knee_angle,
        "right_knee_angle_deg": right_knee_angle,
        "left_knee_flexion_deg": flexion_from_angle(left_knee_angle),
        "right_knee_flexion_deg": flexion_from_angle(right_knee_angle),
        "left_hip_flexion_proxy_deg": flexion_from_angle(left_hip_angle),
        "right_hip_flexion_proxy_deg": flexion_from_angle(right_hip_angle),
        "left_elbow_flexion_deg": flexion_from_angle(left_elbow_angle),
        "right_elbow_flexion_deg": flexion_from_angle(right_elbow_angle),
        "trunk_lean_deg": trunk_lean,
        "left_knee_valgus_proxy": knee_valgus_proxy(joints, "left", confidence),
        "right_knee_valgus_proxy": knee_valgus_proxy(joints, "right", confidence),
        "pelvis_x": float(pelvis[0]) if pelvis is not None else None,
        "pelvis_y": float(pelvis[1]) if pelvis is not None else None,
        "pelvis_z": float(pelvis[2]) if pelvis is not None else None,
        "foot_y": foot_y,
        "mean_confidence": float(np.mean(confidence)),
    }

    if previous is not None:
        dt = max(time_s - previous.time_s, 1e-6)
        prev_pelvis = pelvis_center(previous.joints_3d, previous.confidence)
        if pelvis is not None and prev_pelvis is not None:
            delta = pelvis - prev_pelvis
            horizontal_speed = float(np.linalg.norm(delta[[0, 2]]) / dt)
            vertical_velocity = float((-delta[1]) / dt)
            metrics["pelvis_speed_mps"] = horizontal_speed
            metrics["vertical_velocity_mps"] = vertical_velocity
        else:
            metrics["pelvis_speed_mps"] = None
            metrics["vertical_velocity_mps"] = None
    else:
        metrics["pelvis_speed_mps"] = None
        metrics["vertical_velocity_mps"] = None

    return metrics


def generate_frame_flags(metrics: Dict[str, Optional[float]]) -> List[str]:
    """Generate coach-facing technique flags from frame metrics."""
    flags: List[str] = []

    left_flex = metrics.get("left_knee_flexion_deg")
    right_flex = metrics.get("right_knee_flexion_deg")
    flex_values = [v for v in (left_flex, right_flex) if v is not None]
    if flex_values and min(flex_values) < 18:
        flags.append("stiff_knee_position")

    left_valgus = metrics.get("left_knee_valgus_proxy")
    right_valgus = metrics.get("right_knee_valgus_proxy")
    if any(v is not None and abs(v) > 0.16 for v in (left_valgus, right_valgus)):
        flags.append("knee_alignment_deviation")

    trunk_lean = metrics.get("trunk_lean_deg")
    if trunk_lean is not None and trunk_lean > 28:
        flags.append("large_trunk_lean")

    speed = metrics.get("pelvis_speed_mps")
    if speed is not None and speed > 7.0:
        flags.append("high_speed_frame")

    return flags


def recompute_sequence_metrics(frames: Iterable[AthleteFrame]) -> List[AthleteFrame]:
    """Recompute velocity-dependent metrics after optional sequence scaling."""
    out: List[AthleteFrame] = []
    previous: Optional[AthleteFrame] = None
    for frame in frames:
        metrics = compute_frame_metrics(frame.joints_3d, frame.confidence, frame.time_s, previous)
        flags = generate_frame_flags(metrics)
        new_frame = AthleteFrame(
            frame=frame.frame,
            time_s=frame.time_s,
            person_id=frame.person_id,
            joints_3d=frame.joints_3d,
            keypoints_2d=frame.keypoints_2d,
            confidence=frame.confidence,
            metrics=metrics,
            flags=flags,
        )
        out.append(new_frame)
        previous = new_frame
    return out


def summarize_metrics(frames: Iterable[AthleteFrame]) -> Dict[str, Optional[float]]:
    """Aggregate high-signal summary metrics across a sequence."""
    frame_list = list(frames)
    summary: Dict[str, Optional[float]] = {"frames_analyzed": float(len(frame_list))}
    metric_keys = sorted({key for frame in frame_list for key in frame.metrics})

    for key in metric_keys:
        values = [frame.metrics.get(key) for frame in frame_list]
        valid = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        if len(valid) == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(valid))
        summary[f"{key}_max"] = float(np.max(valid))
        summary[f"{key}_min"] = float(np.min(valid))

    left = summary.get("left_knee_flexion_deg_mean")
    right = summary.get("right_knee_flexion_deg_mean")
    if left is not None and right is not None:
        denom = max(abs(left), abs(right), 1e-6)
        summary["knee_flexion_asymmetry_pct"] = float(abs(left - right) / denom * 100.0)

    flag_counts: Dict[str, int] = {}
    for frame in frame_list:
        for flag in frame.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    for flag, count in sorted(flag_counts.items()):
        summary[f"flag_{flag}_count"] = float(count)

    return summary
