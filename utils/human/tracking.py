"""
3D-aware multi-person tracking using PHALP.

PHALP (Predicting Human Appearance, Location and Pose in 3D) tracks people
across video frames by predicting 3D appearance, location, and pose, then
associating detections through occlusion events.

Paper: "Tracking People by Predicting 3D Appearance, Location & Pose"
(CVPR 2022, brjathu/PHALP, MIT license, pip install phalp)

For Splatline, this gives consistent athlete IDs across the video — critical
for longitudinal analysis where the same person must be tracked through
occlusions (other players, equipment, frame exits/re-entries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PersonTrack:
    """A tracked person across video frames."""

    person_id: int
    frame_indices: List[int] = field(default_factory=list)
    joints_3d: List[np.ndarray] = field(default_factory=list)  # (17, 3) per frame
    keypoints_2d: List[np.ndarray] = field(default_factory=list)  # (17, 4) per frame
    confidence: List[float] = field(default_factory=list)
    smpl_params: List[Optional[Dict]] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def get_frame(self, frame_idx: int) -> Optional[int]:
        """Get the track's position index for a given video frame, or None."""
        try:
            return self.frame_indices.index(frame_idx)
        except ValueError:
            return None


def track_persons_3d(
    frame_paths: List[Path],
    device: str = "default",
    output_dir: Optional[Path] = None,
) -> Dict[int, PersonTrack]:
    """Track all persons across video frames using PHALP.

    Args:
        frame_paths: List of frame image paths.
        device: torch device.
        output_dir: Optional directory to save tracking results.

    Returns:
        Dict mapping person_id → PersonTrack with 3D joints per frame.
    """
    try:
        import torch
        from phalp.track import PHALP
        from phalp.config import get_cfg
    except ImportError as exc:
        raise ImportError(
            "PHALP is not installed. Install with:\n"
            "  pip install phalp\n"
            "  Or: git clone https://github.com/brjathu/PHALP && pip install -e PHALP\n"
            f"Error: {exc}"
        ) from exc

    from utils.splat_models import resolve_torch_device
    device_obj, device_name = resolve_torch_device(device)
    print(f"  PHALP tracker loaded on {device_name}")

    # Configure PHALP
    cfg = get_cfg()
    cfg.DEVICE = str(device_obj)

    tracker = PHALP(cfg)

    # Run tracking on the video frames
    print(f"  Tracking {len(frame_paths)} frames with PHALP...")
    results = tracker.track_frames(frame_paths)

    # Organize results into PersonTrack objects
    tracks: Dict[int, PersonTrack] = {}

    for frame_idx, frame_result in enumerate(results):
        if frame_result is None:
            continue

        for detection in frame_result:
            pid = int(detection["id"])
            if pid not in tracks:
                tracks[pid] = PersonTrack(person_id=pid)

            track = tracks[pid]
            track.frame_indices.append(frame_idx)
            track.joints_3d.append(detection.get("joints_3d", np.zeros((17, 3))))
            track.keypoints_2d.append(detection.get("keypoints_2d", np.zeros((17, 4))))
            track.confidence.append(float(detection.get("confidence", 0.0)))
            track.smpl_params.append(detection.get("smpl_params"))

    print(f"  PHALP: {len(tracks)} person(s) tracked")
    return tracks


def select_primary_athlete(tracks: Dict[int, PersonTrack]) -> Optional[int]:
    """Select the primary athlete from tracked persons.

    Picks the person with the most tracked frames and highest average
    confidence — typically the main subject in a sports video.
    """
    if not tracks:
        return None

    best_id = None
    best_score = -1.0

    for pid, track in tracks.items():
        if track.num_frames < 3:
            continue
        avg_conf = np.mean(track.confidence) if track.confidence else 0.0
        score = track.num_frames * avg_conf
        if score > best_score:
            best_score = score
            best_id = pid

    return best_id
