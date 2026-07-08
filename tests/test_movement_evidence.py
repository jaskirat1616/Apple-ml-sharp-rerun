import unittest

import numpy as np

from utils.biomechanics import AthleteFrame
from utils.movement_evidence import build_evidence_summary, compute_frame_quality, enrich_events_with_quality
from utils.sports_events import SportEvent


def make_frame(frame=0, confidence=1.0):
    return AthleteFrame(
        frame=frame,
        time_s=frame / 30.0,
        person_id=0,
        joints_3d=np.zeros((17, 3), dtype=np.float32),
        keypoints_2d=np.zeros((17, 4), dtype=np.float32),
        confidence=np.full(17, confidence, dtype=np.float32),
        metrics={"pelvis_speed_mps": 2.0},
        flags=[],
    )


class MovementEvidenceTests(unittest.TestCase):
    def test_frame_quality_labels_high_confidence_frames(self):
        quality = compute_frame_quality(make_frame(confidence=0.9))
        self.assertEqual(quality["label"], "high")
        self.assertGreater(quality["score"], 0.75)

    def test_frame_quality_warns_for_low_confidence(self):
        quality = compute_frame_quality(make_frame(confidence=0.1))
        self.assertEqual(quality["label"], "review_only")
        self.assertIn("low_pose_confidence", quality["warnings"])

    def test_enrich_events_adds_review_frames_and_quality(self):
        frames = [make_frame(10), make_frame(11), make_frame(12)]
        event = SportEvent(
            event_type="hard_deceleration",
            start_frame=10,
            end_frame=12,
            start_time_s=0.33,
            end_time_s=0.4,
            peak_frame=11,
            metrics={},
            notes=[],
        )
        enriched = enrich_events_with_quality(frames, [event])[0]
        self.assertEqual(enriched.review_frames, [10, 11, 12])
        self.assertEqual(enriched.quality["label"], "high")

    def test_build_evidence_summary_shape(self):
        frames = [make_frame(0), make_frame(1)]
        event = SportEvent("high_speed_window", 0, 1, 0.0, 0.03, 1, {}, [])
        events = enrich_events_with_quality(frames, [event])
        payload = build_evidence_summary(
            frames,
            events,
            {"frames_analyzed": 2.0, "high_speed_window_count": 1.0},
            {"source": "clip.mp4", "fps": 30.0, "pose_model": "test", "splat_backend": "sharp"},
        )
        self.assertEqual(payload["schema"], "splatline.movement_evidence.v1")
        self.assertEqual(payload["session"]["frames_analyzed"], 2)
        self.assertEqual(len(payload["events"]), 1)


if __name__ == "__main__":
    unittest.main()
