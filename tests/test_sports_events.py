import unittest

import numpy as np

from utils.biomechanics import AthleteFrame
from utils.sports_events import detect_cuts_and_decelerations, detect_jump_landings, estimate_ground_y


def make_frame(frame, time_s, pelvis_x, pelvis_z, foot_y, pelvis_y=0.0, speed=None):
    joints = np.zeros((17, 3), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)
    metrics = {
        "pelvis_x": pelvis_x,
        "pelvis_y": pelvis_y,
        "pelvis_z": pelvis_z,
        "foot_y": foot_y,
        "vertical_velocity_mps": -2.0 if foot_y >= 1.0 else 1.0,
        "pelvis_speed_mps": speed,
        "left_knee_flexion_deg": 30.0,
        "right_knee_flexion_deg": 32.0,
        "trunk_lean_deg": 12.0,
    }
    return AthleteFrame(
        frame=frame,
        time_s=time_s,
        person_id=0,
        joints_3d=joints,
        keypoints_2d=np.zeros((17, 4), dtype=np.float32),
        confidence=conf,
        metrics=metrics,
        flags=[],
    )


class SportsEventTests(unittest.TestCase):
    def test_estimate_ground_y(self):
        frames = [
            make_frame(0, 0.0, 0.0, 0.0, 1.0),
            make_frame(1, 0.1, 0.0, 0.0, 0.8),
            make_frame(2, 0.2, 0.0, 0.0, 1.1),
        ]
        self.assertAlmostEqual(estimate_ground_y(frames), 1.08, places=2)

    def test_detect_jump_landing(self):
        frames = [
            make_frame(0, 0.0, 0.0, 0.0, 1.0, pelvis_y=0.0),
            make_frame(1, 0.1, 0.0, 0.0, 0.80, pelvis_y=-0.2),
            make_frame(2, 0.2, 0.0, 0.0, 0.78, pelvis_y=-0.3),
            make_frame(3, 0.3, 0.0, 0.0, 1.02, pelvis_y=0.0),
        ]
        events = detect_jump_landings(frames, ground_y=1.0, airborne_tolerance_m=0.08)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "jump_landing")
        self.assertEqual(events[0].end_frame, 3)

    def test_detect_change_of_direction(self):
        frames = [
            make_frame(0, 0.0, 0.0, 0.0, 1.0),
            make_frame(1, 0.1, 1.0, 0.0, 1.0),
            make_frame(2, 0.2, 2.0, 0.0, 1.0),
            make_frame(3, 0.3, 2.0, 1.0, 1.0),
            make_frame(4, 0.4, 2.0, 2.0, 1.0),
        ]
        events = detect_cuts_and_decelerations(frames, min_speed_mps=1.0, cut_angle_deg=45.0)
        self.assertTrue(any(event.event_type == "change_of_direction" for event in events))


if __name__ == "__main__":
    unittest.main()

