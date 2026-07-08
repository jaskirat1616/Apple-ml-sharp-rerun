import unittest

import numpy as np

from utils.biomechanics import (
    COCO17_JOINTS,
    AthleteFrame,
    compute_frame_metrics,
    depth_lift_coco_keypoints,
    flexion_from_angle,
    joint_angle,
    rescale_sequence_to_height,
)


class BiomechanicsTests(unittest.TestCase):
    def _joints(self):
        joints = np.zeros((17, 3), dtype=np.float32)
        ids = COCO17_JOINTS
        joints[ids["left_hip"]] = [0.0, 0.0, 2.0]
        joints[ids["left_knee"]] = [0.0, 1.0, 2.0]
        joints[ids["left_ankle"]] = [0.0, 2.0, 2.0]
        joints[ids["right_hip"]] = [0.4, 0.0, 2.0]
        joints[ids["right_knee"]] = [0.4, 1.0, 2.0]
        joints[ids["right_ankle"]] = [0.4, 2.0, 2.0]
        joints[ids["left_shoulder"]] = [0.0, -1.0, 2.0]
        joints[ids["right_shoulder"]] = [0.4, -1.0, 2.0]
        joints[ids["nose"]] = [0.2, -1.35, 2.0]
        return joints

    def test_joint_angle_and_flexion(self):
        ids = COCO17_JOINTS
        conf = np.ones(17, dtype=np.float32)
        joints = self._joints()

        angle = joint_angle(joints, ids["left_hip"], ids["left_knee"], ids["left_ankle"], conf)
        self.assertAlmostEqual(angle, 180.0, places=4)
        self.assertAlmostEqual(flexion_from_angle(angle), 0.0, places=4)

        joints[ids["left_ankle"]] = [1.0, 1.0, 2.0]
        angle = joint_angle(joints, ids["left_hip"], ids["left_knee"], ids["left_ankle"], conf)
        self.assertAlmostEqual(angle, 90.0, places=4)
        self.assertAlmostEqual(flexion_from_angle(angle), 90.0, places=4)

    def test_compute_frame_metrics_has_expected_keys(self):
        conf = np.ones(17, dtype=np.float32)
        metrics = compute_frame_metrics(self._joints(), conf, time_s=0.0)
        self.assertIn("left_knee_flexion_deg", metrics)
        self.assertIn("trunk_lean_deg", metrics)
        self.assertAlmostEqual(metrics["left_knee_flexion_deg"], 0.0, places=4)

    def test_depth_lift_keypoints_uses_gaussian_depth(self):
        keypoints = np.zeros((17, 4), dtype=np.float32)
        keypoints[:, 0] = 0.5
        keypoints[:, 1] = 0.5
        keypoints[:, 3] = 1.0
        gaussians = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.01, 0.0, 2.1],
                [-0.01, 0.0, 1.9],
                [0.0, 0.01, 2.0],
                [0.0, -0.01, 2.0],
            ],
            dtype=np.float32,
        )
        joints = depth_lift_coco_keypoints(keypoints, gaussians, f_px=500.0, img_w=1000, img_h=1000)
        self.assertAlmostEqual(float(joints[0, 2]), 2.0, places=3)

    def test_rescale_sequence_to_height(self):
        conf = np.ones(17, dtype=np.float32)
        frame = AthleteFrame(
            frame=0,
            time_s=0.0,
            person_id=0,
            joints_3d=self._joints(),
            keypoints_2d=np.zeros((17, 4), dtype=np.float32),
            confidence=conf,
            metrics={},
            flags=[],
        )
        scaled, scale = rescale_sequence_to_height([frame], target_height_m=1.8)
        self.assertIsNotNone(scale)
        self.assertEqual(len(scaled), 1)
        self.assertGreater(scale, 0.0)


if __name__ == "__main__":
    unittest.main()

