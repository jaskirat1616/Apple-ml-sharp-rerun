#!/usr/bin/env python3
"""
Splatline SLAM — Monocular Visual SLAM for Video 3D Mapping

Runs a pure-Python ORB-based visual SLAM pipeline on a video file:
  1. Extracts ORB features from each frame
  2. Matches features between consecutive frames
  3. Estimates camera pose via essential matrix decomposition
  4. Triangulates matched features into 3D map points
  5. Visualizes camera trajectory + growing 3D map in Rerun

No CUDA required — works on macOS with MPS or CPU.
Dependencies: opencv-python, numpy, rerun-sdk (all already installed)

Usage:
  python run_slam_3d.py <video_file>
  python run_slam_3d.py /path/to/video.mp4
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class VisualSLAM:
    """Lightweight monocular ORB-based visual SLAM.

    Tracks ORB features across video frames, estimates camera motion
    via essential matrix decomposition, and triangulates 3D points
    to build a sparse map. Uses keyframe-based tracking for stability.
    """

    def __init__(self, video_path, frame_skip=2):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip

        # ORB feature detector — fast, free, no license issues
        self.orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
        # BFMatcher with Hamming distance for ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera intrinsics — estimated from frame size (30mm default, like SHARP)
        self.f_px = None
        self.cx = None
        self.cy = None
        self.K = None  # 3x3 intrinsic matrix
        self.K_inv = None

        # State
        self.camera_poses = []  # list of (R, t) — cumulative camera pose
        self.map_points = []  # list of 3D points (Nx3)
        self.map_colors = []  # list of RGB colors for each map point
        self.map_point_count = 0
        self.frame_count = 0
        self.keyframe_count = 0

        # Previous frame data
        self.prev_kp = None
        self.prev_desc = None
        self.prev_frame_gray = None
        self.prev_R = np.eye(3)
        self.prev_t = np.zeros((3, 1))

        # Keyframe management — insert a keyframe every N frames or when
        # motion is large enough
        self.keyframe_interval = 5
        self.last_keyframe_idx = 0

    def _setup_camera(self, width, height):
        """Estimate camera intrinsics from frame dimensions."""
        # 30mm focal length default (same as SHARP)
        # f_px = 30mm * width / 36mm_sensor
        self.f_px = width * (30.0 / 36.0)
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.K = np.array([
            [self.f_px, 0, self.cx],
            [0, self.f_px, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def _extract_features(self, gray):
        """Detect ORB keypoints and compute descriptors."""
        kp, desc = self.orb.detectAndCompute(gray, None)
        if kp is None or desc is None or len(kp) < 10:
            return None, None
        return kp, desc

    def _match_features(self, desc1, desc2, ratio_threshold=0.75):
        """Match features using Lowe's ratio test."""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        return good_matches

    def _estimate_pose(self, pts1, pts2):
        """Estimate relative camera pose from matched 2D points.

        Uses essential matrix decomposition via RANSAC.
        Returns (R, t, mask) where R is rotation, t is translation,
        mask indicates inlier matches.
        """
        pts1_f = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2_f = np.ascontiguousarray(pts2, dtype=np.float64)
        E, mask = cv2.findEssentialMat(
            pts1_f, pts2_f, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None or mask is None or len(mask) < 5:
            return None, None, None

        # Recover pose from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_f, pts2_f, self.K, mask=mask)
        # Ensure float64 and contiguous
        R = np.ascontiguousarray(R, dtype=np.float64)
        t = np.ascontiguousarray(t, dtype=np.float64)
        return R, t, mask_pose.ravel().astype(bool)

    def _triangulate(self, pts1, pts2, R, t):
        """Triangulate matched 2D points to 3D world points.

        Uses the camera projection matrices P1 = K[I|0] and P2 = K[R|t].
        Returns 3D points in world coordinates.
        """
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])

        pts1_norm = (pts1 - np.array([self.cx, self.cy])) / self.f_px
        pts2_norm = (pts2 - np.array([self.cx, self.cy])) / self.f_px

        # cv2.triangulatePoints expects float32 CV_32F matrices
        R32 = np.ascontiguousarray(R, dtype=np.float32)
        t32 = np.ascontiguousarray(t, dtype=np.float32)
        proj1 = np.ascontiguousarray(np.hstack([np.eye(3, 4)]), dtype=np.float32)
        proj2 = np.ascontiguousarray(np.hstack([R32, t32.reshape(3, 1)]), dtype=np.float32)
        pts1_32 = np.ascontiguousarray(pts1_norm.T, dtype=np.float32)
        pts2_32 = np.ascontiguousarray(pts2_norm.T, dtype=np.float32)

        # Debug: ensure shapes are correct
        assert proj1.shape == (3, 4), f"proj1 shape: {proj1.shape}"
        assert proj2.shape == (3, 4), f"proj2 shape: {proj2.shape}"
        assert pts1_32.shape[0] == 2, f"pts1 shape: {pts1_32.shape}"
        assert pts2_32.shape[0] == 2, f"pts2 shape: {pts2_32.shape}"

        pts4d = cv2.triangulatePoints(proj1, proj2, pts1_32, pts2_32)

        # Convert homogeneous to 3D
        pts3d = (pts4d[:3] / pts4d[3]).T
        return pts3d

    def _get_matched_points(self, kp1, kp2, matches):
        """Extract matched 2D point coordinates from keypoints and matches."""
        pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def _get_colors_for_points(self, frame_bgr, pts2d):
        """Sample colors from the frame at 2D point locations."""
        h, w = frame_bgr.shape[:2]
        u = np.clip(pts2d[:, 0].astype(int), 0, w - 1)
        v = np.clip(pts2d[:, 1].astype(int), 0, h - 1)
        colors = cv2.cvtColor(frame_bgr[v, u].reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        return colors

    def process_frame(self, frame):
        """Process a single video frame through the SLAM pipeline.

        Returns dict with frame data for visualization, or None if skipped.
        """
        h, w = frame.shape[:2]
        if self.K is None:
            self._setup_camera(w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract features
        kp, desc = self._extract_features(gray)
        if kp is None:
            return None

        # First frame — just store features
        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_desc = desc
            self.prev_frame_gray = gray
            self.camera_poses.append((np.eye(3), np.zeros((3, 1))))
            return {
                "frame": frame,
                "keypoints": kp,
                "matches": [],
                "pose": (np.eye(3), np.zeros((3, 1))),
                "new_points": np.array([]),
                "new_colors": np.array([]),
                "is_keyframe": True,
            }

        # Match with previous frame
        matches = self._match_features(self.prev_desc, desc)
        if len(matches) < 20:
            # Not enough matches — skip this frame, use as new reference
            self.prev_kp = kp
            self.prev_desc = desc
            self.prev_frame_gray = gray
            return None

        pts1, pts2 = self._get_matched_points(self.prev_kp, kp, matches)

        # Estimate relative pose
        R_rel, t_rel, inlier_mask = self._estimate_pose(pts1, pts2)
        if R_rel is None:
            self.prev_kp = kp
            self.prev_desc = desc
            self.prev_frame_gray = gray
            return None

        # Filter to inliers — but if mask is too restrictive, use all matches
        if inlier_mask.sum() >= 10:
            pts1_in = pts1[inlier_mask]
            pts2_in = pts2[inlier_mask]
            matches_in = [m for i, m in enumerate(matches) if inlier_mask[i]]
        else:
            # Inlier mask too restrictive — use all matched points
            pts1_in = pts1
            pts2_in = pts2
            matches_in = matches

        # Scale translation (monocular SLAM has scale ambiguity)
        # Use median triangulation depth to set a reasonable scale
        # Accumulate pose: T_world = T_prev @ T_rel
        R_new = self.prev_R @ R_rel
        t_new = self.prev_t + self.prev_R @ t_rel

        # Determine if this should be a keyframe
        is_keyframe = (
            self.frame_count - self.last_keyframe_idx >= self.keyframe_interval
            or np.linalg.norm(t_rel) > 0.3
        )

        new_points = np.array([])
        new_colors = np.array([])

        if is_keyframe and len(pts1_in) >= 5:
            # Triangulate points between previous keyframe and this frame
            # Use the relative pose for triangulation
            pts3d = self._triangulate(pts1_in, pts2_in, R_rel, t_rel)

            # Filter points: keep those with positive depth (in front of camera)
            # and reasonable triangulation angle
            valid = (
                (pts3d[:, 2] > 0) &  # positive depth
                (pts3d[:, 2] < 50) &  # not too far
                (np.abs(pts3d[:, 0]) < 50) &  # reasonable X
                (np.abs(pts3d[:, 1]) < 50)   # reasonable Y
            )

            if valid.sum() > 5:
                pts3d_valid = pts3d[valid]
                pts2_valid = pts2_in[valid]

                # Transform to world coordinates using accumulated pose
                pts3d_world = (self.prev_R @ pts3d_valid.T + self.prev_t).T

                # Get colors from current frame
                new_colors = self._get_colors_for_points(frame, pts2_valid)

                # Add to map
                self.map_points.append(pts3d_world)
                self.map_colors.append(new_colors)
                self.map_point_count += len(pts3d_world)

                new_points = pts3d_world
                self.last_keyframe_idx = self.frame_count
                self.keyframe_count += 1

        # Update state
        self.prev_R = R_new
        self.prev_t = t_new
        self.prev_kp = kp
        self.prev_desc = desc
        self.prev_frame_gray = gray
        self.camera_poses.append((R_new, t_new.copy()))

        return {
            "frame": frame,
            "keypoints": kp,
            "matches": matches_in,
            "pose": (R_new, t_new),
            "new_points": new_points,
            "new_colors": new_colors,
            "is_keyframe": is_keyframe,
        }

    def get_camera_trajectory(self):
        """Return the camera trajectory as Nx3 array of positions."""
        positions = np.array([t.ravel() for _, t in self.camera_poses])
        return positions

    def get_map_points(self):
        """Return all map points as Nx3 array."""
        if not self.map_points:
            return np.array([]).reshape(0, 3)
        return np.vstack(self.map_points)

    def get_map_colors(self):
        """Return all map point colors as Nx3 array."""
        if not self.map_colors:
            return np.array([]).reshape(0, 3)
        return np.vstack(self.map_colors)

    def run(self):
        """Run the SLAM pipeline on the video and visualize in Rerun."""
        import rerun as rr

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("=" * 60)
        print("SPLATLINE SLAM — 3D MAPPING FROM VIDEO")
        print("=" * 60)
        print(f"Video: {self.video_path.name}")
        print(f"Resolution: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
        print(f"Processing every {self.frame_skip} frame(s)")
        print(f"Estimated focal length: {width * 30 / 36:.0f}px (30mm default)")
        print()

        # Initialize Rerun
        rr.init("Splatline SLAM 3D Mapping")
        rr.spawn(memory_limit="2GiB", server_memory_limit="4GiB")

        # Log static camera intrinsics
        self._setup_camera(width, height)
        rr.log("slam/camera_intrinsics",
            rr.Pinhole(
                width=width, height=height,
                focal_length=self.f_px,
            ),
            static=True,
        )

        frame_idx = 0
        processed = 0
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            self.frame_count = processed
            result = self.process_frame(frame)

            if result is not None:
                # Print progress every frame (flush for real-time output)
                if processed % 5 == 0:
                    print(f"\r  Frame {processed}/{total_frames // self.frame_skip} | "
                          f"KFs: {self.keyframe_count} | "
                          f"Map: {self.map_point_count:,} pts", end="", flush=True)
                # Set time
                rr.set_time("frame", sequence=processed)
                rr.set_time("time", duration=processed * self.frame_skip / fps)

                # Log the source frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("slam/source_frame", rr.Image(frame_rgb))

                # Log camera pose as a transform + frustum
                R, t = result["pose"]
                # Camera position in world space
                cam_pos = t.ravel()

                # Log camera trajectory (all positions so far)
                trajectory = self.get_camera_trajectory()
                if len(trajectory) > 1:
                    rr.log("slam/trajectory", rr.LineStrips3D(
                        [trajectory],
                        colors=[(255, 255, 0)],  # yellow trajectory
                        radii=[0.05],
                    ))

                # Log camera position as a point
                rr.log("slam/camera_position", rr.Points3D(
                    positions=[cam_pos],
                    colors=[(255, 0, 0)],  # red camera
                    radii=[0.3],
                ))

                # Log new map points from this keyframe
                if len(result["new_points"]) > 0:
                    pts = result["new_points"]
                    cols = result["new_colors"]
                    rr.log(f"slam/map/keyframe_{self.keyframe_count}",
                        rr.Points3D(
                            positions=pts,
                            colors=cols,
                            radii=[0.05] * len(pts),
                        )
                    )

                # Log all map points (accumulated)
                all_pts = self.get_map_points()
                all_cols = self.get_map_colors()
                if len(all_pts) > 0:
                    rr.log("slam/map/all_points", rr.Points3D(
                        positions=all_pts,
                        colors=all_cols,
                        radii=[0.04] * len(all_pts),
                    ))

                # Log matched keypoints on the source frame
                if result["matches"]:
                    kp = result["keypoints"]
                    match_pts = np.array([kp[m.trainIdx].pt for m in result["matches"]])
                    rr.log("slam/matched_features", rr.Points2D(
                        positions=match_pts,
                        colors=[(0, 255, 0)] * len(match_pts),  # green matches
                        radii=[3] * len(match_pts),
                    ))

                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"  Frame {processed}/{total_frames // self.frame_skip} | "
                          f"Keyframes: {self.keyframe_count} | "
                          f"Map points: {self.map_point_count:,} | "
                          f"Camera: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}) | "
                          f"{elapsed:.1f}s", flush=True)

            frame_idx += 1

        cap.release()

        # Final summary
        trajectory = self.get_camera_trajectory()
        all_pts = self.get_map_points()
        all_cols = self.get_map_colors()

        print()
        print("=" * 60)
        print("SLAM COMPLETE")
        print("=" * 60)
        print(f"Frames processed: {processed}")
        print(f"Keyframes: {self.keyframe_count}")
        print(f"Map points: {self.map_point_count:,}")
        print(f"Trajectory length: {len(trajectory)} positions")
        if len(trajectory) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            print(f"Path length: {path_length:.2f} units")
        print(f"Time: {time.time() - t_start:.1f}s")
        print()
        print("Rerun viewer is open. Explore the 3D map and camera trajectory.")
        print("Press Ctrl+C to exit.")

        # Save map to PLY for later use
        if len(all_pts) > 0:
            output_ply = self.video_path.parent / f"{self.video_path.stem}_slam_map.ply"
            self._save_ply(all_pts, all_cols, output_ply)
            print(f"Map saved to: {output_ply}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")

    def _save_ply(self, points, colors, path):
        """Save point cloud + colors to a PLY file."""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_slam_3d.py <video_file>")
        print()
        print("Example:")
        print("  python run_slam_3d.py /path/to/video.mp4")
        print("  python run_slam_3d.py ~/Downloads/grok-video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    slam = VisualSLAM(video_path, frame_skip=2)
    slam.run()


if __name__ == "__main__":
    main()
