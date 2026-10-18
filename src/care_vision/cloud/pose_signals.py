"""Experimental torso signals for offline comparison; never incident decisions."""

from collections import deque

import numpy as np


class PoseTrajectoryAnalyzer:
    # Declared before evaluation; none of these values is calibrated for falls.
    keypoint_threshold = 0.3
    horizontal_degrees = 60.0
    vertical_degrees = 30.0
    velocity_threshold = 0.15
    straightening_degrees_per_s = 15.0

    def __init__(self):
        self.identity = None
        self.history = deque(maxlen=64)
        self.last_timestamp_ns = None

    @staticmethod
    def empty(status):
        return {
            "pose_status": status,
            "pose_keypoints": [],
            "pose_torso_min_score": None,
            "pose_torso_angle_degrees": None,
            "pose_torso_center_y_normalized": None,
            "pose_orientation": None,
            "pose_vertical_velocity_per_s": None,
            "pose_motion_direction": "unknown",
            "pose_torso_rotation_degrees_per_s": None,
            "pose_straightening": None,
        }

    def update(self, points, track_id, captured_ns, image_shape, namespace):
        identity = (namespace, track_id)
        if identity != self.identity:
            self.history.clear()
            self.last_timestamp_ns = None
            self.identity = identity
        if self.last_timestamp_ns is not None and captured_ns <= self.last_timestamp_ns:
            raise ValueError("pose capture time must increase")
        self.last_timestamp_ns = captured_ns
        if points is None or track_id is None:
            self.history.clear()
            return self.empty("no_confirmed_person")
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (17, 3) or not np.all(np.isfinite(points)):
            raise ValueError("pose must contain 17 finite x/y/score keypoints")
        height, width = image_shape[:2]
        torso = points[[5, 6, 11, 12]]
        result = self.empty("low_torso_confidence")
        result["pose_keypoints"] = np.round(points, 6).tolist()
        result["pose_torso_min_score"] = float(torso[:, 2].min())
        if np.any(torso[:, 2] < self.keypoint_threshold):
            self.history.clear()
            return result
        if (
            np.any(torso[:, 0] < 0)
            or np.any(torso[:, 0] >= width)
            or np.any(torso[:, 1] < 0)
            or np.any(torso[:, 1] >= height)
        ):
            self.history.clear()
            result["pose_status"] = "torso_outside_image"
            return result
        shoulders, hips = torso[:2, :2].mean(axis=0), torso[2:, :2].mean(axis=0)
        dx, dy = shoulders - hips
        if np.hypot(dx, dy) < 0.02 * height:
            self.history.clear()
            result["pose_status"] = "degenerate_torso"
            return result
        angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        center_y = float((shoulders[1] + hips[1]) / 2 / height)
        while self.history and captured_ns - self.history[0][0] > 800_000_000:
            self.history.popleft()
        candidates = [r for r in self.history if 400_000_000 <= captured_ns - r[0] <= 800_000_000]
        velocity = rotation = None
        if candidates:
            previous = min(candidates, key=lambda r: abs(captured_ns - r[0] - 500_000_000))
            seconds = (captured_ns - previous[0]) / 1e9
            velocity, rotation = (center_y - previous[1]) / seconds, (angle - previous[2]) / seconds
        self.history.append((captured_ns, center_y, angle))
        result.update(
            pose_status="ok" if velocity is not None else "insufficient_temporal_context",
            pose_torso_angle_degrees=angle,
            pose_torso_center_y_normalized=center_y,
            pose_orientation="horizontal"
            if angle >= self.horizontal_degrees
            else "vertical"
            if angle <= self.vertical_degrees
            else "intermediate",
            pose_vertical_velocity_per_s=velocity,
            pose_motion_direction="unknown"
            if velocity is None
            else "downward"
            if velocity >= self.velocity_threshold
            else "upward"
            if velocity <= -self.velocity_threshold
            else "stable",
            pose_torso_rotation_degrees_per_s=rotation,
            pose_straightening=rotation <= -self.straightening_degrees_per_s
            if rotation is not None
            else None,
        )
        return result


def pose_scoring_row(row, straightening=False):
    """Adapt pose fields to existing temporal scorer without modifying bbox signals."""
    direction = row["pose_motion_direction"]
    if straightening:
        value = row["pose_straightening"]
        direction = "unknown" if value is None else "upward" if value else "stable"
    return {
        "frame_index": row["frame_index"],
        "phase": row["phase"],
        "primary_motion_direction": direction,
        "primary_orientation": row["pose_orientation"],
        "primary_near_floor": None,
    }
