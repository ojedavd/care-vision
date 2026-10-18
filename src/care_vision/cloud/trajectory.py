"""Deterministic primary-person posture and trajectory signals in source coordinates."""

from collections import defaultdict, deque

from care_vision.common.image_geometry import ImageGeometry


def map_bbox_to_source(bbox, header):
    """Invert the Edge letterbox and optional ROI crop for one detection bbox."""
    return ImageGeometry.from_header(header).to_source(bbox)


class PostureTrajectoryAnalyzer:
    def __init__(self):
        self.namespace = None
        self.history = defaultdict(lambda: deque(maxlen=64))
        self.near_floor_since = {}

    def reset(self, namespace=None):
        self.namespace = namespace
        self.history.clear()
        self.near_floor_since.clear()

    def update(self, person_observation, header, namespace):
        if self.namespace != namespace:
            self.reset(namespace)
        track_id = person_observation.get("primary_track_id")
        bbox = person_observation.get("primary_tracked_bbox")
        empty = {
            "posture_trajectory_status": "no_confirmed_person",
            "primary_source_bbox": None,
            "primary_center_y_normalized": None,
            "primary_bottom_y_normalized": None,
            "primary_bbox_aspect_ratio": None,
            "primary_orientation": None,
            "primary_vertical_velocity_per_s": None,
            "primary_motion_direction": "unknown",
            "primary_near_floor": None,
            "primary_time_near_floor_s": None,
        }
        if not track_id or not bbox:
            return empty
        source_bbox = person_observation.get("primary_tracked_source_bbox")
        if source_bbox is None:
            source_bbox = map_bbox_to_source(bbox, header)
        x, y, width, height = source_bbox
        source_height = header.get("source_height", header["height"])
        center_y = (y + height / 2) / source_height
        bottom_y = (y + height) / source_height
        aspect = width / height if height else None
        orientation = (
            "horizontal"
            if aspect is not None and aspect >= 1.2
            else "vertical"
            if aspect is not None and aspect <= 0.8
            else "intermediate"
        )
        captured = header["captured_mono_ns"]
        history = self.history[track_id]
        while history and captured - history[0][0] > 2_000_000_000:
            history.popleft()
        candidates = [item for item in history if 400_000_000 <= captured - item[0] <= 800_000_000]
        velocity = None
        if candidates:
            previous_ns, previous_center = min(
                candidates, key=lambda item: abs(captured - item[0] - 500_000_000)
            )
            velocity = (center_y - previous_center) / ((captured - previous_ns) / 1e9)
        history.append((captured, center_y))
        near_floor = bottom_y >= 0.85
        if near_floor:
            self.near_floor_since.setdefault(track_id, captured)
            time_near_floor = (captured - self.near_floor_since[track_id]) / 1e9
        else:
            self.near_floor_since.pop(track_id, None)
            time_near_floor = 0.0
        direction = (
            "unknown"
            if velocity is None
            else "downward"
            if velocity >= 0.15
            else "upward"
            if velocity <= -0.15
            else "stable"
        )
        return {
            "posture_trajectory_status": (
                "ok" if velocity is not None else "insufficient_temporal_context"
            ),
            "primary_source_bbox": [round(value, 3) for value in source_bbox],
            "primary_center_y_normalized": round(center_y, 6),
            "primary_bottom_y_normalized": round(bottom_y, 6),
            "primary_bbox_aspect_ratio": round(aspect, 6) if aspect is not None else None,
            "primary_orientation": orientation,
            "primary_vertical_velocity_per_s": round(velocity, 6) if velocity is not None else None,
            "primary_motion_direction": direction,
            "primary_near_floor": near_floor,
            "primary_time_near_floor_s": round(time_near_floor, 6),
        }
