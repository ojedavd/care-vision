"""Verify posture/trajectory signal integrity without claiming fall classification."""

import statistics

from care_vision.check_loop import require

FIELDS = {
    "posture_trajectory_status",
    "primary_source_bbox",
    "primary_center_y_normalized",
    "primary_bottom_y_normalized",
    "primary_bbox_aspect_ratio",
    "primary_orientation",
    "primary_vertical_velocity_per_s",
    "primary_motion_direction",
    "primary_near_floor",
    "primary_time_near_floor_s",
}


def check_trajectory_records(cloud):
    observations = [record for record in cloud if record.get("event") == "observation"]
    require(observations, "missing observations")
    require(
        all(FIELDS <= record.keys() for record in observations),
        "observation lacks posture/trajectory fields",
    )
    live = [record for record in observations if record.get("evidence_kind", "live") == "live"]
    valid = [record for record in live if record["posture_trajectory_status"] == "ok"]
    require(len(valid) >= 20, "insufficient valid live trajectory observations")
    require(
        {record["view_kind"] for record in valid} >= {"full_frame", "roi"},
        "trajectory was not measured across full-frame and ROI views",
    )
    directions = {record["primary_motion_direction"] for record in valid}
    require({"downward", "stable"} <= directions, "missing downward or stable trajectory evidence")
    for record in valid:
        bbox = record["primary_source_bbox"]
        require(
            isinstance(bbox, list)
            and len(bbox) == 4
            and bbox[0] >= 0
            and bbox[1] >= 0
            and bbox[2] > 0
            and bbox[3] > 0
            and bbox[0] + bbox[2] <= record["source_width"] + 0.01
            and bbox[1] + bbox[3] <= record["source_height"] + 0.01,
            "source-space bbox is invalid",
        )
        require(
            0 <= record["primary_center_y_normalized"] <= 1
            and 0 <= record["primary_bottom_y_normalized"] <= 1
            and record["primary_bbox_aspect_ratio"] > 0
            and record["primary_orientation"] in {"vertical", "intermediate", "horizontal"}
            and record["primary_motion_direction"] in {"upward", "stable", "downward"}
            and record["primary_time_near_floor_s"] >= 0,
            "normalized posture/trajectory value is invalid",
        )
    velocity = [record["primary_vertical_velocity_per_s"] for record in valid]
    aspect = [record["primary_bbox_aspect_ratio"] for record in valid]
    return {
        "verified": True,
        "live_observations": len(live),
        "valid_trajectory_observations": len(valid),
        "valid_full_frame_observations": sum(
            record["view_kind"] == "full_frame" for record in valid
        ),
        "valid_roi_observations": sum(record["view_kind"] == "roi" for record in valid),
        "directions": {
            direction: sum(record["primary_motion_direction"] == direction for record in valid)
            for direction in ("upward", "stable", "downward")
        },
        "orientations": {
            orientation: sum(record["primary_orientation"] == orientation for record in valid)
            for orientation in ("vertical", "intermediate", "horizontal")
        },
        "near_floor_observations": sum(record["primary_near_floor"] for record in valid),
        "vertical_velocity_per_s": {
            "minimum": min(velocity),
            "median": statistics.median(velocity),
            "maximum": max(velocity),
        },
        "bbox_aspect_ratio": {
            "minimum": min(aspect),
            "median": statistics.median(aspect),
            "maximum": max(aspect),
        },
        "fall_classification_enabled": False,
    }
