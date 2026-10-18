"""Eligibility guard for experimental bbox recovery, not a posture classifier."""

import math

RECOVERY_RULE_VERSION = "reference-bbox-guard-v2"


def bbox_block_reason(box, width, height):
    try:
        if len(box) != 4 or any(not math.isfinite(v) for v in (*box, width, height)):
            return "invalid_bbox"
        x, y, w, h = box
        if width <= 0 or height <= 0 or w <= 0 or h <= 0:
            return "invalid_bbox"
        # Exact image boundary only: no scene-specific margin or tuned threshold.
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            return "bbox_touches_frame_boundary"
    except (TypeError, ValueError):
        return "invalid_bbox"
    return None


def reference_block_reason(reference, width, height):
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return "reference_invalid_bbox"
    reason = bbox_block_reason(reference, 1, 1)
    if reason:
        return "reference_" + reason
    if reference[2] * width / (reference[3] * height) > 0.8:
        return "reference_aspect_incompatible"
    return None


def recovery_block_reason(reference, observation, person):
    width, height = observation["width"], observation["height"]
    for key in ("tracked_bbox", "bbox"):
        # Older structured observations contain tracked_bbox only.
        if key == "bbox" and key not in person:
            continue
        reason = bbox_block_reason(person.get(key), width, height)
        if reason:
            return "current_" + reason
    reason = reference_block_reason(reference, width, height)
    if reason:
        return reason
    if person.get("association_only", False):
        return "weak_association"
    return None
