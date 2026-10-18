"""Offline pose support diagnostics, not visibility, occlusion or fall estimates.

A model may give strong responses to hidden joints or reflections. Support here
means only that the declared score, geometry and crop checks pass.
"""

import math

import numpy as np


def pose_crop_bbox(bbox):
    """Match the existing RTMPose 192x256, padding=1.25 affine crop."""
    x, y, width, height = bbox
    if not all(math.isfinite(v) for v in bbox) or width <= 0 or height <= 0:
        raise ValueError("invalid bbox")
    crop_width, crop_height = width * 1.25, height * 1.25
    aspect = 192 / 256
    if crop_width > crop_height * aspect:
        crop_height = crop_width / aspect
    else:
        crop_width = crop_height * aspect
    return [x + width / 2 - crop_width / 2, y + height / 2 - crop_height / 2,
            crop_width, crop_height]


def assess_pose_support(points, bbox, image_shape, other_boxes=(), *, threshold=0.3):
    """Inspect full-body support without treating missing keypoints as normal.

    Both shoulders, hips, knees and ankles must meet the existing 0.30 response
    threshold and lie inside the image. Torso and lower-leg segments must have
    nondegenerate 2D length. A crop containing another detection's center is
    marked ambiguous. These are conservative experimental checks, not calibrated
    quality probabilities. Knee angles remain diagnostics, not decision rules.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("invalid threshold")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("invalid image size")
    crop = pose_crop_bbox(bbox)
    cx, cy, cw, ch = crop
    others = []
    for box in other_boxes:
        pose_crop_bbox(box)  # validate even if outside crop
        x, y, w, h = box
        if cx <= x + w / 2 <= cx + cw and cy <= y + h / 2 <= cy + ch:
            others.append(list(box))
    result = {
        "body_support_status": "NO_POSE", "body_support_usable": False,
        "torso_supported": False, "legs_supported": False,
        "torso_min_response": None, "leg_min_response": None,
        "left_knee_angle_degrees": None, "right_knee_angle_degrees": None,
        "crop_bbox": crop, "other_detection_centers_in_crop": others,
        "physical_visibility_verified": False,
    }
    if points is None:
        return result
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (17, 3) or not np.all(np.isfinite(points)):
        raise ValueError("expected 17 finite keypoints")
    inside = ((points[:, 0] >= 0) & (points[:, 0] < width)
              & (points[:, 1] >= 0) & (points[:, 1] < height))
    supported = inside & (points[:, 2] >= threshold)
    torso_indices, leg_indices = [5, 6, 11, 12], [11, 12, 13, 14, 15, 16]
    shoulder = points[[5, 6], :2].mean(axis=0)
    hip = points[[11, 12], :2].mean(axis=0)
    torso_supported = bool(
        supported[torso_indices].all() and np.linalg.norm(shoulder - hip) >= 0.02 * height
    )
    legs_supported = bool(supported[leg_indices].all())
    for label, indices in (("left", [11, 13, 15]), ("right", [12, 14, 16])):
        a, b, c = points[indices, :2]
        upper, lower = a - b, c - b
        upper_length, lower_length = np.linalg.norm(upper), np.linalg.norm(lower)
        valid = supported[indices].all() and min(upper_length, lower_length) >= 0.02 * height
        legs_supported = bool(legs_supported and valid)
        if valid:
            cosine = np.clip(np.dot(upper, lower) / (upper_length * lower_length), -1, 1)
            result[label + "_knee_angle_degrees"] = float(np.degrees(np.arccos(cosine)))
    result.update(
        torso_supported=torso_supported, legs_supported=legs_supported,
        torso_min_response=float(points[torso_indices, 2].min()),
        leg_min_response=float(points[leg_indices, 2].min()),
        body_support_status=(
            "AMBIGUOUS_CROP" if others else "SUPPORTED_KEYPOINTS"
            if torso_supported and legs_supported else "INSUFFICIENT_KEYPOINT_SUPPORT"
        ),
        body_support_usable=torso_supported and legs_supported and not others,
    )
    return result
