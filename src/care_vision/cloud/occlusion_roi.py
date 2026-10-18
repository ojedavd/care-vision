"""Experimental image-plane ROI geometry; not calibrated floor projection."""

import math


def incident_roi(reference_bbox, image_shape, *, expanded=True):
    """Return a causal crop derived only from the pre-descent person box."""
    height, width = image_shape[:2]
    if len(reference_bbox) != 4 or any(not math.isfinite(v) for v in reference_bbox):
        raise ValueError("invalid reference bbox")
    x, y, w, h = reference_bbox
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        raise ValueError("invalid dimensions")
    if expanded:
        left, right = x + w / 2 - 0.75 * h, x + w / 2 + 0.75 * h
        top, bottom = y - 0.15 * h, y + 1.6 * h
    else:
        left, right = x - 0.15 * w, x + 1.15 * w
        top, bottom = y - 0.15 * h, y + 1.15 * h
    x0, x1 = max(0, math.floor(left)), min(width, math.ceil(right))
    y0, y1 = max(0, math.floor(top)), min(height, math.ceil(bottom))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI outside image")
    return [x0, y0, x1 - x0, y1 - y0]
