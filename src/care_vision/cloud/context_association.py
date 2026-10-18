"""Experimental geometric association, restricted to a unique anchored segment.

No confidence ranking, ID reuse, gap bridging, or identity claim. A geometric
match must be unique in both directions between consecutive available frames.
"""

import math

from care_vision.evaluation import intersection_over_union


def eligible(a, b):
    """Use the existing tracker's static geometry gate, without reacquisition."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    distance = math.hypot(ax + aw / 2 - bx - bw / 2, ay + ah / 2 - by - bh / 2)
    distance /= max(aw, ah, bw, bh)
    ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh)
    return intersection_over_union(a, b) >= 0.2 or (distance <= 0.75 and ratio >= 0.15)


def anchored_segment(rows, anchor_frame_id, anchor_index):
    """Inspect only supplied evidence; caller controls cloud arrival availability."""
    if not rows or len(rows) > 512:
        raise ValueError("association requires one to 512 frames")
    if any(
        b["captured_mono_ns"] <= a["captured_mono_ns"] or b["capture_seq"] <= a["capture_seq"]
        for a, b in zip(rows, rows[1:])
    ) or len({r["frame_id"] for r in rows}) != len(rows):
        raise ValueError("association requires unique chronological frames")
    if any(len(r["persons"]) > 32 for r in rows):
        raise ValueError("too many detections")
    anchors = [i for i, r in enumerate(rows) if r["frame_id"] == anchor_frame_id]
    if len(anchors) != 1:
        raise ValueError("missing anchor")
    position = anchors[0]
    anchor = rows[position]
    if (
        not anchor["available"]
        or type(anchor_index) is not int
        or not 0 <= anchor_index < len(anchor["persons"])
    ):
        raise ValueError("unavailable anchor detection")
    selected = {anchor_frame_id: anchor_index}
    boundaries = {}
    for step, direction in ((-1, "before"), (1, "after")):
        current, index = position, anchor_index
        boundary = {"reason": "window_edge", "frame_id": None}
        while 0 <= current + step < len(rows):
            a, b = rows[current], rows[current + step]
            reason = None
            candidates, reverse = [], []
            if abs(a["captured_mono_ns"] - b["captured_mono_ns"]) > 800_000_000:
                reason = "capture_gap"
            elif not b["available"] or not b["persons"]:
                reason = "evidence_unavailable"
            else:
                candidates = [
                    j
                    for j, p in enumerate(b["persons"])
                    if eligible(a["persons"][index]["bbox"], p["bbox"])
                ]
                if len(candidates) != 1:
                    reason = "ambiguous_successor" if candidates else "no_geometric_match"
                else:
                    reverse = [
                        j
                        for j, p in enumerate(a["persons"])
                        if eligible(p["bbox"], b["persons"][candidates[0]]["bbox"])
                    ]
                    if reverse != [index]:
                        reason = "ambiguous_predecessor"
            if reason:
                boundary = {
                    "reason": reason,
                    "frame_id": b["frame_id"],
                    "source_frame_index": b["capture_seq"] - 1,
                    "candidate_indices": candidates,
                    "reverse_indices": reverse,
                }
                break
            index = candidates[0]
            selected[b["frame_id"]] = index
            current += step
        boundaries[direction] = boundary
    return {"selected": selected, "boundaries": boundaries}
