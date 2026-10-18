"""Associate in source pixels while retaining image-space observation fields."""

import copy


class SourceProjectionError(ValueError):
    pass


def track_in_source(tracker, detected, header, namespace, geometry):
    mapped = copy.deepcopy(detected)
    originals = {p["detection_id"]: p["bbox"] for p in mapped["persons"]}
    for person in mapped["persons"]:
        person["bbox"] = geometry.to_source(person["bbox"])
        if min(person["bbox"][2:]) <= 0:
            raise SourceProjectionError("detection has no transmitted source pixels")
    # PersonTracker itself remains independent of camera representations.
    result = tracker.update(mapped, header["captured_mono_ns"], namespace)
    result["tracking_coordinate_space"] = "source_pixels"
    result["tracking_geometry_version"] = "edge-integer-letterbox-v1"
    primary_source = None
    for person in result["persons"]:
        if "tracked_bbox" not in person:
            continue
        person["source_bbox"] = person["bbox"]
        person["tracked_source_bbox"] = person["tracked_bbox"]
        person["bbox"] = originals[person["detection_id"]]
        person["tracked_bbox"] = [
            round(v) for v in geometry.to_image(person["tracked_source_bbox"])
        ]
        if person["track_id"] == result["primary_track_id"]:
            primary_source = person["tracked_source_bbox"]
            result["primary_tracked_bbox"] = person["tracked_bbox"]
    result["primary_tracked_source_bbox"] = primary_source
    for track in result["tracks"]:
        track["source_bbox"] = track["bbox"]
        track["bbox"] = [round(v) for v in geometry.to_image(track["source_bbox"])]
    detected_primary = max(result["persons"], key=lambda p: p["confidence"], default=None)
    # Weak-association mode may have filtered the original primary detection.
    if tracker.association_confidence is not None:
        result["primary_person_bbox"] = detected_primary["bbox"] if detected_primary else None
    return result
