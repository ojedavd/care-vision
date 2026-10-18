import importlib.util
from pathlib import Path

import numpy as np
import pytest

from care_vision.cloud.controller import Controller, Policy
from care_vision.cloud.person import PersonDetection
from care_vision.cloud.vision import MotionVision
from care_vision.common.image_geometry import ImageGeometry, letterbox_geometry
from care_vision.edge.buffer import encode, fit_to_canvas

ROOT = Path(__file__).resolve().parents[1]


class GeometricDetector:
    backend = "synthetic-known-source-boxes"
    boxes = ()

    def detect(self, image):
        return [PersonDetection(tuple(box), .9) for box in self.boxes]


def sample(seq, *, roi=False, width=None, source_width=1280):
    header = {
        "edge_id": "edge", "boot_id": "boot", "session_id": "session",
        "frame_id": f"boot:{seq}", "capture_seq": seq,
        "captured_mono_ns": round(seq * .5e9), "captured_at_utc": "2026-10-14T00:00:00+00:00",
        "applied_command_id": None, "policy_seq": 0, "evidence_kind": "live",
        "view_kind": "roi" if roi else "full_frame", "source_width": source_width,
        "source_height": 720, "width": width or (640 if roi else source_width),
        "height": 360 if roi else 720,
    }
    if roi:
        header.update(roi_x=600, roi_y=50, roi_width=300, roi_height=600)
    jpeg = encode(np.zeros((header["height"], header["width"], 3), np.uint8),
                  header["width"], header["height"])
    return header, jpeg


def observe(vision, detector, seq, boxes, **options):
    header, jpeg = sample(seq, **options)
    geometry = ImageGeometry.from_header(header)
    detector.boxes = [[round(v) for v in geometry.to_image(b)] for b in boxes]
    return vision.observe(header, jpeg)


def test_roi_of_second_person_cannot_associate_to_first_person_at_same_image_location():
    detector = GeometricDetector()
    vision = MotionVision(person_detector=detector)
    a, b = [100, 100, 120, 400], [650, 100, 120, 400]
    observe(vision, detector, 1, [a, b])
    confirmed = observe(vision, detector, 2, [a, b])
    ids = [p["track_id"] for p in confirmed["persons"]]
    cropped = observe(vision, detector, 3, [b], roi=True)
    assert cropped["primary_track_id"] == ids[1]
    assert cropped["primary_tracked_source_bbox"] == b
    assert cropped["persons"][0]["bbox"] == [260, 30, 72, 240]
    restored = observe(vision, detector, 4, [a, b])
    assert [p["track_id"] for p in restored["persons"]] == ids
    assert [p["tracked_bbox"] for p in restored["persons"]] == [a, b]


def test_archived_baseline_reproduces_wrong_identity_for_same_geometric_sequence():
    path = ROOT / "artifacts/source-coordinate-tracking-20261014/baseline/" \
        "src/care_vision/cloud/vision.py"
    spec = importlib.util.spec_from_file_location("legacy_image_tracking", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    detector = GeometricDetector()
    vision = old.MotionVision(person_detector=detector)
    boxes = [[100, 100, 120, 400], [650, 100, 120, 400]]
    observe(vision, detector, 1, boxes)
    confirmed = observe(vision, detector, 2, boxes)
    cropped = observe(vision, detector, 3, [boxes[1]], roi=True)
    assert cropped["primary_track_id"] == confirmed["persons"][0]["track_id"]
    assert cropped["primary_track_id"] != confirmed["persons"][1]["track_id"]


def test_stationary_person_has_no_artificial_source_motion_across_views():
    detector = GeometricDetector()
    vision = MotionVision(person_detector=detector)
    box = [650, 100, 120, 400]
    rows = [observe(vision, detector, seq, [box], roi=(seq in (3, 4))) for seq in range(1, 7)]
    assert len({r["persons"][0]["track_id"] for r in rows}) == 1
    assert all(r["primary_source_bbox"] == box for r in rows[1:])
    assert all(r["primary_vertical_velocity_per_s"] == 0 for r in rows[2:])
    # Pixel motion still requires a new same-view reference after each switch.
    assert not rows[2]["valid"] and not rows[4]["valid"]


def test_source_dimension_change_never_reuses_track_id():
    detector = GeometricDetector()
    vision = MotionVision(person_detector=detector)
    box = [100, 100, 120, 400]
    observe(vision, detector, 1, [box])
    old = observe(vision, detector, 2, [box])
    changed = observe(vision, detector, 3, [box], source_width=1440)
    assert changed["persons"][0]["track_id"] != old["primary_track_id"]
    assert changed["persons"][0]["track_state"] == "tentative"


@pytest.mark.parametrize("shape,output", [((560, 320), (640, 360)), ((437, 403), (640, 360)),
                                         ((513, 769), (641, 359))])
def test_inverse_matches_integer_pixels_actually_rendered_by_edge(shape, output):
    height, width = shape
    ow, oh = output
    rendered = fit_to_canvas(np.full((height, width, 3), 255, np.uint8), ow, oh)
    ys, xs = np.where(rendered[:, :, 0] != 0)
    rw, rh, left, top = letterbox_geometry(width, height, ow, oh)
    assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == (left, top, left + rw, top + rh)
    geometry = ImageGeometry.from_header({"source_width": width, "source_height": height,
                                          "width": ow, "height": oh})
    assert geometry.to_source([left, top, rw, rh]) == [0, 0, width, height]


def test_padding_only_detection_cannot_update_or_expose_identity():
    detector = GeometricDetector()
    vision = MotionVision(person_detector=detector)
    observe(vision, detector, 1, [[650, 100, 120, 400]])
    previous = observe(vision, detector, 2, [[650, 100, 120, 400]])
    header, jpeg = sample(3, roi=True)
    detector.boxes = [[10, 10, 20, 100]]
    rejected = vision.observe(header, jpeg)
    assert rejected["invalid_reason"] == "source_projection_unavailable"
    assert not rejected["confirmed_person_detected"] and rejected["persons"] == []
    restored = observe(vision, detector, 4, [[650, 100, 120, 400]])
    assert restored["primary_track_id"] == previous["primary_track_id"]


def test_crop_never_accumulates_recovery_and_restored_scene_requires_new_dwell(identity):
    detector = GeometricDetector()
    vision = MotionVision(person_detector=detector)
    box = [650, 100, 120, 400]
    observe(vision, detector, 1, [box])
    origin = observe(vision, detector, 2, [box])
    controller = Controller(identity, Policy(experimental_s_only_5fps=True))
    controller._open_episode(origin, origin["primary_track_id"], box)
    controller.state = "UNRESOLVED"
    for seq in range(3, 14):
        obs = observe(vision, detector, seq, [box], roi=seq < 8)
        obs["target_transmit_fps"] = 2
        controller.observe(obs)
        if seq < 13:
            assert controller.episode is not None
        if seq < 9:
            assert controller.episode["upright_since_ns"] is None
    assert controller.episode is None
    assert controller.pop_episode_events()[-1]["captured_mono_ns"] == 6_500_000_000


def test_native_resolution_roi_does_not_feed_full_scene_sensing_probe(identity):
    from test_incident_episode import sample as controller_sample

    controller = Controller(identity, Policy(experimental_s_only_5fps=True))
    for t in (1, 1.5, 2, 2.5, 3):
        obs = controller_sample(t, box=[650, 100 if t < 2 else 200, 120, 400])
        obs["view_kind"] = "roi"
        assert controller.observe(obs) is None
    assert controller.s_probe.change.last_time is None
    assert controller.episode is None and controller.state == "NORMAL"
