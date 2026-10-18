import cv2
import numpy as np

from care_vision.common.contracts import envelope, utc
from care_vision.edge.representation import VisualPolicy


def encoded_test_image():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[40:340, 160:400] = (20, 120, 240)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_default_full_frame_reuses_capture_bytes():
    jpeg = encoded_test_image()
    rendered, metadata = VisualPolicy().render(jpeg)
    assert rendered is jpeg
    assert metadata == {
        "view_kind": "full_frame",
        "source_width": 640,
        "source_height": 360,
        "visual_policy_seq": 0,
        "applied_visual_command_id": None,
    }


def test_roi_is_letterboxed_to_requested_output(identity):
    command = envelope(
        "REQUEST_ROI",
        identity,
        command_id="R1",
        command_seq=2,
        issued_at_utc=utc(),
        caused_by_observation_id="O1",
        based_on_frame_id="boot:1",
        source_width=640,
        source_height=360,
        roi_x=160,
        roi_y=40,
        roi_width=240,
        roi_height=300,
        output_width=640,
        output_height=360,
        reason="test",
        valid_until_edge_mono_ns=10,
    )
    policy = VisualPolicy()
    policy.apply(command)
    rendered, metadata = policy.render(encoded_test_image())
    image = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert image.shape == (360, 640, 3)
    assert metadata["view_kind"] == "roi"
    assert metadata["roi_width"] == 240
    assert metadata["roi_height"] == 300
    assert metadata["visual_policy_seq"] == 2
    assert image[:, :170].mean() < 5
    assert image[:, 180:460].mean() > 20


def test_full_frame_can_reduce_resolution():
    policy = VisualPolicy()
    policy.apply(
        {
            "type": "REQUEST_FULL_FRAME",
            "command_id": "F1",
            "command_seq": 3,
            "output_width": 320,
            "output_height": 180,
        }
    )
    rendered, metadata = policy.render(encoded_test_image())
    image = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (180, 320, 3)
    assert metadata["view_kind"] == "full_frame"


def test_high_resolution_source_can_feed_reduced_full_frame_and_roi(identity):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[100:700, 800:1200] = (20, 120, 240)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    policy = VisualPolicy(1280, 720, 320, 180)
    reduced, metadata = policy.render(jpeg)
    assert cv2.imdecode(np.frombuffer(reduced, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
        180,
        320,
        3,
    )
    assert metadata["source_width"] == 1280
    command = envelope(
        "REQUEST_ROI",
        identity,
        command_id="high-res-roi",
        command_seq=1,
        issued_at_utc=utc(),
        caused_by_observation_id="O1",
        based_on_frame_id="boot:1",
        source_width=1280,
        source_height=720,
        roi_x=800,
        roi_y=100,
        roi_width=400,
        roi_height=600,
        output_width=640,
        output_height=360,
        reason="test",
        valid_until_edge_mono_ns=10,
    )
    policy.apply(command)
    roi, metadata = policy.render(jpeg)
    assert cv2.imdecode(np.frombuffer(roi, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
        360,
        640,
        3,
    )
    assert metadata["roi_x"] == 800


def test_mixed_roi_interleaves_context_on_capture_clock_and_resets():
    jpeg = encoded_test_image()
    policy = VisualPolicy()
    policy.render(jpeg, 30_000_000)
    command = {
        "type": "REQUEST_ROI",
        "command_id": "mixed",
        "command_seq": 1,
        "output_width": 640,
        "output_height": 360,
        "roi_x": 160,
        "roi_y": 40,
        "roi_width": 240,
        "roi_height": 300,
        "context_fps": 2,
    }
    policy.apply(command)
    _, first = policy.render(jpeg, 100_000_000)
    _, context = policy.render(jpeg, 530_000_000)
    _, following = policy.render(jpeg, 580_000_000)
    assert [first["view_kind"], context["view_kind"], following["view_kind"]] == [
        "roi",
        "full_frame",
        "roi",
    ]
    assert context["applied_visual_command_id"] == "mixed"
    assert "roi_x" not in context
    assert policy.view_kind == "roi"
    policy.apply(
        {
            "type": "REQUEST_FULL_FRAME",
            "command_id": "full",
            "command_seq": 2,
            "output_width": 640,
            "output_height": 360,
        }
    )
    assert policy.context_fps == 0
    assert policy.render(jpeg, 600_000_000)[1]["view_kind"] == "full_frame"


def test_mixed_roi_requires_monotonic_capture_time():
    import pytest

    policy = VisualPolicy(context_fps=2)
    with pytest.raises(ValueError):
        policy.render(encoded_test_image())
    policy.render(encoded_test_image(), 100)
    with pytest.raises(ValueError):
        policy.render(encoded_test_image(), 99)
