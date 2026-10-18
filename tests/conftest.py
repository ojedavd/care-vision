import cv2
import numpy as np
import pytest

from care_vision.common.contracts import envelope, utc


@pytest.fixture
def identity():
    return {"edge_id": "test-edge", "boot_id": "boot", "session_id": "session"}


def frame(identity, seq=1, at_ns=1_000_000_000, x=20, policy_seq=0, command_id=None):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (x, 90), (x + 180, 270), (230, 230, 230), -1)
    _, encoded = cv2.imencode(".jpg", image)
    jpeg = encoded.tobytes()
    header = envelope(
        "FRAME",
        identity,
        frame_id=f"{identity['boot_id']}:{seq}",
        capture_seq=seq,
        captured_mono_ns=at_ns,
        sampled_mono_ns=at_ns,
        captured_at_utc=utc(),
        encoding="jpeg",
        width=640,
        height=360,
        payload_bytes=len(jpeg),
        policy_seq=policy_seq,
        applied_command_id=command_id,
        target_transmit_fps=2 if not policy_seq else 20,
    )
    return header, jpeg


def status(identity):
    return envelope(
        "EDGE_STATUS",
        identity,
        capture_fps=30,
        transmit_fps=2,
        window_seconds=1,
        target_transmit_fps=2,
        captured=30,
        selected=2,
        sent=2,
        dropped=0,
        bytes_sent=1000,
        policy_seq=0,
        connection="connected",
    )


def command(identity, seq=1, command_id="C1", fps=20, deadline=10_000_000_000):
    return envelope(
        "SET_TRANSMIT_FPS",
        identity,
        command_id=command_id,
        command_seq=seq,
        caused_by_observation_id="O1",
        based_on_frame_id="boot:1",
        reason="test",
        issued_at_utc=utc(),
        valid_until_edge_mono_ns=deadline,
        transmit_fps=fps,
    )


def roi_command(identity, seq=2, command_id="R1", deadline=10_000_000_000):
    return envelope(
        "REQUEST_ROI",
        identity,
        command_id=command_id,
        command_seq=seq,
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
        reason="inspect_person_detail",
        valid_until_edge_mono_ns=deadline,
    )
