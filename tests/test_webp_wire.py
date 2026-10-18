import cv2
import numpy as np
import pytest
from conftest import frame
from care_vision.common.contracts import pack_frame, unpack_frame, ProtocolError, validate
from care_vision.cloud.vision import MotionVision


def test_webp_packet_reaches_existing_image_decoder(identity):
    header, jpeg = frame(identity)
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    ok, payload = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 50])
    assert ok
    header.update(encoding='webp', payload_bytes=len(payload))
    wire_header, wire_payload = unpack_frame(pack_frame(header, payload.tobytes()))
    assert wire_header['encoding'] == 'webp'
    assert wire_payload == payload.tobytes()
    vision = MotionVision()
    observation = vision.observe(wire_header, wire_payload)
    assert observation['invalid_reason'] == 'insufficient_temporal_context'
    later = {**wire_header, 'captured_mono_ns': wire_header['captured_mono_ns'] + 500_000_000}
    assert vision.observe(later, wire_payload)['valid'] is True


def test_unknown_codec_still_rejected(identity):
    header, _ = frame(identity)
    header['encoding'] = 'h264'
    with pytest.raises(ProtocolError):
        validate(header)
