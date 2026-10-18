import numpy as np
import pytest

from care_vision.cloud.occlusion_fragments import OcclusionFragments
from care_vision.cloud.occlusion_roi import incident_roi


def test_expanded_roi_preserves_space_below_feet_and_on_both_sides():
    box = [600, 100, 160, 400]
    x, y, w, h = incident_roi(box, (720, 1280), expanded=True)
    assert x <= 600 and x + w >= 760
    assert y <= 100 and y + h == 720
    assert w > incident_roi(box, (720, 1280), expanded=False)[2]


def test_roi_is_clipped_to_image():
    assert incident_roi([0, 0, 100, 700], (720, 1280)) == [0, 0, 575, 720]


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError):
        incident_roi([0, 0, 10, 0], (720, 1280))


def test_untransmitted_pixels_cannot_be_foreground_evidence():
    # Synthetic arrays test masking only, not local inference on user footage.
    tracker = OcclusionFragments()
    image = np.full((360, 640, 3), 255, np.uint8)
    _, _, foreground = tracker.update(image, 0, 0, visible_roi=[200, 100, 100, 100])
    assert not foreground[:100].any()
    assert not foreground[200:].any()
    assert not foreground[:, :200].any()
    assert not foreground[:, 300:].any()
    assert foreground[100:200, 200:300].any()
