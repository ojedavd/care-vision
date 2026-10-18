import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "tracking_cache", Path(__file__).parents[1] / "tools/prepare_tracking_cache.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frame(index, timestamp):
    return {"frame_index": index, "captured_mono_ns": timestamp, "capture_seq": index + 1}


def test_shared_frames_are_encoded_once_without_changing_schedules():
    first, second, third = frame(0, 100), frame(2, 200), frame(4, 300)
    run = {
        "schedules": [
            {"frames": [first, third]},
            {"frames": [first, second, third]},
        ]
    }
    assert module.selected_frames(run) == [first, second, third]
    assert run["schedules"][0]["frames"] == [first, third]


@pytest.mark.parametrize(
    "frames",
    [
        [frame(2, 100), frame(1, 200)],
        [frame(0, 100), frame(2, 100)],
        [frame(0, 200), frame(2, 100)],
        [frame(0, 0)],
    ],
)
def test_invalid_schedule_rejected(frames):
    with pytest.raises(ValueError):
        module.selected_frames({"schedules": [{"frames": frames}]})


def test_conflicting_metadata_between_schedules_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        module.selected_frames(
            {
                "schedules": [
                    {"frames": [frame(0, 100)]},
                    {"frames": [frame(0, 101)]},
                ]
            }
        )
