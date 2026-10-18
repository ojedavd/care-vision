"""Verify native-size dense recording and rejection of mid-stream size changes."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("change_size", [False, True])
@pytest.mark.parametrize(
    "scenario", ["natural", "controlled_episode", "controlled_lateral", "daily_negative"]
)
def test_native_capture_integrity(tmp_path, monkeypatch, change_size, scenario):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    capture_tool = importlib.import_module("capture_tapo_dense")
    clock = [0]

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            clock[0] += 500_000_000
            height = 240 if not change_size or clock[0] < 8_000_000_000 else 200
            return True, np.zeros((height, 320, 3), dtype=np.uint8)

        def release(self):
            pass

    class Source:
        def __init__(self, *args, **kwargs):
            pass

        def _open(self):
            return Camera()

    monkeypatch.setattr(capture_tool, "RTSPSource", Source)
    monkeypatch.setattr(
        capture_tool,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0] / 1e9, monotonic_ns=lambda: clock[0]),
    )
    monkeypatch.setattr(
        capture_tool,
        "subprocess",
        SimpleNamespace(
            run=lambda *a, **kw: None, Popen=lambda *a, **kw: SimpleNamespace(poll=lambda: 0)
        ),
    )
    monkeypatch.setenv("CARE_VISION_RTSP_USERNAME", "test")
    monkeypatch.setenv("CARE_VISION_RTSP_PASSWORD", "test")
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--rtsp-ip", "192.0.2.1", "--output", str(output), "--scenario", scenario],
    )
    if change_size:
        with pytest.raises(RuntimeError, match="resolution changed"):
            capture_tool.main()
    else:
        capture_tool.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["recording_resolution"] == [320, 240]
    assert manifest["decoded_frames"] == manifest["recorded_frames"]
    assert manifest["complete"] is not change_size
    if not change_size:
        seconds, cues = capture_tool.SCENARIOS[scenario]
        assert manifest["duration_seconds"] >= seconds - 1
        assert len(manifest["delivered_cues"]) == len(cues)
        assert manifest["scenario"] == scenario
