from pathlib import Path
from runpy import run_path

import pytest

analyze = run_path(str(Path(__file__).resolve().parents[1] / "tools/probe_rtsp_visual_latency.py"))[
    "analyze"
]


@pytest.mark.parametrize("inverted", [False, True])
def test_visual_latency_probe_detects_both_brightness_polarities(inverted):
    start = 100_000_000_000
    flashes = [
        {"mono_ns": start + i * 8_000_000_000, "color": color}
        for i, color in enumerate(("white", "black", "white", "black"))
    ]
    samples = []
    for index in range(-75, 925):
        at = start + index * 40_000_000
        transitions_seen = sum(at >= flash["mono_ns"] + 1_400_000_000 for flash in flashes)
        bright = transitions_seen % 2 == 1
        value = 35 if bright == inverted else 160
        samples.append((at, value))

    result = analyze(samples, flashes)

    assert [row["status"] for row in result] == ["measured"] * 4
    assert [row["screen_to_edge_ms"] for row in result] == [1400.0] * 4
    assert {row["response_polarity"] for row in result} == {"inverted" if inverted else "expected"}
