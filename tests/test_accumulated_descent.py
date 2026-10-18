import math

import pytest

from care_vision.cloud.accumulated_descent import AccumulatedDescent, DescentPolicy


def sample(analyzer, t, box, track="a"):
    return analyzer.update(t, round(t * 1000), {} if box is None else {track: box})


@pytest.mark.parametrize("fps", [2, 20])
def test_gradual_posture_change_with_consistent_identity(fps):
    analyzer = AccumulatedDescent()
    signals = []
    for i in range(2 * fps + 1):
        t = i / fps
        signals += sample(analyzer, t, [10, t * 10, 20, 100 - t * 10])
    first = next(s for s in signals if s["descent_signal"])
    assert first["source_time_seconds"] == 1.5
    assert first["reference_time"] < first["source_time_seconds"]


@pytest.mark.parametrize("kind", ["static", "translation", "centered_shrink", "upward"])
def test_geometry_controls_reject_nonmatching_changes(kind):
    analyzer = AccumulatedDescent()
    for i in range(5):
        t = i / 2
        box = {
            "static": [0, 0, 20, 100],
            "translation": [0, t * 10, 20, 100],
            "centered_shrink": [0, t * 5, 20, 100 - t * 10],
            "upward": [0, 30 - t * 10, 20, 70 + t * 10],
        }[kind]
        assert not sample(analyzer, t, box)[0]["descent_signal"]


@pytest.mark.parametrize("kind", ["missing", "new_identity", "gap"])
def test_never_bridges_unavailable_evidence(kind):
    analyzer = AccumulatedDescent()
    sample(analyzer, 0, [0, 0, 20, 100])
    sample(analyzer, 0.5, [0, 5, 20, 95])
    if kind != "gap":
        sample(analyzer, 1, None if kind == "missing" else [0, 10, 20, 90], "b")
    signal = sample(analyzer, 1.5, [0, 30, 20, 70])[0]
    assert signal["status"] == "insufficient_contiguous_history"
    assert not signal["descent_signal"]


def test_signal_is_scale_and_translation_invariant():
    a, b = AccumulatedDescent(), AccumulatedDescent()
    for i in range(5):
        t = i / 2
        sa = sample(a, t, [0, t * 10, 20, 100 - t * 10])[0]
        sb = sample(b, t, [70, 50 + t * 30, 60, 300 - t * 30])[0]
        assert sa["descent_signal"] == sb["descent_signal"]
        assert sa["top_drop"] == sb["top_drop"]


def test_independent_people_do_not_supply_each_others_history():
    analyzer = AccumulatedDescent()
    for i in range(5):
        t = i / 2
        signals = analyzer.update(
            t,
            i,
            {
                "a": [0, t * 10, 20, 100 - t * 10],
                "b": [100, 0, 20, 100],
            },
        )
    assert signals[0]["descent_signal"]
    assert not signals[1]["descent_signal"]


def test_rejects_bad_input_before_mutation():
    analyzer = AccumulatedDescent()
    sample(analyzer, 0, [0, 0, 20, 100])
    with pytest.raises(ValueError):
        sample(analyzer, 0.5, [0, math.nan, 20, 100])
    assert analyzer.last_time == 0
    with pytest.raises(ValueError):
        sample(analyzer, 0, [0, 0, 20, 100])
    with pytest.raises(ValueError):
        DescentPolicy(window_seconds=0.5)


def test_old_observations_expire_and_horizontal_reference_is_not_standing():
    analyzer = AccumulatedDescent()
    for i in range(30):
        signal = sample(analyzer, i / 2, [0, i * 2, 150, 100 - i])[0]
        assert not signal["descent_signal"]
        assert signal["span_seconds"] is None or signal["span_seconds"] <= 2
    assert len(analyzer.histories["a"]) == 5


def test_opt_in_missing_gap_uses_fresh_box_and_same_identity_only():
    analyzer = AccumulatedDescent(policy=DescentPolicy(maximum_missing_gap_seconds=1.6))
    sample(analyzer, 0, [0, 0, 20, 100])
    sample(analyzer, 0.5, [0, 5, 20, 95])
    assert sample(analyzer, 1, None) == []
    signal = sample(analyzer, 1.5, [0, 30, 20, 70])[0]
    assert signal["descent_signal"]
    assert signal["reference_time"] == 0


@pytest.mark.parametrize("kind", ["expired", "new_identity"])
def test_opt_in_does_not_bridge_long_absence_or_identity(kind):
    analyzer = AccumulatedDescent(policy=DescentPolicy(maximum_missing_gap_seconds=1.6))
    sample(analyzer, 0, [0, 0, 20, 100])
    sample(analyzer, 0.5, None)
    t = 2 if kind == "expired" else 1.5
    signal = sample(analyzer, t, [0, 30, 20, 70], "b" if kind == "new_identity" else "a")[0]
    assert not signal["descent_signal"]
    assert signal["reference_time"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 2.1])
def test_missing_gap_policy_rejects_unbounded_values(value):
    with pytest.raises(ValueError):
        DescentPolicy(maximum_missing_gap_seconds=value)
