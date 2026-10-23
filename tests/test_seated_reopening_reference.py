"""Reopening after experimental closure; synthetic geometry, not fall accuracy."""

import copy

import pytest
from test_seated_closure import SEATED, feed, opened, sample, seconds

LOWER = [600, 340, 170, 240]


def closed(identity):
    c = opened(identity)
    feed(c, seconds(2, 7))
    assert c.state == "NORMAL"
    assert c.seated_reopen_reference is not None
    return c


def test_second_descent_reopens_with_seated_not_upright_reference(identity):
    c = closed(identity)
    assert c.observe(sample(7.5, box=LOWER)) is None
    command = c.observe(sample(8, box=LOWER))
    assert command["transmit_fps"] == 5
    assert command["reason"] == "posture_change_requires_more_evidence"
    assert c.episode["reference_bbox"] == [v / (1280 if i % 2 == 0 else 720)
                                           for i, v in enumerate(SEATED)]
    assert c.seated_reopen_reference is None


@pytest.mark.parametrize("kind", ["stable", "upward", "translation"])
def test_benign_synthetic_continuation_does_not_reopen(identity, kind):
    c = closed(identity)
    for step in range(1, 17):
        box = list(SEATED)
        if kind == "upward":
            box[1] -= min(step * 10, 100)
            box[3] += min(step * 10, 100)
        elif kind == "translation":
            box[1] += min(step * 5, 70)
        assert c.observe(sample(7 + step / 2, box=box)) is None
    assert c.state == "NORMAL" and c.episode is None


@pytest.mark.parametrize("kind", [
    "missing", "other_track", "ambiguous", "weak", "boundary", "invalid",
    "motion_context_only", "tracking_error", "detection_error", "tentative",
    "roi", "resolution", "rate", "policy", "visual_policy", "gap",
])
def test_disqualified_sample_discards_seed_and_is_not_a_new_reference(identity, kind):
    c = closed(identity)
    item = copy.deepcopy(sample(7.5))
    if kind == "missing":
        item["persons"] = []
    elif kind == "other_track":
        item["persons"][0]["track_id"] = "b"
    elif kind == "ambiguous":
        item["persons"].append(dict(item["persons"][0], track_id="b"))
    elif kind == "weak":
        item["persons"][0]["association_only"] = True
    elif kind == "boundary":
        item["persons"][0]["tracked_bbox"] = [0, 250, 170, 300]
    elif kind in {"invalid", "motion_context_only"}:
        item["valid"] = False
        item["invalid_reason"] = ("insufficient_temporal_context"
                                  if kind == "motion_context_only" else "injected")
    elif kind in {"tracking_error", "detection_error"}:
        item[f"person_{kind.split('_')[0]}_status"] = "error"
    elif kind == "tentative":
        item["persons"][0]["track_state"] = "tentative"
    elif kind == "roi":
        item["view_kind"] = "roi"
    elif kind == "resolution":
        item["width"] = 640
    elif kind == "rate":
        item["target_transmit_fps"] = 5
    elif kind == "policy":
        item["applied_command_id"] = "different"
    elif kind == "visual_policy":
        item["applied_visual_command_id"] = "different"
    else:
        item = sample(7.9)
    assert c.observe(item) is None
    assert c.seated_reopen_reference is None
    assert not c.s_probe.change.histories
    start = item["captured_mono_ns"] / 1e9
    for step in range(1, 7):
        assert c.observe(sample(start + step / 2, box=LOWER)) is None
    assert c.episode is None


def test_seed_expires_without_clearing_new_observation_history(identity):
    c = closed(identity)
    for t in (7.5, 8, 8.5, 9, 9.5):
        c.observe(sample(t))
    assert c.seated_reopen_reference is None
    assert c.s_probe.change.histories
    # A later descent can still use fresh seated samples.
    assert c.observe(sample(10, box=LOWER))["transmit_fps"] == 5


def test_async_old_diagnostic_cannot_seed_current_detector(identity):
    c = opened(identity)
    feed(c, seconds(2, 6.5))
    old = sample(7)
    c.observe(old)
    c.observe(sample(7.5))
    result = c.consider_seated_closure(
        old, {"episode_id": c.episode["episode_id"], "observation_id": old["observation_id"],
              "seated_pose_candidate": True}, result_age_seconds=0.5)
    assert result["closed"]
    assert c.seated_reopen_reference is None
    assert not c.s_probe.change.histories


def test_standing_closure_does_not_seed_seated_reference(identity):
    c = opened(identity)
    for t in seconds(2, 5):
        c.observe(sample(t, box=[600, 40, 150, 500]))
    assert c.episode is None
    assert c.seated_reopen_reference is None
    assert not c.s_probe.change.histories or c.s_probe.change.last_time >= 4


def test_seed_does_not_alias_mutable_observation(identity):
    c = closed(identity)
    before = copy.deepcopy(c.s_probe.change.histories)
    item = copy.deepcopy(sample(7.5))
    c.observe(item)
    item["persons"][0]["tracked_bbox"][1] = 999
    assert c.s_probe.change.histories["a"][0] == before["a"][0]
