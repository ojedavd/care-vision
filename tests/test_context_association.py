import copy
import runpy
from pathlib import Path

import pytest
from test_event_context import live_rows, observation, request

from care_vision.cloud.context_association import anchored_segment
from care_vision.cloud.event_context import EventContext
from care_vision.event_context_evaluation import replay_event_contexts


def row(n, boxes, *, available=True):
    return {
        "frame_id": str(n),
        "capture_seq": n + 1,
        "captured_mono_ns": n * 100_000_000,
        "available": available,
        "persons": [{"bbox": b} for b in boxes],
    }


A = [0, 0, 10, 20]
B = [100, 0, 10, 20]


def test_distant_second_person_does_not_invalidate_unique_anchor_segment():
    rows = [row(i, [A, B]) for i in range(4)]
    result = anchored_segment(rows, "1", 0)
    assert result["selected"] == {str(i): 0 for i in range(4)}
    assert all(b["reason"] == "window_edge" for b in result["boundaries"].values())


def test_competing_successors_stop_instead_of_ranking_confidence():
    rows = [row(0, [A]), row(1, [A]), row(2, [A, [1, 0, 10, 20]]), row(3, [A])]
    rows[2]["persons"][1]["confidence"] = 0.01
    result = anchored_segment(rows, "1", 0)
    assert set(result["selected"]) == {"0", "1"}
    assert result["boundaries"]["after"]["reason"] == "ambiguous_successor"


def test_reverse_competition_blocks_an_otherwise_unique_successor():
    rows = [row(0, [A, [1, 0, 10, 20]]), row(1, [A])]
    result = anchored_segment(rows, "0", 0)
    assert result["selected"] == {"0": 0}
    assert result["boundaries"]["after"]["reason"] == "ambiguous_predecessor"


@pytest.mark.parametrize("cause", ["empty", "unavailable", "gap", "different_person"])
def test_no_bridge_over_missing_evidence_or_new_person(cause):
    rows = [row(i, [A]) for i in range(4)]
    if cause == "empty":
        rows[2]["persons"] = []
    elif cause == "unavailable":
        rows[2]["available"] = False
    elif cause == "gap":
        rows[2]["captured_mono_ns"] += 1_000_000_000
        rows[3]["captured_mono_ns"] += 1_000_000_000
    else:
        rows[2]["persons"] = [{"bbox": B}]
    result = anchored_segment(rows, "1", 0)
    assert set(result["selected"]) == {"0", "1"}


def test_detection_order_and_prior_ids_do_not_select_another_person():
    rows = [row(i, [A, B]) for i in range(4)]
    rows[2]["persons"].reverse()
    for r in rows:
        for p in r["persons"]:
            p.update(track_id="same-poisoned-id", confidence=0.01)
    result = anchored_segment(rows, "1", 0)
    for r in rows:
        assert r["persons"][result["selected"][r["frame_id"]]]["bbox"] == A


def test_late_ambiguous_frame_retracts_segment_without_mutating_previous_revision():
    rows = [row(i, [A]) for i in (0, 2, 3)]
    before = anchored_segment(rows, "2", 0)
    after = anchored_segment([rows[0], row(1, [A, A]), *rows[1:]], "2", 0)
    assert set(before["selected"]) == {"0", "2", "3"}
    assert set(after["selected"]) == {"2", "3"}


def test_context_keeps_single_person_policy_as_default_and_unique_is_opt_in():
    c = EventContext(request(), observation(1), association_policy="mutual_unique")
    strict = EventContext(request(), observation(1))
    for r in live_rows():
        r["persons"].append({"bbox": [85, 5, 10, 20], "confidence": 0.99, "track_id": "other"})
        r["person_count"] = 2
        # Constructor already stored the anchor; keep its exact original contents.
        if r["capture_seq"] == 101:
            continue
        c.add(r)
        strict.add(r)
    assert strict.snapshot(110_000_000_000)["identity_status"] == "ambiguous_multiple_detections"
    result = c.snapshot(110_000_000_000)
    assert result["identity_status"] == "experimental_unique_segment"
    assert result["episodes"]
    assert not result["commands_enabled"]


def test_unique_replay_preserves_prefix_causality():
    rows = live_rows()
    prefix = rows[:3] + [request()] + rows[3:]
    delayed = observation(0.75, kind="historical", local_mono_ns=111_000_000_000)
    delayed["persons"].append(copy.deepcopy(delayed["persons"][0]))
    delayed["person_count"] = 2
    a = replay_event_contexts(prefix, association_policy="mutual_unique")
    b = replay_event_contexts(prefix + [delayed], association_policy="mutual_unique")
    for x, y in zip(a["snapshots"], b["snapshots"]):
        assert {k: v for k, v in x.items() if k != "replay_processing_ms"} == {
            k: v for k, v in y.items() if k != "replay_processing_ms"
        }
    assert b["snapshots"][-1]["identity_status"] == "anchor_unconfirmed_in_context"


def test_invalid_policy_or_order_is_rejected():
    with pytest.raises(ValueError):
        EventContext(request(), observation(1), association_policy="highest_score")
    with pytest.raises(ValueError):
        anchored_segment([row(1, [A]), row(0, [A])], "1", 0)
    with pytest.raises(ValueError):
        anchored_segment([row(0, [A])] * 513, "0", 0)


@pytest.mark.parametrize("case", ["correct", "swapped_identity", "missing_detection"])
def test_identity_score_excludes_anchor_and_reports_wrong_or_missing_targets(case):
    evaluate = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/evaluate_context_identity.py")
    )["evaluate"]
    rows = [row(i, [A, B]) for i in range(4)]
    video = {
        "width": 200,
        "height": 100,
        "samples": [
            {
                "frame_index": index,
                "persons": [{"identity_id": "a", "bbox": A}, {"identity_id": "b", "bbox": B}],
            }
            for index in (0, 3)
        ],
    }
    if case == "swapped_identity":
        video["samples"][1]["persons"][0]["identity_id"] = "b"
        video["samples"][1]["persons"][1]["identity_id"] = "a"
    elif case == "missing_detection":
        rows[-1]["persons"] = []
    report = evaluate({"width": 200, "height": 100, "rows": rows}, video)
    assert report["anchor_count"] == 4
    assert report["matched_anchors"] == (2 if case == "missing_detection" else 4)
    expected = {"correct": 0, "wrong_identity": 0, "unmatched_detection": 0, "abstained": 0}
    expected[
        {
            "correct": "correct",
            "swapped_identity": "wrong_identity",
            "missing_detection": "abstained",
        }[case]
    ] = 2 if case == "missing_detection" else 4
    assert report["target_counts"] == expected
