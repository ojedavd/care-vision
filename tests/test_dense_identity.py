import copy

import pytest

from care_vision.dense_identity_evaluation import associate_pair, evaluate_dense

A, B = [10, 10, 30, 80], [35, 10, 30, 80]


def fixture():
    rows = [
        {
            "capture_seq": i + 1,
            "captured_mono_ns": i * 100_000_000,
            "available": True,
            "persons": [{"bbox": A}, {"bbox": B}],
        }
        for i in range(3)
    ]
    detections = {
        "source_sha256": "source",
        "dataset_sha256": "collection",
        "width": 100,
        "height": 100,
        "rows": rows,
    }
    dataset = {
        "source": {"width": 100, "height": 100, "frame_count": 3, "sha256": "source"},
        "collection_dataset_sha256": "collection",
        "windows": [
            {
                "window_id": "one",
                "samples": [
                    {
                        "frame_index": i,
                        "persons": [
                            {"identity_id": "a", "bbox": A},
                            {"identity_id": "b", "bbox": B},
                        ],
                    }
                    for i in range(3)
                ],
            }
        ],
    }
    return copy.deepcopy(detections), copy.deepcopy(dataset)


def test_ranked_candidate_resolves_distinct_boxes_while_broad_geometry_abstains():
    assert associate_pair([A, B], [A, B], "mutual_unique")[0] == {}
    assert associate_pair([A, B], [A, B], "mutual_best_iou")[0] == {0: 0, 1: 1}


@pytest.mark.parametrize("side", ["left", "right"])
def test_duplicate_detections_require_abstention_in_both_directions(side):
    left, right = [A], [A]
    (left if side == "left" else right).append(A)
    assert associate_pair(left, right, "mutual_best_iou")[0] == {}


def test_order_changes_only_indices_and_disjoint_boxes_do_not_link():
    assert associate_pair([B, A], [A, B], "mutual_best_iou")[0] == {0: 1, 1: 0}
    assert associate_pair([A], [[200, 10, 30, 80]], "mutual_best_iou")[0] == {}


def test_dense_scoring_excludes_anchor_and_reports_chains_separately():
    d, a = fixture()
    report = evaluate_dense(d, a)
    assert report["coverage"]["matched_persons"] == 6
    assert report["policies"]["mutual_unique"]["adjacent_counts"]["abstained"] == 4
    p = report["policies"]["mutual_best_iou"]
    assert p["adjacent_counts"]["correct"] == 4
    assert p["chain_counts"]["correct"] == 4
    assert sum(p["chain_counts"].values()) == 4


def test_wrong_identity_is_an_error_not_an_abstention():
    d, a = fixture()
    people = a["windows"][0]["samples"][-1]["persons"]
    people[0]["identity_id"], people[1]["identity_id"] = "b", "a"
    result = evaluate_dense(d, a)["policies"]["mutual_best_iou"]
    assert result["adjacent_counts"]["wrong_identity"] == 2
    assert result["chain_counts"]["wrong_identity"] == 2


@pytest.mark.parametrize("cause", ["empty", "unavailable", "gap"])
def test_chain_does_not_resume_after_a_missing_link(cause):
    d, a = fixture()
    if cause == "empty":
        d["rows"][1]["persons"] = []
    elif cause == "unavailable":
        d["rows"][1]["available"] = False
    else:
        d["rows"][1]["captured_mono_ns"] += 1_000_000_000
        d["rows"][2]["captured_mono_ns"] += 1_000_000_000
    result = evaluate_dense(d, a)["policies"]["mutual_best_iou"]
    assert result["chain_counts"]["abstained"] == 4
    assert result["adjacent_counts"]["correct"] == (2 if cause == "gap" else 0)


@pytest.mark.parametrize(
    "damage", ["hash", "identity", "box", "duplicate_frame", "missing_frame", "clock"]
)
def test_invalid_provenance_annotations_and_chronology_are_rejected(damage):
    d, a = fixture()
    if damage == "hash":
        d["source_sha256"] = "wrong"
    elif damage == "identity":
        a["windows"][0]["samples"][0]["persons"][1]["identity_id"] = "a"
    elif damage == "box":
        a["windows"][0]["samples"][0]["persons"][0]["bbox"] = [-1, 0, 10, 10]
    elif damage == "duplicate_frame":
        d["rows"].append(d["rows"][-1])
    elif damage == "missing_frame":
        d["rows"].pop()
    else:
        d["rows"][1]["captured_mono_ns"] = 0
    with pytest.raises(ValueError):
        evaluate_dense(d, a)
