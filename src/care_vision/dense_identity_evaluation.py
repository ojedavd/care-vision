"""Offline comparison of conservative geometry and mutual IoU ranking.

This evaluator is not imported by the cloud controller. The candidate's fixed
0.2 IoU and 0.2 margin are experimental, not calibrated identity thresholds.
"""

import math
from collections import Counter
from types import SimpleNamespace

from care_vision.cloud.context_association import eligible
from care_vision.evaluation import intersection_over_union, match_detection_pairs

POLICIES = ("mutual_unique", "mutual_best_iou")


def associate_pair(left, right, policy):
    """Return index-to-index links from boxes only; never use labels or confidence."""
    if policy not in POLICIES:
        raise ValueError("unsupported diagnostic policy")
    if len(left) > 32 or len(right) > 32:
        raise ValueError("too many detections")
    for box in [*left, *right]:
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
            raise ValueError("invalid detection box")
    matrix = [[intersection_over_union(a, b) for b in right] for a in left]
    links = {}
    for i, row in enumerate(matrix):
        if not row:
            continue
        if policy == "mutual_unique":
            candidates = [j for j, b in enumerate(right) if eligible(left[i], b)]
            if len(candidates) == 1:
                j = candidates[0]
                if sum(eligible(a, right[j]) for a in left) == 1:
                    links[i] = j
        else:
            j = max(range(len(row)), key=row.__getitem__)
            best = row[j]
            row_second = max((v for k, v in enumerate(row) if k != j), default=0)
            column_second = max((r[j] for k, r in enumerate(matrix) if k != i), default=0)
            if best >= 0.2 and best - row_second >= 0.2 and best - column_second >= 0.2:
                links[i] = j
    return links, matrix


def validate_annotations(dataset):
    source = dataset["source"]
    if any(
        type(source[k]) is not int or source[k] <= 0 for k in ("width", "height", "frame_count")
    ):
        raise ValueError("invalid source dimensions")
    if not dataset["windows"] or len(dataset["windows"]) > 32:
        raise ValueError("invalid windows")
    frame_ids, window_ids = set(), set()
    for window in dataset["windows"]:
        if window["window_id"] in window_ids:
            raise ValueError("duplicate window")
        window_ids.add(window["window_id"])
        samples = window["samples"]
        if not 2 <= len(samples) <= 512:
            raise ValueError("window needs two to 512 samples")
        previous = -1
        for sample in samples:
            index = sample["frame_index"]
            if (
                type(index) is not int
                or not previous < index < source["frame_count"]
                or index in frame_ids
            ):
                raise ValueError("invalid or duplicate sample index")
            previous = index
            frame_ids.add(index)
            identities = set()
            for p in sample["persons"]:
                box = p["bbox"]
                if (
                    not isinstance(p["identity_id"], str)
                    or not p["identity_id"]
                    or p["identity_id"] in identities
                ):
                    raise ValueError("invalid or duplicate identity")
                identities.add(p["identity_id"])
                if (
                    len(box) != 4
                    or any(type(v) is not int for v in box)
                    or min(box[:2]) < 0
                    or min(box[2:]) <= 0
                    or box[0] + box[2] > source["width"]
                    or box[1] + box[3] > source["height"]
                ):
                    raise ValueError("invalid annotation box")


def evaluate_dense(detections, dataset):
    validate_annotations(dataset)
    if (
        detections["source_sha256"] != dataset["source"]["sha256"]
        or detections["dataset_sha256"] != dataset["collection_dataset_sha256"]
    ):
        raise ValueError("source or collection provenance mismatch")
    rows = detections["rows"]
    by_index = {r["capture_seq"] - 1: r for r in rows}
    if len(by_index) != len(rows) or any(
        b["capture_seq"] <= a["capture_seq"] or b["captured_mono_ns"] <= a["captured_mono_ns"]
        for a, b in zip(rows, rows[1:])
    ):
        raise ValueError("invalid detection chronology")
    width, height = dataset["source"]["width"], dataset["source"]["height"]
    scale = min(detections["width"] / width, detections["height"] / height)
    left = (detections["width"] - round(width * scale)) // 2
    top = (detections["height"] - round(height * scale)) // 2
    labels, coverage = {}, Counter()
    for window in dataset["windows"]:
        for sample in window["samples"]:
            index = sample["frame_index"]
            if index not in by_index:
                raise ValueError("missing annotated frame")
            row = by_index[index]
            truth = []
            for p in sample["persons"]:
                x, y, w, h = p["bbox"]
                truth.append(
                    {
                        "identity_id": p["identity_id"],
                        "bbox": [x * scale + left, y * scale + top, w * scale, h * scale],
                    }
                )
            pairs = (
                match_detection_pairs([SimpleNamespace(**p) for p in row["persons"]], truth, 0.5)
                if row["available"]
                else []
            )
            labels[index] = {pred: truth[gt]["identity_id"] for pred, gt, _ in pairs}
            coverage.update(
                annotated_persons=len(truth),
                matched_persons=len(pairs),
                raw_detections=len(row["persons"]),
            )

    def status(chosen, target_index, identity):
        if chosen is None:
            return "abstained"
        actual = labels[target_index].get(chosen)
        return (
            "correct"
            if actual == identity
            else "wrong_identity"
            if actual
            else "unmatched_detection"
        )

    reports = {}
    for policy in POLICIES:
        transitions, chains = [], []
        for window in dataset["windows"]:
            samples = window["samples"]
            mappings = []
            for a, b in zip(samples, samples[1:]):
                ai, bi = a["frame_index"], b["frame_index"]
                ar, br = by_index[ai], by_index[bi]
                allowed = (
                    ar["available"]
                    and br["available"]
                    and br["captured_mono_ns"] - ar["captured_mono_ns"] <= 800_000_000
                )
                links, matrix = (
                    associate_pair(
                        [p["bbox"] for p in ar["persons"]],
                        [p["bbox"] for p in br["persons"]],
                        policy,
                    )
                    if allowed
                    else ({}, [])
                )
                mappings.append(links)
                outcomes = []
                for identity in sorted(
                    {p["identity_id"] for p in a["persons"]}
                    & {p["identity_id"] for p in b["persons"]}
                ):
                    seed = next((j for j, name in labels[ai].items() if name == identity), None)
                    outcomes.append(
                        {
                            "identity_id": identity,
                            "status": "anchor_missed"
                            if seed is None
                            else status(links.get(seed), bi, identity),
                        }
                    )
                transitions.append(
                    {
                        "window_id": window["window_id"],
                        "from_frame": ai,
                        "to_frame": bi,
                        "links": [
                            {"from_detection": i, "to_detection": j} for i, j in links.items()
                        ],
                        "iou_matrix": matrix,
                        "outcomes": outcomes,
                    }
                )
            first_index = samples[0]["frame_index"]
            for p in samples[0]["persons"]:
                seed = next(
                    (j for j, name in labels[first_index].items() if name == p["identity_id"]), None
                )
                current = seed
                targets = []
                for links, sample in zip(mappings, samples[1:]):
                    current = links.get(current) if current is not None else None
                    if any(q["identity_id"] == p["identity_id"] for q in sample["persons"]):
                        targets.append(
                            {
                                "frame_index": sample["frame_index"],
                                "status": "anchor_missed"
                                if seed is None
                                else status(current, sample["frame_index"], p["identity_id"]),
                            }
                        )
                chains.append(
                    {
                        "window_id": window["window_id"],
                        "anchor_frame": first_index,
                        "identity_id": p["identity_id"],
                        "targets": targets,
                    }
                )

        def counts(outcomes):
            c = Counter(o["status"] for o in outcomes)
            return {
                k: c[k]
                for k in (
                    "correct",
                    "wrong_identity",
                    "unmatched_detection",
                    "abstained",
                    "anchor_missed",
                )
            }

        reports[policy] = {
            "adjacent_counts": counts(o for t in transitions for o in t["outcomes"]),
            "chain_counts": counts(o for c in chains for o in c["targets"]),
            "transitions": transitions,
            "chains": chains,
        }
    return {
        "coverage": dict(coverage),
        "policies": reports,
        "commands_enabled": False,
        "scope": "Conditional offline identity diagnostic on provisional development labels.",
    }
