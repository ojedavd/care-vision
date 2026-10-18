"""Small deterministic helpers shared by CARE-Vision evaluation tools."""


def intersection_over_union(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0


def match_detection_pairs(predictions, expected, threshold):
    """Greedily match detections to truth and retain the paired indexes."""
    candidates = sorted(
        (
            (
                intersection_over_union(prediction.bbox, truth["bbox"]),
                prediction_index,
                truth_index,
            )
            for prediction_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(expected)
        ),
        reverse=True,
    )
    matched_predictions = set()
    matched_truth = set()
    matches = []
    for score, prediction_index, truth_index in candidates:
        if score < threshold:
            break
        if prediction_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_truth.add(truth_index)
        matches.append((prediction_index, truth_index, score))
    return matches


def match_detections(predictions, expected, threshold):
    matches = match_detection_pairs(predictions, expected, threshold)
    return (
        {prediction_index for prediction_index, _, _ in matches},
        {truth_index for _, truth_index, _ in matches},
        [score for _, _, score in matches],
    )


def person_detection_metrics(rows):
    """Aggregate fixed-IoU person detection results from evaluated frames."""
    if not rows:
        raise ValueError("detection metric rows must not be empty")
    true_positive = sum(len(row["matched_ious"]) for row in rows)
    predicted = sum(row["predicted_person_count"] for row in rows)
    expected = sum(row["expected_person_count"] for row in rows)
    false_positive = predicted - true_positive
    false_negative = expected - true_positive
    precision = true_positive / predicted if predicted else 0
    recall = true_positive / expected if expected else 0
    matched_ious = [score for row in rows for score in row["matched_ious"]]
    ordered_ious = sorted(matched_ious)
    middle = len(ordered_ious) // 2
    median_iou = None
    if ordered_ious:
        median_iou = (
            ordered_ious[middle]
            if len(ordered_ious) % 2
            else (ordered_ious[middle - 1] + ordered_ious[middle]) / 2
        )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "matched_iou_median": median_iou,
    }


def frame_localization_diagnostics(rows):
    """Separate frame-level presence failures from fixed-IoU localization failures."""
    if not rows:
        raise ValueError("localization diagnostic rows must not be empty")
    positive = [row for row in rows if row["expected_person_count"] > 0]
    negative = [row for row in rows if row["expected_person_count"] == 0]
    detected = [row for row in positive if row["predicted_person_count"] > 0]
    localized = [row for row in positive if row["matched_ious"]]
    return {
        "positive_frames": len(positive),
        "positive_frames_with_any_detection": len(detected),
        "positive_frames_with_iou_match": len(localized),
        "positive_frames_detection_without_iou_match": sum(
            row["predicted_person_count"] > 0 and not row["matched_ious"] for row in positive
        ),
        "positive_frames_without_detection": sum(
            row["predicted_person_count"] == 0 for row in positive
        ),
        "negative_frames": len(negative),
        "negative_frames_with_detection": sum(
            row["predicted_person_count"] > 0 for row in negative
        ),
        "presence_recall": len(detected) / len(positive) if positive else 0,
        "localized_frame_recall": len(localized) / len(positive) if positive else 0,
    }


def positive_segments(samples):
    segments = []
    current = []
    for sample in samples:
        if sample["persons"]:
            if current and sample["second"] != current[-1]["second"] + 1:
                segments.append(current)
                current = []
            current.append(sample)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def primary_track_continuity(track_ids):
    """Summarize identity continuity while treating missing observations as gaps."""
    observed = [track_id for track_id in track_ids if track_id is not None]
    switches = sum(left != right for left, right in zip(observed, observed[1:]))
    return {
        "frames_with_confirmed_track": len(observed),
        "unique_primary_track_ids": len(set(observed)),
        "primary_track_switches": switches,
        "primary_track_ids": observed,
    }


def identity_tracking_metrics(assignments):
    """Summarize ground-truth identity coverage and fragmentation."""
    if not assignments:
        raise ValueError("identity assignments must not be empty")
    grouped = {}
    for assignment in assignments:
        grouped.setdefault(assignment["identity_id"], []).append(assignment)

    identities = []
    for identity_id, values in sorted(grouped.items()):
        values.sort(key=lambda item: (item["video_id"], item["frame_index"]))
        observed = [item["track_id"] for item in values if item["track_id"] is not None]
        adjacent_switches = sum(
            left["track_id"] is not None
            and right["track_id"] is not None
            and left["track_id"] != right["track_id"]
            for left, right in zip(values, values[1:])
        )
        identities.append(
            {
                "identity_id": identity_id,
                "annotated_frames": len(values),
                "matched_frames": len(observed),
                "coverage": len(observed) / len(values),
                "unique_track_ids": len(set(observed)),
                "adjacent_track_switches": adjacent_switches,
                "track_ids": observed,
            }
        )
    return {
        "ground_truth_identities": len(identities),
        "annotated_identity_instances": len(assignments),
        "matched_identity_instances": sum(item["matched_frames"] for item in identities),
        "identity_coverage": sum(item["matched_frames"] for item in identities) / len(assignments),
        "unique_track_ids_across_identities": len(
            {track_id for item in identities for track_id in item["track_ids"]}
        ),
        "adjacent_track_switches": sum(item["adjacent_track_switches"] for item in identities),
        "identities": identities,
    }


def _maximum_assignment_weight(weights):
    """Return the optimal one-to-one weight for a small evaluation matrix."""
    if not weights or not weights[0]:
        return 0
    if len(weights[0]) > len(weights):
        weights = [list(row) for row in zip(*weights)]
    columns = len(weights[0])
    if columns > 20:
        raise ValueError("identity metric supports at most 20 tracks or identities")
    states = {0: 0}
    for row in weights:
        updated = dict(states)
        for used, total in states.items():
            for column, weight in enumerate(row):
                bit = 1 << column
                if used & bit:
                    continue
                key = used | bit
                updated[key] = max(updated.get(key, 0), total + weight)
        states = updated
    return max(states.values())


def multi_object_tracking_metrics(samples):
    """Compute sampled CLEAR MOT and global IDF1 metrics from matched frames."""
    if not samples:
        raise ValueError("tracking metric samples must not be empty")
    ordered = sorted(samples, key=lambda item: (item["video_id"], item["frame_index"]))
    ground_truth_total = sum(item["expected_persons"] for item in ordered)
    predicted_total = sum(item["confirmed_persons"] for item in ordered)
    matches = [match for item in ordered for match in item["matches"]]
    false_negatives = ground_truth_total - len(matches)
    false_positives = predicted_total - len(matches)

    last_track_by_identity = {}
    identity_switches = 0
    for item in ordered:
        for match in item["matches"]:
            identity = match["identity_id"]
            track_id = match["track_id"]
            previous = last_track_by_identity.get(identity)
            if previous is not None and previous != track_id:
                identity_switches += 1
            last_track_by_identity[identity] = track_id

    identities = sorted({match["identity_id"] for match in matches})
    track_ids = sorted({match["track_id"] for match in matches})
    identity_index = {value: index for index, value in enumerate(identities)}
    track_index = {value: index for index, value in enumerate(track_ids)}
    weights = [[0] * len(track_ids) for _ in identities]
    for match in matches:
        weights[identity_index[match["identity_id"]]][track_index[match["track_id"]]] += 1
    identity_true_positive = _maximum_assignment_weight(weights)
    identity_false_positive = predicted_total - identity_true_positive
    identity_false_negative = ground_truth_total - identity_true_positive
    idf1_denominator = (
        2 * identity_true_positive + identity_false_positive + identity_false_negative
    )
    return {
        "ground_truth_detections": ground_truth_total,
        "predicted_detections": predicted_total,
        "matches": len(matches),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "identity_switches": identity_switches,
        "mota": (
            1 - (false_negatives + false_positives + identity_switches) / ground_truth_total
            if ground_truth_total
            else 0
        ),
        "motp": sum(match["iou"] for match in matches) / len(matches) if matches else None,
        "identity_true_positive": identity_true_positive,
        "identity_false_positive": identity_false_positive,
        "identity_false_negative": identity_false_negative,
        "idf1": 2 * identity_true_positive / idf1_denominator if idf1_denominator else 0,
    }


def person_count_metrics(expected_counts, predicted_counts):
    """Compute presence and count metrics from manually reviewed frame counts."""
    if len(expected_counts) != len(predicted_counts) or not expected_counts:
        raise ValueError("expected and predicted counts must have the same non-zero length")
    if any(type(value) is not int or value < 0 for value in [*expected_counts, *predicted_counts]):
        raise ValueError("person counts must be non-negative integers")

    pairs = list(zip(expected_counts, predicted_counts))
    true_positive = sum(expected > 0 and predicted > 0 for expected, predicted in pairs)
    false_positive = sum(expected == 0 and predicted > 0 for expected, predicted in pairs)
    true_negative = sum(expected == 0 and predicted == 0 for expected, predicted in pairs)
    false_negative = sum(expected > 0 and predicted == 0 for expected, predicted in pairs)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0
    recall = true_positive / recall_denominator if recall_denominator else 0
    absolute_errors = [abs(expected - predicted) for expected, predicted in pairs]
    exact_matches = sum(expected == predicted for expected, predicted in pairs)
    return {
        "samples": len(expected_counts),
        "presence": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        },
        "count": {
            "exact_match_frames": exact_matches,
            "exact_match_rate": exact_matches / len(expected_counts),
            "overcount_frames": sum(predicted > expected for expected, predicted in pairs),
            "undercount_frames": sum(predicted < expected for expected, predicted in pairs),
            "mean_absolute_error": sum(absolute_errors) / len(absolute_errors),
            "maximum_absolute_error": max(absolute_errors),
        },
    }
