from care_vision.cloud.posture_evidence import PostureEvidence


def sample(t, orientation=None, track="person"):
    return {
        "captured_mono_ns": round(t * 1e9),
        "confirmed_person_detected": orientation is not None,
        "primary_track_id": track,
        "primary_orientation": orientation,
    }


def test_recent_horizontal_posture_tolerates_short_detection_gap():
    evidence = PostureEvidence()
    for i in range(11):
        orientation = "horizontal" if i in (1, 2, 3, 4, 9, 10) else None
        evidence.observe(sample(i * 0.2, orientation))
    result = evidence.classify(2_000_000_000)
    assert result["outcome"] == "horizontal_posture_observed"
    assert result["horizontal_frames"] == 6


def test_upright_posture_and_insufficient_evidence_are_distinct():
    upright = PostureEvidence()
    sparse = PostureEvidence()
    for i in range(10):
        upright.observe(sample(i * 0.2, "vertical"))
        sparse.observe(sample(i * 0.2, "horizontal" if i < 5 else None))
    assert upright.classify(1_800_000_000)["outcome"] == "upright_posture_observed"
    assert sparse.classify(1_800_000_000)["outcome"] == "unresolved"


def test_tracks_cannot_be_combined_to_create_posture_evidence():
    evidence = PostureEvidence()
    for i in range(10):
        evidence.observe(sample(i * 0.2, "horizontal", "a" if i % 2 else "b"))
    assert evidence.classify(1_800_000_000)["outcome"] == "unresolved"
