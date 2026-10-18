import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "artifacts/pose-observation-live-20261014/sit-return-live-1"
spec = importlib.util.spec_from_file_location(
    "pose_live_timing", ROOT / "tools/diagnose_pose_live_timing.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_live_rejection_is_receiver_window_and_does_not_rewrite_evidence():
    paths = [TRIAL / name for name in ("edge.jsonl", "cloud.jsonl", "strict-report.json")]
    before = [path.read_bytes() for path in paths]
    report = module.diagnose(TRIAL)
    for policy in report["policies"]:
        assert all(w["within_original_tolerance"] for w in policy["send"]["windows"])
    rejected = [(p["policy_seq"], w["fps"]) for p in report["policies"]
                for w in p["receive"]["windows"] if not w["within_original_tolerance"]]
    assert rejected == [(0, 2.4)]
    assert not report["full_demo_passed"]
    assert before == [path.read_bytes() for path in paths]


def test_all_eighteen_missing_references_reproduce_from_saved_metadata():
    context = module.diagnose(TRIAL)["temporal_context"]
    assert context["categories"] == {
        "startup_context": 2,
        "view_change_context": 6,
        "same_view_no_reference_in_400_800ms": 10,
    }
    assert not context["reference_mismatches"]
    assert context["all_missing_context_have_zero_candidates"]


def test_diagnostic_detects_forged_reference_across_roi_change():
    observations = [r for r in module.read_rows(TRIAL / "cloud.jsonl")
                    if r["event"] == "observation"]
    altered = copy.deepcopy(observations)
    first_roi = next(r for r in altered if r["view_kind"] == "roi")
    first_roi["reference_frame_id"] = observations[0]["frame_id"]
    context = module.temporal_context(altered)
    assert context["reference_mismatches"] == [{
        "capture_seq": first_roi["capture_seq"], "expected": None,
        "actual": observations[0]["frame_id"],
    }]
