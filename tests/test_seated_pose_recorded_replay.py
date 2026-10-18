import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "seated_pose_replay", ROOT / "tools/replay_seated_pose_candidates.py"
)
replayer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replayer)


def test_original_known_confusers_filtered_with_recorded_references():
    result = replayer.replay_recorded(
        ROOT / "evaluation/seated-pose-20261014/recorded-plan.json",
        ROOT / "artifacts/seated-pose-temporal-20261014/recorded-poses",
    )
    by_key = {(r["run"], r["model"]): r for r in result["conditions"]}
    for model in ("RTMPose", "YOLO11n-pose"):
        sitting = by_key[("seated-hold-1", model)]
        assert sitting["summary"]["pose_candidates"] > 0
        for run, count in (("crouch-near-seat-1", 8), ("kneel-near-seat-1", 11)):
            condition = by_key[(run, model)]
            assert condition["summary"]["geometric_candidates"] == count
            assert condition["summary"]["pose_candidates"] == 0
            source = ROOT / f"artifacts/seated-confusers-live-20261014/{run}/cloud.jsonl"
            opens = [
                r
                for r in map(json.loads, source.read_text().splitlines())
                if r["event"] == "incident_episode" and r["reason"] == "episode_opened"
            ]
            assert condition["episodes"][0]["recorded_reference_bbox"] == opens[0]["reference_bbox"]
        for row in result["conditions"]:
            for sample in row["timeline"]:
                assert not sample["seat_support_verified"] and not sample["would_close_episode"]
                if sample["suggested_observation"]:
                    assert sample["suggested_observation"]["dispatched"] is False


def test_original_kneeling_counterexample_remains_pose_uncertain():
    result = replayer.replay_recorded(
        ROOT / "evaluation/seated-pose-20261014/recorded-plan.json",
        ROOT / "artifacts/seated-pose-temporal-20261014/recorded-poses",
    )
    for condition in result["conditions"]:
        if condition["run"] != "kneel-near-seat-1":
            continue
        sample = next(r for r in condition["timeline"] if r["capture_seq"] == 302)
        assert sample["observation_id"] == "c6369763-f4ed-4682-a505-0d174310f4c9"
        assert sample["geometric_candidate"] and not sample["seated_pose_candidate"]
        assert sample["evidence_status"] == "INSUFFICIENT_EVIDENCE"
        assert sample["suggested_observation"] is not None
