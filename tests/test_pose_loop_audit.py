import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/pose-observation-loop-20261014/run-1/sit-return-1"
SOURCE = ROOT / "artifacts/reference-recovery-live-20261014/sit-return-1"
spec = importlib.util.spec_from_file_location(
    "pose_loop_audit", ROOT / "tools/audit_pose_observation_loop.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_historical_evidence_is_accounted_separately_from_live_loop():
    report = module.summarize(FOLDER, SOURCE, "sit-return-1")
    assert report["checks"]["same_sent_received_processed_headers"]
    assert report["historical_sent_received_processed"] == [1, 1, 1]
    assert report["sent_received_processed"] == [143, 143, 143]


def test_matching_frame_count_and_command_id_cannot_hide_different_crop(monkeypatch):
    read = module.records

    def changed(path):
        rows = read(path)
        if path == FOLDER / "edge.jsonl":
            rows = copy.deepcopy(rows)
            row = next(r for r in rows if r["event"] == "frame_sent"
                       and r.get("view_kind") == "roi")
            row["roi_x"] += 1
        return rows

    monkeypatch.setattr(module, "records", changed)
    report = module.summarize(FOLDER, SOURCE, "sit-return-1")
    assert report["sent_received_processed"] == [143, 143, 143]
    assert not report["checks"]["same_sent_received_processed_headers"]
    assert not report["scoped_loop_passed"]
