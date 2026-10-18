import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "tools/compare_tracking_cache.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cache(base, *, mismatched=False):
    frame = {
        "run": "trial",
        "frame_index": 0,
        "frame_id": "trial:1",
        "captured_mono_ns": 100,
        "jpeg_sha256": "pinned",
    }
    (base / "plan.json").write_text(json.dumps({"frames": [frame]}))
    row = {**frame, "observation": {"person_detection_status": "ok", "persons": []}}
    if mismatched:
        row["captured_mono_ns"] = 101
    (base / "detections.jsonl").write_text(json.dumps(row) + "\n")
    (base / "cache-runtime.json").write_text(
        json.dumps(
            {
                "cache_sha256": module.sha(base / "detections.jsonl"),
                "plan_sha256": module.sha(base / "plan.json"),
            }
        )
    )


def test_empty_detection_frames_are_retained(tmp_path):
    cache(tmp_path)
    _, _, rows = module.load_cache(tmp_path)
    assert rows[("trial", 0)]["observation"]["persons"] == []


def test_changed_timestamp_rejected_even_with_valid_file_digest(tmp_path):
    cache(tmp_path, mismatched=True)
    with pytest.raises(ValueError, match="metadata"):
        module.load_cache(tmp_path)


def test_corrupted_detection_file_rejected(tmp_path):
    cache(tmp_path)
    with (tmp_path / "detections.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="digest"):
        module.load_cache(tmp_path)


def test_changed_schedule_rejected(tmp_path):
    cache(tmp_path)
    (tmp_path / "plan.json").write_text('{"frames": []}')
    with pytest.raises(ValueError, match="plan digest"):
        module.load_cache(tmp_path)


def test_compatibility_edit_preserves_numpy_float64_and_logic(tmp_path):
    source = tmp_path / "bytetrack-upstream/yolox/tracker"
    source.mkdir(parents=True)
    original = "import torch\nimport torch.nn.functional as F\na = np.float\nb = np.float64\n"
    (source / "byte_tracker.py").write_text(original)
    edits = module.install_numpy_adapter(tmp_path)
    adapted = (tmp_path / "bytetrack-numpy/yolox/tracker/byte_tracker.py").read_text()
    assert adapted == "a = float\nb = np.float64\n"
    assert (source / "byte_tracker.py").read_text() == original
    assert edits[0]["upstream_sha256"] != edits[0]["adapted_sha256"]
