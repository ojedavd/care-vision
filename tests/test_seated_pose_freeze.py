import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reserved_prepare", Path(__file__).parents[1] / "tools/prepare_seated_pose_reserved.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def freeze(root):
    names = (
        "src/care_vision/cloud/seated_pose_candidate.py",
        "src/care_vision/cloud/seated_candidate.py",
        "tools/run_tapo_spoken_trial.py",
    )
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"frozen")
    return {
        "schema_version": 1,
        "shadow_only": True,
        "file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        },
    }


def test_freeze_detects_changed_or_missing_code_before_capture(tmp_path):
    manifest = freeze(tmp_path)
    assert tool.verify_freeze(manifest, tmp_path)
    p = tmp_path / "src/care_vision/cloud/seated_pose_candidate.py"
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen inputs changed"):
        tool.verify_freeze(manifest, tmp_path)
    p.unlink()
    with pytest.raises(ValueError):
        tool.verify_freeze(manifest, tmp_path)


def test_empty_manifest_cannot_claim_a_frozen_shadow_version(tmp_path):
    with pytest.raises(ValueError):
        tool.verify_freeze({"schema_version": 1, "shadow_only": True, "file_sha256": {}}, tmp_path)
