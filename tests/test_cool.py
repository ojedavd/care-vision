import copy
import hashlib

import pytest

from care_vision.check_loop import CheckFailed, check_records
from care_vision.common.cool import OPERATIONS, validate_evidence, verify_runtime


def evidence():
    build = "test fixture: KleidiCV -mcpu=neoverse-v2"
    return {
        "native_module": "/opt/cool/python_3.11/site-packages/cv2/python-3.11/cv2.so",
        "native_sha256": "a" * 64,
        "optimized": True,
        "build_information": build,
        "build_sha256": hashlib.sha256(build.encode()).hexdigest(),
        "loaded_libraries": [
            {"path": f"/opt/cool/cpp_sdk/lib/libopencv_{name}.so.5.1.0", "sha256": "b" * 64}
            for name in ("core", "imgproc", "imgcodecs")
        ],
        "executed_operations": sorted(OPERATIONS),
    }


def test_manifest_requires_native_paths_build_and_operations():
    validate_evidence(evidence(), "5.1.0-dev")


@pytest.mark.parametrize(
    "change",
    [
        {"native_module": "/tmp/.venv/site-packages/cv2.abi3.so"},
        {"optimized": False},
        {"build_information": "ordinary OpenCV"},
        {"build_sha256": "bad"},
        {"native_sha256": "bad"},
        {"loaded_libraries": []},
        {"executed_operations": ["resize"]},
    ],
)
def test_incomplete_or_standard_runtime_is_not_cool(change):
    with pytest.raises(RuntimeError):
        validate_evidence({**evidence(), **change}, "5.1.0-dev")


def test_major_version_four_cannot_pass():
    with pytest.raises(RuntimeError, match="major version 5"):
        validate_evidence(evidence(), "4.14.0")


def test_mixed_linked_libraries_cannot_pass():
    value = copy.deepcopy(evidence())
    value["loaded_libraries"].append(
        {"path": "/usr/lib/libopencv_core.so.5.1.0", "sha256": "c" * 64}
    )
    with pytest.raises(RuntimeError, match="exclusively"):
        validate_evidence(value, "5.1.0-dev")


def test_actual_local_standard_opencv_fails_closed():
    with pytest.raises(RuntimeError, match="standard OpenCV wheel"):
        verify_runtime()


def test_flag_without_cool_evidence_is_rejected():
    startup = {
        "event": "server_ready",
        "phase": 3,
        "cool_verified": True,
        "runtime": {
            "system": "Linux",
            "machine": "aarch64",
            "opencv_version": "5.1.0-dev",
            "ec2": {"architecture": "arm64"},
        },
    }
    with pytest.raises(CheckFailed, match="native cv2 module"):
        check_records([], [startup])
