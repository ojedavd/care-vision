"""Evidence that the motion workload is executing the installed COOL SDK.

This verifies the loaded distribution and operations, not a speedup or that
every call selected a particular optimized kernel. No package-name-only check.
"""

import hashlib
import json
import re
from pathlib import Path

OPERATIONS = {"imdecode", "resize", "cvtColor", "absdiff", "mean"}
SDK = Path("/opt/cool")


def validate_evidence(evidence: dict, version: str):
    if not version.startswith("5."):
        raise RuntimeError("COOL phase requires OpenCV major version 5")
    native = Path(evidence.get("native_module", ""))
    if not native.is_relative_to(SDK) or native.suffix != ".so":
        raise RuntimeError("native cv2 module is not loaded from the COOL SDK")
    if not evidence.get("optimized"):
        raise RuntimeError("OpenCV optimizations are disabled")
    build = evidence.get("build_information", "")
    if "KleidiCV" not in build or "neoverse-v2" not in build:
        raise RuntimeError("expected COOL Graviton4 build markers are absent")
    expected_build_hash = hashlib.sha256(build.encode()).hexdigest()
    if evidence.get("build_sha256") != expected_build_hash:
        raise RuntimeError("build information hash mismatch")
    libraries = evidence.get("loaded_libraries", [])
    for component in ("core", "imgproc", "imgcodecs"):
        matches = [
            item
            for item in libraries
            if Path(item["path"]).name.startswith(f"libopencv_{component}.so.")
        ]
        if not matches or any(
            not Path(item["path"]).is_relative_to(SDK / "cpp_sdk/lib") for item in matches
        ):
            raise RuntimeError(f"{component} is not loaded exclusively from the COOL SDK")
    for digest in [
        evidence.get("native_sha256", ""),
        *(item.get("sha256", "") for item in libraries),
    ]:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("missing binary hash")
    if set(evidence.get("executed_operations", [])) != OPERATIONS:
        raise RuntimeError("required vision operations were not verified")


def verify_runtime() -> dict:
    import cv2
    import numpy as np

    native = Path(getattr(cv2, "_native", cv2).__file__).resolve()
    if not native.is_relative_to(SDK):
        raise RuntimeError("standard OpenCV wheel cannot satisfy phase 3")
    # Exercise exactly the operation types used by MotionVision in this process.
    image = np.full((360, 640, 3), 64, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("COOL JPEG preparation failed")
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2.resize(decoded, (160, 90)), cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray, gray + np.uint8(10))
    if gray.shape != (90, 160) or cv2.mean(difference)[0] != 10:
        raise RuntimeError("COOL operation correctness probe failed")
    loaded = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libopencv_" in line
        }
    )
    libraries = [
        {
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path in loaded
        if any(f"libopencv_{name}.so." in path for name in ("core", "imgproc", "imgcodecs"))
    ]
    build = cv2.getBuildInformation()
    evidence = {
        "native_module": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "build_information": build,
        "build_sha256": hashlib.sha256(build.encode()).hexdigest(),
        "loaded_libraries": libraries,
        "optimized": cv2.useOptimized(),
        "cpu_features": cv2.getCPUFeaturesLine(),
        "executed_operations": sorted(OPERATIONS),
        "per_kernel_acceleration_measured": False,
        "distribution_source": "preinstalled /opt/cool SDK on the supplied AMI",
    }
    validate_evidence(evidence, cv2.__version__)
    evidence["runtime_id"] = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode()
    ).hexdigest()
    evidence["verified"] = True
    return evidence
