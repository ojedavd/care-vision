"""Record the execution environment without reading credentials from EC2 metadata."""

import json
import platform
import urllib.request

import cv2


def ec2_identity() -> dict:
    base = "http://169.254.169.254/latest/"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        base + "api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with opener.open(request, timeout=3) as response:
        token = response.read().decode()
    request = urllib.request.Request(
        base + "dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with opener.open(request, timeout=3) as response:
        document = json.load(response)
    # Exclude account ID, private addresses, and all credential endpoints.
    return {
        key: document[key]
        for key in ("instanceId", "instanceType", "region", "architecture", "imageId")
    }


def runtime_info(phase=1) -> dict:
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "opencv_module": cv2.__file__,
    }
    if phase in {2, 3}:
        if info["system"] != "Linux" or info["machine"] not in {"aarch64", "arm64"}:
            raise RuntimeError("remote phase requires Linux ARM64 on EC2")
        info["ec2"] = ec2_identity()
        if info["ec2"]["architecture"] != "arm64":
            raise RuntimeError("EC2 metadata does not identify an ARM64 instance")
    if phase == 3:
        from care_vision.common.cool import verify_runtime

        info["cool"] = verify_runtime()
    return info
