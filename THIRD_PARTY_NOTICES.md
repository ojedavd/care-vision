# External dependencies and assets

CARE-Vision has no selected open-source license. Private competition review does
not assign a new license to the project's code or third-party material.

Python dependencies and their exact resolved versions/hashes are recorded in
`pyproject.toml` and `uv.lock`. Their upstream license terms apply. Dependencies
are installed from their registries rather than vendored into this repository.

The COOL SDK, ONNX model weights, datasets, personal recordings, native libraries
and benchmark input bundles are excluded. Access/provisioning and redistribution
rights for any externally supplied asset must be checked separately. Do not infer
a model weight's redistribution terms solely from a parent repository's license.

Selected upstream references:

- OpenCV: https://github.com/opencv/opencv
- OpenCV Zoo: https://github.com/opencv/opencv_zoo
- MMPose: https://github.com/open-mmlab/mmpose
- Gitleaks (separate development tool): https://github.com/gitleaks/gitleaks

The optional explicit Lab model setup fetches YOLOX-S from the pinned OpenCV Zoo
revision and RTMPose-t Body7 from the official OpenMMLab ONNX package, checking
their SHA-256 digests. See [asset URLs and hashes](docs/full-pipeline-lab.md).
These assets stay in an external cache and are not redistributed with this code.
