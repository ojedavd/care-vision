# Choose local processing or your own AWS Graviton

The English Lab offers three explicitly selected modes. Cloning never connects
to the project owner's AWS account, Graviton, CloudFront service or credentials.

| Mode | Edge | Vision and deterministic controller | Requirements |
| --- | --- | --- | --- |
| Local · quick motion demo | This computer | Local motion diagnostic | Generated scene or camera; no models |
| Local · full pipeline | This computer | Separate local service: YOLOX person detection, tracking, temporal episode policy and passive RTMPose | Download public models once; webcam or your own video |
| My AWS Graviton · COOL | This computer | Your Graviton: existing OpenCV 5 + COOL person/episode pipeline, pose observer and bounded pose observation policy | Your authorized instance, entitled installed SDK, external models/scene config and SSH profile |

The local full pipeline is a **functional reference**, not full equivalence to a
calibrated AWS deployment. It uses standard OpenCV, a portable passive pose
observer and the existing person/episode controller. Scene-specific pose ROI
actions and seating closure are not enabled in this reference. The competition's
primary CV workload remains on Graviton. Neither mode establishes validated fall
accuracy, medical decisions or a production notification service.

## 1. Full local reference — no AWS account

Use macOS or Linux with Python 3.11, `uv` and `curl`:

```sh
git clone https://github.com/ojedavd/care-vision.git
cd care-vision
uv sync --frozen
uv run --frozen care-lab --setup-models
uv run --frozen care-lab
```

Choose **Local · full pipeline**, **Local webcam / USB camera**, duration and
device index, then **Start camera**. Choosing a mode does not open hardware.
Preview is optional and memory-only. The separate local vision process checks
both model hashes and warms up inference before starting capture. Missing/corrupt
models fail the run; they never cause a silent motion-only fallback.

For your own recorded source:

```sh
uv run --frozen care-lab --video /ABSOLUTE/PATH/your-video.mp4
```

Select **Configured local video**. It is paced using the file FPS (up to 30) and
bounded by the selected duration. Its original timestamps are not a benchmark
replay; use the dedicated recorded runners for that purpose. No file path or
upload can be submitted through the browser API. The full pipeline sends 1280×720
representations, preserving the existing controller's expected input geometry.

Observe a single person in view, then inspect person counts, confirmed tracks,
pose samples and actual policy decisions. Ordinary activity need not generate a
command. The experimental person/episode profile uses 2→5 FPS, rather than the
motion fixture's 2→20 FPS. It never forces an event just to animate the panel.
Pose requires one confirmed person and usable input; missing/ambiguous pose is
shown as unavailable. Keypoint responses are not calibrated probabilities.

Models are downloaded from upstream into `~/.cache/care-vision/models`, outside
Git. The download command is explicit and uses HTTPS with certificate verification,
bounded sizes and SHA-256 checks; it never installs archive code or extracts paths.
It needs approximately 50 MB of downloads and 47 MiB for the two final ONNX files.

| Asset | Upstream | Expected SHA-256 |
| --- | --- | --- |
| YOLOX-S640 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox) | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| RTMPose-t Body7 | [Official OpenMMLab ONNX package](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip) | `a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f` |

Upstream asset terms apply; weights and SDK binaries are not redistributed here.
Use `--models /PRIVATE/models` for a different explicit model cache.

## 2. Prepare your own Graviton

This path uses **your own AWS account and instance**. It creates no AWS resources,
changes no security groups and installs no COOL SDK. Obtain the entitled COOL SDK
through the official competition/vendor process. The currently supported runtime
checks expect the evaluated Linux ARM64 Graviton4/OpenCV5 COOL build and its
CLASSIC DNN capability. A standard OpenCV wheel cannot satisfy the check. Setup
from a blank AMI and compatibility with other native SDK builds are not validated.

Prerequisites on your instance:

- Python 3.11 and uv 0.11.32; `curl`, Git and SSH access.
- Installed COOL at `/opt/cool/python_3.11/site-packages` and
  `/opt/cool/cpp_sdk/lib`, with the matching FFmpeg libraries at `/opt/ffmpeg/lib`.
- A trusted SSH host key in the evaluator's `known_hosts`. Verify its fingerprint
  through your own AWS/administrator channel; host-key verification stays enabled.
- An SSH alias/key belonging to you. The worker only binds remote 127.0.0.1;
  there is no public inference port or shared project credential.

On that instance, clone the **same source revision** as your local checkout:

```sh
git clone https://github.com/ojedavd/care-vision.git "$HOME/care-vision"
cd "$HOME/care-vision"
UV_PROJECT_ENVIRONMENT=.venv-cloud uv sync --frozen --no-default-groups --python /usr/bin/python3.11
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:/opt/cool/python_3.11/site-packages"
export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
.venv-cloud/bin/python -m care_vision.lab.models --directory "$HOME/care-assets/models"
.venv-cloud/bin/python -m care_vision.lab.pose_config --model "$HOME/care-assets/models/pose.onnx" --output "$HOME/care-assets/scene.json"
```

Keep the standard OpenCV wheel out of `.venv-cloud`. The default scene config is
**unknown**, with unreviewed placeholders: the pipeline can run, but scene-based
pose inference/actions will wait for eligible reviewed context and an active
episode. It must not invent a seat location. To enable the existing pose policy,
independently review the actual camera view and create a **new** config with
`--reviewed-scene --seat-region X Y WIDTH HEIGHT --hip-region X Y WIDTH HEIGHT`.
Both regions use normalized full-frame coordinates in [0,1] and need positive
size. Move the camera and the context must be reviewed again. The Lab expects
edge identity `edge-1` and 1280×720. The helper never overwrites an earlier config.
Scene-specific experimental seating closure and external test-alert thresholds
remain disabled; the enabled pose policy requests evidence, not safe closure.

Record the printed scene SHA-256 for the private profile. Person SHA-256 is in
the table above. All source files and the dependency lock are also compared to
the local checkout before input starts. Keep local/remote revisions in sync.

## 3. Connect the local Lab to your instance

Create `/PRIVATE/cloud-profile.json` **outside Git**, with your own values:

```json
{
  "ssh_host": "my-care-graviton",
  "ssh_identity": "/PRIVATE/my-graviton.pem",
  "remote_root": "/home/ubuntu/care-vision",
  "remote_python": "/home/ubuntu/care-vision/.venv-cloud/bin/python",
  "person_model": "/home/ubuntu/care-assets/models/person.onnx",
  "person_sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
  "pose_config": "/home/ubuntu/care-assets/scene.json",
  "pose_config_sha256": "REPLACE_WITH_THE_64_CHARACTER_SCENE_SHA256"
}
```

The alias must resolve using your existing SSH configuration. Paths are explicit;
no project-owner IP address, AMI, key, asset directory or calibration is supplied.
The profile and SSH key must be owner-only (`chmod 600`). On the local computer:

```sh
chmod 600 /PRIVATE/cloud-profile.json /PRIVATE/my-graviton.pem
uv run --frozen care-lab --cloud-profile /PRIVATE/cloud-profile.json --check-cloud
uv run --frozen care-lab --cloud-profile /PRIVATE/cloud-profile.json
```

`--check-cloud` verifies native COOL operations on EC2 ARM64, source/lock equality,
person/scene/pose hashes and model warm-up using blank images. It starts no camera,
Edge or long-running inference worker. A failed check points to private local
diagnostics. It never falls back to a standard wheel or a different host.

In the browser, choose **My AWS Graviton · COOL**, your camera or configured local
video, and **Start**. The run repeats checks, starts its own bounded cloud worker
on an ephemeral loopback port, opens an authenticated encrypted SSH tunnel, then
opens input. Selected visual frames go to **your configured instance**; commands
return over the same bidirectional protocol. Credentials, host paths and raw logs
are not served to the browser. There is no use of the owner's CloudFront endpoint.

## What the Lab shows and what Stop means

The Edge chart measures capture, requested transmission and actual transmission
separately. Cloud telemetry shows received/processed counts, person detections,
model readiness, scene context, passive pose, processing time, controller state,
policy reasons and command→ACK→new-perception chains. Remote timings are compared
only within the remote clock; Edge charts use their own local clock. A pose sample
is labeled with its capture number; it is not overlaid on a newer source frame.

Preview is opt-in, bounded, memory-only and expires. Camera images/video are not
saved by these runners. Structured per-run logs stay in owner-only local temporary
directories; they are not public artifacts. No recording is uploaded as a file.
In AWS mode selected frames necessarily leave the Edge for your instance.

Closing the browser does not bypass deadlines. Stop terminates the owned local
process group; SSH EOF stops the owned remote worker, with an independent remote
deadline as a backstop. A run lasts 30–120 seconds plus bounded startup/drain.
Your EC2 instance remains running. This is a bounded evaluation tool, not a
continuous multi-household service. Reconnection or stale data cannot establish
fresh cloud vision or a person's wellbeing. A completed session is not an accuracy
audit. The existing deployed CloudFront review portal remains separate.

## Verification scope

The local full mode was exercised with the actual upstream ONNX weights and an
existing controlled development recording. Tests cover private profiles, source
integrity, no fallback/no camera on remote failure and owned-process cleanup.
No new evaluator instance was supplied during implementation, and the owner's
Graviton was deliberately not used. Therefore the new SSH launcher has not yet
been accepted end-to-end on a second Graviton. The existing cloud pipeline has
its separately documented historical evidence; that evidence does not validate
this new launcher or provisioning a blank AMI.
