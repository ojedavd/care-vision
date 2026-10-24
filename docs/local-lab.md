# Local Lab: visible Edge and agent behavior

This guide describes **Local · quick motion demo**. For real person/pose inference
on this computer or your own Graviton, use the
[two-mode full pipeline guide](full-pipeline-lab.md). The default remains local;
an AWS run requires an explicit private profile and selection.

Run the data-free, English-language lab from a fresh clone:

```sh
git clone https://github.com/ojedavd/care-vision.git
cd care-vision
uv sync --frozen
uv run --frozen care-lab
```

Use macOS or Linux, Python 3.11 and `uv`. Native Windows is not supported by
the current POSIX worker-group lifecycle. The command opens your browser at
`http://127.0.0.1:8781`. Choose **Start synthetic run**. If the browser does not
open, use the printed URL. If the port is occupied:

```sh
uv run --frozen care-lab --port 0 --no-browser
```

No AWS account, camera, private credentials, model weights or existing recording
is required. Initial dependency installation needs Internet access. For the generated demo, no camera is used. The default
local OpenCV wheel is standard OpenCV, **not COOL**. The fixture contains a moving
rectangle, not a person. A successful run verifies the local observation/action
loop; it does not establish fall-detection accuracy or cloud performance.

## Connect a local webcam

In **Lab & judges**, choose **Local webcam / USB camera**, a device index (usually
0) and a duration. Preview is off by default. Optionally enable **local preview**,
then press **Start camera**. Selecting an input alone does not open hardware.
The OS may ask for camera permission. Input errors remain visible; there is no
silent fallback to a generated scene or to AWS.

This local motion diagnostic sends frames to a freshly created vision process on
`127.0.0.1`. It uses standard OpenCV, not models, person/fall classification or
COOL. It does not connect to the project owner's AWS instance or use AWS credits.
The source and processing location are displayed explicitly. The primary claimed
competition CV workload remains on Graviton/OpenCV 5/COOL; this local diagnostic
is not a replacement for that pipeline.

A camera session is bounded to the selected 30, 45, 60 or 120 seconds (the internal
worker supports 1..120 seconds). Its watchdog allows bounded startup/drain and
cloud checks; input capture remains limited to the selected duration.
Preview is capped at approximately 2 FPS on a separate bounded thread/pipe, with
old frames dropped without blocking camera acquisition. Images stay in memory,
expire after three seconds, and are cleared when the run ends or stops. No
personal image/video file is written. Structured motion/transport logs remain
in the private temporary evidence directory. Preview is the locally captured
image, not a rendering of cloud-processed pixels.

The fixed synthetic schedule has a full loop auditor; spontaneous camera movement
has no predetermined 2→20→2 outcome. A camera session's completion is shown
separately and is **not** labeled as a passed incident or accuracy test.

Camera worker tests use a generated virtual input; physical webcams and their OS
permission behavior still require a run on the target device. RTSP cameras remain
available through `care-edge`, but are not yet selectable in this lab UI.

## Does cloning connect to the owner's AWS instance?

**No.** There are no shared AWS credentials or project-owner Graviton destination.
The start API rejects arbitrary host, URL, profile and file-path fields. The
optional **My AWS Graviton · COOL** mode uses only the private profile supplied
when launching the Lab. It connects to the evaluator's own configured instance.

**This does not replace the CloudFront page.** The existing authenticated,
image-free judge portal is a separate deployment. Its code, origin, distribution
and access controls are not changed by running or cloning this local lab. A
future cloud version of these views must be adapted to authenticated backend
projections; this unauthenticated loopback API must not be deployed as that origin.

## Four English perspectives

- **Lab & judges:** the generated source preview, measured Edge capture and live
  transmit FPS, requested FPS, application bytes, buffer occupancy, cloud counts,
  motion observations, policy reasons and command inspection.
- **Caregiver:** a focused event summary, whether human review was requested,
  who has taken attention and whether a notification was sent.
- **Family:** a plain-language update and the human handoff, without technical
  metrics or continuous video.
- **Operations:** source availability, technical issues and unassigned review
  requests, kept distinct from incident classification.

**Live local run** reads actual logs from the lab-owned Edge and vision
processes. A measured chart separates capture, requested transmit and actual
live-frame transmission. Values can differ around changes and measurement-window
boundaries. The generated source preview reconstructs the latest transmitted
capture from the fixture; it is not a rendering of transmitted/processed pixels
or an inference overlay. Rates are samples from the Edge, not inferred from the
agent's requested rate. Counts distinguish live and historical evidence.

**Scenario walkthrough** uses explicitly scripted examples for verification,
unresolved human review and connection interruption. It does not run inference
or inject events into a live run. Taking/completing human review changes only the
browser's walkthrough state and appears in the other perspectives. It never
changes the detector's decision, contacts a person or sends an external alert.
The local perspectives preview product roles; tab visibility is not authorization.

## Inspect a real command

Select a numbered command to see its policy reason and source capture. A sensing
cycle requires a seen source observation, issued command, applied ACK and later
valid live evidence linked to that command. An ACK alone is insufficient.
Historical buffer retrieval instead shows its own ACK and full processing
completion. It is not counted as another live sensing cycle.

A generated run takes about 35 seconds. The final audit includes source sampling, actual
rate changes, command causality and pre-event buffer retrieval. On completion,
stop, failure or stale telemetry the UI ceases to indicate current evidence;
retained values remain available for inspection. Start again for a separate run.
**Stop** and terminal shutdown terminate only the lab-owned worker process group.
A 90-second watchdog bounds generated runs independently of browser polling;
camera runs use their configured duration plus 20 seconds.

## Local boundaries and future delivery

The lab binds only to `127.0.0.1`, validates Host/Origin and a local mutation token,
serves a fixed asset/API allowlist, accepts only bounded local device indices
and accepts no upload, URL or file-path parameters. Only its generated fixture
or an explicitly opted-in local camera has a preview route. Evidence goes to
an owner-only temporary directory printed by the command, outside the checkout;
retain it for debugging or delete it after stopping the server.

This local lab is **not a CloudFront origin or a production caregiver service**.
The existing authenticated, image-free judge portal remains separate and unchanged.
The primary claimed CV/controller architecture remains AWS Graviton/OpenCV 5/COOL.
Authenticated organization/household permissions, durable human-case history,
notification delivery receipts, escalation and production operation are later work.
Do not expose this loopback demo through a tunnel or publish its local API as a
caregiver service.

## Verification

```sh
uv run --frozen pytest -q tests/test_local_lab.py tests/test_local_camera_lab.py
```

This includes an actual generated Edge/vision run, preview access, measured
telemetry, command causality, historical completion, freshness, restart and stop.
It also exercises HTTP boundaries and same-origin mutations. The lab launches no
AWS jobs and does not modify vision/controller policy.
