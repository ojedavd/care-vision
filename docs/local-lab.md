# Local Lab: visible Edge and agent behavior

Run the data-free, English-language lab from a fresh clone:

```sh
git clone https://github.com/ojedavd/care-vision.git
cd care-vision
uv sync --frozen
uv run --frozen care-lab
```

Use Python 3.11 and `uv`. The command opens your browser at
`http://127.0.0.1:8781`. Choose **Start synthetic run**. If the browser does not
open, use the printed URL. If the port is occupied:

```sh
uv run --frozen care-lab --port 0 --no-browser
```

No AWS account, camera, private credentials, model weights or existing recording
is required. Initial dependency installation needs Internet access. The default
local OpenCV wheel is standard OpenCV, **not COOL**. The fixture contains a moving
rectangle, not a person. A successful run verifies the local observation/action
loop; it does not establish fall-detection accuracy or cloud performance.

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

**Live synthetic run** reads actual logs from the lab-owned Edge and vision
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

A run takes about 35 seconds. The final audit includes source sampling, actual
rate changes, command causality and pre-event buffer retrieval. On completion,
stop, failure or stale telemetry the UI ceases to indicate current evidence;
retained values remain available for inspection. Start again for a separate run.
**Stop** and terminal shutdown terminate only the lab-owned worker process group.
A 90-second watchdog bounds each run independently of browser polling.

## Local boundaries and future delivery

The lab binds only to `127.0.0.1`, validates Host/Origin and a local mutation token,
serves a fixed asset/API allowlist and accepts no camera, upload or file-path
parameters. Only its own generated fixture has a preview route. Evidence goes to
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
uv run --frozen pytest -q tests/test_local_lab.py
```

This includes an actual generated Edge/vision run, preview access, measured
telemetry, command causality, historical completion, freshness, restart and stop.
It also exercises HTTP boundaries and same-origin mutations. The lab launches no
AWS jobs and does not modify vision/controller policy.
