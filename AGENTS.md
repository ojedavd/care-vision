# CARE-Vision — Codex Project Context

Publication copy: use `docs/approved-proposal.md` as the text reference and the
aggregate `docs/obstacle-and-evidence-register.md`. Original experiments, videos,
credentials and exact private input profiles remain outside this repository.

## 1. Project

**Project:** CARE-Vision
**Subtitle:** Adaptive Agentic Vision for Safer Independent Living
**Competition:** OpenCV AI Competition 2026
**Competition paths:** COOL + Agentic Vision
**Team:** Individual project — Victor Ojeda, Argentina

CARE-Vision was selected for an **AWS Compute Grant** for the OpenCV AI Competition 2026.

The approved competition proposal should be treated as the project's scope and architectural reference.

Before making architectural changes, read the proposal and this file.

---

## 2. Problem

CARE-Vision is a computer vision safety-support system for older adults living independently.

The system aims to detect and intelligently verify potentially dangerous situations such as:

* falls,
* prolonged inactivity,
* unusual posture,
* unusual movement,
* lack of recovery after a suspected incident.

CARE-Vision is **not a medical diagnostic system**.

The objective is to provide useful incident information while reducing:

* unnecessary video transmission,
* cloud computation,
* bandwidth consumption,
* unnecessary exposure of private visual information.

---

## 3. Core Architectural Principle

The primary computer vision workload MUST run in AWS on **Graviton using OpenCV 5 + COOL**.

Do not move the main CV pipeline to the Edge Gateway without explicitly identifying that this conflicts with the approved competition architecture.

High-level architecture:

Camera
→ Edge Gateway
→ AWS Graviton
→ OpenCV 5 + COOL
→ Structured Observation
→ Agentic Controller
→ Action
→ Edge Gateway
→ New Visual Evidence
→ OpenCV 5 + COOL

This forms the closed loop:

PERCEIVE
→ UNDERSTAND
→ DECIDE
→ ACT
→ PERCEIVE AGAIN

---

## 4. Edge Gateway Responsibilities

The Edge Gateway is intentionally lightweight.

Its responsibilities include:

* camera/video acquisition,
* local high-density ring buffer,
* frame sampling,
* resizing,
* ROI cropping when requested,
* video/image encoding and compression,
* monitoring network conditions,
* transmitting visual evidence,
* receiving commands from AWS,
* executing sensing/transmission commands,
* store-and-forward during connectivity problems.

The Edge Gateway does NOT perform the primary computer vision workload.

An optional lightweight offline safety fallback may eventually be evaluated, but it must not replace the Graviton + COOL pipeline.

---

## 5. Camera Capture vs Transmission FPS

This distinction is fundamental.

The camera can capture continuously at a stable rate such as:

30 FPS

The Edge Gateway does not necessarily transmit all captured frames.

Instead, the Agentic Controller controls the:

**effective frame transmission rate**

For example:

Camera:

30 FPS capture

NORMAL:

2 FPS transmitted

SUSPICIOUS:

10 FPS transmitted

VERIFYING:

20 FPS transmitted

INCIDENT:

up to 30 FPS transmitted

Therefore:

SET_TRANSMIT_FPS(20)

means:

Select approximately 20 frames per second from the locally captured stream and transmit them to AWS.

It does NOT necessarily mean:

Reconfigure the physical camera to capture at 20 FPS.

Prefer frame sampling/dropping at the Edge over dynamically reconfiguring camera hardware.

The numeric FPS values above are initial examples and must eventually be calibrated experimentally.

---

## 6. Ring Buffer

The Edge Gateway maintains a short rolling high-density video buffer.

Example:

camera capture = 30 FPS
ring buffer = last 10 seconds

The purpose is to preserve recent visual context even when only a small number of frames are currently being transmitted.

Example:

The system is in NORMAL mode and transmitting 2 FPS.

AWS detects suspicious movement at time T.

The Agentic Controller can request:

SEND_BUFFER(
from = T - 5 seconds,
fps = 20
)

The Edge can then recover frames that occurred before the anomaly was detected.

This is important for fall verification because the beginning of the event may have occurred before the cloud increased observation.

---

## 7. Core Vision Pipeline

The main computer vision workload executes on AWS Graviton using:

**OpenCV 5 + COOL**

Planned capabilities include:

* person detection,
* person tracking,
* pose/posture analysis,
* motion analysis,
* temporal analysis,
* activity analysis,
* fall/incident detection,
* recovery detection,
* scene/context analysis where useful.

The system should not classify frames independently when temporal context is required.

For example:

standing
→ rapid downward motion
→ horizontal posture
→ remains on floor
→ no recovery

is much stronger evidence than a single image of a horizontal person.

---

## 8. Structured Observations

The vision pipeline should produce structured observations rather than directly controlling the camera.

Conceptual example:

{
"person_detected": true,
"track_id": 7,
"state": "possible_fall",
"fall_probability": 0.68,
"pose_confidence": 0.81,
"tracking_confidence": 0.93,
"occlusion": 0.42,
"motion": "downward",
"time_on_floor": 1.8,
"bbox": [400, 200, 500, 700]
}

These observations are passed to the Agentic Controller.

---

## 9. Agentic Controller

The Agentic Controller runs in AWS, preferably alongside the vision workload on Graviton.

Its purpose is NOT simply to describe what the camera sees.

Its purpose is to decide:

**What should the system observe next?**

The initial implementation does NOT require an LLM.

Prefer:

* deterministic policy,
* finite-state machine,
* temporal state,
* uncertainty estimation,
* explicit actions.

This makes the system:

* reproducible,
* measurable,
* testable,
* explainable,
* suitable for safety-oriented evaluation.

Do not introduce an LLM into critical incident decisions unless there is a demonstrated technical reason.

---

## 10. Agent State

Initial conceptual states:

NORMAL

SUSPICIOUS

POSSIBLE_FALL

VERIFYING

INCIDENT / ASSISTANCE_REQUIRED

RECOVERY

The exact state machine can evolve during implementation.

A possible transition is:

NORMAL
→ SUSPICIOUS
→ POSSIBLE_FALL
→ VERIFYING
→ INCIDENT

or:

VERIFYING
→ RECOVERY
→ NORMAL

---

## 11. Uncertainty

The agent should reason about uncertainty.

Possible inputs include:

* fall probability,
* pose confidence,
* tracking confidence,
* occlusion,
* motion consistency,
* time in current state,
* temporal evidence,
* network condition.

Do not assume that a single model confidence value is sufficient.

Initial uncertainty logic can be deterministic and rule-based.

Thresholds must eventually be calibrated experimentally.

---

## 12. Agent Action Space

Initial actions may include:

KEEP_OBSERVING

SET_TRANSMIT_FPS(n)

SET_RESOLUTION(...)

REQUEST_ROI(...)

REQUEST_FULL_FRAME

REQUEST_RECENT_BUFFER

WAIT

TRIGGER_ALERT

RETURN_TO_BASELINE

Optional future action:

CHANGE_CAMERA_VIEW

if PTZ or multi-camera hardware is introduced.

---

## 13. ROI Active Perception

The Edge does not need to detect the person locally to generate an ROI.

AWS already has the detected bounding box.

Example:

OpenCV + COOL:

person bbox =
[x=400, y=200, w=500, h=700]

Agent decides that more detail is required.

Command:

{
"action": "REQUEST_ROI",
"x": 400,
"y": 200,
"width": 500,
"height": 700,
"transmit_fps": 20
}

The Edge Gateway crops the corresponding region from subsequent locally captured frames and transmits it.

Cropping is an execution operation, not the primary CV inference workload.

---

## 14. Active Perception

Active Perception means:

**The system actively selects its next visual observation based on uncertainty about the current state.**

It is not simply camera zoom.

The agent may control:

* transmitted FPS,
* transmitted resolution,
* ROI,
* temporal window,
* pre-event buffer retrieval,
* full frame vs crop,
* camera viewpoint if supported later.

Initial MVP should prioritize digital active perception:

FPS

* resolution
* ROI
* temporal buffer

before introducing PTZ hardware.

---

## 15. Observation Policy

Initial conceptual policy:

| State                 | Observation policy        | Example transmission                            |
| --------------------- | ------------------------- | ----------------------------------------------- |
| NORMAL                | Low observation           | 2 FPS, reduced resolution                       |
| SUSPICIOUS            | Increased observation     | 10 FPS, 720p                                    |
| UNCERTAIN / VERIFYING | Targeted observation      | 20 FPS, high-quality person ROI + recent buffer |
| INCIDENT              | Maximum relevant evidence | Up to 30 FPS, high quality + alert              |
| RECOVERY              | Return to low observation | 2 FPS, reduced resolution                       |

These values are examples.

They are NOT final benchmark results or validated thresholds.

---

## 16. Example Agentic Loop

Example:

NORMAL

Edge:
2 FPS / low resolution

↓

OpenCV detects rapid downward movement.

↓

Observation:

possible_fall = true
confidence = 0.68

↓

Agent:

insufficient evidence

↓

Action:

SET_TRANSMIT_FPS(20)

REQUEST_ROI(person_bbox)

REQUEST_RECENT_BUFFER(last_5_seconds)

↓

Edge executes request.

↓

New evidence reaches AWS.

↓

OpenCV 5 + COOL processes new evidence.

↓

Agent evaluates updated state.

↓

Either:

TRIGGER_ALERT

or:

RETURN_TO_BASELINE

This is the key Agentic Vision behavior that must be visible in the final demonstration.

---

## 17. Event-Driven Streaming

CARE-Vision should not continuously send maximum-quality 1080p/30 FPS video unless required.

Normal operation uses low observation.

The agent escalates observation when additional evidence is useful.

Conceptually:

NORMAL
→ low observation

SUSPICIOUS
→ increased observation

UNCERTAINTY
→ targeted observation

INCIDENT
→ maximum relevant evidence

RECOVERY
→ low observation

This should reduce:

* transmitted frames,
* network bandwidth,
* AWS processing,
* unnecessary private video exposure.

---

## 18. Network Awareness

Home Internet connectivity cannot be assumed to be reliable.

The system should explicitly model:

GOOD

DEGRADED

OFFLINE

In GOOD mode:

normal active-perception behavior.

In DEGRADED mode:

adapt:

* transmitted FPS,
* resolution,
* compression,
* ROI size,
* amount of requested historical evidence.

In OFFLINE mode:

* detect connectivity loss,
* continue buffering locally,
* retain relevant evidence,
* retry connection,
* synchronize evidence after reconnection when appropriate.

The system must:

**fail safely, not fail silently.**

Do not pretend cloud vision remains operational while AWS is unreachable.

---

## 19. AWS Architecture

Current planned AWS components:

### AWS Graviton

Runs:

* OpenCV 5,
* COOL,
* core vision pipeline,
* Agentic Controller.

### Amazon S3

Possible uses:

* controlled evaluation data,
* selected encrypted incident evidence,
* benchmark artifacts.

### Amazon DynamoDB

Possible uses:

* system state,
* event metadata,
* temporal state.

### Amazon SQS

Possible use:

* decoupling events/messages where appropriate.

### Amazon CloudWatch

Used for:

* CPU,
* memory,
* processing latency,
* errors,
* throughput,
* benchmarking.

Do not add AWS services merely for architectural complexity.

If a different service is technically justified, explain the reason before introducing it.

---

## 20. Edge ↔ AWS Communication

The exact transport protocol was not fully fixed in the original proposal.

This remains an implementation decision.

The system needs two logical communication paths:

### Edge → AWS

Visual evidence / video frames.

### AWS → Edge

Agent commands.

Examples:

SET_TRANSMIT_FPS

SET_RESOLUTION

REQUEST_ROI

REQUEST_RECENT_BUFFER

RETURN_TO_BASELINE

When selecting protocols, prioritize:

* low latency,
* persistent connections where appropriate,
* simplicity,
* AWS compatibility,
* observability,
* reproducibility for the competition demo.

Do not assume RTSP, WebRTC, MQTT, WebSocket, or AWS IoT Core without evaluating the trade-offs first.

---

## 21. COOL Evaluation

The project follows the COOL competition path.

The claimed core workload must execute using:

OpenCV 5 + COOL
on
AWS Graviton

Benchmark structure:

### A. Baseline

Reference implementation.

Fixed sensing.

### B. Graviton + COOL

Equivalent fixed sensing.

Purpose:

isolate COOL / Arm impact.

### C. Graviton + COOL + Active Perception

Adaptive sensing.

Purpose:

measure the combined effect of optimized cloud vision and adaptive observation.

Do not fabricate benchmark improvements.

All claimed performance improvements must come from measured results.

---

## 22. Evaluation Metrics

### Computer Vision

* precision,
* recall,
* F1,
* false-positive rate,
* false-negative rate.

### Temporal

* incident detection latency,
* confirmation latency,
* recovery detection latency.

### COOL / Cloud

* FPS,
* processing latency,
* CPU utilization,
* memory,
* throughput,
* compute cost.

### Active Perception

Compare fixed sensing against adaptive sensing using:

* frames transmitted,
* frames processed,
* data transmitted,
* cloud compute,
* detection performance,
* confirmation latency.

### Network Resilience

Test:

* normal network,
* restricted bandwidth,
* increased latency,
* packet loss,
* complete outage.

---

## 23. Critical Experiment

A particularly important comparison is:

FIXED:

30 FPS
1080p
continuous

versus

CARE-Vision:

NORMAL → low rate
SUSPICIOUS → increased rate
VERIFYING → targeted high-quality evidence
INCIDENT → maximum relevant evidence
RECOVERY → baseline

Measure:

* total frames transmitted,
* GB transmitted,
* frames processed,
* AWS compute consumption,
* detection/confirmation latency,
* accuracy,
* cost.

The objective is not merely to show lower resource consumption.

The important question is:

**Can adaptive perception reduce visual processing and transmission while maintaining acceptable incident-detection performance?**

---

## 24. Judge Demonstration

The final demo should make the perception-decision-action loop visible.

Display:

* camera input,
* current vision state,
* agent state,
* confidence/uncertainty,
* agent decision,
* selected action,
* camera capture FPS,
* effective transmitted FPS,
* resolution,
* ROI,
* network usage,
* processing latency,
* Graviton/COOL metrics.

Example demo:

Normal activity
→ 2 FPS transmission

Suspicious movement
→ detection

Agent requests:
20 FPS + ROI + recent buffer

Edge changes transmission

OpenCV + COOL receives new evidence

Agent confirms or rejects incident

If confirmed:
caregiver notification

If recovery:
return to baseline.

---

## 25. Midpoint MVP

CARE-Vision received an AWS Compute Grant.

As of October 2, 2026, the competition's Devpost page lists the midpoint check-in window as October 7–14, 2026 (https://opencv26.devpost.com/). The original grant email and OpenCV competition page still list September 21–October 2; confirm the individual meeting time with the organizers.

The MVP should prioritize proving the architecture rather than implementing every final feature.

Minimum useful demonstration:

Camera/video source

→ Edge Gateway

→ low-rate frame transmission

→ AWS Graviton

→ OpenCV 5 + COOL processing

→ structured observation

→ simple deterministic Agentic Controller

→ command returned to Edge

→ transmission rate changes

→ new frames arrive

→ visible second perception cycle.

If possible, also demonstrate the ring buffer.

Do not block the MVP on:

* LLM integration,
* PTZ cameras,
* sophisticated reinforcement learning,
* production caregiver applications,
* unnecessary infrastructure.

---

## 26. Implementation Priority

Preferred implementation order:

1. Create basic repository structure.
2. Implement Edge Gateway with video source.
3. Implement fixed local capture.
4. Implement ring buffer.
5. Implement configurable frame sampler.
6. Establish Edge → AWS communication.
7. Run basic OpenCV workload on Graviton.
8. Produce structured observations.
9. Implement deterministic Agentic Controller.
10. Establish AWS → Edge command channel.
11. Implement SET_TRANSMIT_FPS.
12. Close the first complete perception-action-perception loop.
13. Add resolution control.
14. Add ROI requests.
15. Add historical ring-buffer requests.
16. Implement state machine.
17. Implement uncertainty policy.
18. Add network-aware behavior.
19. Add metrics and CloudWatch instrumentation.
20. Run baseline / COOL / Active Perception benchmarks.
21. Build judge-facing visualization/demo.
22. Only then consider optional advanced features.

---

## 27. LLM Policy

An LLM is NOT required for the initial Agentic Controller.

Critical incident decisions should initially remain deterministic and reproducible.

An LLM may later be considered for non-critical functionality such as:

* human-readable incident explanations,
* caregiver interaction,
* summarizing event history,
* natural-language interfaces.

Do not make an LLM responsible for determining whether a fall occurred unless the architecture is deliberately redesigned and validated.

---

## 28. Engineering Rules for Codex

Before implementing a major component:

1. Read this file.
2. Read the approved competition proposal.
3. Inspect existing code before creating new abstractions.
4. Preserve the Edge/AWS responsibility boundary.
5. Keep the core CV workload on Graviton + OpenCV 5 + COOL.
6. Prefer simple, measurable implementations before sophisticated ones.
7. Do not fabricate benchmark numbers.
8. Clearly distinguish experimental thresholds from measured thresholds.
9. Do not introduce unnecessary AWS services.
10. Do not introduce an LLM simply to make the project appear more agentic.
11. Preserve the perception → decision → action → new perception loop.
12. If a requested implementation conflicts with the approved architecture, identify the conflict before changing code.

---

## 29. Current Source of Truth

Priority when resolving architectural questions:

1. Approved CARE-Vision competition proposal.
2. This AGENTS.md.
3. Current implementation and documented engineering decisions.
4. New experimental findings.

If implementation evidence demonstrates that an architectural assumption should change, document:

* the original assumption,
* experimental evidence,
* proposed change,
* impact on the competition claims,

before making the change.

---

## 30. Immediate Goal

The immediate engineering goal is not to build the complete final product.

It is to prove this loop end-to-end:

Camera
→ Edge
→ AWS Graviton
→ OpenCV 5 + COOL
→ Observation
→ Agent
→ SET_TRANSMIT_FPS
→ Edge
→ Different transmission behavior
→ New observation

Once this works reliably, extend the system incrementally with ROI, ring-buffer retrieval, uncertainty, network resilience, benchmarking, and the final judge demonstration.

---

## 31. Experimental obstacle register

Read [`docs/obstacle-and-evidence-register.md`](docs/obstacle-and-evidence-register.md)
when preparing demos, summarizing progress, or continuing experiments. It indexes
the encountered limitations, mitigations, unresolved findings, and underlying
artifacts. After a meaningful new trial or fix, update that register with the
measured outcome and its scope, including failed runs. Do not turn a component
success into a claim that the full demo or fall detection passed.
