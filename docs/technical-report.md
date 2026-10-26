# CARE-Vision
Adaptive Agentic Vision for Safer Independent Living

Technical report | 26 October 2026 | OpenCV AI Competition 2026
Victor Ojeda, Argentina | Individual project | AWS Compute Grant recipient


## 1. Problem, users and implemented architecture

Older adults living independently may need support after an unusual posture or suspected incident. Continuously transmitting maximum-quality video consumes bandwidth, cloud computation and privacy. CARE-Vision explores a narrower engineering question: can visual uncertainty change the evidence acquired next, while keeping the main vision workload on AWS Graviton?

Intended users are independently living older adults, their supporters and system operators. Current evaluation is a development study with one participant/view in the latest physical-camera trial. It does not establish performance in the intended population, clinical benefit or reliable fall detection.

The Edge Gateway acquires video, samples frames, buffers recent context, resizes/crops on command, encodes evidence and acknowledges applied sensing policies. Person detection, tracking, pose/posture analysis, temporal interpretation and the deterministic controller execute in AWS using OpenCV 5 + COOL. The Edge does not run the primary CV inference.

<!-- ARCHITECTURE -->

The public portal provides image-free saved episode review and a separately labeled illustrative workflow. Live operator controls remain authenticated. CloudFront reaches the supervised portal through a restricted VPC origin. The private CV WebSocket is reached by the Edge through an authorized SSH tunnel. Browser login and Start session launch a bounded cloud worker; they do not connect the camera. A separate local Edge must supply evidence. No LLM is used for incident decisions.

<!-- PAGE -->

## 2. OpenCV implementation and the agentic loop

The current demonstrated worker uses a COOL-supplied OpenCV 5.1.0-dev build on Linux aarch64, an AWS m8g.4xlarge, eight OpenCV threads and CLASSIC person inference with YOLOX-S. An external RTMPose model and scene configuration supply the experimental seated-pattern path. Structured observations carry validity, person/posture/temporal information and causal frame references to the controller.

The latest camera trial used nominal 25 FPS acquisition, 1280x720 full-frame JPEG transmission and requested rates of 2/5 FPS. It had no diagnostic byte-budget cap. Existing episode continuity was enabled; pose-driven ROI and alerts were disabled. The five-second seated criterion is an explicit scene-specific experimental option, off by default. It does not verify physical seat support.

| Capture-relative time | Evidence and controller behavior |
| --- | --- |
| 0-15.948 s | Baseline observation at requested 2 FPS |
| 15.948 s | Posture change opens VERIFYING; request 5 FPS |
| 22.068 s | Return sensing to 2 FPS; episode remains UNRESOLVED |
| 24.863 s | Sustained seated pattern closes episode; NORMAL |

Returning to baseline sensing is separate from resolving an episode. This explains why the earlier person-only portal could remain UNRESOLVED after sitting. The integrated pose-assisted path later closed the episode while the participant remained seated in the reviewed samples.

Each command must reference a valid causal observation; an Edge ACK must confirm application; a subsequent valid observation must come from a frame captured under the new policy. Two such chains passed in the latest run. Measured send-completion rates were 2.007598, 5.001160 and 2.001002 FPS. These are achieved transmission measurements, not camera rates or proof of resource savings. The companion causal-trace.json contains a minimized, derived trace with original evidence hashes.

Implemented action families include rate changes, full frame/ROI and historical-buffer requests. Only rate changes are exercised in this latest profile; historical trials cover other actions. Missing or uncertain evidence is not treated as recovery. Alert handling remains test-only, without external caregiver notification.

<!-- PAGE -->

## 3. Evaluation, failed trials and limits

Five completed runs from 21 October were re-audited on 22 October using original cloud and Edge logs: frame identity/order, capture timestamps, dimensions/view, uniqueness, count agreement, drain, command cause, ACK and later valid evidence. This was a reanalysis, not a new inference run.

| Run / profile | Sent = received = processed | Verified chains | Functional result |
| --- | --- | --- | --- |
| Recorded recovery / person-only | 140 | 2 | NORMAL, known recording |
| Physical camera, 90 s / person-only | 239 | 6 | UNRESOLVED at end |
| Recorded seated / integrated pose | 140 | 2 | NORMAL; closure 24.39 s |
| First physical integrated trial | 121 | 0 | UNKNOWN; episode not exercised |
| Restarted physical trial, 60 s | 140 | 2 | NORMAL; closure 24.863 s |

The final trial captured 1,501 frames locally and processed the 140 transmitted frames. No declared drops or connection errors occurred; two initial observations lacked sufficient temporal context. Transport success in the first integrated trial is not a pass of seated closure: that trial opened no episode. Activity review used sampled original frames, not continuous independent human labels.

A separate frozen constrained-link regression on known development recordings passed five conditions, with 414 matched frames and 20 causal commands. That experiment used WebP80, a 1.34 FPS regular full-frame cap, 112 KiB/s application admission and a 128 KiB/s evaluation proxy. It is a different sensing profile from the latest JPEG 2/5 FPS trial. Do not pool their results or attribute improvements to one factor when sampling and encoding changed together.

Retained failures include a stalled pong/backlog; a lower-quality JPEG profile that missed an expected test alert; WebP50 losing the original identity and failing recovery; an earlier continuity case with 311 transmitted versus 310 processed frames; and a geometric portal probe that achieved 14 FPS against 20 requested. HTTPS alone does not correct those workload/transport failures.

No independent precision, recall, F1, false-alert rate, fall-confirmation latency or billed-cost reduction has been established. The new fixed5/adaptive study below measures resource reductions with a retained failure; full incident-performance equivalence remains unestablished. The second physical descent after seated closure remains unevaluated. Different participants, rooms, lighting, occlusion, chair positions and outages need broader evaluation. Scene regions and the five-second criterion are development choices, not calibrated thresholds.

<!-- PAGE -->

## 4. Paired fixed/adaptive transmission study - 25 October

Eighteen recorded replays compare fixed 5 FPS against adaptive 2/5 FPS: three repeated pairs per known recovery, no-recovery and sitting recording. Both arms run person-only YOLOX-S640/CLASSIC, tracking and identical incident/recovery rules on the same Graviton + COOL. Pose, ROI and seated closure are disabled in both. No thresholds were tuned after execution.

| Known recording | Less JPEG | Less cloud CPU | Pairs passing all gates |
| --- | --- | --- | --- |
| Recovery | 53.54% | 53.12% | 3/3 |
| No recovery | 53.47% | 53.07% | 2/3 |
| Sitting | 47.17% | 46.61% | 3/3 |

Figures are medians of three paired percentage reductions, including every attempted pair. Direct process CPU counters bracket first frame receipt and confirmed drain; pre-first-frame model startup is excluded. JPEG counts payload, not WebSocket/SSH/TCP or control overhead. Full ranges, RSS, source hashes, inputs, method and scalar verifier are in the paired study attachment.

Eight of nine pairs met all registered gates. The second adaptive no-recovery run reconnected: 171 locally completed sends, 160 received/processed, including 19 historical frames. Eleven sends have no confirmed cloud receipt. It ended VERIFYING with zero expected test alerts, versus one in fixed5. The failed pair remains in summaries and its resources are connection-confounded. Lower resource use does not establish equal incident performance.

Sitting remained UNRESOLVED without alerts in both arms, a limitation rather than correct benign classification. Matched adaptive-minus-fixed event delays were +0.284 to +0.524 seconds for three recovery pairs and -0.160 to +0.400 seconds for two no-recovery pairs. The missing alert is not imputed. The two-second additional-delay margin is experimental, not a validated safety threshold.

The fixed controller needs an explicit evaluation adapter because requested FPS also acts as its logical baseline/verification label. Actual raw 5 FPS metadata is retained; a copied controller input carries logical 2/5 mode. Sensing proposals are logged as suppressed, with no physical change or invented ACK. Adaptive uses actual commands, applied ACKs and later evidence. This is a declared experimental intervention, not an unchanged deployed fixed protocol.

These known clips and repetitions are not independent incidents or dense ground truth. They do not establish general fall accuracy, a sustained 30 FPS/1080p comparison, isolated COOL gains or billed AWS savings. The separate 15 October pose/ROI resource study is not pooled with this person-only study.

<!-- PAGE -->

## 5. COOL on Graviton: execution and comparison

Runtime provenance records the native COOL module, aarch64 architecture, eight threads and the executing person/pose workload. Visible OpenCV version is 5.1.0-dev; the supplied build stamp is 2026-07-13T03:56:14Z. The exact SDK source revision is unknown. A dependency version such as KleidiCV must not be presented as the COOL SDK version.

Native OpenCV module SHA-256:
`cd6d9cfeefbbb352c4d814c3b52035f27bda6930f241df8bf4dac35b37c60518`

| Experiment on Graviton | Standard OpenCV 5 | COOL-supplied OpenCV 5 | Interpretation |
| --- | --- | --- | --- |
| Original pipeline, median elapsed | 18.036657 s | 18.085676 s | Output equivalence rejected |
| Shared-input-blob control, median forward | 67.701247 ms | 67.733231 ms | Exact tensor/detection equality; no speedup |

The original 17 October comparison used 240 input frames per process and 12 processes in total. Its output comparison disagreed on 102/240 frames; dependency/build differences also limit causal attribution. It cannot establish an equivalent-workload acceleration claim.

The later control isolated forward inference by supplying the same blob to both runtimes, across eight alternating runs of 240 frames. Raw tensors and detections were exactly equal on all 1,920 comparisons; measured forward medians were effectively similar with overlapping ranges. This resolves equivalence for that controlled stage, not for the original preprocessing/full pipeline.

The report supports execution of the core workload on the required Arm path and a measured, reproducible comparison method. It does not claim COOL acceleration, superior Arm/x86 performance, energy reduction or cost savings. The deployed main workload is Arm; the local Edge and developer workstation perform acquisition/transport or tooling rather than a parallel x86 inference baseline.

Benchmark method, input hashes, original measurement workers and scalar results accompany this report in benchmark-method.md and benchmark-tools/. Application replay instructions are in docs/portable-cloud-runtime-20261017.md. Private inputs are identified by hash in the retained evidence, but are not redistributed in the code archive. An entitled native COOL installation and approved external model assets are prerequisites. Blank-host SDK provisioning and compatibility with a different SDK build remain unvalidated.

<!-- PAGE -->

## 6. Build, deployment, tests and current availability

The historical seated demonstration is frozen separately from the new paired study and reconnect-review candidate. The 18-condition study used public application commit f1f55d7a1cc7837001dd5ec9c07c4671e3079f7b; exact source and 93 source hashes accompany it. The candidate correction adds durable explicit human-review status when an open episode is restored. It does not relink physical identity or recover incident sensitivity. Candidate source and a separate Graviton controlled-close evaluation are retained. Original: 173 received, 172 processed, one explicit shutdown drop, no review event. Corrected: 170 received, 169 processed, one explicit shutdown drop, review emitted before new evidence and persisted. Both kept identity unresolved and emitted no test alert. This is application-close evaluation, not WAN-loss validation or a deployed production fix.

Python dependencies are pinned in uv.lock. For source review and focused tests, use Python 3.11 and the locked development environment. The default standard-cv dependency is for local tools/tests; it is not evidence of COOL execution. The cloud environment must omit the default standard OpenCV group and load the entitled COOL libraries instead.

```sh
uv sync --frozen
uv run --frozen python -m pytest -q \
  tests/test_judge_https.py tests/test_judge_portal.py \
  tests/test_judge_seated_profile.py tests/test_seated_closure.py \
  tests/test_seated_reopening_reference.py tests/test_s_only_geometry.py \
  tests/test_session_shutdown_accounting.py
```

```sh
UV_PROJECT_ENVIRONMENT=.venv-cloud uv sync --frozen --no-default-groups
PYTHONPATH=/opt/cool/python_3.11/site-packages \
LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib \
.venv-cloud/bin/python -c \
 'from care_vision.common.cool import verify_runtime; print(verify_runtime()["verified"])'
```

See docs/judge-https-deployment.md for the private-origin deployment, authenticated roles, security checks and rollback; docs/demo-profile-20261021.md for the selected options. The operator supplies a private cloud-options JSON with experimental-s-only-5fps=true, pose-shadow-config and experimental-seated-closure-seconds=5. Camera credentials, models, scene configuration, COOL entitlement and SSH identity are external. Start the bounded cloud worker, connect the separate Edge, verify fresh evidence and causal ACKs, then stop and verify drain/stale state. Never expose the CV WebSocket publicly to repair a missing stream.

The historical focused set passed 116 tests. The new reconnect-review candidate passed 48 focused continuity, projection, portal and shutdown tests; these test protocol behavior rather than fall-detection accuracy. The full suite retains failures and omitted private-fixture dependencies. A source archive alone does not reproduce the camera experiment or install the SDK on an empty host.

The earlier 23 October outage remains historical. On 25 October the HTTPS portal and saved episode review were responding; the new benchmark summary is available on the public landing page. Public review starts no camera, cloud worker or external notification. During isolated experiments the portal remained IDLE and protected configuration/state hashes were unchanged. Availability is a dated check, not an uptime guarantee or proof of live incident detection.

<!-- PAGE -->

## 7. Responsible use, evidence and judge demonstration

CARE-Vision is an experimental safety-support system, not a medical diagnostic product. A seated-pattern resolution means the controller met its current rule; it does not prove health, physical support or absence of risk. A stale connection must be visible and must not be interpreted as a safe scene. Alerts are disabled in the latest demo, and no real caregiver service is connected.

The public episode review displays estimated body pose/detection boxes on an empty background. Live telemetry excludes imagery, body coordinates, track identifiers and raw logs. Input frames are still processed in the cloud. Presentation head blur is applied after inference and reviewed by samples; it does not anonymize the original capture, clothing or room, and does not automatically protect a live screen-share. Personal footage remains restricted to authorized private judge review. Source archives exclude recordings, model weights, SDK binaries, credentials and scene calibration.

Operator/viewer separation, explicit opt-ins, bounded start/stop, freshness and cleanup make operation observable. A historical local synthetic test exercised 13 operator controls and 27 focused tests, including invalid opt-in rejection and clean interruption. That does not establish caregiver usability, remote emergency intervention or production safety.

| Companion item | What a judge can inspect |
| --- | --- |
| Architecture figure and causal trace | Edge/cloud boundary; two commands with ACK and later valid evidence |
| Paired study attachment | Full method, 18 runs, all outcomes, failures, source hashes and offline verifier |
| Frozen source archives | Historical implementation and separately identified review-status candidate |
| Updated presentation and private video | Seven editable slides; dated demonstration with new benchmark and failure ending |

The video identifies the individual team, shows architecture, application behavior, principal results and limitations. The 60.04 s camera segment reconstructs telemetry from original frames/logs; it is not a screen recording of the portal. Earlier footage remains marked with its date/profile. Do not add the new 140 frames to historical totals displayed elsewhere in the video. Narration can accompany the captioned file during a live presentation.

The current public code and saved episode review are linked in Devpost. Original personal recordings and the private presentation video retain restricted access. The draft has completed Additional info with both special awards selected; final submission and terms acceptance remain the owner's pending action. No public/unlisted publication is implied by this package. If using a screen-share instead of the endpoint, arrange access and rehearse a functioning session; a local video file alone is not an arranged live demonstration.

The original 8/9 benchmark and its missed alert remain unchanged by the later review-status correction. It preserves the distinction between implemented behavior, a successful development trial, historical failures and remaining delivery conditions.
