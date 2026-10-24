# Obstacle and evidence register — code-only summary

Only aggregate engineering results are published here. Raw recordings, frame or
person traces, state snapshots, host/account identifiers and private input manifests
remain outside GitHub. These are known development scenes, not an independent
validation of incident accuracy or clinical benefit.

| Trial | Outcome | Scope / retained limitation |
| --- | --- | --- |
| Archived v3 five-condition replay | 758/758 frames processed; 20 causal commands | Known scenes on the existing Graviton/COOL host |
| v4 five-condition replay | 4/5 conditions; 762 received = 761 processed + 1 explicit shutdown drop | Pong timeout and reconnect; historical cause not proven |
| JPEG quality25 + byte pacing | Transport processed 638/638; functional 4/5 | Expected TEST alert missing; rejected profile |
| WebP quality50 + byte pacing | Transport processed 653/653; functional 4/5 | Original identity lost, recovery remained unresolved; rejected profile |
| WebP quality80 + regular full-frame1.34FPS | 5/5; 414/414 processed; no drops or connection errors; 20/20 causal commands | Quality and sampling changed together; controlled local proxy limit; no general WAN/accuracy claim |
| Local outage/buffer recovery | Reconnection observed | One tested recovery lacked 766.21ms of the requested prefix; not lossless history recovery |
| COOL comparisons | Core SDK workload on Graviton documented | No acceleration demonstrated; fresh SDK provisioning and exact upstream native revision remain limitations |

The selected experimental profile passed recovery by sustained evidence of the
same track, a separate TEST-alert scenario, continued observation without recovery,
and two pose/ROI conditions. It used OpenCV5 + COOL / CLASSIC, eight threads,
1280×720, WebP80, a112KiB/s application admission budget and a128KiB/s local TCP
proxy before the SSH tunnel. The alert contacted no caregiver. Targets2/5FPS are
requests; full-frame sampling was capped at1.34FPS. Legacy status cap fields describe
RTT limiting; actual transmit FPS and experimental frame headers expose the extra cap.

## Rules for future evaluation

- Freeze the source/profile before a reserved cohort. Annotate real activities
  before inspecting predictions; clips used for tuning become development inputs.
- Keep scene calibration separate. A different viewpoint has unknown context until
  independently calibrated; no inherited seat/current label from an earlier view.
- Keep uncertainty, explicit drops, identity loss and failed conditions visible.
  A newly tracked person cannot establish recovery of an earlier person by assumption.
- Record new personal scenes outside GitHub. No real fall enactment is needed for
  the proposed everyday-activity negative clips; those negatives cannot establish
  incident recall/F1.
- A demo needs a recent rehearsal with its exact chosen source and network conditions.
  Private presentation copies and blur need review; post-processing does not protect
  a raw live display or anonymize original inference inputs.

The original detailed register and raw evidence remain in the owner's private
experiment workspace. The repository adds privacy guards and sanitized documentation;
its own test results must be reported separately from archived AWS measurements.

## Publication CI: wall-clock retention assertion

The first private GitHub Actions run passed privacy/protocol/sampling checks and
29 of 30 synthetic/core tests, including the closed loop. The offline spool test
failed because it assumed two seconds of video always complete within a two-second
capture interval: CI retained seven samples versus the test's fixed maximum five.
Retention uses monotonic capture timestamps; SQLite/encryption and runner load can
extend playback. The test now bounds sample count using measured capture elapsed
time and retains its restart, identity and decryption checks. Core source is
unchanged. This is a test portability correction, not a new AWS/network result.

## Private live review deployment

A new image-free authenticated portal was deployed separately on the existing
Graviton host, on loopback only. It launches a bounded phase3 worker without
changing the original CV/controller modules. Generated geometric frames originated
at the local Edge; no personal videos were used or uploaded. Initial cloud launch
failed because resolving the venv Python symlink selected system Python without
websockets; preserving the explicit interpreter fixed that launch. An earlier uv
preflight falsely rejected the platform suffix of the pinned uv version; the
private procedure now compares version tokens. Both failures remain documented.

The real COOL motion/transport probe processed383/383 frames, no declared drops,
with two observed complete live command chains and an eventual NORMAL state.
The full auditor **failed** the requested20FPS check (achieved14FPS). This probe
uses different launch/transport settings from the frozen WebP80 regression; no
full regression success, person/fall accuracy or COOL acceleration is claimed.
Original protected episode files remained unchanged. The current portal has no
visual preview or external HTTPS URL; viewer access is role-based rather than
individual accounts. See [portal guide](private-judge-portal.md).

## HTTPS preparation while AWS console session is expired

Prepared a disabled-by-default CloudFront/VPC-origin template, cache-off and
viewer Host/Origin/cookie forwarding, with a dedicated ingress group restricted
to the verified AWS-managed origin group. An opt-in RFC1918-only portal bind
requires an HTTPS viewer origin; the default remains loopback. No real account
identifiers, credentials or recordings are in the template. The optional schema
checker is pinned and locked; previously locked dependency versions are retained.

28 local tests and cfn-lint1.57.2 passed. These are local preparation checks, not
an AWS deployment or eligibility result. The console session expired; authenticated
network/AZ/subnet/group-union discovery is pending. No CloudFront resource, group
attachment, ingress rule or new external endpoint was created. The existing SSH
portal still denies unauthenticated API access. See [HTTPS runbook](judge-https-deployment.md).

## HTTPS account staging after restored sign-in

The authenticated preflight passed for the existing Arm instance and supported AZ.
The original subnet is publicly routed; AWS reports the VPC origin deployed
without modifying shared routes, the public address or SSH. A disabled distribution
and dedicated no-ingress group were created. External ingress, group attachment,
distribution enablement and external acceptance remain pending; this is not yet a
working web endpoint.

A separate supervised private-interface portal passed role/Host/Origin/CSRF/cookie,
no-store/private-file-route and real COOL worker start/stop checks. The approved
existing person-model hash matched; native CLASSIC inference on a generated blank
frame returned zero people. Person detection is configured, but no inherited pose
calibration or full evaluated sensing profile is claimed. All 87 staged package
files matched, no owned worker remained and the three protected episode files
checked were unchanged. No camera or personal media was used. The original SSH
portal remains available. The earlier 14-versus-20FPS failure remains unresolved.
See the [staging result and acceptance checklist](judge-https-deployment.md).

## HTTPS activated and externally checked

After explicit owner confirmation, the restricted CloudFront-source rule and
portal group attachment were applied while preserving the original SSH group.
The stack completed and the HTTPS distribution was enabled/deployed. Real browser
and external API checks passed authentication, read-only viewer isolation, cookie/
Origin/CSRF and no-store controls, missing private routes, real phase3 COOL worker
start, no false live status without Edge input, native expiry at120.0009 seconds,
restart, explicit stop and logout. The public origin port was not reachable from
the test client; the security-group union grants it only to the managed origin.

The harness retained three failed assumptions: a system Python issuer-store
problem, CloudFront400 traversal rejection instead of origin404, and native clean
COMPLETED at the time limit before the portal watchdog TIME_LIMIT state. TLS stayed
validated; native stop reason/duration were checked from the private log and a
separate external continuation completed lifecycle checks. No deployed code or
security protection was changed to satisfy these checks.

The endpoint and guided, credential-free testing instructions were saved in
Devpost. The app serves no images or videos; judge access delivery and full
new-scene/profile rehearsal remain pending. No incident accuracy, COOL acceleration
or resolution of the earlier14-versus20FPS failure is inferred from this deployment.

## Recorded inputs through the HTTPS portal — person-only scope

Four known development recordings were replayed by the local Edge into the actual
portal-owned Graviton/OpenCV5/COOL worker. The WebP80 adapter used a1.34FPS full-frame
cap,112KiB/s admission and128KiB/s local proxy. Four completed sessions each sent,
received and processed81 frames (324 total), with no declared drops or connection
errors in those completed sessions. Fourteen commands passed source-observation,
matching cloud/Edge ACK and subsequent valid live-evidence checks. Recovery and
sit-return ended NORMAL; no-recovery and sit-hold remained UNRESOLVED. The latter
is an open episode, not a confirmed incident. Pose/scene calibration was disabled,
so this is not the original five-condition person/pose regression or independent
incident accuracy. No TEST20 override or caregiver notification was used.

An earlier no-recovery attempt failed after25 frames sent and24 received/processed,
with a connection error and refused reconnect to the worker's loopback port. The
last UI sample had23 processed; final raw logs had24. No worker stderr or matching
kernel OOM/segfault entry was found. Its cause remains undetermined; a later run
completed unchanged. The failure and unmatched in-flight frame remain part of the
evidence, and completion on repetition does not establish a network fix.

The capped profile applies requested2/5FPS changes but stays near1.34FPS for full
frames; its command chains alone do not prove higher achieved observation density.
A separate unshaped original-Edge/JPEG80 recovery trial completed140/140 frames,
zero declared drops/errors, two causal commands and NORMAL. Within-policy live
send-completion cadence measured1.997 →5.000 →1.999FPS while replay acquisition
retained the original nominal25FPS timeline. JPEG/codec and shaping differ from
the capped profile; no isolated bandwidth/accuracy improvement or general WAN
resilience is inferred. The earlier requested20FPS probe remains unresolved.

The recording runs exposed a portal-only freshness defect: native normal closure
uses session_finished, while the projection recognized session_ended. The live
indicator could linger until its five-second age cutoff. It now treats actual
session_finished/session_error and existing terminal events as disconnected
immediately. Thirty-two focused portal/HTTPS tests passed; native log re-projection
verified live before closure and false immediately afterward. The CV/controller,
model, scene calibration and thresholds were not changed. Raw clips, personal
traces, private inputs and screenshots remain outside GitHub.

After deploying only the projection change, one live recorded-source acceptance
run repeated the separate unshaped recovery condition:140/140 frames, two causal
commands, NORMAL and no declared drops/errors. Before manual stop, the worker was
still RUNNING but vision_live was already false at normal recording EOF. Actual
within-policy transmission was1.997 →4.998 →2.000FPS. This verifies the display
fix on the real connection; it does not add an independent recording or validate
incident accuracy. All eight original video/capture files remained unchanged.

## 23 October - integrated seated profile and delivery reconciliation

The portal explicitly accepts the scene-specific pose configuration and five-second
experimental seated closure; defaults remain off, with pose ROI and alerts disabled
in the demonstrated profile. The restarted physical-camera development run captured
1,501 frames and sent/received/processed 140. Two causal sensing commands passed;
measured transmission was 2.007598 / 5.001160 / 2.001002 FPS. The episode resolved
at source time 24.863 s while sampled original frames remained seated. The first
integrated trial was UNKNOWN with no episode and did not exercise the function.
The second physical descent after closure remains unevaluated. No general accuracy,
fixed/adaptive saving or clinical benefit is inferred from these development runs.

The seven implementation files preserve the recorded deployed bytes. The focused
HTTPS, portal, seated, geometry and shutdown set passes 116 tests locally. The full
suite retains omitted private-fixture requirements and previous failures. Current
report and diagram text now distinguish this JPEG 2/5 FPS trial from the historical
WebP capped regression, and retain the negative COOL comparison.

On 23 October the HTTPS root and app.js returned 504; SSH to the prior destination
expired. A cached, signed-out console page showed a user-initiated instance stop
on 21 October, but current instance state was not authenticated/verified. Do not
interpret recorded success as current live availability. Endpoint restoration and
private judge access remain delivery conditions. No new capture, inference or AWS
mutation was performed for this reconciliation. Personal traces, exported diagrams,
benchmark evidence and presentation media remain outside Git.

## 24 October — English Local Lab and product-perspective previews

Added `care-lab`: a loopback-only, English UI around the existing bounded
synthetic Edge/vision demo. It shows actual Edge rate samples, counts, application
bytes, buffer occupancy, motion observations, deterministic reasons, command
source/ACK/later-perception checks and an owned generated-source preview.
Capture, requested transmit and achieved transmit rates remain separate. Historical
buffer completion is shown separately from live causal sensing cycles.

Caregiver, Family and Operations perspectives include explicitly scripted browser
walkthroughs for recovery, human review and outage. Human attention is separate
from detection and notification delivery. They are product previews, not production
roles, person monitoring or an alert-delivery service. No AWS resources, critical
vision/controller policy, models or image routes in the existing private judge
portal were changed. Local runtime is standard OpenCV, NOT COOL.

Validation: the six new lab tests passed (36.98 s), including an actual generated
run, preview, restart and stop; 63 privacy/protocol/sampling tests passed; 37
lab/portal/HTTPS tests passed with the new paced lab test excluded (4.49 s).
These sets overlap. The initial combined run had 74 passing tests and one failed
assertion that incorrectly expected live re-perception for a historical buffer
request; the local loop auditor itself passed. The UI/test now treat historical
completion independently. No detector success is inferred from this correction.

A separate browser-driven local run passed its full local loop auditor with
397 sent/received/observed frames and COOL unverified. Browser checks covered the
four English views, measured Edge data, scripted human handoff, outage, 360px
light/dark layouts and no JavaScript errors. Selected text/background pairs had
contrast >=5.66:1; this is not a complete accessibility certification. UI assets
were verified in the built wheel. Private generated evidence, failed-test summary,
reports and screenshots are retained in the owner's local evidence workspace.
The README now documents a fresh-clone quick start. CloudFront deployment,
production authorization, durable cases and external notification delivery remain
separate future work; the earlier endpoint outage is not resolved by this change.

## 24 October — Explicit local webcam diagnostic, isolated from AWS

The English Local Lab now offers a generated scene or an explicitly started local
webcam/USB device. Start options accept only local indices and bounded durations;
remote URLs/AWS source options are rejected. All views expose THIS_COMPUTER and
aws_connected=false. No Graviton destination, shared credentials, AWS call,
CloudFront distribution/origin or deployed judge-portal route was introduced or
changed. The competition's primary CV workload remains Graviton/OpenCV5/COOL;
this phase1 standard-OpenCV diagnostic is explicitly separate.

Camera preview is opt-in, bounded to a latest-frame queue/pipe at approximately
2 FPS, kept in memory and expired/cleared on stale/ended/stopped sessions.
It does not block the acquisition thread or save camera images/video files.
Structured diagnostic logs remain in an owner-only temporary directory. The
source preview is not claimed as processed/cloud imagery. A spontaneous camera
session reports completion without claiming the fixed synthetic-loop audit or
incident accuracy passed.

Tests used generated virtual input, never a physical webcam. The actual camera
worker, Edge and local phase1 vision process completed that virtual trial without
image/video output, alongside preview framing/expiry, source validation and worker
launch tests. An initial 44-test lab/camera/judge set passed (43.96 s). After adding
API-boundary cases, 51 tests passed with the paced synthetic test excluded
(11.56 s); the sets overlap. The separate 63 privacy/protocol/sampling tests passed.
Browser checks verified explicit camera selection/start payload, opt-in preview,
location banner, stop state and mobile light/dark layout without JavaScript errors.
Browser camera start/stop transport was stubbed to avoid opening hardware; these
checks are not physical-camera validation. Target hardware, OS camera permissions
and real-camera cadence still require a separate authorized trial.

## 24 October — Distinct visual identity deployed on the CloudFront portal

The existing authenticated judge portal now uses an ivory/plum/copper palette,
serif brand treatment, stronger type hierarchy, clearer metric cards and responsive
layout. The deployed update replaced only the inline CSS, preserving all non-style
HTML bytes, element IDs, JavaScript, permissions, runtime and authentication.
An owner-only rollback copy was retained on the server; original file mode was
preserved. The portal service stayed active with the same process ID; no restart,
vision worker, camera, inference session or AWS infrastructure change was required.
The local lab's blue/teal identity and local diagnostic boundary remain separate.

Before editing, CloudFront already returned HTTP200 with verified TLS. The first
SSH attempt used a stale previously documented address and timed out. The current
instance address was read from refreshed authenticated EC2 details, and SSH host
identity was verified against the existing strict host-key alias; no network rule
or credential change was used to regain access.

After deployment, HTTPS root returned200 with no-store and the expected HTML/CSS
hash; anonymous /api/state remained401. The browser's existing authenticated
session rendered the new style in IDLE with zero processed frames and no fresh
evidence. No worker was started. The live page fit360px without horizontal overflow
and reported no browser JavaScript errors. Static dashboard/login layout was also
reviewed locally, and32 existing judge/HTTPS tests passed (3.46 s). These checks
verify appearance, delivery and preserved access behavior, not a CV/COOL or
incident-detection result. Deployment metadata, original/revised hashes, local
layout previews and live screenshots remain in the owner's private evidence.

## 24 October — Public source reconciliation for the demonstrated seated closure

Inspection of public main `1f40f60` found no `consider_seated_closure` or closure
policy option in the controller. The demonstrated implementation had been frozen
in local commit `433a235` without publication. It is now integrated into branch
`codex/publish-seated-closure` on that public baseline. The only cherry-pick conflict
was this aggregate register; both historical entries were retained. The seven
cloud/judge Python/JavaScript implementation files match the demonstrated commit.
The portal HTML adds the profile status while preserving current main's CSS
exactly. Local Lab and Edge sources have no diff against that baseline.

The closure remains opt-in, with an uncalibrated five-second development profile.
No primary CV workload moves off Graviton/OpenCV5/COOL. The documented physical
trial remains dated evidence from one participant/view; no new physical camera,
AWS workload, deployment or accuracy evaluation was performed for this integration.
Private model weights, scene settings, recordings, credentials and COOL SDK remain
external. The historical 23 October delivery archive is unchanged.

Local verification:

- The documented focused portal/HTTPS/closure/geometry/shutdown set: 116 passed.
- Existing CI privacy/protocol/sampling, Local Lab/local-camera diagnostic and
  synthetic closed-loop sets together: 145 passed in 86.95 seconds. These overlap
  with the focused set; the counts must not be summed as unique tests.
- The added CI closure/profile/reopening/geometry/shutdown step: 84 passed (a
  subset of those 116, not additional independent checks).
- Additional pose-actuation, pose-shadow, episode-continuity, incident-episode and
  reference-guard checks: 62 passed, one failed because the historical private
  `artifacts/reference-recovery-guard-20261014/baseline/` controller is absent.
  This is not a new closure assertion failure; the full suite is not claimed green.
  An earlier command named nonexistent pose/episode test files and collected no
  tests; the subsequent command used the repository's actual filenames.
- Ruff on all modified Python implementation/test files, JavaScript syntax,
  `git diff --check`, tracked/history hygiene and Gitleaks: passed.

The source workflow now runs the 84 closure-specific regression tests on pushes
and pull requests alongside its existing privacy and synthetic-loop tests. These
checks validate source behavior with synthetic inputs, not physical seat support,
fall accuracy, a new live demo or local COOL execution. Publication is reviewed in
the reconciliation PR; main still needs that PR merged to expose the implementation
at the default-branch code URL.
