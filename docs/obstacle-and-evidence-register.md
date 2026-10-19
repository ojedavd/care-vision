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
