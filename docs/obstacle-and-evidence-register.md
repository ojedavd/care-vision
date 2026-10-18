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
