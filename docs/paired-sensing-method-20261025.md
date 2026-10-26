# CARE-Vision paired transmission rate evaluation

This experiment compares fixed 5 FPS transmission with adaptive 2/5 FPS on three
existing development recordings. It measures resource consumption and agreement
of incident-rule outcomes on known inputs. It does not estimate fall accuracy,
clinical benefit, independent incident sensitivity or AWS billing savings.

## Frozen implementation and configuration

The application source is an immutable Git archive of commit
`f1f55d7a1cc7837001dd5ec9c07c4671e3079f7b`. Detection, tracking, temporal
observations and incident rules execute on the existing AWS Graviton using native
COOL OpenCV 5.1.0-dev, CLASSIC YOLOX-S640 and eight OpenCV threads. The local Edge
only replays, buffers, samples, encodes and transmits. No new AWS resources or
camera recordings are required. Model, source, input and execution-adapter hashes
are locked before inference in `run-1/plan.json` and `plan-lock.json`.

Both arms use 1280x720 JPEG80, the same weak/strong detection association settings,
episode continuity with a fresh state file and a test-only 20-second horizontal
evidence alert. Pose, ROI and experimental seated closure are disabled in both
arms to isolate the transmission-rate intervention. This profile is separate
from the earlier pose/ROI resource study and the later pose-assisted seated demo.

## Fixed sensing intervention

The deployed controller uses requested FPS as a logical baseline/verification
mode label. A fixed physical stream reporting 5 FPS cannot enter its unmodified
2 FPS baseline branch. The evaluation adapter therefore separates two values:

- Raw frame and observation logs retain the actual fixed 5 FPS target, capture
  times, applied command IDs and geometry.
- A copy supplied to the unchanged incident-rule implementation carries the
  controller's logical requested baseline/verification label of 2 or 5 FPS.
  Every projection is logged by observation reference.

The fixed controller logs each sensing proposal as suppressed and never sends it
to the Edge. It clears the non-executed pending proposal without acknowledging
it; actual expected/applied command IDs remain unchanged. No ACK or later frame
under a fictitious physical policy is manufactured. The adaptive arm retains
real command dispatch, applied ACK validation and later evidence under the new
rate. The fixed arm is an explicit non-actuating experimental adapter, not an
unmodified production transport mode. This distinction must accompany results.

## Conditions and outcome gates

There are three repeated pairs for each recording: recovery, no recovery and
sitting. Pair order alternates; nine pairs cannot balance first position exactly.
Every run starts fresh and is bounded to 100 cloud seconds and 95 Edge seconds.
Failed runs remain in the record and are not silently retried or used to tune
thresholds. Repetitions characterize runtime variation, not independent events.

The known recovery input is expected to exercise an episode and horizontal
evidence, return to NORMAL and produce no test alert. The known no-recovery input
is expected to exercise an episode, emit one test alert and stay open in
ASSISTANCE_REQUIRED. Sitting is expected to remain UNRESOLVED without a test
alert in this person-only profile. That unresolved benign activity is a retained
limitation; matching it is not successful benign classification.

Paired final state and alert count must agree. Adaptive recovery/test-alert event
delay relative to fixed5 must be no greater than two seconds. This provisional
engineering comparison was fixed before these trials; it is not a calibrated
safety threshold. Times are measured from original capture timestamps, not an
independently annotated fall onset or absolute camera-to-caregiver latency.
Negative delays and all per-pair differences are reported.

## Measurements and audit

The ledger reconciles sent, received and processed frame identities, order,
capture timestamps, geometry and actual policy metadata. The audit requires
complete recorded capture mapping, source-time interval preservation, zero
unaccounted frames, confirmed drain and stopped owned workers. Adaptive commands
require a valid causal observation, matching applied ACKs and a later valid
capture under the new policy. Fixed runs must have no executed sensing commands
or command ACKs, and their suppression records must explicitly deny execution.

Cloud CPU uses direct cumulative process counters at the first received frame
and confirmed drain. It excludes process/model startup before that boundary and
includes the first inference and session overhead. RSS samples and process peak
RSS are reported separately. JPEG bytes are encoded payload; packed application
bytes are reconstructed from logged wire headers plus the four-byte framing
length and payload size. Both exclude WebSocket/SSH/TCP framing and control
traffic. Lower CPU time is not a measured reduction in an allocated EC2 bill.

The report presents each run and per-recording medians/ranges of paired resource
reductions and event-time differences. Three repeats do not justify population
confidence intervals. Native runtime hashes establish execution provenance,
not a COOL speedup; that remains a separate standard-versus-COOL comparison.

## Reproduction requirements

Use the exact source commit, locked dependencies, hash-verified person model,
three authorized recordings and their original capture timestamp files. Original
recordings, private logs, SDK and model binaries are not public attachment assets.
An entitled matching Linux ARM64 COOL environment remains necessary.

Run `harness/paired_cloud.py --mode fixed5` or `--mode adaptive` with the shared
cloud options in the frozen plan, and a fresh loopback port/log/state per run.
Connect through an authorized SSH tunnel. Run `harness/paired_edge.py` with the
matching mode, video, capture timestamps, zero replay sampling phase and 720p
capture/transmission geometry. It rejects physical camera input. Preserve logs,
recorded capture mappings, runtime records, drain and process-stop confirmation.

`harness/run_paired.py` is the private bounded orchestrator used for this host.
`harness/audit_paired.py` evaluates the stored records without rerunning inference.
The adapter and ledger/resource audit have focused positive and negative tests.
The source profile and production state remain unchanged throughout the study.
