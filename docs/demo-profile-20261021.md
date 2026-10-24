# Demonstrated portal profile - 21 October 2026

The current technical report is [technical-report.md](technical-report.md).
The private worker combines OpenCV 5 + COOL on Graviton, eight threads, CLASSIC
YOLOX-S, S-only 1280x720 and the existing episode-continuity path. External
RTMPose model/scene configuration enables the five-second experimental seated
closure. Pose-driven ROI and alerts are disabled. The global closure default is off.
The local Edge captures nominally 25 FPS and transmits full-frame JPEG at requested
2/5 FPS, with no diagnostic byte-budget cap in the latest camera trial.

Configure the private cloud-options JSON with:

```json
{
  "experimental-s-only-5fps": true,
  "pose-shadow-config": "/PRIVATE/reviewed-scene.json",
  "experimental-seated-closure-seconds": 5
}
```

Use the HTTPS runbook for the portal command and external access checks. Starting
the cloud session does not connect an Edge or enable camera capture. Supply the
model, scene-specific regions and credentials privately. Camera/chair movement
requires reviewing the scene configuration. No physical seat support is verified.

The restarted physical-camera development trial captured 1,501 frames and sent,
received and processed 140, with two verified causal command chains. It resolved
at source time 24.863 s while the reviewed activity remained seated. The first
integrated camera trial opened no episode and did not exercise closure. The second
physical descent after closure remains unevaluated. These are not accuracy rates.

The earlier WebP80, 1.34 FPS capped constrained-link regression is a separate
profile. Preserve its results and failures without merging totals. Private source
inputs/configuration, model weights, native SDK and recordings are not included
in Git. Source-snapshot-manifest.json in the delivery archive identifies the exact
source commit and bytes; older source-provenance.json refers to an earlier baseline.
The focused test set is documented in the current report; the full suite is not
claimed green. The archive supports source review, not blank-host SDK provisioning.

## Public source integration — 24 October 2026

The implementation originally frozen in local commit `433a235` is integrated on
public `main` baseline `1f40f60` in the source-reconciliation branch. This preserves
the later Local Lab, explicit local-camera diagnostic and ivory/plum portal CSS.
See the [integration record](obstacle-and-evidence-register.md#24-october--public-source-reconciliation-for-the-demonstrated-seated-closure)
for measured source checks; the historical delivery archive remains unchanged.

The implementation is in `cloud/controller.py` (`consider_seated_closure`),
`cloud/server.py` (pose-result integration) and `cloud/__main__.py` (explicit
`--experimental-seated-closure-seconds`). The judge launcher validates the profile
and its image-free projection reports the experimental resolution. All paths are
relative to `src/care_vision/`. The CI workflow runs closure, reopening-reference,
profile, geometry and shutdown regression tests without private models or footage.

These checks establish source integration and synthetic behavior. They do not
repeat the physical-camera trial, run COOL locally or validate fall accuracy.
