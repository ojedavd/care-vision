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
