# CARE-Vision paired sensing results

25 October 2026

Study completed with failed gates. The study completed 18 recorded replays and nine paired comparisons on three known development inputs. 8/9 pairs met the declared outcome/alert and additional-delay criteria. These repetitions are not independent incidents or a fall-accuracy validation.

## Retained failures

Retained failures: No recovery, repetition 2, adaptive: 2 cloud sessions, 2 connection errors, 171 sent / 160 received / 160 processed packets (19 historical sends), final VERIFYING, 0 test alerts. Failed checks: ledger_identity_reconciled, sent_received_order, all_frames_processed_without_drops, one_cloud_session, one_controller_final, no_stale_or_session_error, actual_causal_cycles. A missing expected alert and unconfirmed cloud receipt are failures; they are not repaired by eventual drain or lower resource use. No thresholds were tuned and no failed condition was retried to replace it.

## Resource measurements

| Known recording | Pairs meeting all gates | Frames reduction | JPEG reduction | Cloud CPU reduction | Adaptive event delay |
| --- | --- | --- | --- | --- | --- |
| Recovery | 3/3 | 53.33% (53.33–53.33) | 53.54% (53.54–53.55) | 53.12% (52.97–53.23) | +0.284 s (+0.284 to +0.524); n=3 matched |
| No recovery | 2/3 | 53.33% (43.00–53.33) | 53.47% (43.17–53.48) | 53.07% (46.01–53.26) | +0.120 s (-0.160 to +0.400); n=2 matched |
| Sitting | 3/3 | 47.00% (44.67–47.00) | 47.17% (44.83–47.17) | 46.61% (44.53–47.01) | No matched event |

Reported resource reductions are medians of three paired percentage reductions per recording; parentheses show their minimum and maximum. CPU is the difference of direct process counters from first frame receipt through confirmed drain, excluding pre-first-frame process/model startup and including first inference/session overhead. JPEG bytes count payload only. Packed application byte reconstruction is reconciled against Edge counters; neither measure includes WebSocket/SSH/TCP framing or control traffic. Process RSS and peak RSS remain in study-summary.json. Both policies use COOL, so these savings are not a standard-versus-COOL speedup. Resource medians include all three attempted pairs per recording, including any failed or connection-confounded pair. The gate count must be read with the resource figures; they cannot establish complete outcome equivalence when a pair fails. Event-time summaries include only pairs with both matching events, without imputing missing alerts.

## Recorded outcomes

| Recording | Fixed5 states, repetitions 1–3 | Adaptive states, repetitions 1–3 | Fixed test alerts total | Adaptive test alerts total |
| --- | --- | --- | --- | --- |
| Recovery | NORMAL, NORMAL, NORMAL | NORMAL, NORMAL, NORMAL | 0 | 0 |
| No recovery | ASSISTANCE_REQUIRED, ASSISTANCE_REQUIRED, ASSISTANCE_REQUIRED | ASSISTANCE_REQUIRED, VERIFYING, ASSISTANCE_REQUIRED | 3 | 2 |
| Sitting | UNRESOLVED, UNRESOLVED, UNRESOLVED | UNRESOLVED, UNRESOLVED, UNRESOLVED | 0 | 0 |

Sitting remains UNRESOLVED in this person-only profile; matching that limitation is not correct benign classification. Recovery and no-recovery are known voluntary development scenarios, with no dense independent fall labels. Event times use original source clocks; signed adaptive-minus-fixed delays are not camera-to-caregiver or annotated fall-confirmation latency. The two-second margin is provisional, not a calibrated safety requirement. Three repeats report medians/ranges rather than population confidence intervals. No independent fall precision/recall/F1, 30 FPS comparison, COOL acceleration, clinical benefit or billed AWS savings follows.

## Method and execution

Both arms used the same frozen person detector, tracker, incident/recovery rules, 720p JPEG80, CLASSIC engine, eight OpenCV threads and test-only T20 alert configuration on the same Graviton/COOL runtime. Pose, ROI and seated closure were disabled in both. The fixed arm kept physical transmission at 5 FPS through an explicitly declared non-actuating controller adapter: actual raw metadata stayed truthful, a copied controller input carried the logical 2/5 verification-mode label, and sensing proposals were logged as suppressed without invented ACKs. The adaptive arm used real commands, applied ACKs and later valid evidence. This is an experimental fixed-sensing intervention, not an unchanged production transport profile.

Application commit: `f1f55d7a1cc7837001dd5ec9c07c4671e3079f7b`. Preregistered plan SHA-256: `d3b8934881ee68fc2cfa9e51bbbe2830d52fae8203adafcb578921192c99cf3c`. Adaptive causal cycles passed: 25/27. Each raw log remains private and is identified by hash in the scalar export. Native module and input/source/model hashes are included. The exact COOL SDK source revision remains unknown.

## Reproduction and evidence

Read [the method](paired-sensing-method-20261025.md) and [all scalar results](evidence/paired-sensing-20261025.json). The exact public Git source archive, original execution adapters, offline audit and focused tests accompany the report. An entitled matching native COOL environment, authorized original recordings/timestamps and external model remain prerequisites for fresh inference. The offline verifier checks source hashes and scalar arithmetic without inference. SDK/model binaries and original camera media are not redistributed.
