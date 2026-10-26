# Explicit human review after an interrupted episode

The 25 October paired study retained one adaptive no-recovery failure: reconnecting
left identity unlinked and the expected test alert absent. Its 8/9 paired result
remains unchanged. The controller deliberately cannot infer physical identity
from a new tracker, including when the Edge boot ID is unchanged.

The candidate adds persistent `review_required` and a fixed allowlisted reason
when an open episode is restored. Cloud emits `episode_review_required` and saves
its metadata before processing any new frame. The telemetry projection latches
that status independently of evidence freshness or the current sensing state.
A separate live-view banner makes the need for human review explicit. It does
not claim a human was notified, acknowledged or resolved anything.

It preserves existing incident/recovery thresholds and identity gates. New frames
cannot relink the original person, clear the pending episode, establish recovery
or manufacture an incident alert. This improves observability and escalation
status; automatic incident sensitivity after identity loss remains unresolved.

## Separate controlled application-close probe

One known no-recovery recording per version ran primary person inference on the
existing AWS Graviton using native OpenCV 5 + COOL, CLASSIC YOLOX-S640, eight
threads and the same 720p JPEG80/2-5 FPS profile. The diagnostic hook closed the
WebSocket three source seconds after opening an episode. This is an application
interruption, not WAN packet loss. Source, adapter, plan, model and input hashes
were frozen before execution, with no new camera capture or threshold tuning.

| Version | Sent / received | Processed / explicit shutdown drops | Review status |
| --- | --- | --- | --- |
| Original f1f55d7 | 173 / 173 | 172 / 1 | No explicit review event/metadata |
| Candidate | 170 / 170 | 169 / 1 | Emitted before fresh evidence, persisted and projected |

Each run had two sessions, one forced-close error and one restored original
episode. All sent frame identities, metadata and order reconcile with observations
plus the explicit shutdown drop. Both finished UNRESOLVED with zero test alerts
and unlinked identity. No external notification was sent. These are not passing
incident-detection trials or resource comparisons.

Forty-eight focused continuity, review-projection, portal and shutdown checks
passed, including restoration without any subsequent frame and persistence
across another restart. Full-suite success is not claimed. Production configuration
and state hashes were unchanged; the portal remained IDLE. Both owned workers
stopped, and remote records matched local hashes before workspace removal.

The core candidate is not deployed to the production portal. The public landing
page separately displays the original measured benchmark and its retained failure.
Original camera inputs and private logs remain restricted. The exact controlled
plan/log hashes remain in the owner's evidence package.
