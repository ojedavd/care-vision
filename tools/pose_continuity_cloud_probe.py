"""Experiment-only cloud launcher: disconnect once after matched ROI evidence.

Uses the existing Session diagnostic callback, after the episode checkpoint.
The production CLI, defaults, perception and policy are unchanged. Run only in
an isolated workspace with both opt-ins and a fresh private episode state path.
"""

from care_vision.cloud import __main__ as cloud_cli
from care_vision.cloud.server import CloudServer
from care_vision.common.metrics import EventLog


class ProbeLog(EventLog):
    roi_observation_id = None

    def emit(self, event, **fields):
        super().emit(event, **fields)
        if event == "pose_observation_cycle" and fields.get("stage") == "roi_evidence_processed":
            self.roi_observation_id = fields["observation_id"]


class ProbeServer(CloudServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.experimental_episode_continuity and self.experimental_pose_observation
                and self.episode_store.path and isinstance(self.log, ProbeLog)):
            raise ValueError("probe requires pose, durable continuity and ProbeLog")
        if self.diagnostic_disconnect_after_episode_seconds is not None:
            raise ValueError("probe cannot combine phase and elapsed-time disconnects")

    def diagnostic_disconnect_due(self, controller, observation):
        episode = controller.episode
        if (self.diagnostic_disconnect_done or episode is None
                or episode.get("identity_link_status") != "session_track_only"
                or not episode.get("pose_observation_requested")
                or observation.get("invalid_reason") == "processing_stale"
                or observation.get("view_kind") != "roi"
                or observation.get("evidence_kind") != "live"
                or self.log.roi_observation_id is None
                or observation.get("observation_id") != self.log.roi_observation_id):
            return False
        self.diagnostic_disconnect_done = True
        self.log.emit("pose_continuity_probe_trigger", episode_id=episode["episode_id"],
                      observation_id=observation["observation_id"],
                      trigger="first_fresh_matched_roi_evidence", test_only=True)
        return True


if __name__ == "__main__":
    cloud_cli.CloudServer = ProbeServer
    cloud_cli.EventLog = ProbeLog
    cloud_cli.main()
