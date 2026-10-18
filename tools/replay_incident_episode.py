"""Replay saved cloud observations through episode decisions, without new inference."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from care_vision.cloud.controller import Controller, Policy


def replay(rows, *, remove_recovery=False):
    controller = Controller({}, Policy(experimental_s_only_5fps=True))
    commands, events = [], []
    for row in rows:
        observation = copy.deepcopy(row)
        # A counterfactual missing-evidence control, not another real video trial.
        if remove_recovery and controller.episode and controller.episode["horizontal_observed"]:
            if observation.get("primary_orientation") != "horizontal":
                observation.update(persons=[], confirmed_person_detected=False)
        # Simulate applied ACKs for the replay's newly generated command IDs.
        observation["applied_command_id"] = controller.expected_command
        command = controller.observe(observation)
        events.extend(controller.pop_episode_events())
        if command:
            commands.append(
                {
                    "captured_mono_ns": observation["captured_mono_ns"],
                    "transmit_fps": command["transmit_fps"],
                }
            )
            controller.acknowledge(
                {
                    "command_id": command["command_id"],
                    "result": "applied",
                    "active_policy_seq": command["command_seq"],
                    "active_transmit_fps": command["transmit_fps"],
                }
            )
    return {
        "observations": len(rows),
        "commands": commands,
        "events": events,
        "final_state": controller.state,
        "pending_episode": controller.finish_episode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [
        r
        for r in map(json.loads, args.cloud_log.read_text().splitlines())
        if r["event"] == "observation"
    ]
    result = {
        "source": str(args.cloud_log),
        "sha256": hashlib.sha256(args.cloud_log.read_bytes()).hexdigest(),
        "scope": "Full saved observation replay; no new inference or transport trial",
        "observed_recovery": replay(rows),
        "missing_recovery_control": replay(rows, remove_recovery=True),
    }
    assert result["observed_recovery"]["final_state"] == "NORMAL"
    assert any(
        e["incident_status"] == "recovery_observed" for e in result["observed_recovery"]["events"]
    )
    assert result["missing_recovery_control"]["final_state"] == "UNRESOLVED"
    assert result["missing_recovery_control"]["pending_episode"]["incident_status"] == "unresolved"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
