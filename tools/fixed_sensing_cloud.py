"""Evaluation-only fixed sensing: real cloud vision/queues, no agent decisions.

The fixed arm measures transmission and vision capacity, not episode accuracy.
Run with a fixed-rate Edge; never use this launcher for monitoring a person.
"""

from care_vision.cloud import __main__ as cloud_cli
from care_vision.cloud import server
from care_vision.cloud.controller import Controller


class NoActuationController(Controller):
    def observe(self, observation):
        return None


if __name__ == "__main__":
    server.Controller = NoActuationController
    cloud_cli.main()
