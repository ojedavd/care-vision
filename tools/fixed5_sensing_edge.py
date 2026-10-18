"""Recorded-evaluation launcher for the existing supported 5 FPS sampler rate.

The normal Edge CLI exposes only initial2/20. This isolated launcher sets initial5
before run/reconnect and retains the real sampler, transport and network monitor.
"""

import sys

from care_vision.edge import __main__ as edge_cli
from care_vision.edge.runtime import Edge


class Fixed5Edge(Edge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_transmit_fps = 5
        self.sampler.set_rate(5, 0)


if __name__ == "__main__":
    if "--video" not in sys.argv or "--replay-captures" not in sys.argv:
        raise SystemExit("fixed5 evaluation requires recorded video and timestamps")
    edge_cli.Edge = Fixed5Edge
    edge_cli.main()
