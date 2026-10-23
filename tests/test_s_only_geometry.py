"""An unsupported full-frame profile must fail visibly before policy is skipped."""
import asyncio

import cv2
import pytest
from conftest import frame, status
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from care_vision.check_loop import read_log
from care_vision.cloud.server import CloudServer
from care_vision.common.contracts import dumps, pack_frame
from care_vision.common.metrics import EventLog


@pytest.mark.parametrize("enabled,resolution,rejected", [
    (True, (640, 360), True),
    (True, (1280, 720), False),
    (False, (640, 360), False),
])
def test_profile_geometry_is_checked_before_inference(
    tmp_path, identity, enabled, resolution, rejected
):
    async def scenario():
        path = tmp_path / "cloud.jsonl"
        log = EventLog("cloud", path)
        cloud = CloudServer(log, experimental_s_only_5fps=enabled)
        try:
            async with serve(cloud.handler, "127.0.0.1", 0) as server:
                url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(url) as client:
                    await client.send(dumps(status(identity)))
                    header, jpeg = frame(identity)
                    if resolution != (640, 360):
                        import numpy as np
                        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        ok, encoded = cv2.imencode(".jpg", cv2.resize(image, resolution))
                        assert ok
                        jpeg = encoded.tobytes()
                        header.update(width=resolution[0], height=resolution[1],
                                      payload_bytes=len(jpeg))
                    await client.send(pack_frame(header, jpeg))
                    if rejected:
                        with pytest.raises(ConnectionClosed):
                            async with asyncio.timeout(2):
                                await client.recv()
                    else:
                        async with asyncio.timeout(2):
                            while not any(r["event"] == "observation" for r in read_log(path)):
                                await asyncio.sleep(.005)
        finally:
            log.close()
        rows = read_log(path)
        if rejected:
            assert not any(r["event"] == "observation" for r in rows)
            assert any(r["event"] == "frame_rejected" and
                       r["reason"] == "s_only_profile_requires_1280x720_full_frame"
                       for r in rows)
            assert any(r["event"] == "session_error" and
                       "S-only profile requires 1280x720" in r["reason"] for r in rows)
        else:
            assert not any(r["event"] in {"frame_rejected", "session_error"} for r in rows)

    asyncio.run(scenario())
