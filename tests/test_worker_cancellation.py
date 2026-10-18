import asyncio
from types import SimpleNamespace

import pytest
from conftest import status

from care_vision.edge.runtime import Edge


@pytest.mark.parametrize("worker", ["send", "report"])
def test_completed_socket_operation_cannot_swallow_session_cancellation(worker, identity):
    async def scenario():
        loop = asyncio.get_running_loop()
        task = None
        first = True
        waiting = asyncio.Event()

        class Socket:
            async def send(self, data):
                # The operation completes just as session shutdown cancels its worker.
                loop.call_soon(task.cancel)

            async def ping(self):
                pong = loop.create_future()

                def complete_and_cancel():
                    pong.set_result(0.1)
                    task.cancel()

                loop.call_soon(complete_and_cancel)
                return pong

        async def get():
            nonlocal first
            if first:
                first = False
                return "control", None
            await waiting.wait()

        edge = Edge.__new__(Edge)
        edge.outbox = SimpleNamespace(get=get, put_control=lambda data: None)
        edge.observe_network_rtt = lambda *args: None
        edge.status = lambda: status(identity)
        edge.log = SimpleNamespace(emit=lambda *args, **kwargs: None)
        task = asyncio.create_task(getattr(edge, worker)(Socket()))
        try:
            done, _ = await asyncio.wait([task], timeout=2)
            assert task in done, (
                "session cancellation was swallowed by a completed socket operation"
            )
            assert task.cancelled()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_history_wakeup_preserves_session_cancellation():
    from care_vision.common.transport import Outbox

    async def scenario():
        outbox = Outbox()
        outbox.put_history([(b"history", None)], 20)
        outbox.history_due = asyncio.get_running_loop().time() + 0.05
        task = asyncio.create_task(outbox.get())
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            outbox.ready.set()
            asyncio.get_running_loop().call_soon(task.cancel)
            done, _ = await asyncio.wait([task], timeout=0.2)
            assert task in done and task.cancelled()
            assert len(outbox.history) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
