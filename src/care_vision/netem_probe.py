"""Opt-in Linux diagnostic segment. Never changes an existing interface/qdisc.

SSH -> namespace relay -> host relay -> loopback cloud. Only namespace veth
egress loses packets (both application directions, on two distinct TCP legs).
The supervisor runs as root; relays run as the invoking user. No default route,
forwarding, firewall change, public listener, or payload capture is installed.
"""

import argparse
import asyncio
import ipaddress
import json
import math
import os
import select
import signal
import subprocess
import time
import uuid
from pathlib import Path

HOST_IP, PEER_IP = "198.19.240.1", "198.19.240.2"
NETWORK = ipaddress.ip_network("198.19.240.0/30")


def validate_loss(loss):
    if not math.isfinite(loss) or not 0 <= loss <= 5:
        raise ValueError("diagnostic loss must be finite and between 0 and 5 percent")


def check_routes(routes):
    for route in routes:
        dst = route.get("dst", "default")
        if dst == "default":
            continue
        network = ipaddress.ip_network(dst, strict=False)
        if network.version == 4 and network.overlaps(NETWORK):
            raise ValueError(f"diagnostic subnet overlaps existing route {dst}")


async def relay(bind, port, destination, target):
    async def handle(reader, writer):
        upstream = None
        tasks = []
        async def copy(source, sink):
            while data := await source.read(65536):
                sink.write(data)
                await sink.drain()
        try:
            other, upstream = await asyncio.open_connection(destination, target)
            tasks = [asyncio.create_task(copy(reader, upstream)),
                     asyncio.create_task(copy(other, writer))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            if upstream:
                upstream.close()
    server = await asyncio.start_server(handle, bind, port)
    print("RELAY_READY", flush=True)
    async with server:
        await asyncio.sleep(300)


def supervise(run, cloud_port, loss):
    import fcntl
    import pwd

    validate_loss(loss)
    if os.geteuid() != 0 or not os.environ.get("SUDO_UID"):
        raise ValueError("invoke via sudo from the deployment user")
    uid = int(os.environ["SUDO_UID"])
    user = pwd.getpwuid(uid)
    run = run.resolve(strict=True)
    expected = Path(user.pw_dir) / "care-vision-phase3/releases"
    if not run.is_relative_to(expected) or run.name != "run" or run.stat().st_uid != uid:
        raise ValueError("expected an owned phase-3 release/run directory")
    if not 1 <= cloud_port <= 65535:
        raise ValueError("invalid cloud port")
    # Root-only lock serializes our reserved subnet without following user symlinks.
    lock = os.open("/run/care-vision-netem.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    namespace = "cvn-" + uuid.uuid4().hex[:8]
    host_dev, peer_dev = namespace + "h", namespace + "p"
    children = []
    created_namespace = created_link = False
    def command(args, check=True):
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=check)
    def inside(args):
        return ["ip", "netns", "exec", namespace, *args]
    def emit(event, **fields):
        log.write(json.dumps({"event": event, "local_mono_ns": time.monotonic_ns(), **fields})
                  + "\n")
        log.flush()
    def snapshot(event):
        q = command(inside(["tc", "-s", "-j", "qdisc", "show", "dev", peer_dev]))
        sockets = command(inside(["ss", "-tinH"]))
        emit(event, qdisc=json.loads(q.stdout), sockets=sockets.stdout)
    def interrupted(*_):
        raise InterruptedError("diagnostic supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (run / "netem.jsonl").open("w") as log:
        try:
            check_routes(json.loads(command(["ip", "-j", "route", "show", "table", "all"]).stdout))
            command(["ip", "netns", "add", namespace])
            created_namespace = True
            command(["ip", "link", "add", host_dev, "type", "veth", "peer", "name", peer_dev])
            created_link = True
            command(["ip", "link", "set", peer_dev, "netns", namespace])
            command(["ip", "addr", "add", HOST_IP + "/30", "dev", host_dev])
            command(inside(["ip", "addr", "add", PEER_IP + "/30", "dev", peer_dev]))
            for prefix, dev in (([], host_dev), (["ip", "netns", "exec", namespace], peer_dev)):
                command([*prefix, "ip", "link", "set", dev, "mtu", "1500", "up"])
                command([*prefix, "ip", "link", "set", dev,
                         "gso_max_size", "1500", "gso_max_segs", "1"])
                command([*prefix, "ethtool", "-K", dev, "tso", "off", "gso", "off", "gro", "off"])
                emit("offload_configuration", device=dev,
                     features=command([*prefix, "ethtool", "-k", dev]).stdout,
                     link=json.loads(command([*prefix, "ip", "-d", "-j", "link", "show",
                                              "dev", dev]).stdout))
            command(inside(["ip", "link", "set", "lo", "up"]))
            command(inside(["tc", "qdisc", "add", "dev", peer_dev, "root", "netem",
                            "limit", "1000", "loss", "random", str(loss) + "%"]))
            emit("configured", namespace=namespace, host_device=host_dev, peer_device=peer_dev,
                 loss_percent=loss, seed=None, scope="namespace egress, two inner TCP legs",
                 routes=json.loads(command(inside(["ip", "-j", "route"])).stdout))
            # Both children drop privilege before handling application bytes.
            for prefix, bind, port, destination, target in (
                ([], HOST_IP, 8767, "127.0.0.1", cloud_port),
                (["ip", "netns", "exec", namespace], PEER_IP, 8766, HOST_IP, 8767),
            ):
                child = subprocess.Popen([
                    *prefix, "runuser", "-u", user.pw_name, "--", "python3", __file__,
                    "relay", bind, str(port), destination, str(target),
                ], stdout=subprocess.PIPE, start_new_session=True)
                children.append(child)
                readable, _, _ = select.select([child.stdout], [], [], 10)
                if not readable or child.stdout.readline().strip() != b"RELAY_READY":
                    raise RuntimeError("diagnostic relay readiness failed")
            snapshot("initial")
            (run / "netem-ready.json").write_text(json.dumps({"host": PEER_IP, "port": 8766}))
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline and not (run / "netem.stop").exists():
                if any(p.poll() is not None for p in children):
                    raise RuntimeError("diagnostic relay exited early")
                snapshot("sample")
                time.sleep(.5)
            snapshot("final")
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=3)
                child.stdout.close()
            cleanup = []
            if created_link:
                cleanup.append(command(["ip", "link", "del", host_dev], check=False).returncode)
            if created_namespace:
                cleanup.append(command(["ip", "netns", "del", namespace], check=False).returncode)
            emit("cleanup", success=all(code == 0 for code in cleanup), returncodes=cleanup)
            (run / "netem-done").touch()
            os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    listener = sub.add_parser("relay")
    listener.add_argument("bind", choices=[HOST_IP, PEER_IP])
    listener.add_argument("port", type=int)
    listener.add_argument("destination", choices=[HOST_IP, "127.0.0.1"])
    listener.add_argument("target", type=int)
    parent = sub.add_parser("supervise")
    parent.add_argument("run", type=Path)
    parent.add_argument("cloud_port", type=int)
    parent.add_argument("loss", type=float)
    args = parser.parse_args()
    if args.mode == "relay":
        asyncio.run(relay(args.bind, args.port, args.destination, args.target))
    else:
        supervise(args.run, args.cloud_port, args.loss)


if __name__ == "__main__":
    main()
