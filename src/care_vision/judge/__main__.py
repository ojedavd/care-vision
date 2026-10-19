"""Authenticated loopback portal. Expose only behind a reviewed HTTPS proxy."""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import signal
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from care_vision.judge.session import Sessions


class Auth:
    def __init__(self, credentials):
        if (
            set(credentials) != {"viewer", "operator"}
            or any(not isinstance(v, str) or len(v) < 32 for v in credentials.values())
            or credentials["viewer"] == credentials["operator"]
        ):
            raise ValueError("two distinct credentials of at least 32 characters required")
        self.pepper = secrets.token_bytes(32)
        self.credentials = {role: self.digest(value) for role, value in credentials.items()}
        self.sessions = {}
        self.attempts = []
        self.lock = threading.Lock()

    def digest(self, value):
        return hmac.new(self.pepper, value.encode(), hashlib.sha256).digest()

    def login(self, value):
        with self.lock:
            now = time.monotonic()
            self.attempts = [t for t in self.attempts if now - t < 60]
            if len(self.attempts) >= 20:
                return None
            self.attempts.append(now)
            role = None
            candidate = self.digest(value)
            for name, expected in self.credentials.items():
                if hmac.compare_digest(candidate, expected):
                    role = name
            if role is None:
                return None
            self.sessions = {k: v for k, v in self.sessions.items() if v[2] > now}
            if len(self.sessions) >= 100:
                return None
            token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self.sessions[token] = (role, csrf, now + 1800)
            return token, role, csrf

    def get(self, cookie):
        try:
            parsed = SimpleCookie(cookie or "")
            token = parsed["care_session"].value
        except (KeyError, ValueError):
            return None, None
        with self.lock:
            session = self.sessions.get(token)
            if not session or session[2] <= time.monotonic():
                self.sessions.pop(token, None)
                return None, None
            return token, session


class Portal(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, sessions, auth, origin=None):
        super().__init__(address, Handler)
        self.sessions, self.auth = sessions, auth
        self.origin = origin or f"http://127.0.0.1:{self.server_port}"
        self.host = urlsplit(self.origin).netloc
        self.secure = self.origin.startswith("https://")
        self.stopping = threading.Event()
        self.slots = threading.BoundedSemaphore(20)
        self.ticker = threading.Thread(target=self.tick, daemon=True)
        self.ticker.start()

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def tick(self):
        while not self.stopping.wait(1):
            self.sessions.snapshot()

    def server_close(self):
        self.stopping.set()
        self.sessions.stop()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "CARE-Vision"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass  # Do not put credentials, URLs, client addresses or log records in access logs.

    def reply(self, status, value, kind="application/json", cookie=None):
        body = json.dumps(value, allow_nan=False).encode() if kind == "application/json" else value
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            ),
        )
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if self.server.secure:
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        return self.headers.get("Host") == self.server.host

    def do_GET(self):
        if not self.allowed_host():
            return self.reply(403, {"error": "host rejected"})
        if self.path in {"/", "/app.js"}:
            name = "index.html" if self.path == "/" else "app.js"
            return self.reply(
                200,
                Path(__file__).with_name(name).read_bytes(),
                "text/html" if name.endswith("html") else "application/javascript",
            )
        _, session = self.server.auth.get(self.headers.get("Cookie"))
        if not session:
            return self.reply(401, {"error": "sign in required"})
        if self.path == "/api/state":
            value = self.server.sessions.snapshot(operator=session[0] == "operator")
            value["csrf"] = session[1]
            return self.reply(200, value)
        return self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed_host() or self.headers.get("Origin") != self.server.origin:
            return self.reply(403, {"error": "origin rejected"})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "JSON required"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get("Transfer-Encoding"):
                raise ValueError()
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError()
        except (ValueError, TypeError):
            return self.reply(400, {"error": "invalid request"})
        if self.path == "/api/login":
            credential = value.get("credential", "")
            result = self.server.auth.login(credential) if isinstance(credential, str) else None
            if not result:
                return self.reply(401, {"error": "invalid or rate-limited login"})
            token, role, csrf = result
            cookie = f"care_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=1800"
            if self.server.secure:
                cookie += "; Secure"
            return self.reply(200, {"role": role, "csrf": csrf}, cookie=cookie)
        token, session = self.server.auth.get(self.headers.get("Cookie"))
        if not session:
            return self.reply(401, {"error": "sign in required"})
        if not hmac.compare_digest(self.headers.get("X-CSRF-Token", ""), session[1]):
            return self.reply(403, {"error": "request token rejected"})
        if self.path == "/api/logout":
            with self.server.auth.lock:
                self.server.auth.sessions.pop(token, None)
            return self.reply(
                200,
                {"ok": True},
                cookie="care_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0",
            )
        if session[0] != "operator":
            return self.reply(403, {"error": "operator required"})
        try:
            if self.path == "/api/start":
                self.server.sessions.start()
            elif self.path == "/api/stop":
                self.server.sessions.stop()
            else:
                return self.reply(404, {"error": "not found"})
        except (OSError, ValueError):
            return self.reply(
                409, {"error": "session unavailable; operator must inspect private worker log"}
            )
        return self.reply(200, {"ok": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--synthetic-local", action="store_true")
    parser.add_argument("--cloud-python", type=Path)
    parser.add_argument("--cloud-source", type=Path)
    parser.add_argument("--cloud-options", type=Path)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument(
        "--https-origin", help="exact external HTTPS origin; trusted proxy must preserve Host"
    )
    args = parser.parse_args()
    if not 40 <= args.seconds <= 600:
        parser.error("session lifetime must be 40..600 seconds")
    if args.credentials_file.stat().st_mode & 0o077:
        parser.error("credentials file must have private owner-only permissions")
    origin = args.https_origin
    if origin and (
        urlsplit(origin).scheme != "https"
        or not urlsplit(origin).netloc
        or urlsplit(origin).path
        or urlsplit(origin).username
        or urlsplit(origin).query
        or urlsplit(origin).fragment
    ):
        parser.error("exact HTTPS origin required, without path or credentials")
    if not args.synthetic_local and (not args.cloud_python or not args.cloud_source):
        parser.error("explicit Graviton cloud Python and source required")
    if args.synthetic_local and (args.cloud_options or args.cloud_python or args.cloud_source):
        parser.error("synthetic local mode cannot accept cloud options")
    options = json.loads(args.cloud_options.read_text()) if args.cloud_options else {}
    allowed = {
        "person-model",
        "person-detector",
        "person-input-size",
        "pose-shadow-config",
        "experimental-s-only-5fps",
        "experimental-weak-association",
        "experimental-yolox-classic-engine",
        "experimental-episode-continuity",
        "experimental-pose-observation",
        "experimental-alert-after-seconds",
    }
    if not isinstance(options, dict) or set(options) - allowed:
        parser.error("unsupported cloud option")
    argv = []
    for key, value in options.items():
        if value is True:
            argv.append("--" + key)
        elif type(value) in (str, int, float) and value is not False:
            argv.extend(("--" + key, str(value)))
        else:
            parser.error("invalid cloud option value")
    os.umask(0o077)
    sessions = Sessions(
        args.state_root,
        synthetic=args.synthetic_local,
        cloud_python=str(args.cloud_python.absolute()) if args.cloud_python else None,
        cloud_source=args.cloud_source,
        cloud_args=argv,
        seconds=args.seconds,
    )
    server = Portal(
        ("127.0.0.1", args.port),
        sessions,
        Auth(json.loads(args.credentials_file.read_text())),
        origin,
    )
    print(
        f"Private judge portal listening on loopback port {server.server_port}; no images served.",
        flush=True,
    )

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
