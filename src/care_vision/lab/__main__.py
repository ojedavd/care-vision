"""Launch the English, loopback-only local observability lab."""

import argparse
import json
import os
import secrets
import signal
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from care_vision.lab.session import LabSessions

ASSETS = Path(__file__).parent


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, sessions, port=0):
        super().__init__(('127.0.0.1', port), Handler)
        self.sessions = sessions
        self.origin = f'http://127.0.0.1:{self.server_port}'
        self.host = f'127.0.0.1:{self.server_port}'
        self.csrf = secrets.token_urlsafe(32)
        self.stopping = threading.Event()
        self.slots = threading.BoundedSemaphore(12)
        self.ticker = threading.Thread(target=self.tick, daemon=True)
        self.ticker.start()

    def tick(self):
        while not self.stopping.wait(1):
            self.sessions.snapshot()

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

    def server_close(self):
        self.stopping.set()
        self.sessions.stop()
        super().server_close()
        self.ticker.join(timeout=2)


class Handler(BaseHTTPRequestHandler):
    server_version = 'CARE-Vision Local Lab'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def reply(self, status, content, kind='application/json'):
        if isinstance(content, dict):
            content = json.dumps(content, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Content-Security-Policy',
                         "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                         "base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(content)

    def allowed_host(self):
        return self.headers.get('Host') == self.server.host

    def do_GET(self):
        if not self.allowed_host():
            return self.reply(403, {'error': 'Untrusted host.'})
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path in assets:
            name, kind = assets[self.path]
            return self.reply(200, (ASSETS / name).read_bytes(), kind)
        if self.path == '/api/state':
            result = self.server.sessions.snapshot()
            result['csrf'] = self.server.csrf
            return self.reply(200, result)
        if self.path == '/api/preview.jpg':
            payload = self.server.sessions.preview()
            if payload:
                return self.reply(200, payload, 'image/jpeg')
            return self.reply(404, {'error': 'No opted-in current preview is available.'})
        return self.reply(404, {'error': 'Not found.'})

    def do_POST(self):
        if (not self.allowed_host() or self.headers.get('Origin') != self.server.origin
                or not secrets.compare_digest(
                    self.headers.get('X-CSRF-Token', ''), self.server.csrf)):
            return self.reply(403, {'error': 'Same-origin request and local token required.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.reply(400, {'error': 'Invalid request length.'})
        if not 0 <= length <= 1024:
            return self.reply(413, {'error': 'Request too large.'})
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, UnicodeError):
            return self.reply(400, {'error': 'JSON object required.'})
        if not isinstance(body, dict):
            return self.reply(400, {'error': 'JSON object required.'})
        try:
            if self.path == '/api/start':
                if set(body) - {'source', 'camera_index', 'seconds', 'preview'}:
                    return self.reply(400, {'error': 'Unsupported source or connection option.'})
                self.server.sessions.start(**body)
            elif self.path == '/api/stop':
                self.server.sessions.stop()
            else:
                return self.reply(404, {'error': 'Not found.'})
        except ValueError as error:
            if str(error) == 'session already running':
                return self.reply(409, {'error': 'Stop the active run before starting another.'})
            return self.reply(400, {'error': str(error)})
        except OSError:
            return self.reply(503, {'error': 'Local worker could not start.'})
        return self.reply(200, {'ok': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8781)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('The local lab requires macOS or Linux (POSIX process groups).')
    if not 0 <= args.port <= 65535:
        parser.error('port must be in 0..65535')
    root = Path(tempfile.mkdtemp(prefix='care-vision-lab-'))
    sessions = LabSessions(root)
    try:
        server = LabServer(sessions, args.port)
    except OSError as error:
        parser.exit(1, f'Cannot open the local port: {error}\nTry --port 0.\n')
    print(f'CARE-Vision Local Lab: {server.origin}', flush=True)
    print(f'Generated run evidence: {root}', flush=True)
    print('Local processing only. No automatic AWS connection or external notifications.',
          flush=True)
    if not args.no_browser:
        webbrowser.open(server.origin)

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
