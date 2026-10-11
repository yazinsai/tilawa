# /// script
# requires-python = ">=3.12"
# ///
"""Serve docs/benchmarks locally and open the scoreboard dashboard.

    uv run tools/scoreboard_serve.py [--port 8765] [--no-open]

Also serves, read-only and on 127.0.0.1 only, the private per-clip data behind the mistake review:
/clips/ (pnpm scoreboard's clip dump), /audio/help/ (help clip wavs) and /audio/leadin/ (tlog lead-in/out wavs).
POST /api/replay runs a bounded local component test. It does not update benchmark files.
"""

import argparse
import functools
import http.server
import json
import subprocess
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs/benchmarks"
MOUNTS = {
    "/clips/": ROOT / ".scratch/scoreboard/clips",
    "/audio/help/": ROOT / ".scratch/help/wav",
    "/audio/leadin/": ROOT / ".scratch/leadin/audio",
}
REPLAY_LOCK = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if urlsplit(self.path).path != "/api/replay":
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed or (origin and origin != f"http://{host}"):
            self.reply_json(403, {"error": "Replay must be opened from this local scoreboard."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self.reply_json(415, {"error": "Replay needs a JSON request."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                self.reply_json(413, {"error": "Replay request is too large or empty."})
                return
            request = json.loads(self.rfile.read(size))
        except (ValueError, json.JSONDecodeError):
            self.reply_json(400, {"error": "Replay request is invalid."})
            return
        if not REPLAY_LOCK.acquire(blocking=False):
            self.reply_json(409, {"error": "Another replay is running. Try again when it finishes."})
            return
        try:
            result = subprocess.run(
                ["node", "--import", "tsx", "src/bench/component-replay-cli.ts"],
                cwd=ROOT / "packages/core", input=json.dumps(request), text=True,
                capture_output=True, timeout=120,
            )
            try:
                data = json.loads(result.stdout)
            except ValueError:
                data = {"error": "Replay could not start. Install the project dependencies and build the core package."}
            self.reply_json(400 if result.returncode or "error" in data else 200, data)
        except subprocess.TimeoutExpired:
            self.reply_json(504, {"error": "Replay took too long. Try a shorter recording."})
        except OSError:
            self.reply_json(503, {"error": "Replay needs Node.js and the project dependencies."})
        finally:
            REPLAY_LOCK.release()

    def reply_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, base in MOUNTS.items():
            if clean.startswith(prefix):
                target = (base / clean[len(prefix):]).resolve()
                return str(target) if target.is_relative_to(base.resolve()) else str(base / "__denied__")
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/scoreboard.html"
    handler = functools.partial(Handler, directory=str(DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Scoreboard: {url}", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
