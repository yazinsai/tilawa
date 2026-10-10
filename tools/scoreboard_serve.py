# /// script
# requires-python = ">=3.12"
# ///
"""Serve docs/benchmarks locally and open the scoreboard dashboard.

    uv run tools/scoreboard_serve.py [--port 8765] [--no-open]

Also serves, read-only and on 127.0.0.1 only, the private per-clip data behind the mistake review:
/clips/ (pnpm scoreboard's clip dump), /audio/help/ (help clip wavs) and /audio/leadin/ (tlog lead-in/out wavs).
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs/benchmarks"
MOUNTS = {
    "/clips/": ROOT / ".scratch/scoreboard/clips",
    "/audio/help/": ROOT / ".scratch/help/wav",
    "/audio/leadin/": ROOT / ".scratch/leadin/audio",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, base in MOUNTS.items():
            if clean.startswith(prefix):
                target = (base / clean[len(prefix):]).resolve()
                return str(target) if target.is_relative_to(base.resolve()) else str(base / "__denied__")
        return super().translate_path(path)


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
