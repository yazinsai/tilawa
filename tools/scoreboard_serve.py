# /// script
# requires-python = ">=3.12"
# ///
"""Serve docs/benchmarks locally and open the scoreboard dashboard.

    uv run tools/scoreboard_serve.py [--port 8765] [--no-open]
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs/benchmarks"

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--no-open", action="store_true")
args = parser.parse_args()

url = f"http://127.0.0.1:{args.port}/scoreboard.html"
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIR))
server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
print(f"Scoreboard: {url}", flush=True)
if not args.no_open:
    webbrowser.open(url)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
