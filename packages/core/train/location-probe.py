"""Read-only availability probe; avoids shell quoting of inline Python."""
import collections
import json
import os
from pathlib import Path

root = Path(os.environ["LAB_DATA"]) / "help-train"
rows = [json.loads(s) for s in (root / "labels.jsonl").read_text().splitlines()]
print(json.dumps({"recordings": len(rows), "splits": dict(collections.Counter(r.get("split") for r in rows)),
                  "kinds": dict(collections.Counter(r.get("kind") for r in rows)), "sample": rows[:2]}))
v = os.statvfs("/scratch")
print("scratch available bytes", v.f_bavail * v.f_frsize)
