"""Lab CLI (.scratch/lab/spec.md).

    python tools/lab.py run [worktree] --shards 4 --cmd '<command>'   # runs modal/lab_run.py from the worktree
"""

import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="op", required=True)
r = sub.add_parser("run", help="run a command over shards on Modal against a worktree's source")
r.add_argument("worktree", nargs="?", default=".")
r.add_argument("--shards", type=int, default=1)
r.add_argument("--name", default="adhoc")
r.add_argument("--cmd", required=True)
a = p.parse_args()

if a.op == "run":
    wt = Path(a.worktree).resolve()
    sys.exit(subprocess.run(["modal", "run", "modal/lab_run.py", "--shards", str(a.shards), "--name", a.name, "--cmd", a.cmd], cwd=wt).returncode)
