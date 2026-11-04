"""`lab new` after an interrupted one: python3 -m pytest tools/test_lab_new.py"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab  # noqa: E402


def g(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_only_a_leftover_branch_with_no_work_is_dropped(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    g(r, "init", "-q", "-b", "main")
    g(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "base")
    g(r, "branch", "exp/left")  # what an interrupted `new` leaves
    assert lab.drop_leftover_branch("exp/left", "main", cwd=r) and not g(r, "branch", "--list", "exp/left")
    g(r, "checkout", "-q", "-b", "exp/work")
    g(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "work")
    g(r, "checkout", "-q", "main")
    assert not lab.drop_leftover_branch("exp/work", "main", cwd=r) and g(r, "branch", "--list", "exp/work")
    g(r, "worktree", "add", "-q", "-b", "exp/busy", str(tmp_path / "busy"), "main")
    assert not lab.drop_leftover_branch("exp/busy", "main", cwd=r) and g(r, "branch", "--list", "exp/busy")
    assert not lab.drop_leftover_branch("exp/none", "main", cwd=r)
