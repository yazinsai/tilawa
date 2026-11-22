"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_jumps_build.py"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jumps_build import Clip, plan  # noqa: E402

LOCAL = 2


def pool() -> list[Clip]:
    """Two long covered stretches and several short ones, one clip per ayah ordinal, 10 s each."""
    clips = []
    for start, n in [(100, 30), (400, 25), (900, 6), (1200, 7), (1500, 8), (2000, 6), (2500, 9)]:
        for o in range(start, start + n):
            # A missing ayah inside the first stretch: still local.
            if o == 110:
                continue
            clips.append(Clip(id=f"c{o}", ordinals=[o], duration_s=10.0, speaker=str(o % 3)))
    return clips


def sessions():
    return plan(pool(), random.Random(0), hard=set(), min_s=180, max_s=600, min_run_s=50)


def steps(s):
    segs = s["segments"]
    return [b["ordinals"][0] - a["ordinals"][-1] for a, b in zip(segs, segs[1:])]


def test_contiguous_sessions_only_step_locally_and_fit_the_length():
    contiguous = [s for s in sessions() if s["kind"] == "contiguous"]
    assert contiguous
    for s in contiguous:
        assert all(1 <= d <= LOCAL for d in steps(s))
        assert 180 <= sum(g["duration_s"] for g in s["segments"]) <= 600


def test_jump_sessions_plant_one_to_three_far_jumps():
    jumps = [s for s in sessions() if s["kind"] == "jump"]
    assert jumps
    for s in jumps:
        far = [d for d in steps(s) if not 1 <= d <= LOCAL]
        assert 1 <= len(far) <= 3
        assert all(abs(d) > LOCAL for d in far)


def test_no_clip_is_used_twice():
    used = [g["clip"] for s in sessions() for g in s["segments"]]
    assert len(used) == len(set(used))
