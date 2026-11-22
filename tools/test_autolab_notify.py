# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_autolab_notify.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_notify as N  # noqa: E402
import autolab_state as S  # noqa: E402


def test_each_kind_sends_once_per_key():
    st, sent = S.empty(), []
    sender = lambda cfg, kind, title, body: sent.append(kind)
    for kind in N.PRIORITY:
        assert N.notify(st, {}, kind, "k", "t", "b", sender=sender)
        assert not N.notify(st, {}, kind, "k", "t", "b", sender=sender)
    assert sent == list(N.PRIORITY)
    assert N.notify(st, {}, "needs-human", "other", "t", "b", sender=sender)


def test_ntfy_without_a_topic_sends_nothing():
    N.ntfy({"notify": {"topic": ""}}, "digest", "t", "b")


def test_digest_counts_the_day():
    st = S.empty()
    st["main"] = {"practice": {"wrongCalls": 85, "missed": 60, "falseWarnings": 25}}
    st["needsHuman"] = [{"id": "x", "what": "w", "at": "t"}]
    ev = [S.event("a", "folding", "folded", "2026-11-01T03:00:00+00:00"), S.event("b", "scoring", "discarded", "2026-11-01T04:00:00+00:00"),
          S.event("c", "scoring", "discarded", "2026-10-31T04:00:00+00:00")]
    title, body = N.digest(st, ev, "2026-11-01")
    assert title == "Autolab 2026-11-01: 1 kept, 1 dropped"
    assert "Practice wrong calls now: 85" in body and "Kept: a." in body and "Waiting on you: 1" in body
