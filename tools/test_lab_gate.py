# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_gate.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_gate as G  # noqa: E402

COMPONENTS = ("listener", "tracker", "checker")


def session(id, reader, surah, misses=0, hits=2, warnings=(), component="checker"):
    """A practice session: `misses` missed mistakes charged to `component`, `hits` caught ones, false warnings as (word, timeS)."""
    return {"id": id, "reader": reader, "surah": surah, "misses": misses, "hits": hits, "warnings": list(warnings), "component": component}


def report(spec, run="r", sha="abc1234", studio=10, realuser=None, exam=None, ts="lab-v9"):
    results, cases = [], []
    att_m, att_f = dict.fromkeys(COMPONENTS, 0), dict.fromkeys(COMPONENTS, 0)
    for s in spec:
        n = s["misses"] + s["hits"]
        for i in range(n):
            miss = i < s["misses"]
            cases.append({"key": f"{s['id']}/phone/mistake:wrong_word/{s['surah']}:1:{i + 1}/{i}.0", "kind": "mistake", "session": s["id"],
                          "variant": "phone", "split": "practice", "errorClass": "wrong_word", "word": f"{s['surah']}:1:{i + 1}",
                          "startS": float(i), "endS": i + 0.5, "outcome": "miss" if miss else "hit", "component": s["component"] if miss else None})
        for word, t in s["warnings"]:
            cases.append({"key": f"{s['id']}/phone/fw/{word}/{t}", "kind": "false_warning", "session": s["id"], "variant": "phone",
                          "split": "practice", "errorClass": "wrong_word", "word": word, "timeS": t, "component": s["component"]})
        att = {c: {"missed": 0, "falseWarnings": 0} for c in COMPONENTS}
        att[s["component"]] = {"missed": s["misses"], "falseWarnings": len(s["warnings"])}
        att_m[s["component"]] += s["misses"]
        att_f[s["component"]] += len(s["warnings"])
        results.append({"id": s["id"], "split": "practice", "speaker": s["reader"], "durationS": 100.0, "recitedS": 90.0,
                        "variants": {"phone": {"mistakes": n, "missed": s["misses"], "falseWarnings": len(s["warnings"]), "attribution": att},
                                     "studio": {"mistakes": n, "missed": 0, "falseWarnings": 0}}})
    missed, fw = sum(att_m.values()), sum(att_f.values())
    practice = {"overall": {"wrongCalls": missed + fw, "missed": missed, "falseWarnings": fw},
                "attribution": {**{c: att_m[c] + att_f[c] for c in COMPONENTS}, "total": missed + fw}}
    splits = {"practice": practice, "exam": exam or {"overall": {"wrongCalls": 0}, "attribution": dict.fromkeys(COMPONENTS, 0)}}
    if exam:
        results.append({"id": "exam-1", "split": "exam", "speaker": "secret", "durationS": 1.0, "recitedS": 1.0, "variants": {}})
        cases.append({"key": "exam-1/x", "split": "exam", "session": "exam-1", "variant": "phone", "kind": "mistake"})
    return {"kind": "lab-score", "testSet": ts, "run": run, "sha": sha, "splits": splits, "results": results, "cases": cases,
            "slices": {"studio": {"practice": {"overall": {"wrongCalls": studio}}, "exam": {"overall": {"wrongCalls": 99}}}},
            "realuser": {"practice": {"sessions": 3 if realuser is not None else 0, "falseWarnings": realuser or 0}}}


CARD = {"id": "x", "component": "checker"}
CODE = "diff --git a/packages/core/src/checker.ts b/packages/core/src/checker.ts\n@@ -10 +10,2 @@\n-  return a;\n+  if (b) return c;\n+  return a;\n"


def spread_base():
    """Ten sessions over five readers and five surahs, each with one missed mistake, plus two clean sessions."""
    return [session(f"p{i}", f"r{i % 5}", str(10 + i % 5), misses=1) for i in range(10)] + \
        [session("c1", "r0", "50", hits=0), session("c2", "r1", "51", hits=0)]


def recover(spec, ids):
    return [{**s, "misses": 0, "hits": s["hits"] + s["misses"]} if s["id"] in ids else s for s in spec]


def names(res, ok=False):
    return [c["name"] for c in res["checks"] if c["ok"] == ok]


def blocking(res):
    return [c["name"] for c in res["checks"] if not c["ok"] and c["blocking"]]


def test_spread_gain_passes():
    base = spread_base()
    res = G.gate(report(base, "b"), report(recover(base, {"p0", "p1", "p2", "p3", "p4", "p5"}), "n"), CARD, CODE)
    assert res["verdict"] == "pass", G.format_gate(res)
    assert res["checks"][0]["change"] == {"overall": -6, "checker": -6}


def test_gain_from_one_reader_fails_spread():
    base = spread_base() + [session("big", "r9", "99", misses=6, hits=0)]
    res = G.gate(report(base, "b"), report(recover(base, {"big"}), "n"), CARD, CODE)
    assert res["verdict"] == "fail"
    assert blocking(res) == ["spread"]
    assert {"significance", "surah spread"} <= set(names(res))


def test_sparse_deterministic_gain_passes_with_notes():
    """Three incidents won in three sessions, none lost: real on these sessions, though no bootstrap clears 0."""
    base = [session(f"p{i}", f"r{i}", "26", misses=1 if i < 3 else 0) for i in range(30)]
    res = G.gate(report(base, "b"), report(recover(base, {"p0", "p1", "p2"}), "n"), CARD, CODE)
    assert res["verdict"] == "pass", G.format_gate(res)
    assert {"significance", "surah spread"} <= set(names(res))


def test_small_gain_fails():
    base = spread_base()
    res = G.gate(report(base, "b"), report(recover(base, {"p0", "p1"}), "n"), CARD, CODE)
    assert "gain" in blocking(res)


def test_gain_charged_to_another_component_passes_if_the_card_component_holds():
    base = [{**s, "component": "listener"} for s in spread_base()]
    res = G.gate(report(base, "b"), report(recover(base, {f"p{i}" for i in range(6)}), "n"), CARD, CODE)
    assert res["checks"][0]["change"] == {"overall": -6, "checker": 0}
    assert res["verdict"] == "pass"


def test_component_worse_than_noise_fails():
    """Nine listener misses recovered while the checker loses three: overall -6, but the card's component is +3."""
    base = [{**s, "component": "listener"} for s in spread_base()] + [session("l", "r3", "60", hits=3, component="checker")]
    new = [{**s, "misses": 3, "hits": 0} if s["id"] == "l" else s for s in recover(base, {f"p{i}" for i in range(9)})]
    res = G.gate(report(base, "b"), report(new, "n"), CARD, CODE)
    assert res["checks"][0]["change"] == {"overall": -6, "checker": 3}
    assert "gain" in blocking(res)


def test_new_warning_on_clean_session_fails_collateral():
    base = spread_base()
    new = recover(base, {f"p{i}" for i in range(6)})
    new = [{**s, "warnings": [("50:1:1", 3.0)]} if s["id"] == "c1" else s for s in new]
    res = G.gate(report(base, "b"), report(new, "n"), CARD, CODE)
    coll = next(c for c in res["checks"] if c["name"] == "collateral")
    assert not coll["ok"] and coll["cleanNewWarnings"] == 1


def test_a_trade_needs_net_five():
    base = spread_base() + [session("t", "r4", "70", hits=1)]
    lose_one = lambda new: [{**s, "misses": 1, "hits": 0} if s["id"] == "t" else s for s in new]
    small = G.gate(report(base, "b"), report(lose_one(recover(base, {f"p{i}" for i in range(4)})), "n"), CARD, CODE)
    assert "collateral" in blocking(small)
    big = G.gate(report(base, "b"), report(lose_one(recover(base, {f"p{i}" for i in range(6)})), "n"), CARD, CODE)
    assert big["verdict"] == "pass", G.format_gate(big)


def test_studio_and_realuser_regressions_fail_collateral():
    base = spread_base()
    new = recover(base, {f"p{i}" for i in range(6)})
    assert "collateral" in names(G.gate(report(base, "b", studio=10), report(new, "n", studio=13), CARD, CODE))
    assert "collateral" in names(G.gate(report(base, "b", realuser=1), report(new, "n", realuser=3), CARD, CODE))
    assert G.gate(report(base, "b", studio=10, realuser=1), report(new, "n", studio=12, realuser=2), CARD, CODE)["verdict"] == "pass"


def test_tuning_diff_needs_the_higher_bar():
    tuning = "diff --git a/packages/core/src/checker.ts b/packages/core/src/checker.ts\n@@ -3 +3 @@\n-const SUPPORT = 10;\n+const SUPPORT = 20.5;\n"
    assert G.is_tuning(tuning) and not G.is_tuning(CODE)
    base = spread_base()
    new = recover(base, {f"p{i}" for i in range(4)})
    assert G.gate(report(base, "b"), report(new, "n"), CARD, CODE)["checks"][0]["ok"]
    res = G.gate(report(base, "b"), report(new, "n"), CARD, tuning)
    assert res["tuning"] and "gain" in blocking(res)


def test_tuning_ignores_tests_and_docs():
    diff = ("diff --git a/packages/core/test/x.test.ts b/packages/core/test/x.test.ts\n@@ -1 +1 @@\n-expect(1)\n+expect(2)\n"
            "diff --git a/.scratch/lab/experiments/x.md b/.scratch/lab/experiments/x.md\n@@ -1 +1 @@\n-3\n+4\n")
    assert not G.is_tuning(diff)


def test_hygiene_protected_paths_and_literals():
    diff = ("diff --git a/packages/core/src/bench/lab.ts b/packages/core/src/bench/lab.ts\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/packages/core/src/checker.ts b/packages/core/src/checker.ts\n@@ -1,0 +1,3 @@\n"
            "+if (session === 'lab-v1.5-p17') skip();\n+const W = '2:8:1';\n+// see 26:115:1 in the write-up\n")
    h = G.hygiene_check(diff)
    assert not h["ok"]
    assert h["protected"] == ["packages/core/src/bench/lab.ts"]
    assert sorted({l["kind"] for l in h["literals"]}) == ["session id", "word reference"]
    assert G.hygiene_check(CODE)["ok"]
    assert not G.hygiene_check(None)["ok"]


def test_practice_drops_every_exam_record():
    r = G.practice(report(spread_base(), exam={"overall": {"wrongCalls": 7}, "attribution": dict.fromkeys(COMPONENTS, 2)}))
    assert all(x["split"] == "practice" for x in r["results"] + r["cases"])
    assert "exam" not in str(r)


def test_exam_tripwire_says_only_pass_or_blocked():
    t = lambda o, c: {"wrongCalls": o, "checker": {"wrongCalls": c}}
    assert G.exam_tripwire(t(20, 10), t(22, 12), "checker") == "pass"
    assert G.exam_tripwire(t(20, 10), t(23, 10), "checker") == "blocked"
    assert G.exam_tripwire(t(20, 10), t(19, 13), "checker") == "blocked"


def test_bootstrap_is_seeded():
    d = [-1, 0, 0, -1, 1, -1, 0, 0, -2, 0]
    assert G.bootstrap_upper(d) == G.bootstrap_upper(d)
    assert G.bootstrap_upper([-1] * 10) == -10
