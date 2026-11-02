"""Autolab fold gate (.scratch/autolab/spec.md, decision 5; ticket 02). Pure logic over two scorer reports (the main line's
and the experiment's merged score.json), the experiment's card and its diff (`git diff -U0 <main> <tip>`).

    gate(base, new, card, diff) -> {"verdict": "pass" | "fail", "tuning": bool, "checks": [{"name", "ok", "why", ...}]}

Reports are reduced to the practice split on entry (`practice`), so no exam number reaches a check or its output. The exam
tripwire (`exam_tripwire`) takes the two exam totals separately and answers only "pass" or "blocked".
"""

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_compare_incidents import compare as compare_incidents  # noqa: E402
from lab_ledger import count  # noqa: E402

GAIN = 3
TUNING_GAIN = 5
NOISE = 2
SPREAD = 1
STUDIO_SLACK = 2
REALUSER_SLACK = 1
EXAM_SLACK = 2
COLLATERAL_SHARE = 0.3
RESAMPLES = 10_000
SEED = 20261101

PROTECTED = (
    re.compile(r"^packages/core/src/bench/lab[^/]*\.ts$"),
    re.compile(r"^tools/(lab|autolab)[^/]*$"),
    re.compile(r"^modal/(lab_|review_|autolab_)[^/]*\.py$"),
    re.compile(r"^modal/lab_web/"),
    re.compile(r"^docs/benchmarks/lab-"),
    re.compile(r"^\.scratch/(lab|autolab)/(spec\.md|issues/)"),
)
PRODUCTION = re.compile(r"^packages/[^/]+/src/(?!bench/).+\.(ts|tsx|js|mjs)$")
LITERALS = (
    ("session id", re.compile(r"lab-v\d+(\.\d+)*-[a-z]\d+")),
    ("reader id", re.compile(r"mp3quran-\d+")),
    ("word reference", re.compile(r"\b\d{1,3}:\d{1,3}:\d{1,3}\b")),
    ("ayah reference", re.compile(r"[\"'`]\d{1,3}:\d{1,3}[\"'`]")),
)
NUMBER = re.compile(r"(?<![\w.])-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?(?![\w.])")


# Reports


def practice(report: dict) -> dict:
    """The report with every exam record dropped: results, cases, the exam split and every exam slice."""
    out = {k: report[k] for k in ("testSet", "run", "sha")}
    out["results"] = [r for r in report["results"] if r["split"] == "practice"]
    out["cases"] = [c for c in report["cases"] if c["split"] == "practice"]
    out["practice"] = report["splits"]["practice"]
    out["studio"] = report["slices"]["studio"]["practice"]["overall"]
    ru = (report.get("realuser") or {}).get("practice") or {}
    out["realuser"] = ru.get("falseWarnings") if ru.get("sessions") else None
    if not out["results"]:
        raise ValueError(f"report {report['run']} has no practice sessions")
    return out


def sessions(report: dict, variant: str = "phone") -> dict[str, dict]:
    """Per practice session: reader, surah (most common among its cases), clean, and wrong calls overall and per component."""
    surah = defaultdict(Counter)
    for c in report["cases"]:
        if c.get("word"):
            surah[c["session"]][c["word"].split(":")[0]] += 1
    out = {}
    for r in report["results"]:
        v = r["variants"][variant]
        out[r["id"]] = {"reader": r["speaker"], "surah": surah[r["id"]].most_common(1)[0][0] if surah[r["id"]] else f"({r['id']})",
                        "clean": v["mistakes"] == 0, "overall": v["missed"] + v["falseWarnings"],
                        **{k: a["missed"] + a["falseWarnings"] for k, a in v["attribution"].items()}}
    return out


def totals(report: dict) -> dict:
    p = report["practice"]
    return {"overall": p["overall"]["wrongCalls"], **{k: p["attribution"][k] for k in ("listener", "tracker", "checker")}}


# Diff


def diff_files(diff: str) -> list[str]:
    return sorted({m.group(2) for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", diff, re.M)})


def hunks(diff: str) -> dict[str, list[tuple[list[str], list[str]]]]:
    """file -> [(removed lines, added lines)] per hunk of a -U0 diff."""
    out, file, cur = defaultdict(list), None, None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            file, cur = line.split(" b/", 1)[1], None
        elif line.startswith("@@"):
            cur = ([], [])
            out[file].append(cur)
        elif cur is not None and line.startswith("-") and not line.startswith("--- "):
            cur[0].append(line[1:])
        elif cur is not None and line.startswith("+") and not line.startswith("+++ "):
            cur[1].append(line[1:])
    return out


def is_tuning(diff: str) -> bool:
    """Every production hunk replaces lines one for one, each differing from its old version only in numeric literals."""
    prod = {f: hs for f, hs in hunks(diff).items() if PRODUCTION.match(f)}
    if not prod:
        return False
    norm = lambda s: NUMBER.sub("#", s).strip()
    return all(len(old) == len(new) and old != new and all(norm(a) == norm(b) for a, b in zip(old, new))
               for hs in prod.values() for old, new in hs)


def literals(diff: str) -> list[dict]:
    """Benchmark identifiers on lines a diff adds to production code."""
    out = []
    for f, hs in hunks(diff).items():
        if not PRODUCTION.match(f):
            continue
        for _, added in hs:
            for line in added:
                if line.lstrip().startswith(("//", "*", "/*")):
                    continue
                for kind, rx in LITERALS:
                    for m in rx.finditer(line):
                        out.append({"file": f, "kind": kind, "text": m.group(0)})
    return out


# Checks


def check(name: str, ok: bool, why: str, blocking: bool = True, **data) -> dict:
    """A check's result; a non-blocking check is reported but doesn't decide the verdict."""
    return {"name": name, "ok": bool(ok), "why": why, "blocking": blocking, **data}


def gain_check(b: dict, n: dict, component: str, need: int) -> dict:
    """Overall improves by `need`; the card's component gets no worse than noise (a fix in one component can show up in
    another's attribution, as checker-listener-noise did: overall -5, checker +0, as its card predicted)."""
    ch = {k: n[k] - b[k] for k in ("overall", component)}
    ok = -ch["overall"] >= need and ch[component] <= NOISE
    return check("gain", ok, f"overall {ch['overall']:+d} (needs -{need} or better), {component} {ch[component]:+d} (needs +{NOISE} or better)",
                 change=ch, need=need)


def bootstrap_upper(deltas: list[int], resamples: int = RESAMPLES, seed: int = SEED) -> float:
    """97.5th percentile of the resampled total change (sessions drawn with replacement)."""
    rng, n = random.Random(seed), len(deltas)
    sums = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return sums[int(0.975 * resamples)]


def significance_check(bs: dict, ns: dict) -> dict:
    """Reported, not blocking: scoring is deterministic, so a change of 3 incidents won and 0 lost is real on these
    sessions, yet a session bootstrap can't clear 0 when only 3 of 100 sessions move. Generalization is the exam
    tripwire's and the test set refresh's job."""
    deltas = [ns[s]["overall"] - bs[s]["overall"] for s in bs]
    hi = bootstrap_upper(deltas)
    moved = sum(d != 0 for d in deltas)
    return check("significance", hi < 0, f"95% upper bound of the overall change {hi:+d} over {len(deltas)} sessions ({moved} moved)",
                 blocking=False, upper=hi, sessionsMoved=moved)


def leave_one_out(bs: dict, ns: dict, key: str) -> dict:
    total = sum(ns[s]["overall"] - bs[s]["overall"] for s in bs)
    by = defaultdict(int)
    for s in bs:
        by[bs[s][key]] += ns[s]["overall"] - bs[s]["overall"]
    g, d = min(by.items(), key=lambda kv: kv[1], default=(None, 0))
    return {"left out": g, "remaining": total - d}


def spread_check(bs: dict, ns: dict) -> dict:
    """A gain left after leaving out the reader who gained most (readers are what the exam holds out)."""
    w = leave_one_out(bs, ns, "reader")
    return check("spread", -w["remaining"] >= SPREAD, f"without reader {w['left out']}: {w['remaining']:+d} (needs -{SPREAD} or better)", worst=w)


def surah_check(bs: dict, ns: dict) -> dict:
    """Reported, not blocking: the practice set is about a third surah 26, so leaving a surah out isn't a fair test."""
    w = leave_one_out(bs, ns, "surah")
    return check("surah spread", -w["remaining"] >= SPREAD, f"without surah {w['left out']}: {w['remaining']:+d}", blocking=False, worst=w)


def incident_key(x: dict) -> str:
    return x.get("key") or "/".join(str(x.get(k, "")) for k in ("session", "variant", "kind", "word", "class"))


def wins(base: dict, new: dict) -> list[str]:
    """The phone incidents a run fixes against its base (recovered misses and removed false warnings), by key."""
    phone = compare_incidents(practice(base), practice(new))["variants"]["phone"]
    return sorted(incident_key(x) for x in phone["recoveries"] + phone["removedWarnings"])


def collateral_check(b: dict, n: dict, bs: dict, inc: dict) -> dict:
    phone = inc["variants"]["phone"]
    clean_new = [w for w in phone["newWarnings"] if bs[w["session"]]["clean"]]
    studio = n["studio"]["wrongCalls"] - b["studio"]["wrongCalls"]
    realuser = None if b["realuser"] is None or n["realuser"] is None else n["realuser"] - b["realuser"]
    lost = len(phone["newMisses"]) + len(phone["newWarnings"])
    won = len(phone["recoveries"]) + len(phone["removedWarnings"])
    problems = []
    if clean_new:
        problems.append(f"{len(clean_new)} new false warning(s) on fully clean sessions")
    if studio > STUDIO_SLACK:
        problems.append(f"studio {studio:+d}")
    if realuser is not None and realuser > REALUSER_SLACK:
        problems.append(f"real users' clean false warnings {realuser:+d}")
    if lost > COLLATERAL_SHARE * won:
        problems.append(f"{lost} incidents lost against {won} won")
    elif lost and won - lost < TUNING_GAIN:
        problems.append(f"a trade ({won} won, {lost} lost) needs a net gain of {TUNING_GAIN}")
    why = "; ".join(problems) or (f"{won} incidents won, {lost} lost; studio {studio:+d}"
                                  + (f"; real users {realuser:+d}" if realuser is not None else ""))
    return check("collateral", not problems, why, cleanNewWarnings=len(clean_new), studio=studio, realuser=realuser, won=won, lost=lost,
                 wonKeys=sorted(incident_key(x) for x in phone["recoveries"] + phone["removedWarnings"]))


def hygiene_check(diff: str | None) -> dict:
    if diff is None:
        return check("hygiene", False, "no diff available")
    files = diff_files(diff)
    prot = [f for f in files if any(rx.match(f) for rx in PROTECTED)]
    lits = literals(diff)
    problems = ([f"protected: {', '.join(prot)}"] if prot else []) + ([f"benchmark literals: {', '.join(sorted({l['text'] for l in lits}))}"] if lits else [])
    return check("hygiene", not problems, "; ".join(problems) or f"{len(files)} files, none protected, no benchmark literals",
                 protected=prot, literals=lits)


def gate(base: dict, new: dict, card: dict, diff: str | None, *, tests_ok: bool | None = None) -> dict:
    """The fold gate on the practice split. `tests_ok` None means not run here (the supervisor's validation step runs them)."""
    b, n = practice(base), practice(new)
    if b["testSet"] != n["testSet"]:
        raise ValueError(f"test sets differ: {b['testSet']} vs {n['testSet']}")
    bs, ns = sessions(b), sessions(n)
    if bs.keys() != ns.keys():
        raise ValueError("the reports score different practice sessions")
    tuning = diff is not None and is_tuning(diff)
    checks = [
        gain_check(totals(b), totals(n), card["component"], TUNING_GAIN if tuning else GAIN),
        spread_check(bs, ns),
        collateral_check(b, n, bs, compare_incidents(b, n)),
        hygiene_check(diff),
        significance_check(bs, ns),
        surah_check(bs, ns),
    ]
    if tests_ok is not None:
        checks.append(check("tests", tests_ok, "build and tests pass" if tests_ok else "build or tests fail"))
    return {"verdict": verdict(checks), "tuning": tuning, "checks": checks}


def verdict(checks: list[dict]) -> str:
    return "pass" if all(c["ok"] for c in checks if c["blocking"]) else "fail"


def exam_tripwire(base_exam: dict, new_exam: dict, component: str) -> str:
    """'pass' or 'blocked' from two ledger exam scores (`lab_ledger.score`); never the numbers themselves."""
    worse = any(count(new_exam, k) - count(base_exam, k) > EXAM_SLACK for k in ("overall", component))
    return "blocked" if worse else "pass"


def format_gate(res: dict) -> list[str]:
    head = f"gate: {res['verdict']}" + (" (tuning diff: higher gain bar)" if res["tuning"] else "")
    mark = lambda c: "ok  " if c["ok"] else "FAIL" if c["blocking"] else "note"
    return [head] + [f"  {mark(c)} {c['name']:12} {c['why']}" for c in res["checks"]]
