# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1"]
# ///
"""Teacher/student PER breakdown on the tlog-v2 clean-match test split: edit types, phoneme pairs, normalised PER.

    modal run modal/student_decode_cpu.py --run e2-libri-full
    cp /tmp/e2-libri-full-clean-match-test.jsonl .scratch/bench/per-breakdown/
    uv run tools/per_breakdown.py               # e2-libri-full -> teacher-per-breakdown-v1
    uv run tools/per_breakdown.py <run>         # any other run -> per-breakdown-<run>-v1
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
RUN = sys.argv[1] if len(sys.argv) > 1 else "e2-libri-full"
SRC = ROOT / f".scratch/bench/per-breakdown/{RUN}-clean-match-test.jsonl"
OUT = ROOT / ("docs/benchmarks/teacher-per-breakdown-v1" if RUN == "e2-libri-full" else f"docs/benchmarks/per-breakdown-{RUN}-v1")

VOWELS = set("\u064e\u064f\u0650\u06ea\u0619")
MADD = set("اۥۦ")
NASAL = set("نمں۾")
QALQALA = "ڇ"
# Ikhfa nasal vs plain nasal is always a spelling choice. Idgham and iqlab pairs count only inside a run
# (مِللَ vs مِننَ), so a lone ل→ن stays a substitution.
NASAL_SPELLING = {frozenset(p) for p in ("نں", "م۾", "ں۾")}
ASSIMILATION = {frozenset(p) for p in ("نم", "نل", "نر", "ںو", "ںي", "مب", "۾ب", "ںم")}
CONFUSABLE = {frozenset(p) for p in ("حه", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس")}
ANCHOR = 3

REPRESENTATION = ("edge", "madd_length", "ghunna_length", "assimilation", "qalqala")
AMBIGUOUS = ("gemination", "vowel_insdel", "madd_insdel", "madd_vowel_sub")
MISHEARING = ("vowel_sub", "confusable_sub", "consonant_sub", "consonant_insdel")
KINDS = REPRESENTATION + AMBIGUOUS + MISHEARING


def collapse(s: str) -> str:
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def canon(s: str) -> str:
    """Collapse repeats, write ikhfa nasals as plain ن/م, drop qalqala."""
    return collapse(s.replace("ں", "ن").replace("۾", "م").replace(QALQALA, ""))


def anchors(ref: str, hyp: str) -> tuple[int, int]:
    """Ref span between the first and last exact match of >= ANCHOR phonemes; edits outside it are clip-edge edits."""
    eq = [o for o in Levenshtein.opcodes(ref, hyp) if o.tag == "equal" and o.src_end - o.src_start >= ANCHOR]
    return (eq[0].src_start, eq[-1].src_end) if eq else (len(ref) + 1, -1)


def neighbours(s: str, k: int, skip: bool) -> set[str]:
    """Phonemes either side of position k in s; with `skip`, s[k] is the edited phoneme and is left out."""
    return {s[x] for x in ((k - 1, k + 1) if skip else (k - 1, k)) if 0 <= x < len(s)}


def run_at(s: str, k: int) -> int:
    if not 0 <= k < len(s):
        return 0
    lo = hi = k
    while lo > 0 and s[lo - 1] == s[k]:
        lo -= 1
    while hi + 1 < len(s) and s[hi + 1] == s[k]:
        hi += 1
    return hi - lo + 1


def kind(op: str, a: str, b: str, ref: str, hyp: str, i: int, j: int) -> str:
    if op == "replace":
        pair = frozenset((a, b))
        near = neighbours(ref, i, True) | neighbours(hyp, j, True)
        if pair in (NASAL_SPELLING) or (pair in ASSIMILATION and (a in near or b in near)):
            return "assimilation"
        if pair in CONFUSABLE:
            return "confusable_sub"
        if a in VOWELS and b in VOWELS:
            return "vowel_sub"
        if (a in MADD | VOWELS) and (b in MADD | VOWELS):
            return "madd_vowel_sub"
        if a in "وي" and b in MADD and run_at(ref, i) >= 2:
            return "madd_vowel_sub"
        return "consonant_sub"
    c = a or b
    near = neighbours(ref, i, op == "delete") | neighbours(hyp, j, op == "insert")
    if c == QALQALA:
        return "qalqala"
    if c in near and c in MADD:
        return "madd_length"
    if c in near and (c in NASAL or (c in "وي" and max(run_at(ref, i), run_at(hyp, j)) >= 3)):
        return "ghunna_length"
    if c in near and c not in VOWELS:
        return "gemination"
    if c in VOWELS:
        return "vowel_insdel"
    if c in MADD:
        return "madd_insdel"
    return "consonant_insdel"


def edits(ref: str, hyp: str):
    lo, hi = anchors(ref, hyp)
    for op, i, j in Levenshtein.editops(ref, hyp):
        a = ref[i] if op != "insert" else ""
        b = hyp[j] if op != "delete" else ""
        zone = "start" if i < lo or (op == "insert" and i == lo) else "end" if i >= hi else "interior"
        yield dict(op={"replace": "sub", "insert": "ins", "delete": "del"}[op], a=a, b=b, zone=zone,
                   kind="edge" if zone != "interior" else kind(op, a, b, ref, hyp, i, j))


def spliced(ref: str, teacher: str) -> str:
    """Reference between the teacher-anchored span, with the teacher's own phonemes outside it."""
    eq = [o for o in Levenshtein.opcodes(ref, teacher) if o.tag == "equal" and o.src_end - o.src_start >= ANCHOR]
    if not eq:
        return teacher
    return teacher[: eq[0].dest_start] + ref[eq[0].src_start : eq[-1].src_end] + teacher[eq[-1].dest_end :]


def edge_kind(ref: str, hyp: str, zone: str) -> str:
    """Coarse label for a clip's start or end region."""
    ops = [o for o in Levenshtein.opcodes(ref, hyp) if o.tag == "equal" and o.src_end - o.src_start >= ANCHOR]
    if not ops:
        return "no_anchor"
    if zone == "start":
        r, h = ref[: ops[0].src_start], hyp[: ops[0].dest_start]
        if "بِسمِللَااهِ" in h and "بِسمِللَااهِ" not in r:
            return "basmala"
        if "ءَعُۥۥذُ" in h and "ءَعُۥۥذُ" not in r:
            return "istiadha"
        if h.startswith("ء") and not r.startswith("ء") and len(h) <= len(r) + 2:
            return "hamzat_wasl"
    else:
        r, h = ref[ops[-1].src_end :], hyp[ops[-1].dest_end :]
        if r and r[-1] in VOWELS and Levenshtein.distance(r[:-1], h) < Levenshtein.distance(r, h) and len(r) - len(h) <= 2:
            return "waqf"
    return "ref_longer" if len(r) > len(h) else "hyp_longer" if len(h) > len(r) else "same_length"


def per(pairs, f=lambda s: s, interior=False) -> float:
    e = n = 0
    for h, r in pairs:
        h, r = f(h), f(r)
        n += len(r)
        if interior:
            e += sum(1 for x in edits(r, h) if x["zone"] == "interior")
        else:
            e += Levenshtein.distance(h, r)
    return e / max(1, n)


def ladder(pairs) -> dict:
    return {
        "raw": per(pairs),
        "collapse": per(pairs, collapse),
        "canon": per(pairs, canon),
        "raw_interior": per(pairs, interior=True),
        "canon_interior": per(pairs, canon, interior=True),
    }


def breakdown(rows, hyp_key: str) -> dict:
    n_ref = sum(len(r["ref"]) for r in rows)
    by_kind, by_op, by_pair, by_zone = Counter(), Counter(), Counter(), Counter()
    pair_kind = {}
    edge = Counter()
    examples = defaultdict(list)
    for r in rows:
        ref, hyp = r["ref"], r[hyp_key]
        for x in edits(ref, hyp):
            by_kind[x["kind"]] += 1
            by_op[x["op"]] += 1
            by_zone[x["zone"]] += 1
            if x["zone"] == "interior":
                p = (x["op"], x["a"], x["b"])
                by_pair[p] += 1
                pair_kind[p] = x["kind"]
                if len(examples[x["kind"]]) < 3:
                    examples[x["kind"]].append(r["key"])
        for z in ("start", "end"):
            ek = edge_kind(ref, hyp, z)
            ne = sum(1 for x in edits(ref, hyp) if x["zone"] == z)
            if ne:
                edge[f"{z}:{ek}"] += ne
    total = sum(by_kind.values())

    def share(c):
        return {k: {"edits": v, "perPoints": v / n_ref} for k, v in c.most_common()}

    return {
        "edits": total,
        "refPhonemes": n_ref,
        "per": total / n_ref,
        "byOp": share(by_op),
        "byZone": share(by_zone),
        "byKind": {k: {"edits": by_kind[k], "perPoints": by_kind[k] / n_ref} for k in KINDS},
        "groups": {g: sum(by_kind[k] for k in ks) / n_ref for g, ks in
                   (("representation", REPRESENTATION), ("ambiguous", AMBIGUOUS), ("mishearing", MISHEARING))},
        "edge": share(edge),
        "topInteriorPairs": [{"op": o, "ref": a, "hyp": b, "kind": pair_kind[(o, a, b)], "edits": v}
                             for (o, a, b), v in by_pair.most_common(30)],
        "examples": dict(examples),
    }


def clip_exact(rows, hyp_key: str, f) -> float:
    return sum(f(r[hyp_key]) == f(r["ref"]) for r in rows) / len(rows)


def main():
    rows = [json.loads(line) for line in open(SRC)]
    pairs = {
        "teacherVsRef": [(r["teacher"], r["ref"]) for r in rows],
        "studentVsRef": [(r["student"], r["ref"]) for r in rows],
        "studentVsTeacher": [(r["student"], r["teacher"]) for r in rows],
        "teacherVsSpliced": [(r["teacher"], spliced(r["ref"], r["teacher"])) for r in rows],
        "studentVsSpliced": [(r["student"], spliced(r["ref"], r["teacher"])) for r in rows],
    }
    starts_mid = sum(r["teacher"][:1] in VOWELS | MADD for r in rows)
    interior_clean = sum(not any(x["zone"] == "interior" for x in edits(canon(r["ref"]), canon(r["teacher"]))) for r in rows)
    out = {
        "version": "v1",
        "evalSet": "tlog-v2 clean-match test, 3000 clips, cluster-held-out",
        "student": f"{RUN}, greedy, chunk 16 / left 128, CPU fp32",
        "normalisations": {
            "collapse": "merge runs of the same phoneme into one",
            "canon": "ں→ن, ۾→م, drop ڇ (qalqala), then collapse",
            "interior": "count only edits between the first and last exact match of >= 3 phonemes",
            "spliced": "reference = teacher phonemes outside the teacher-anchored span + reference inside it",
        },
        "per": {k: ladder(v) for k, v in pairs.items()},
        "clipExact": {k: {"raw": clip_exact(rows, k, lambda s: s), "canon": clip_exact(rows, k, canon)}
                      for k in ("teacher", "student")},
        "teacherStartsOnVowel": starts_mid,
        "teacherCanonInteriorExact": interior_clean,
        "teacher": breakdown(rows, "teacher"),
        "student": breakdown(rows, "student"),
    }
    OUT.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: o[k] for k in ("per", "clipExact")}, indent=1))
    for who in ("teacher", "student"):
        b = o[who]
        print(who, json.dumps({k: b[k] for k in ("per", "byOp", "byZone", "groups", "edge")}, ensure_ascii=False))
        print({k: round(v["perPoints"] * 100, 2) for k, v in b["byKind"].items()})
        for p in b["topInteriorPairs"][:25]:
            print(" ", p)
