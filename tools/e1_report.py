# /// script
# requires-python = ">=3.12"
# dependencies = ["quran-transcript==0.6.4", "rapidfuzz==3.14.1", "pyarrow", "numpy"]
# ///
"""E1 teacher-bias report: per v1 pronunciation class, how often the teacher shows a labelled error.

Reads the teacher pass from modal/e1_pass.py. Each clip's labelled "as said" transcript and the
teacher's greedy decode are aligned against the ayah's canonical phonemes. A site counts only where
the dataset's most common label for that ayah agrees with the canonical, so systematic labelling or
alphabet-mapping differences are never counted as errors.

    modal volume get tilawa-bench e1 .scratch/bench/
    uv run --python cpython-3.12-macos-aarch64-none tools/e1_report.py
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import INVENTORY, PROFILE  # noqa: E402

REPORT_VERSION = "v1"
ROOT = Path(__file__).resolve().parent.parent
PASS = ROOT / ".scratch/bench/e1"
OUT = ROOT / "docs/benchmarks" / f"e1-teacher-bias-{REPORT_VERSION}"

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
VOWELS = (FATHA, DAMMA, KASRA)
PAIRS = [("ح", "ه"), ("ع", "ء"), ("ص", "س"), ("ض", "د"), ("ط", "ت"), ("ق", "ك"), ("ذ", "ز"), ("ث", "س")]
# Madd, length and ghunna marks (ں ikhfa noon, ۾ ikhfa meem) repeat for duration, not gemination.
NOT_CONSONANT = set(VOWELS) | {"ا", "ۥ", "ۦ", "ڇ", "ں", "۾", "\u06ea", "\u0640", "\u0672", "\u06dc", "\u0619"}
MAX_CLIP_PER = 0.5
MIN_POSITIVES = 20
VIABLE = dict(recall=0.5, fp=0.02)
POSTERIOR_ONLY = dict(recall=0.5, fp=0.05)
LOG_FLOOR = np.log(1e-6)

# azizbekphd/ikhlas_recitations uses its own 25-symbol alphabet (see its README).
IKHLAS_MAP = {
    "ﹸ": DAMMA, "ﹶ": FATHA, "َ": FATHA, "ࣵ": FATHA, "ﹺ": KASRA, "ِ": KASRA,
    "ھ": "ه", "ه": "ه", "ڶ": "ل", "ۮ": "دڇ", "ݚ": "د", "ࢧ": "م", "ڽ": "ن",
    "ٞ": FATHA + "اا", "ﹰ": FATHA + "اا", "ٌ": DAMMA + "ۥۥ", "ﱞ": DAMMA + "ۥۥۥ",
}


def ikhlas_to_teacher(label: str) -> str:
    out = ""
    for ch in label:
        m = IKHLAS_MAP.get(ch, ch)
        # The dataset marks one fatha with two codepoints in places (يَﹶ); keep one.
        if m == FATHA and out.endswith(FATHA):
            continue
        out += m
    return out


def canonicals() -> dict[str, str]:
    moshaf = MoshafAttributes(**PROFILE)
    out = {f"112:{a}": quran_phonetizer(Aya(112, a).get().uthmani, moshaf, remove_spaces=True, sura_idx=112).phonemes for a in range(1, 5)}
    return out


def align(ref: str, hyp: str) -> list[int | None]:
    """For each ref index, the hyp index it is aligned to (None when deleted)."""
    pos: list[int | None] = [None] * len(ref)
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag in ("equal", "replace"):
            for k in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                pos[op.src_start + k] = op.dest_start + k
    return pos


def runs(s: str) -> list[tuple[int, int]]:
    """Maximal runs of one consonant of length >= 2, as (start, end)."""
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i >= 2 and s[i] not in NOT_CONSONANT:
            out.append((i, j))
        i = j
    return out


def log_probs(row: dict) -> np.ndarray:
    n = row["n_frames"]
    ids = np.frombuffer(row["top5_ids"], dtype=np.uint8).reshape(n, 5)
    p = np.frombuffer(row["top5_p"], dtype=np.float16).reshape(n, 5).astype(np.float64)
    lp = np.full((n, len(INVENTORY) + 1), LOG_FLOOR)
    np.put_along_axis(lp, ids.astype(np.int64), np.log(np.maximum(p, 1e-12)), axis=1)
    return lp


TOKEN = {c: i + 1 for i, c in enumerate(INVENTORY)}


def viterbi(lp: np.ndarray, seq: str) -> tuple[float, list[tuple[int, int]] | None]:
    """Best CTC path score for seq, and each token's frame span [start, end)."""
    labels = [0]
    for c in seq:
        labels += [TOKEN[c], 0]
    T, S = lp.shape[0], len(labels)
    if T < len(seq):
        return -np.inf, None
    score = np.full((T, S), -np.inf)
    back = np.zeros((T, S), dtype=np.int8)
    score[0, 0] = lp[0, labels[0]]
    if S > 1:
        score[0, 1] = lp[0, labels[1]]
    lab = np.array(labels)
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[2:] = (lab[2:] != 0) & (lab[2:] != lab[:-2])
    for t in range(1, T):
        prev = score[t - 1]
        cand = np.stack([prev, np.r_[-np.inf, prev[:-1]], np.where(skip_ok, np.r_[-np.inf, -np.inf, prev[:-2]], -np.inf)])
        back[t] = cand.argmax(axis=0)
        score[t] = cand.max(axis=0) + lp[t, lab]
    end = S - 1 if S < 2 or score[-1, S - 1] >= score[-1, S - 2] else S - 2
    best = score[-1, end]
    if not np.isfinite(best):
        return -np.inf, None
    spans: list[list[int]] = [[-1, -1] for _ in seq]
    s = end
    for t in range(T - 1, -1, -1):
        if lab[s] != 0:
            k = (s - 1) // 2
            spans[k][0] = t
            if spans[k][1] < 0:
                spans[k][1] = t + 1
        s -= int(back[t, s]) if t > 0 else 0
    return float(best), [tuple(x) for x in spans]


def classes() -> list[dict]:
    return [{"name": "short vowel", "key": "vowel"}] + [{"name": f"{a}/{b}", "key": f"{a}{b}", "pair": (a, b)} for a, b in PAIRS] + [{"name": "missing shadda", "key": "shadda"}]


def class_of(ch: str) -> str | None:
    if ch in VOWELS:
        return "vowel"
    for a, b in PAIRS:
        if ch in (a, b):
            return f"{a}{b}"
    return None


def confusables(ch: str) -> list[str]:
    if ch in VOWELS:
        return [v for v in VOWELS if v != ch]
    return [b if ch == a else a for a, b in PAIRS if ch in (a, b)]


def empty() -> dict:
    return {"pos": 0, "greedy_hit": 0, "greedy_canon": 0, "greedy_other": 0, "top5_hit": 0, "llr_hit": 0,
            "neg": 0, "greedy_ok": 0, "greedy_fp": 0, "top5_fp": 0, "llr_fp": 0, "examples": []}


def score_clip(row: dict, canon: str, modal: str, stats: dict[str, dict], sites: set[int] | None = None) -> None:
    """Score every class site in the clip, or only the canonical indices in sites."""
    said, hyp = row["said"], row["greedy"]
    a_said, a_hyp, a_modal = align(canon, said), align(canon, hyp), align(canon, modal)
    lp = log_probs(row)
    base, spans = viterbi(lp, said)
    if spans is None:
        row["skipped"] = "too few frames"
        return

    def at(seq, a, i):
        return seq[a[i]] if a[i] is not None else None

    # Substitution classes: short vowels and letter pairs.
    for i, c in enumerate(canon):
        key = class_of(c)
        if key is None or at(modal, a_modal, i) != c or (sites is not None and i not in sites):
            continue
        s = at(said, a_said, i)
        st = stats[key]
        if s == c:
            j = a_said[i]
            g = at(hyp, a_hyp, i)
            st["neg"] += 1
            st["greedy_ok"] += g == c
            st["greedy_fp"] += g in confusables(c)
            lo, hi = spans[j]
            top = np.argsort(lp[lo:hi], axis=1)[:, -5:]
            alts = {TOKEN[x] for x in confusables(c)}
            st["top5_fp"] += bool(alts & set(top.ravel().tolist()))
            st["llr_fp"] += any(viterbi(lp, said[:j] + x + said[j + 1 :])[0] > base for x in confusables(c))
        elif s in confusables(c):
            j = a_said[i]
            g = at(hyp, a_hyp, i)
            st["pos"] += 1
            st["greedy_hit"] += g == s
            st["greedy_canon"] += g == c
            st["greedy_other"] += g not in (s, c)
            lo, hi = spans[j]
            top = np.argsort(lp[lo:hi], axis=1)[:, -5:]
            st["top5_hit"] += TOKEN[s] in set(top.ravel().tolist())
            st["llr_hit"] += viterbi(lp, said[:j] + c + said[j + 1 :])[0] < base
            st["examples"].append({"id": row["id"], "ayah": row["ayah"], "site": i, "canonical": c, "said": s, "greedy": g, "label": row["label"], "hyp": hyp})

    # Missing shadda: a doubled consonant said as one.
    st = stats["shadda"]
    modal_runs = {r[0] for r in runs(modal)}
    for lo, hi in runs(canon):
        c = canon[lo]
        if a_modal[lo] is None or a_modal[lo] not in modal_runs or (sites is not None and lo not in sites):
            continue
        said_idx = [a_said[k] for k in range(lo, hi) if a_said[k] is not None]
        kept = sum(said[k] == c for k in said_idx)
        g = sum(hyp[a_hyp[k]] == c for k in range(lo, hi) if a_hyp[k] is not None)
        if not said_idx:
            continue
        s0, s1 = min(said_idx), max(said_idx) + 1
        if kept >= 2:
            st["neg"] += 1
            st["greedy_ok"] += g >= 2
            st["greedy_fp"] += g == 1
            st["llr_fp"] += viterbi(lp, said[:s0] + c + said[s1:])[0] > base
        elif kept <= 1:
            st["pos"] += 1
            st["greedy_hit"] += g <= 1 and any(a_hyp[k] is not None for k in range(lo, hi))
            st["greedy_canon"] += g >= 2
            st["greedy_other"] += all(a_hyp[k] is None for k in range(lo, hi))
            st["llr_hit"] += viterbi(lp, said[:s0] + canon[lo:hi] + said[s1:])[0] < base
            g_idx = [a_hyp[k] for k in range(lo, hi) if a_hyp[k] is not None]
            g_span = hyp[min(g_idx) : max(g_idx) + 1] if g_idx else None
            st["examples"].append({"id": row["id"], "ayah": row["ayah"], "site": lo, "canonical": canon[lo:hi], "said": said[s0:s1], "greedy": g_span, "label": row["label"], "hyp": hyp})


def verdict(st: dict) -> str:
    if st["pos"] < MIN_POSITIVES:
        return "unsupported"
    rate = lambda k, n: st[k] / st[n] if st[n] else 0.0  # noqa: E731
    if rate("greedy_hit", "pos") >= VIABLE["recall"] and rate("greedy_fp", "neg") <= VIABLE["fp"]:
        return "viable"
    if rate("llr_hit", "pos") >= POSTERIOR_ONLY["recall"] and rate("llr_fp", "neg") <= POSTERIOR_ONLY["fp"]:
        return "posterior-only"
    return "unsupported"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} {100 * n / d:.1f}%" if d else "—"


def main() -> None:
    canon = canonicals()
    clips = []
    for name in ("qdat_bench", "ikhlas_recitations"):
        for r in pq.read_table(PASS / f"{name}.parquet").to_pylist():
            r["said"] = r["label"].replace(" ", "") if name == "qdat_bench" else ikhlas_to_teacher(r["label"])
            clips.append(r)
    qdat_modal = Counter(r["said"] for r in clips if r["ayah"] == "5:109").most_common(1)[0][0]
    canon["5:109"] = qdat_modal
    modal = {a: Counter(r["said"] for r in clips if r["ayah"] == a).most_common(1)[0][0] for a in canon}

    unknown = {c for r in clips for c in r["said"] + r["greedy"]} - set(INVENTORY)
    assert not unknown, unknown

    per_ds = {}
    stats = {c["key"]: empty() for c in classes()}
    for r in clips:
        ds = r["dataset"]
        d = per_ds.setdefault(ds, Counter())
        d["clips"] += 1
        per = Levenshtein.distance(r["said"], r["greedy"]) / max(1, len(r["said"]))
        if per > MAX_CLIP_PER:
            d["skipped: teacher decode too far from label"] += 1
            continue
        score_clip(r, canon[r["ayah"]], modal[r["ayah"]], stats)
        d["skipped: too few frames" if r.get("skipped") else "scored"] += 1

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    result = {
        "version": REPORT_VERSION,
        "commit": commit,
        "model": "obadx/muaalem-model-v3_2",
        "thresholds": {"min_positives": MIN_POSITIVES, "viable": VIABLE, "posterior_only": POSTERIOR_ONLY, "max_clip_per": MAX_CLIP_PER},
        "datasets": {k: dict(v) for k, v in per_ds.items()},
        "canonicals": canon,
        "classes": [],
    }
    for c in classes():
        st = stats[c["key"]]
        result["classes"].append({"name": c["name"], "verdict": verdict(st), **st})
    OUT.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    OUT.with_suffix(".md").write_text(render(result))
    print(OUT.with_suffix(".md").read_text())


def render(r: dict) -> str:
    t = r["thresholds"]
    lines = [
        f"# E1 teacher-bias report {r['version']}",
        "",
        f"Report {r['version']}, teacher `{r['model']}`, commit {r['commit']}. Data: `obadx/qdat_bench` (5:109 fragment) and `azizbekphd/ikhlas_recitations` (112:1–4).",
        "",
        "| dataset | clips | scored | skipped |",
        "|---|---|---|---|",
    ]
    for ds, d in r["datasets"].items():
        skipped = ", ".join(f"{v} {k.removeprefix('skipped: ')}" for k, v in d.items() if k.startswith("skipped")) or "0"
        lines.append(f"| `{ds}` | {d['clips']} | {d.get('scored', 0)} | {skipped} |")
    lines += [
        "",
        "## Per class",
        "",
        "Positives are labelled errors at sites where the dataset's usual label matches the canonical. Negatives are correctly said tokens of the class.",
        "",
        "| class | positives | sources | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | negatives | greedy FP | top-5 FP | posterior FP | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in r["classes"]:
        shadda = c["name"] == "missing shadda"
        sources = len({(e["ayah"], e["site"]) for e in c["examples"]})
        lines.append(
            f"| {c['name']} | {c['pos']} | {sources} | {pct(c['greedy_hit'], c['pos'])} | {pct(c['greedy_canon'], c['pos'])} | {pct(c['greedy_other'], c['pos'])} | "
            f"{'—' if shadda else pct(c['top5_hit'], c['pos'])} | {pct(c['llr_hit'], c['pos'])} | {c['neg']} | {pct(c['greedy_fp'], c['neg'])} | "
            f"{'—' if shadda else pct(c['top5_fp'], c['neg'])} | {pct(c['llr_fp'], c['neg'])} | {c['verdict']} |"
        )
    lines += [
        "",
        "## Definitions",
        "",
        "- **in greedy**: the greedy decode at the aligned site is the phoneme actually said.",
        "- **canonicalized**: the error was said, but the greedy decode emitted the canonical phoneme.",
        "- **greedy other**: the site was deleted or decoded as some third phoneme.",
        "- **in top-5**: the said phoneme is in the top-5 of at least one frame of its CTC forced-aligned span.",
        "- **posterior prefers error**: the CTC Viterbi score of the labelled sequence beats the same sequence with the site corrected to canonical. For negatives (posterior FP), some confusable variant beats the labelled sequence.",
        "- **FP** columns are measured on negatives, against the class's confusables: the pair partner, or either other short vowel. For shadda, the FP is the doubled consonant decoded or scored as single.",
        f"- Clips whose greedy decode is more than {t['max_clip_per']:.0%} phoneme edits from the label are skipped, because many Ikhlas clips are truncated.",
        "",
        "## Verdict rule",
        "",
        f"- Fewer than {t['min_positives']} positives: `unsupported` until scripted-error recordings (ticket 05) supply positives.",
        f"- `viable`: greedy recall ≥ {t['viable']['recall']:.0%} and greedy FP ≤ {t['viable']['fp']:.0%}.",
        f"- `posterior-only`: posterior-preference recall ≥ {t['posterior_only']['recall']:.0%} and posterior FP ≤ {t['posterior_only']['fp']:.0%}. Top-5 presence is reported but not used: a confusable sits in the top-5 of most correct tokens, so it can't separate errors from correct speech.",
        "- Otherwise `unsupported`.",
        "",
        "## Caveats",
        "",
        "- Both datasets are a single passage each (one 5:109 fragment, al-Ikhlas), so **sources** (distinct canonical sites behind the positives) is the real sample size, not the positive count. A verdict built on a handful of sites is provisional until ticket 05's scripted recordings re-run it.",
        "- Neither dataset labels letter-pair errors. qdat_bench labels madd, ghunna and qalqalah; Ikhlas labels mostly qalqalah, madd and a few vowel and idgham errors. Ikhlas's one \"wrong makhraj\" symbol never occurs in its transcripts.",
        "- The only missing-shadda positives are an unassimilated noon in يَكُن لَّهُۥ (idgham dropped), not a dropped shadda inside a word.",
        "- Negatives still measure each pair's false-positive floor on correct speech, which the checker needs whatever the recall turns out to be.",
        "",
        "## Positives",
        "",
        "| class | clip | ayah | canonical | said | greedy |",
        "|---|---|---|---|---|---|",
    ]
    for c in r["classes"]:
        for e in c["examples"]:
            lines.append(f"| {c['name']} | {e['id']} | {e['ayah']} | {e['canonical']} | {e['said']} | {e['greedy'] or '(deleted)'} |")
    return "\n".join(lines) + "\n"


E1B_VERSION = "v1"
E1B_OUT = ROOT / "docs/benchmarks" / f"e1b-spliced-{E1B_VERSION}"
E1B_CLASSES = [c for c in classes() if c["key"] != "vowel"]
# Below this share of controls decoded as the canonical, the seam itself garbles the phoneme.
MIN_CONTROL_INTACT = 0.8
MIN_SITES = 30


def add(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] += v


def e1b_verdict(sp: dict, ctl: dict, sites: int) -> str:
    """Spliced positives judged against control negatives, so both sides carry the same seam."""
    if ctl["neg"] and ctl["greedy_ok"] / ctl["neg"] < MIN_CONTROL_INTACT:
        return "void"
    base = verdict({**sp, **{k: ctl[k] for k in ("neg", "greedy_fp", "top5_fp", "llr_fp")}})
    if base != "unsupported":
        return base
    if sp["pos"] < MIN_POSITIVES or sites < MIN_SITES:
        return "insufficient"
    heard = max(sp["greedy_hit"], sp["llr_hit"]) / sp["pos"] >= POSTERIOR_ONLY["recall"]
    return "inconclusive" if heard else "dead"


def e1b_main() -> None:
    clean = pq.read_table(PASS / "e1b-clean.parquet").to_pylist()
    rows = clean + pq.read_table(PASS / "e1b-spliced.parquet").to_pylist()
    # Keyed by class, and by "class:canonical letter" for one pair direction.
    stats = {kind: defaultdict(empty) for kind in ("spliced", "control", "clean")}
    sites: dict[str, set] = defaultdict(set)
    counts = Counter()
    for r in rows:
        counts[r["kind"]] += 1
        if Levenshtein.distance(r["label"], r["greedy"]) / max(1, len(r["label"])) > MAX_CLIP_PER:
            counts[f"{r['kind']} skipped"] += 1
            continue
        r["said"] = r["label"]
        tmp = {c["key"]: empty() for c in classes()}
        score_clip(r, r["canonical"], r["canonical"], tmp, set(r["sites"]))
        if r["kind"] == "clean":
            for k, v in tmp.items():
                add(stats["clean"][k], v)
            continue
        # س is in two pairs, so a clip's class is the one it was spliced for, not class_of's first match.
        shadda = r["cls"] == "shadda"
        total = empty()
        for k, v in tmp.items():
            if (k == "shadda") == shadda and k != "vowel":
                add(total, v)
        keys = [r["cls"]] if shadda else [r["cls"], f"{r['cls']}:{r['canonical'][r['sites'][0]]}"]
        for k in keys:
            add(stats[r["kind"]][k], total)
            if total["pos"]:
                sites[k].add((r["ayah"], r["ayah_site"]))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    result = {
        "version": E1B_VERSION,
        "commit": commit,
        "model": "obadx/muaalem-model-v3_2",
        "thresholds": {"min_positives": MIN_POSITIVES, "min_sites": MIN_SITES, "min_control_intact": MIN_CONTROL_INTACT,
                       "viable": VIABLE, "posterior_only": POSTERIOR_ONLY, "max_clip_per": MAX_CLIP_PER},
        "clips": dict(counts),
        "classes": [],
    }
    def entry(k: str, name: str, clean_key: str | None) -> dict:
        sp, ctl = stats["spliced"][k], stats["control"][k]
        out = {"name": name, "key": k, "verdict": e1b_verdict(sp, ctl, len(sites[k])), "sites": len(sites[k]),
               "spliced": {x: v for x, v in sp.items() if x != "examples"}, "control": {x: v for x, v in ctl.items() if x != "examples"}}
        if clean_key:
            out["clean"] = {x: v for x, v in stats["clean"][clean_key].items() if x != "examples"}
        return out

    for c in E1B_CLASSES:
        k = c["key"]
        e = entry(k, c["name"], k)
        if "pair" in c:
            e["directions"] = [entry(f"{k}:{a}", f"{a}→{b}", None) for a, b in (c["pair"], c["pair"][::-1])]
            # Pooling can hide a void or inconclusive direction behind a dead one.
            if e["verdict"] == "dead" and any(d["verdict"] != "dead" for d in e["directions"]):
                e["verdict"] = "mixed"
        result["classes"].append(e)
    E1B_OUT.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    E1B_OUT.with_suffix(".md").write_text(render_e1b(result))
    print(E1B_OUT.with_suffix(".md").read_text())


def render_e1b(r: dict) -> str:
    t = r["thresholds"]
    n = r["clips"]
    lines = [
        f"# E1b spliced-error probe {r['version']}",
        "",
        f"Report {r['version']}, teacher `{r['model']}`, commit {r['commit']}. Built by `tools/e1b_build.py`, scored by `tools/e1_report.py --e1b`.",
        "",
        "**Every number here is an upper bound.** A splice is a clean, isolated phoneme from the same reciter, so it is easier to hear than a learner's error. "
        "A class that fails here is `dead`; a class that passes still needs ticket 05's real recordings.",
        "",
        f"Source: everyayah Husary and Minshawy murattal, juz 29–30 (surahs 67–114). {n.get('spliced', 0)} spliced clips, {n.get('control', 0)} controls, "
        f"{n.get('clean', 0)} clean ayahs; skipped as too far from the label: {n.get('spliced skipped', 0)} spliced, {n.get('control skipped', 0)} control, {n.get('clean skipped', 0)} clean.",
        "",
        "## Per class",
        "",
        "| class | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(c: dict) -> str:
        sp, ctl, cl = c["spliced"], c["control"], c.get("clean")
        shadda = c["key"] == "shadda"
        clean = f"{pct(cl['greedy_fp'], cl['neg'])} | {pct(cl['llr_fp'], cl['neg'])}" if cl else "— | —"
        return (
            f"| {c['name']} | {sp['pos']} | {c['sites']} | {pct(sp['greedy_hit'], sp['pos'])} | {pct(sp['greedy_canon'], sp['pos'])} | {pct(sp['greedy_other'], sp['pos'])} | "
            f"{'—' if shadda else pct(sp['top5_hit'], sp['pos'])} | {pct(sp['llr_hit'], sp['pos'])} | {ctl['neg']} | {pct(ctl['greedy_ok'], ctl['neg'])} | "
            f"{pct(ctl['greedy_fp'], ctl['neg'])} | {pct(ctl['llr_fp'], ctl['neg'])} | {clean} | {c['verdict']} |"
        )

    lines += [row(c) for c in r["classes"]]
    lines += [
        "",
        "## Per direction",
        "",
        "A direction is canonical→said: ت→ط is a ت in the text recited as ط. Controls are the same canonical letter spliced from another word.",
        "",
        "| direction | positives | sites | in greedy | canonicalized | greedy other | in top-5 | posterior prefers error | controls | control intact | control greedy FP | control posterior FP | clean greedy FP | clean posterior FP | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [row(d) for c in r["classes"] for d in c.get("directions", [])]
    lines += [
        "",
        "## How the clips are made",
        "",
        "- Each clean ayah's canonical phonemes are CTC force-aligned on the teacher's own posteriors (`viterbi` in `tools/e1_report.py`). Each blank gap between two tokens is split at its midpoint, giving contiguous per-phoneme segments.",
        "- **Letter pairs**: a pair consonant plus its short vowel is replaced by the partner consonant plus the same vowel, cut from another ayah by the same reciter (segment duration within 0.6–1.6×). Cuts snap to zero crossings and join with an 8 ms raised-cosine crossfade, loudness-matched.",
        "- **Missing shadda**: the middle of a doubled consonant is cut out until it lasts as long as that reciter's median single consonant of the same letter. Only geminates at least 1/0.7 that long are used.",
        "- **Controls** carry the same seam with no error. Letters: the same consonant plus vowel from another ayah. Shadda: the same kind of internal cut at the hold's midpoint, removing only 15%, so the consonant stays doubled.",
        "- **Clean** is the unspliced ayah scored at the same kinds of sites: the teacher's floor without any seam.",
        "- Clips are trimmed at pauses at least 1.5 s from the splice and scored only at the spliced site.",
        "",
        "## Verdict rule",
        "",
        f"- Positives come from splices; negatives come from **controls**, so the FP side carries the same seam as the positives.",
        f"- `void`: fewer than {t['min_control_intact']:.0%} of controls decode as the canonical phoneme. The seam garbles the sound, so the class's numbers mean nothing.",
        f"- `viable`: greedy recall ≥ {t['viable']['recall']:.0%} and control greedy FP ≤ {t['viable']['fp']:.0%}.",
        f"- `posterior-only`: posterior-preference recall ≥ {t['posterior_only']['recall']:.0%} and control posterior FP ≤ {t['posterior_only']['fp']:.0%}.",
        f"- `dead`: neither, with ≥ {t['min_positives']} positives from ≥ {t['min_sites']} sites, and both greedy and posterior recall under {t['posterior_only']['recall']:.0%}. The teacher can't hear the class even on clean splices.",
        "- `mixed`: a pair that pools to `dead` but has a direction that isn't. Read its directions instead.",
        "- `inconclusive`: the recall is there, but control FP is over the bound. The teacher reacts to the seam, so the splice can't separate the error from the artifact either way.",
        "- Clean FP for ث/س counts only ث sites: a clean س is filed under ص/س. Its FP still counts both of س's confusables (ص and ث), as in E1.",
        "- Definitions of the other columns match `e1-teacher-bias-v1.md`. **control intact** is the share of controls whose greedy decode keeps the canonical phoneme (for shadda, both copies).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    e1b_main() if "--e1b" in sys.argv else main()
