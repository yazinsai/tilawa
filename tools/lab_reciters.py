# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1"]
# ///
"""Held-out mp3quran Hafs reciters for the lab test set and their practice/exam split (lab ticket 01).

    uv run --python cpython-3.12-macos-aarch64-none tools/lab_reciters.py [--refresh]

Fetches JSON metadata only (mp3quran API, HF hub/datasets-server, everyayah.com index, one HEAD per held-out reciter),
cached under `.scratch/lab/cache/`. Writes `docs/benchmarks/lab-reciters-v1.json`; method in `lab-reciters-v1.md`.
"""

import argparse
import json
import random
import re
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".scratch/lab/cache"
OUT = ROOT / "docs/benchmarks/lab-reciters-v1.json"
MP3Q = "https://www.mp3quran.net/api/v3"
ROWS = "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length=100"
SEED = 20261028
SPEED_SURAH = 67
# Shorter than this share of the median Al-Mulk duration is hadr-like; fast recitation is unsupported.
FAST = 0.75
SIM = 0.75
LOW_KBPS = 110

# Sources a held-out reciter must not appear in. `student` is what modal/student_train.py actually trained on.
SOURCES = {
    "muaalem-annotated-v3": "teacher",
    "tlog": "student",
    "QUD": "planned",
    "zaibihassan": "planned",
    "everyayah": "planned",
}
# Manual decisions on automatic matches, keyed by (mp3quran read id, source, source name): "reject" or "uncertain".
OVERRIDES: dict[tuple[int, str, str], tuple[str, str]] = {}
# Matches automation misses (spelling too far apart), with the evidence for the call.
MANUAL: list[tuple[int, str, str, str]] = [
    (123, "muaalem-annotated-v3", "Mishary Rashid Al-Fasi",
     "teacher reciter_id 19 'مشاري راشد الفاسي' is a misspelling of العفاسي: moshaf 19.0 is islamway collection 213, "
     "Alafasy's Cairo murattal (المصحف المرتل تسجيل القاهرة استماع المعصراوي), country KW"),
]

AR_DIAC = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي", "ک": "ك", "ی": "ي"})
AR_DROP = {"بن", "ابن", "بنت", "الشيخ", "شيخ", "القارئ", "الدكتور"}
EN_DROP = {"al", "el", "ar", "as", "ash", "ad", "az", "at", "ath", "an", "bin", "ibn", "ben", "sheikh", "shaik", "shaikh",
           "shaykh", "qari", "the", "murattal", "mujawwad", "muallim", "mualim", "kids", "children", "warsh"}


def ar_tokens(name: str) -> list[str]:
    s = AR_DIAC.sub("", name).translate(AR_MAP)
    s = re.sub(r"\b(عبد|ابو)\s+", r"\1", s)
    out = []
    for t in s.split():
        if t in AR_DROP:
            continue
        if t.startswith("ال") and len(t) > 3:
            t = t[2:]
        out.append(t)
    return out


def skeleton(t: str) -> str:
    t = re.sub(r"^a([lrsdtzn])\1", r"\1", t)
    if t.startswith("al") and len(t) >= 6:
        t = t[2:]
    t = re.sub(r"^abd(ul|el|al|ur|ar|us|as|u|e|i)?", "abd", t)
    for a, b in (("dh", "t"), ("th", "t"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("ph", "f"), ("q", "k"), ("c", "k"), ("d", "t")):
        t = t.replace(a, b)
    head, rest = t[:1], t[1:]
    if head in "aeiou":
        head = "a"
    rest = re.sub(r"[aeiouyw]", "", rest)
    return re.sub(r"(.)\1+", r"\1", head + rest)


def en_tokens(name: str) -> list[str]:
    s = re.sub(r"\([^)]*\)", " ", name.lower())
    s = re.sub(r"[’'`‘]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t not in EN_DROP]
    merged: list[str] = []
    for t in toks:
        if merged and re.fullmatch(r"abd(ul|el|al|ur|ar|us|as|u|e|i)?", merged[-1]):
            merged[-1] += t
        else:
            merged.append(t)
    return [skeleton(t) for t in merged]


def tok_eq(a: str, b: str) -> float:
    # Short tokens differ by one letter between distinct names (محمد/محمود), so they must match exactly.
    if min(len(a), len(b)) <= 4:
        return 1.0 if a == b else 0.0
    return Levenshtein.normalized_similarity(a, b)


def same_person(a: list[str], b: list[str]) -> tuple[str, float] | None:
    """How two tokenised names match: first+last, or surname (a one-token name equals the other's surname)."""
    if not a or not b:
        return None
    if len(a) == 1 or len(b) == 1:
        one, other = (a, b) if len(a) == 1 else (b, a)
        s = tok_eq(one[0], other[-1])
        return ("surname", s) if s >= SIM and len(one[0]) >= 3 else None
    first, last = tok_eq(a[0], b[0]), tok_eq(a[-1], b[-1])
    if first >= SIM and last >= SIM:
        return "first+last", min(first, last)
    return None


def nearest(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return statistics.mean(max(tok_eq(t, u) for u in b) for t in a)


def fetch(url: str, name: str, refresh: bool) -> str:
    path = CACHE / name
    if path.exists() and not refresh:
        return path.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": "tilawa-lab/1"})
    body = urllib.request.urlopen(req, timeout=60).read().decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return body


def fetch_json(url: str, name: str, refresh: bool):
    return json.loads(fetch(url, name, refresh))


def hf_rows(dataset: str, config: str, split: str, refresh: bool) -> list[dict]:
    rows, off = [], 0
    while True:
        tag = f"{dataset.replace('/', '__')}-{config}-{off}.json"
        d = fetch_json(ROWS.format(dataset, config, split, off), tag, refresh)
        rows += [r["row"] for r in d["rows"]]
        off += 100
        if off >= d["num_rows_total"]:
            return rows


def head_size(url: str, name: str, refresh: bool) -> int | None:
    path = CACHE / name
    if path.exists() and not refresh:
        return int(path.read_text()) or None
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "tilawa-lab/1"})
    try:
        n = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length") or 0)
    except Exception:
        n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(n))
    return n or None


def source_names(refresh: bool) -> dict[str, list[dict]]:
    """Reciter names per training source: [{name, ar?, en?}]."""
    teacher = [{"name": r["english_name"], "ar": r["arabic_name"], "en": r["english_name"], "detail": f"reciter_id {r['id']}, moshafs {','.join(sorted(r['moshaf_set_ids']))}"}
               for r in hf_rows("obadx/muaalem-annotated-v3", "reciters_metadata", "train", refresh)]
    qud = {}
    for r in hf_rows("QUD-Technologies/quranic-universal-ayahs", "mushafs", "all", refresh):
        q = qud.setdefault(r["reciter_id"], {"name": r["name_en"], "ar": r["name_ar"], "en": r["name_en"], "slugs": []})
        q["slugs"].append(f"{r['slug']} ({r['riwayah']}, {r['style']})")
    for q in qud.values():
        q["detail"] = "; ".join(q.pop("slugs"))
    tree = fetch_json("https://huggingface.co/api/datasets/zaibihassan/Quranic-Recitation-Data/tree/main", "zaibihassan-tree.json", refresh)
    zaib = [{"name": x["path"], "en": x["path"], "detail": "top-level folder"} for x in tree if x["type"] == "directory"]
    ea = fetch_json("https://everyayah.com/data/recitations.js", "everyayah-recitations.json", refresh)
    seen = {}
    for k, v in ea.items():
        if k == "ayahCount":
            continue
        en = re.sub(r"(QuranExplorer\.Com|KetabAllah\.Net|_?\d+Kbps|\(Iran\))", "", v["name"].split("/")[-1]).replace("_", " ")
        seen.setdefault(v["name"], {"name": v["name"], "en": en, "detail": f"subfolders {v['subfolder']}"})
    return {"muaalem-annotated-v3": teacher, "tlog": [], "QUD": list(qud.values()), "zaibihassan": zaib, "everyayah": list(seen.values())}


def mp3quran_reads(refresh: bool) -> list[dict]:
    reads = fetch_json(f"{MP3Q}/ayat_timing/reads", "mp3quran-reads.json", refresh)
    by = {}
    for lang in ("eng", "ar"):
        for rc in fetch_json(f"{MP3Q}/reciters?language={lang}", f"mp3quran-reciters-{lang}.json", refresh)["reciters"]:
            for m in rc["moshaf"]:
                key = m["server"].replace("http://", "https://").rstrip("/")
                by.setdefault(key, {})[lang] = (rc, m)
    out = []
    for r in reads:
        m = by[r["folder_url"].replace("http://", "https://").rstrip("/")]
        (en, men), (_, mar) = m["eng"], m["ar"]
        out.append({"readId": r["id"], "reciterId": en["id"], "nameAr": r["name"], "nameEn": en["name"].strip(),
                    "moshaf": mar["name"], "moshafEn": men["name"], "moshafType": men["moshaf_type"],
                    "timedSurahs": r["soar_count"], "server": r["folder_url"]})
    return out


def leakage(read: dict, sources: dict[str, list[dict]]) -> list[dict]:
    hits = []
    ra, re_ = ar_tokens(read["nameAr"]), en_tokens(read["nameEn"])
    for src, names in sources.items():
        for s in names:
            how = []
            if s.get("ar") and (m := same_person(ra, ar_tokens(s["ar"]))):
                how.append(f"ar {m[0]} {m[1]:.2f}: {read['nameAr']} ~ {s['ar']}")
            if s.get("en") and (m := same_person(re_, en_tokens(s["en"]))):
                how.append(f"en {m[0]} {m[1]:.2f}: {read['nameEn']} ~ {s['en']}")
            if not how:
                continue
            decision, note = OVERRIDES.get((read["readId"], src, s["name"]), ("match", ""))
            if decision == "reject":
                continue
            surname_only = all(" surname " in h for h in how)
            hits.append({"source": src, "role": SOURCES[src], "name": s["name"], "detail": s["detail"],
                         "decision": "uncertain" if decision == "uncertain" or surname_only else "match",
                         "evidence": how + ([note] if note else [])})
    for rid, src, name, note in MANUAL:
        if rid == read["readId"]:
            hits.append({"source": src, "role": SOURCES[src], "name": name, "detail": "manual", "decision": "match", "evidence": [note]})
    return hits


def closest(read: dict, sources: dict[str, list[dict]]) -> dict:
    best = (0.0, "", "")
    ra, re_ = ar_tokens(read["nameAr"]), en_tokens(read["nameEn"])
    for src, names in sources.items():
        for s in names:
            sc = max(nearest(ra, ar_tokens(s["ar"])) if s.get("ar") else 0.0, nearest(re_, en_tokens(s["en"])) if s.get("en") else 0.0)
            if sc > best[0]:
                best = (sc, src, s["name"])
    return {"score": round(best[0], 2), "source": best[1], "name": best[2]}


def speed(read: dict, refresh: bool) -> float | None:
    t = fetch_json(f"{MP3Q}/ayat_timing?surah={SPEED_SURAH}&read={read['readId']}", f"timing-{read['readId']}-{SPEED_SURAH}.json", refresh)
    ayat = [a for a in t if a["ayah"] >= 1]
    if len(ayat) != 30 or any(b["start_time"] < a["start_time"] for a, b in zip(ayat, ayat[1:])):
        return None
    return (ayat[-1]["end_time"] - ayat[0]["start_time"]) / 1000


def split(held: list[dict], seed: int) -> None:
    """Matched pairs: within each bitrate band sort by speed, pair neighbours, coin-flip each pair. Odd ones out from
    the bands are paired the same way; a last odd one goes to practice."""
    rng = random.Random(seed)
    by_speed = lambda rs: sorted(rs, key=lambda r: (r["speedRatio"], r["readId"]))
    low = [r for r in held if (r["kbps"] or 0) < LOW_KBPS]
    bands = [by_speed(low), by_speed([r for r in held if r not in low])]
    pairs, rest = [], []
    for band in bands:
        pairs += [band[i:i + 2] for i in range(0, len(band) - 1, 2)]
        rest += band[len(band) // 2 * 2:]
    rest = by_speed(rest)
    pairs += [rest[i:i + 2] for i in range(0, len(rest) - 1, 2)]
    if len(rest) % 2:
        rest[-1]["split"] = "practice"
    for pair in pairs:
        rng.shuffle(pair)
        pair[0]["split"], pair[1]["split"] = "practice", "exam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    sources = source_names(args.refresh)
    reads = mp3quran_reads(args.refresh)
    hafs = [r for r in reads if "حفص" in r["moshaf"]]
    durs = {r["readId"]: speed(r, args.refresh) for r in hafs if r["moshafType"] == 11}
    median = statistics.median(d for d in durs.values() if d)
    held, excluded = [], []
    for r in hafs:
        hits = leakage(r, sources)
        reasons = []
        for role in ("teacher", "student", "planned"):
            if any(h["role"] == role and h["decision"] == "match" for h in hits):
                reasons.append(f"excluded_{role}")
        if not reasons and any(h["decision"] == "uncertain" for h in hits):
            reasons.append("uncertain")
        if r["moshafType"] != 11:
            reasons.append("style_mujawwad" if "مجود" in r["moshaf"] else "style_other")
        d = durs.get(r["readId"])
        ratio = round(d / median, 3) if d else None
        if r["moshafType"] == 11 and ratio is None:
            reasons.append("timing_unusable")
        elif ratio is not None and ratio < FAST:
            reasons.append("fast")
        base = {"readId": r["readId"], "nameAr": r["nameAr"], "nameEn": r["nameEn"], "moshaf": r["moshaf"]}
        if reasons:
            excluded.append({**base, "reason": reasons[0], "reasons": reasons, "matchedSource": sorted({h["source"] for h in hits}),
                             "evidence": [f"{h['source']} [{h['decision']}] {h['name']} ({h['detail']}): " + " | ".join(h["evidence"]) for h in hits]
                             + ([f"Al-Mulk {d:.0f}s = {ratio} x median {median:.0f}s"] if "fast" in reasons else [])
                             + ([f"moshaf_type {r['moshafType']}: {r['moshafEn']}"] if r["moshafType"] != 11 else [])})
            continue
        url = r["server"].rstrip("/") + f"/{SPEED_SURAH:03d}.mp3"
        size = head_size(url, f"head-{r['readId']}-{SPEED_SURAH}.txt", args.refresh)
        held.append({**base, "timedSurahs": r["timedSurahs"], "speedRatio": ratio, "almulkSeconds": round(d, 1),
                     "kbps": round(size * 8 / d / 1000) if size else None, "nearestSourceName": closest(r, sources), "server": r["server"]})
    split(held, SEED)
    held.sort(key=lambda h: (h["split"], h["readId"]))
    count = lambda key: sum(key in e["reasons"] for e in excluded)
    balance = {}
    for sp in ("practice", "exam"):
        g = [h for h in held if h["split"] == sp]
        balance[sp] = {"n": len(g), "meanSpeedRatio": round(statistics.mean(h["speedRatio"] for h in g), 3),
                       "meanKbps": round(statistics.mean(h["kbps"] for h in g if h["kbps"])),
                       f"under{LOW_KBPS}Kbps": sum((h["kbps"] or 0) < LOW_KBPS for h in g), "timedSurahs": sum(h["timedSurahs"] for h in g)}
    out = {
        "version": 1,
        "source": {"api": f"{MP3Q}/ayat_timing/reads", "fetchedAt": datetime.fromtimestamp((CACHE / "mp3quran-reads.json").stat().st_mtime, timezone.utc).isoformat(timespec="seconds")},
        "criteria": [
            "mp3quran read with ayah timings whose moshaf is Hafs (moshaf name contains حفص)",
            "moshaf_type 11 (Hafs murattal); mujawwad (12) and featured recitation (14) excluded",
            f"surah {SPEED_SURAH} timings complete and monotonic; recited span at least {FAST} x the median across Hafs murattal reads (else hadr-like, unsupported)",
            "reciter not in teacher training data (obadx/muaalem-annotated-v3 reciters_metadata), Arabic or English name",
            "reciter not in the student's actual training data (tlog-v2 only; no names, see risks)",
            "reciter not in planned training sources named by the ticket: QUD (all riwayat), zaibihassan, everyayah.com index (superset of tarteel-ai/everyayah and its mirrors)",
            "uncertain name matches excluded",
            f"practice/exam: matched pairs within bitrate bands (<{LOW_KBPS} kbps on the surah {SPEED_SURAH} mp3, rest) by speed, seeded coin flip per pair",
        ],
        "counts": {"timedReads": len(reads), "timedHafs": len(hafs), "excluded_teacher": count("excluded_teacher"),
                   "excluded_student": count("excluded_student"), "excluded_planned": count("excluded_planned"),
                   "uncertain": count("uncertain"), "style": count("style_mujawwad") + count("style_other"), "fast": count("fast"),
                   "heldOut": len(held), "practice": sum(h["split"] == "practice" for h in held), "exam": sum(h["split"] == "exam" for h in held),
                   "medianAlMulkSeconds": round(median, 1)},
        "balance": balance,
        "heldOut": held,
        "excluded": excluded,
        "risks": [
            "tlog (the student's only training audio) is crowd-sourced with no reciter names. A held-out professional reciter could appear in it (users playing recordings); unverifiable.",
            "Teacher base model facebook/w2v-bert-2.0 was pretrained self-supervised on ~4.5M h of unlabelled multilingual audio; Quran recordings of any reciter may be in it. Unverifiable; no labels involved.",
            "obadx/qdat (planned) is anonymous learner audio (speaker ids only), not professional reciters; not checkable, low risk.",
            "Matching is by name only. Two reciters with the same name, or one reciter under very different spellings, can be mis-sorted; see nearestSourceName for each held-out reciter.",
            "Speed and recording quality come from one surah's timings and the mp3 bitrate, not from listening.",
        ],
        "seed": SEED,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out["counts"], ensure_ascii=False))
    for h in held:
        print(h["split"], h["readId"], h["nameAr"], h["nameEn"], h["speedRatio"], h["kbps"], h["nearestSourceName"])


if __name__ == "__main__":
    main()
