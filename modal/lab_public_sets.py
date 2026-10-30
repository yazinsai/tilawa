"""Public error sets onto the tilawa-lab volume (.scratch/lab, ticket 14). Nothing is scored.

    modal run modal/lab_public_sets.py [--only ikhlas,retasy,quranmb]

Writes /data/public/<dataset>/ on the tilawa-lab volume: orig/ (source audio), wav16k/ (16 kHz mono s16 WAV),
manifest.json (provenance, counts, issues, items). Item paths are relative to the dataset directory.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict

import modal

SR = 16_000
VOL = "/lab"
PUBLIC = f"{VOL}/data/public"
IKHLAS_URL = "https://data.mendeley.com/public-files/datasets/sxtmmr6mvk/files/ec1b3a42-700d-4424-b6cd-6844828f4637/file_downloaded"
IKHLAS_API = "https://data.mendeley.com/public-api/datasets/sxtmmr6mvk"
IKHLAS_SHA = "de8042ac8cc137e83fdcdbed375a24bd51f873c94f7f2cca1e39c60b188c0068"
IKHLAS_MIRROR = "azizbekphd/ikhlas_recitations"
RETASY = "RetaSy/quranic_audio_dataset"
QURANMB = "IqraEval/QuranMB.v2"
QURANMB_GT = "IqraEval/IqraEval_Test_GT"

# 112:1-4, Uthmani (Tanzil).
IKHLAS = ["قُلْ هُوَ ٱللَّهُ أَحَدٌ", "ٱللَّهُ ٱلصَّمَدُ", "لَمْ يَلِدْ وَلَمْ يُولَدْ", "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ"]
MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
RETASY_LABELS = {
    "correct": "correct",
    "in_correct": "incorrect_pronunciation",
    "in_complete": "incomplete_verse",
    "not_match_aya": "wrong_verse",
    "multiple_aya": "multiple_verses",
    "not_related_quran": "unrelated_or_empty",
}


def norm(w: str) -> str:
    w = MARKS.sub("", unicodedata.normalize("NFC", str(w)))
    return w.translate(str.maketrans("ٱأإآى", "ااااي")).strip()


def empty(v) -> bool:
    return v is None or v == 0 or (isinstance(v, str) and not v.strip()) or v != v


def split_list(v) -> list[str]:
    return [] if empty(v) else [p.strip() for p in re.split("[,،]", str(v)) if p.strip()]


def error_class(t: str) -> str:
    c = [n for k, n in (("حركات", "harakat"), ("تجويد", "tajweed")) if k in t]
    return "+".join(c) or f"other:{t}"


def ikhlas_file(name: str) -> tuple[int, int, bool] | None:
    m = re.fullmatch(r"ID(\d+)V(\d)([TF])\.wav", str(name).strip())
    return (int(m[1]), int(m[2]), m[3] == "T") if m else None


def parse_ikhlas_row(r: dict) -> dict:
    """One Dataset.xlsx row (header-keyed, keys stripped) to a normalised item with per-row issues."""
    file = str(r.get("Verse location") or "").strip()
    flag, verse = r.get("True or False"), r.get("Verse number")
    words = split_list(r.get("Error location"))
    types = split_list(r.get("Error type"))
    it = {
        "id": f"ikhlas-{int(r['ID']):04d}",
        "sheet_id": r["ID"],
        "file": file,
        "ref": f"112:{verse}",
        "label": {1: "error", 0: "correct"}.get(flag),
        "error_classes": [error_class(t) for t in types],
        "error_words": words,
        "explanations": split_list(r.get("Error explanation")),
        "error_code": None if empty(r.get("Error number")) else r.get("Error number"),
        "speaker": None,
        "issues": [],
    }
    iss = it["issues"]
    for k in ("ID", "Surah name", "Verse number", "Verse location", "True or False"):
        if r.get(k) is None or r.get(k) == "":
            iss.append(f"missing:{k}")
    if it["label"] is None:
        iss.append(f"bad_flag:{flag}")
    f = ikhlas_file(file)
    if not f:
        iss.append("bad_filename")
    else:
        it["speaker"] = f[0]
        if f[1] != verse:
            iss.append(f"verse_vs_filename:{verse}/{f[1]}")
        if it["label"] and f[2] != (it["label"] == "correct"):
            iss.append(f"flag_vs_filename:{it['label']}/{'T' if f[2] else 'F'}")
    has = [bool(types), bool(words), bool(it["explanations"])]
    if it["label"] == "error" and not all(has):
        iss.append("error_missing:" + ",".join(n for n, h in zip(("type", "location", "explanation"), has) if not h))
    if it["label"] == "correct" and any(has):
        iss.append("correct_with_error_fields")
    if types and words and len(types) != len(words):
        iss.append(f"types_vs_locations:{len(types)}/{len(words)}")
    vw = norm(IKHLAS[verse - 1]).split() if isinstance(verse, int) and 1 <= verse <= 4 else []
    allw = {w for v in IKHLAS for w in norm(v).split()}
    for w in words:
        n = norm(w)
        if n not in vw:
            iss.append(f"location_other_verse:{w}" if n in allw else f"location_not_in_surah:{w}")
    return it


def repair_ikhlas_files(items: list[dict], files: list[str]) -> list[dict]:
    """Rows that share a filename: if the row's place in the sheet (neighbour speaker +/- 1, same verse and T/F) names
    a zip file no row references, point the row at it. Mutates items; returns the repairs."""
    used = Counter(it["file"] for it in items)
    free = set(files) - set(used)
    out = []
    for k, it in enumerate(items):
        f = ikhlas_file(it["file"])
        if used[it["file"]] < 2 or not f:
            continue
        for j, step in ((k - 1, 1), (k + 1, -1)):
            p = ikhlas_file(items[j]["file"]) if 0 <= j < len(items) else None
            if p and p[1] == f[1]:
                cand = f"ID{p[0] + step}V{f[1]}{'T' if f[2] else 'F'}.wav"
                if cand in free:
                    out.append({"id": it["id"], "sheet_file": it["file"], "file": cand})
                    used[it["file"]] -= 1
                    free.discard(cand)
                    it.update(sheet_file=it["file"], file=cand, speaker=p[0] + step)
                    it["issues"].append(f"file_repaired:{it['sheet_file']}->{cand}")
                break
    return out


def label_key(it: dict) -> tuple:
    return (it["label"], tuple(it["error_classes"]), tuple(norm(w) for w in it["error_words"]), tuple(it["explanations"]))


def dedup(hashes: dict[str, str]) -> dict[str, str]:
    """id -> canonical id (first id in sorted order with the same hash); only duplicates are returned."""
    groups = defaultdict(list)
    for i in sorted(hashes):
        groups[hashes[i]].append(i)
    return {i: g[0] for g in groups.values() for i in g[1:]}


def ikhlas_cross_issues(items: list[dict], file_hash: dict[str, str]) -> list[dict]:
    """Rows sharing a filename or identical audio. `file_hash` is file -> audio hash for files in the zip."""
    out = []
    by_file = defaultdict(list)
    for it in items:
        by_file[it["file"]].append(it)
    for f, its in sorted(by_file.items()):
        if len(its) > 1:
            out.append({"kind": "duplicate_filename_rows", "file": f, "ids": [i["id"] for i in its], "conflicting": len({label_key(i) for i in its}) > 1})
        if f not in file_hash:
            out.append({"kind": "row_without_audio", "file": f, "ids": [i["id"] for i in its]})
    for f in sorted(set(file_hash) - set(by_file)):
        out.append({"kind": "audio_without_row", "file": f})
    by_hash = defaultdict(set)
    for f, h in file_hash.items():
        by_hash[h].add(f)
    for h, fs in by_hash.items():
        if len(fs) > 1:
            its = [i for f in sorted(fs) for i in by_file.get(f, [])]
            out.append({"kind": "duplicate_audio", "files": sorted(fs), "ids": [i["id"] for i in its], "conflicting": len({label_key(i) for i in its}) > 1})
    return out


NOT_QURAN = {"Adhan", "Iqamah Prayer", "At-Tahiyyat", "Subhanaka", "Salawat", "Adhkar after prayer", "Dua Qunoot", "Dua for Protection"}


def retasy_quranic(surah, aya) -> bool:
    """False for prayer texts and the isti'adha; 'Dua from the Quran' and 'Ayat Ramadan' are Quranic verses."""
    return surah not in NOT_QURAN and not norm(aya or "").startswith("اعوذ")


def retasy_label(v) -> str:
    return "unlabeled" if empty(v) else RETASY_LABELS.get(str(v).strip(), f"other:{v}")


def retasy_annotators(meta) -> list[dict]:
    try:
        d = json.loads(meta) if meta else {}
    except (TypeError, json.JSONDecodeError):
        return []
    n = sorted({int(m[1]) for k in d if (m := re.fullmatch(r"label_(\d+)", k))})
    return [{"label": retasy_label(d.get(f"label_{i}")), "annotator": d.get(f"annotator{i}_id"), "mcc": d.get(f"annotator{i}_MCC")} for i in n]


def phone_edits(ref: list[str], hyp: list[str]) -> dict:
    """Levenshtein alignment counts of hyp (as said) against ref (canonical)."""
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, s, dl, ins = d[i - 1][j - 1]
            sub = (c, s, dl, ins) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, dl, ins)
            c, s, dl, ins = d[i - 1][j]
            de = (c + 1, s, dl + 1, ins)
            c, s, dl, ins = d[i][j - 1]
            inn = (c + 1, s, dl, ins + 1)
            d[i][j] = min(sub, de, inn)
    _, s, dl, ins = d[n][m]
    return {"sub": s, "del": dl, "ins": ins}


# ---- Modal ----

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("huggingface_hub", "pyarrow", "openpyxl", "requests", "numpy")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache/hf"})
)
app = modal.App("tilawa-lab-public-sets", image=image)
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
fn = dict(cpu=8, memory=16384, volumes={VOL: vol, "/cache": cache_vol}, secrets=[hf_secret], timeout=2 * 60 * 60)


def convert(src: str, dst: str) -> dict:
    """Source audio file to 16 kHz mono s16 WAV; returns the PCM hash, duration and peak."""
    import os
    import subprocess
    import wave

    import numpy as np

    p = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"], capture_output=True)
    if p.returncode:
        return {"error": p.stderr.decode()[-300:]}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with wave.open(dst, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(p.stdout)
    x = np.frombuffer(p.stdout, dtype=np.int16)
    return {"pcm_sha256": hashlib.sha256(p.stdout).hexdigest(), "duration_s": round(len(x) / SR, 3), "peak": int(np.abs(x).max()) if len(x) else 0}


def convert_all(jobs: list[tuple[str, str, str]]) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as ex:
        return dict(zip([j[0] for j in jobs], ex.map(lambda j: convert(j[1], j[2]), jobs)))


def audio_issues(a: dict) -> list[str]:
    if "error" in a:
        return ["decode_failed"]
    return ["silent"] * (a["peak"] < 100) + ["too_short"] * (a["duration_s"] < 0.5) + ["very_long"] * (a["duration_s"] > 60)


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_manifest(d: str, m: dict) -> dict:
    with open(f"{d}/manifest.json", "w") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    vol.commit()
    return {k: v for k, v in m.items() if k != "items"}


def hf_revision(repo: str) -> tuple[str, dict]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True)
    return info.sha, {"files": {s.rfilename: s.size for s in info.siblings}, "card_licence": (info.card_data or {}).get("license") if info.card_data else None}


@app.function(**fn)
def ikhlas() -> dict:
    import io
    import os
    import wave
    import zipfile

    import openpyxl
    import requests

    d = f"{PUBLIC}/ikhlas"
    os.makedirs(f"{d}/raw", exist_ok=True)
    z = f"{d}/raw/Dataset and Sounds.zip"
    if not os.path.exists(z) or file_sha(z) != IKHLAS_SHA:
        with requests.get(IKHLAS_URL, stream=True, timeout=600) as r, open(z + ".tmp", "wb") as f:
            r.raise_for_status()
            for c in r.iter_content(1 << 20):
                f.write(c)
        os.replace(z + ".tmp", z)
    sha = file_sha(z)
    assert sha == IKHLAS_SHA, f"sha256 {sha} != {IKHLAS_SHA}"
    meta = requests.get(IKHLAS_API, timeout=60).json()
    with open(f"{d}/raw/mendeley.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    os.makedirs(f"{d}/orig", exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        wavs = [n for n in zf.namelist() if n.lower().endswith(".wav")]
        for n in wavs:
            with open(f"{d}/orig/{os.path.basename(n)}", "wb") as f:
                f.write(zf.read(n))
        xlsx = zf.read("Dataset and Sounds/Dataset.xlsx")
    files = sorted(os.path.basename(n) for n in wavs)
    assert len(files) == len(set(files)), "duplicate wav basenames in zip"
    with open(f"{d}/raw/Dataset.xlsx", "wb") as f:
        f.write(xlsx)

    ws = openpyxl.load_workbook(io.BytesIO(xlsx), data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h) for h in rows[0]]
    items = [parse_ikhlas_row({h.strip(): v for h, v in zip(header, r)}) for r in rows[1:] if any(v is not None for v in r)]
    sheet_cross = ikhlas_cross_issues(items, {f: f for f in files})
    repairs = repair_ikhlas_files(items, files)

    audio = convert_all([(f, f"{d}/orig/{f}", f"{d}/wav16k/{f[:-4]}.wav") for f in files])
    src_fmt = {}
    for f in files:
        try:
            with wave.open(f"{d}/orig/{f}") as w:
                src_fmt[f] = f"{w.getframerate()}Hz/{w.getnchannels()}ch/{8 * w.getsampwidth()}bit"
        except Exception as e:
            src_fmt[f] = f"unreadable:{type(e).__name__}"
    byte_hash = {f: file_sha(f"{d}/orig/{f}") for f in files}
    pcm_hash = {f: a["pcm_sha256"] for f, a in audio.items() if "pcm_sha256" in a}
    cross = ikhlas_cross_issues(items, byte_hash) + [{**c, "kind": "pcm_" + c["kind"]} for c in ikhlas_cross_issues(items, pcm_hash) if c["kind"] == "duplicate_audio"]
    dup = dedup(pcm_hash)
    for it in items:
        a = audio.get(it["file"], {"error": "missing"})
        it.update(wav=f"wav16k/{it['file'][:-4]}.wav" if "pcm_sha256" in a else None, orig=f"orig/{it['file']}", duration_s=a.get("duration_s"), audio_sha256=byte_hash.get(it["file"]), pcm_sha256=a.get("pcm_sha256"), dup_of=dup.get(it["file"]))
        it["issues"] += audio_issues(a)

    mirror = {}
    try:
        from huggingface_hub import hf_hub_download

        os.makedirs("/tmp/mirror", exist_ok=True)
        with zipfile.ZipFile(hf_hub_download(IKHLAS_MIRROR, "data/ikhlas_recitations.zip", repo_type="dataset")) as mz:
            mnames = {}
            for n in mz.namelist():
                if n.lower().endswith(".wav") and not os.path.basename(n).startswith("._"):
                    b = os.path.basename(n)
                    with open(f"/tmp/mirror/{b}", "wb") as f:
                        f.write(mz.read(n))
                    mnames[b] = file_sha(f"/tmp/mirror/{b}")
        mpcm = {b: a.get("pcm_sha256") for b, a in convert_all([(b, f"/tmp/mirror/{b}", f"/tmp/mirror16k/{b}") for b in mnames]).items()}
        mirror = {
            "repo": IKHLAS_MIRROR, "files": len(mnames), "same_name": len(set(mnames) & set(files)),
            "same_bytes": sum(byte_hash.get(n) == h for n, h in mnames.items()), "same_pcm_same_name": sum(h is not None and pcm_hash.get(n) == h for n, h in mpcm.items()),
            "pcm_found_anywhere": sum(h in set(pcm_hash.values()) for h in mpcm.values() if h), "appledouble_entries_skipped": True,
            "only_in_mirror": sorted(set(mnames) - set(files)), "only_in_mendeley": sorted(set(files) - set(mnames)),
        }
    except Exception as e:
        mirror = {"repo": IKHLAS_MIRROR, "error": f"{type(e).__name__}: {e}"[:300]}

    lab = Counter(it["label"] for it in items)
    return write_manifest(d, {
        "dataset": "ikhlas",
        "source": "Mendeley Data sxtmmr6mvk v1 (doi 10.17632/sxtmmr6mvk.1), 'Dataset and Sounds.zip'",
        "url": IKHLAS_URL,
        "sha256": sha,
        "licence": (meta.get("data_licence") or {}).get("short_name"),
        "sheet_columns": header,
        "counts": {
            "rows": len(items), "error": lab["error"], "correct": lab["correct"], "zip_wavs": len(files),
            "unique_filenames_in_sheet": len({i.get("sheet_file", i["file"]) for i in items}), "unique_filenames_after_repair": len({i["file"] for i in items}), "unique_audio_bytes": len(set(byte_hash.values())), "unique_audio_pcm": len(set(pcm_hash.values())),
            "speakers": len({i["speaker"] for i in items if i["speaker"]}), "by_verse": dict(sorted(Counter(i["ref"] for i in items).items())),
            "error_classes": dict(Counter(c for i in items for c in i["error_classes"]).most_common()),
            "rows_by_class_set": dict(Counter("+".join(sorted(set(i["error_classes"]))) or "none" for i in items).most_common()),
            "error_words": dict(Counter(norm(w) for i in items for w in i["error_words"]).most_common()),
            "explanations": dict(Counter(e for i in items for e in i["explanations"]).most_common(40)),
            "row_issues": dict(Counter(s.split(":")[0] for i in items for s in i["issues"]).most_common()),
            "source_formats": dict(Counter(src_fmt.values()).most_common()),
            "duration_s": {"total": round(sum(i["duration_s"] or 0 for i in items), 1), "max": max(i["duration_s"] or 0 for i in items)},
        },
        "sheet_cross_issues": sheet_cross,
        "filename_repairs": repairs,
        "cross_issues": cross,
        "row_issues": [{"id": i["id"], "file": i["file"], "issues": i["issues"]} for i in items if i["issues"]],
        "mirror": mirror,
        "items": items,
    })


@app.function(**fn)
def retasy() -> dict:
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    d = f"{PUBLIC}/retasy"
    rev, info = hf_revision(RETASY)
    readme = open(hf_hub_download(RETASY, "README.md", repo_type="dataset", revision=rev)).read()
    os.makedirs(f"{d}/orig", exist_ok=True)
    with open(f"{d}/README.md", "w") as f:
        f.write(readme)
    items, jobs, seen = [], [], Counter()
    for shard in sorted(f for f in info["files"] if f.endswith(".parquet")):
        pf = pq.ParquetFile(hf_hub_download(RETASY, shard, repo_type="dataset", revision=rev))
        for b in pf.iter_batches(batch_size=256):
            for r in b.to_pylist():
                a = r.pop("audio") or {}
                stem, ext = os.path.splitext(os.path.basename(a.get("path") or "")) if a.get("path") else (f"row{len(items):05d}", ".wav")
                seen[stem] += 1
                rid = stem if seen[stem] == 1 else f"{stem}-{seen[stem]}"
                orig = f"orig/{rid}{ext or '.wav'}"
                with open(f"{d}/{orig}", "wb") as f:
                    f.write(a.get("bytes") or b"")
                jobs.append((rid, f"{d}/{orig}", f"{d}/wav16k/{rid}.wav"))
                items.append({
                    "id": rid, "orig": orig, "shard": shard,
                    "label": retasy_label(r.get("final_label")), "label_raw": r.get("final_label"),
                    "surah": r.get("Surah"), "aya_text": r.get("Aya"), "quranic": retasy_quranic(r.get("Surah"), r.get("Aya")), "golden": r.get("golden"), "judgments": r.get("judgments_num"),
                    "annotators": retasy_annotators(r.get("annotation_metadata")),
                    "reciter_id": r.get("reciter_id"), "reciter_country": r.get("reciter_country"), "reciter_gender": r.get("reciter_gender"),
                    "reciter_age": r.get("reciter_age"), "reciter_qiraah": r.get("reciter_qiraah"), "duration_ms": r.get("duration_ms"), "create_date": r.get("create_date"),
                })
    audio = convert_all(jobs)
    dup = dedup({i: a["pcm_sha256"] for i, a in audio.items() if "pcm_sha256" in a})
    for it in items:
        a = audio[it["id"]]
        it.update(wav=f"wav16k/{it['id']}.wav" if "pcm_sha256" in a else None, duration_s=a.get("duration_s"), pcm_sha256=a.get("pcm_sha256"), dup_of=dup.get(it["id"]), issues=audio_issues(a))
        if it["label"] != "unlabeled" and not it["annotators"] and not it["golden"]:
            it["issues"].append("label_without_annotations")
        if it["annotators"] and it["label"] != "unlabeled" and Counter(x["label"] for x in it["annotators"]).most_common(1)[0][0] != it["label"] and not it["golden"]:
            it["issues"].append("final_label_not_majority")
    labelled = [i for i in items if i["label"] != "unlabeled"]
    groups = defaultdict(list)
    for i in items:
        groups[i["dup_of"] or i["id"]].append(i)
    dups = [g for g in groups.values() if len(g) > 1]
    canon = [g[0] for g in groups.values()]
    return write_manifest(d, {
        "dataset": "retasy",
        "source": f"https://huggingface.co/datasets/{RETASY}",
        "revision": rev,
        "licence": info["card_licence"] or ("none in card metadata" + ("" if re.search(r"licen[cs]e", readme, re.I) else " or README")),
        "counts": {
            "rows": len(items), "by_label": dict(Counter(i["label"] for i in items).most_common()), "labelled": len(labelled),
            "golden": sum(bool(i["golden"]) for i in items), "golden_by_label": dict(Counter(i["label"] for i in items if i["golden"]).most_common()),
            "reciters": len({i["reciter_id"] for i in items}), "reciters_labelled": len({i["reciter_id"] for i in labelled}),
            "surahs": dict(Counter(i["surah"] for i in items).most_common()), "surahs_labelled": dict(Counter(i["surah"] for i in labelled).most_common()),
            "qiraah": dict(Counter(i["reciter_qiraah"] for i in items).most_common()), "judgments": dict(Counter(i["judgments"] for i in labelled).most_common()),
            "quranic_rows": sum(i["quranic"] for i in items), "by_label_quranic": dict(Counter(i["label"] for i in items if i["quranic"]).most_common()),
            "by_label_not_quranic": dict(Counter(i["label"] for i in items if not i["quranic"]).most_common()),
            "aya_texts": len({norm(i["aya_text"] or "") for i in items}), "aya_texts_labelled": len({norm(i["aya_text"] or "") for i in labelled}),
            "unique_audio_pcm": len({i["pcm_sha256"] for i in items if i["pcm_sha256"]}), "duplicate_rows": len(dup),
            "duplicate_groups": len(dups), "duplicate_group_sizes": dict(Counter(len(g) for g in dups).most_common()),
            "duplicate_groups_same_reciter": sum(len({i["reciter_id"] for i in g}) == 1 for g in dups),
            "duplicate_groups_same_text": sum(len({norm(i["aya_text"] or "") for i in g}) == 1 for g in dups),
            "duplicate_groups_conflicting_labels": sum(len({i["label"] for i in g} - {"unlabeled"}) > 1 for g in dups),
            "duplicate_groups_labelled_and_unlabelled": sum(len({i["label"] for i in g} - {"unlabeled"}) >= 1 and any(i["label"] == "unlabeled" for i in g) for g in dups),
            "by_label_after_dedup": dict(Counter(next((i["label"] for i in g if i["label"] != "unlabeled"), "unlabeled") for g in groups.values()).most_common()),
            "largest_duplicate_groups": [{"ids": [i["id"] for i in g][:5], "size": len(g), "duration_s": g[0]["duration_s"], "texts": len({i["aya_text"] for i in g}), "labels": dict(Counter(i["label"] for i in g))} for g in sorted(dups, key=len, reverse=True)[:5]],
            "issues": dict(Counter(s for i in items for s in i["issues"]).most_common()),
            "issues_labelled": dict(Counter(s for i in labelled for s in i["issues"]).most_common()),
            "duration_s": {"total": round(sum(i["duration_s"] or 0 for i in items), 1), "labelled": round(sum(i["duration_s"] or 0 for i in labelled), 1)},
        },
        "items": items,
    })


@app.function(**fn)
def quranmb() -> dict:
    import csv
    import os

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    d = f"{PUBLIC}/quranmb_v2"
    rev, info = hf_revision(QURANMB)
    os.makedirs(f"{d}/orig", exist_ok=True)
    shards = sorted(f for f in info["files"] if f.endswith(".parquet"))
    paths = {s: hf_hub_download(QURANMB, s, repo_type="dataset", revision=rev) for s in shards}
    shard_sha = {s: file_sha(p) for s, p in paths.items()}
    test = next(s for s in shards if "test" in s)
    items, jobs = [], []
    for b in pq.ParquetFile(paths[test]).iter_batches(batch_size=256):
        for r in b.to_pylist():
            a = r.pop("audio") or {}
            ext = os.path.splitext(a.get("path") or "")[1] or ".wav"
            orig = f"orig/{r['ID']}{ext}"
            with open(f"{d}/{orig}", "wb") as f:
                f.write(a.get("bytes") or b"")
            jobs.append((r["ID"], f"{d}/{orig}", f"{d}/wav16k/{r['ID']}.wav"))
            items.append({"id": r["ID"], "orig": orig, "audio_path": a.get("path")})
    audio = convert_all(jobs)
    dup = dedup({i: a["pcm_sha256"] for i, a in audio.items() if "pcm_sha256" in a})
    for it in items:
        a = audio[it["id"]]
        it.update(wav=f"wav16k/{it['id']}.wav" if "pcm_sha256" in a else None, duration_s=a.get("duration_s"), pcm_sha256=a.get("pcm_sha256"), dup_of=dup.get(it["id"]), issues=audio_issues(a))

    gt = {"repo": QURANMB_GT}
    try:
        gt_rev, gt_info = hf_revision(QURANMB_GT)
        p = hf_hub_download(QURANMB_GT, "labels_test.csv", repo_type="dataset", revision=gt_rev)
        os.makedirs(f"{d}/gt", exist_ok=True)
        with open(p, encoding="utf-8") as f:
            text = f.read()
        with open(f"{d}/gt/labels_test.csv", "w") as f:
            f.write(text)
        rows = list(csv.DictReader(text.splitlines()))
        cols = list(rows[0]) if rows else []
        idc = next(c for c in cols if c.lower() == "id")
        refc = next((c for c in cols if "ref" in c.lower()), None)
        annc = next((c for c in cols if "annot" in c.lower() or "said" in c.lower() or "hyp" in c.lower()), None)
        spkc = next((c for c in cols if "speaker" in c.lower() or "spk" in c.lower()), None)
        by_id = {r[idc]: r for r in rows}
        tot = Counter()
        for it in items:
            g = by_id.get(it["id"])
            it["gt"] = g
            if g and refc and annc:
                e = phone_edits(g[refc].split(), g[annc].split())
                it["phone_edits"] = e
                tot.update(e)
                tot["items_with_errors"] += any(e.values())
                tot["ref_phones"] += len(g[refc].split())
            if not g:
                it["issues"].append("no_gt")
        gt.update(revision=gt_rev, licence=gt_info["card_licence"], columns=cols, rows=len(rows), joined=sum(1 for i in items if i.get("gt")), gt_without_audio=len(set(by_id) - {i["id"] for i in items}),
                  ref_column=refc, said_column=annc, speaker_column=spkc, speakers=len({r[spkc] for r in rows}) if spkc else None,
                  verses=len({r[refc] for r in rows}) if refc else None, phone_errors=dict(tot))
    except Exception as e:
        gt["error"] = f"{type(e).__name__}: {e}"[:600]

    return write_manifest(d, {
        "dataset": "quranmb_v2",
        "source": f"https://huggingface.co/datasets/{QURANMB}",
        "revision": rev,
        "licence": info["card_licence"] or "none in card metadata or README",
        "shards": {s: {"bytes": info["files"][s], "sha256": shard_sha[s]} for s in shards},
        "counts": {
            "rows": len(items), "unique_ids": len({i["id"] for i in items}), "unique_audio_pcm": len({i["pcm_sha256"] for i in items if i["pcm_sha256"]}),
            "id_prefixes": dict(Counter(i["id"].split("_")[0] for i in items).most_common()), "id_samples": [i["id"] for i in items[:10]],
            "id_suffixes": len({i["id"].split("_")[-1] for i in items}),
            "id_suffix_spread": dict(Counter(n for n in Counter(i["id"].split("_")[-1] for i in items).values()).most_common()),
            "issues": dict(Counter(s for i in items for s in i["issues"]).most_common()),
            "duration_s": {"total": round(sum(i["duration_s"] or 0 for i in items), 1), "max": max(i["duration_s"] or 0 for i in items)},
        },
        "gt": gt,
        "items": items,
    })


@app.local_entrypoint()
def main(only: str = "ikhlas,retasy,quranmb"):
    fns = {"ikhlas": ikhlas, "retasy": retasy, "quranmb": quranmb}
    calls = {k: fns[k].spawn() for k in only.split(",")}
    for k, c in calls.items():
        try:
            print(f"== {k}\n" + json.dumps(c.get(), ensure_ascii=False, indent=1, default=str))
        except Exception as e:
            print(f"== {k} FAILED: {type(e).__name__}: {e}")
