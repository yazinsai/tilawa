"""Phase 0 on Modal: leak audit, leak-free manifests, TLOG inference filter.

Outputs live on volume `zipformer-ctc-training` under `/vol/phase0/` (per-clip
scores and id lists stay off git: they are derived from gated / benchmark data).

  # held-out clip durations for twin checks (downloads q-lab audio once)
  modal run scripts/phase0_modal.py --action qlab-clips
  # scan every *_cuts_fbank manifest; --write-rx also writes <src>_rx_cuts_fbank
  modal run scripts/phase0_modal.py --action leak-audit [--write-rx]
  # greedy CTC + forced score over all staged TLOG train clips and tlog_holdout
  modal run --detach scripts/phase0_modal.py --action tlog-filter --model v3 --shards 64
  # bucket counts + tlog_clean manifest from the per-clip scores
  modal run scripts/phase0_modal.py --action tlog-merge --model v3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

LAB = Path(__file__).resolve().parent.parent
for _p in (LAB, Path("/app")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MODELS = {
    "v3": "/vol/reference/zipformer_p_arabic_v3.onnx",
    "v31": "/vol/reference/zipformer_p_arabic_v3.1.onnx",
    "interp-gentle-a0.5": "/vol/exports/interp-gentle-a0.5/model.onnx",
}
QLAB_REPO = "Quran-Lab/quranic-asr-benchmark"
OUT = Path("/vol/phase0")
MAN = Path("/vol/manifests")
AUDIT_SOURCES = ("everyayah", "everyayah_multi", "qua", "iqra", "retasy", "tlog")
# holdout slice each training source is twin-checked against; "enforce" twins
# fail the train-time assertion, "report" twins are listed only (a big studio
# source like QUA has many recitations per ayah, so 60 ms twins can be chance).
TWIN_PLAN = {
    "everyayah": (("everyayah_heldout", "enforce"),),
    "everyayah_multi": (),
    "qua": (("everyayah_heldout", "report"), ("qul_alnufais", "report")),
    "tlog": (("tlog_holdout", "enforce"),),
}

vol = modal.Volume.from_name("zipformer-ctc-training")
app = modal.App("tilawa-phase0")


def _data(rel: str) -> str:
    from shared.paths import resolve_data_file

    return str(resolve_data_file(rel))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "numpy<2",
        "onnxruntime==1.20.1",
        "soundfile",
        "python-Levenshtein",
        "rapidfuzz",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
    )
    .pip_install("torch", "torchaudio", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "OMP_NUM_THREADS": "1"})
    .add_local_file(str(LAB / "shared" / "paths.py"), "/app/shared/paths.py")
    .add_local_file(str(LAB / "shared" / "fbank.py"), "/app/shared/fbank.py")
    .add_local_file(str(LAB / "shared" / "phoneme_labels.py"), "/app/shared/phoneme_labels.py")
    .add_local_file(str(LAB / "shared" / "leak_guard.py"), "/app/shared/leak_guard.py")
    .add_local_file(str(LAB / "shared" / "zipformer_score.py"), "/app/shared/zipformer_score.py")
    .add_local_file(str(LAB / "experiments" / "zipformer-ctc" / "tokens.txt"), "/app/tokens.txt")
    .add_local_file(_data("zipformer/quran.json"), "/app/data/zipformer/quran.json")
    .add_local_file(
        str(LAB / "benchmark" / "test_corpus_qlab" / "manifest.json"),
        "/app/benchmark/test_corpus_qlab/manifest.json",
    )
)

_KW = dict(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("huggingface")])


def _boot() -> None:
    sys.path.insert(0, "/app")
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")
    OUT.mkdir(parents=True, exist_ok=True)


def _iter_jsonl_gz(path: Path):
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- q-lab clips


@app.function(cpu=4, memory=8192, timeout=3600, **_KW)
def qlab_clips() -> dict:
    """Download q-lab audio to the volume and record (source, id, surah, ayah, duration)."""
    _boot()
    import soundfile as sf
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(QLAB_REPO, repo_type="dataset", local_dir=str(OUT / "qlab"),
                                  allow_patterns=["audio/**", "benchmark.jsonl"]))
    ours = json.loads(Path("/app/benchmark/test_corpus_qlab/manifest.json").read_text(encoding="utf-8"))["samples"]
    sa = {s["id"].replace("__", "/"): (s["surah"], s["ayah"]) for s in ours}
    clips = []
    for line in (root / "benchmark.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        wav = root / o["audio"]
        if not wav.is_file() or o["id"] not in sa:
            continue
        s, a = sa[o["id"]]
        clips.append({"source": o["source"], "id": o["id"], "surah": s, "ayah": a,
                      "duration": round(sf.info(str(wav)).duration, 4), "wav": str(wav)})
    (OUT / "qlab_clips.json").write_text(json.dumps(clips) + "\n", encoding="utf-8")
    vol.commit()
    return {"n": len(clips)}


def _load_qlab_clips():
    from shared.leak_guard import HeldoutClip

    rows = json.loads((OUT / "qlab_clips.json").read_text(encoding="utf-8"))
    return [HeldoutClip(r["source"], r["id"], r["surah"], r["ayah"], r["duration"]) for r in rows]


# --------------------------------------------------------------------------- leak audit


@app.function(cpu=2, memory=8192, timeout=3 * 3600, **_KW)
def audit_source(source: str, write_rx: bool = False) -> dict:
    _boot()
    import gzip

    from shared.leak_guard import build_twin_index, scan_cuts

    path = MAN / f"{source}_cuts_fbank.jsonl.gz"
    if not path.is_file():
        return {"source": source, "missing": str(path)}
    clips = _load_qlab_clips()
    plan = next((v for k, v in sorted(TWIN_PLAN.items(), key=lambda kv: -len(kv[0])) if source.startswith(k)), ())
    enforce = build_twin_index(c for c in clips if (c.source, "enforce") in plan)
    report_only = build_twin_index(c for c in clips if (c.source, "report") in plan)

    cuts = list(_iter_jsonl_gz(path))
    rep = scan_cuts(source, cuts, twins=enforce)
    rep_only = scan_cuts(source, cuts, twins=report_only) if report_only else None
    out = rep.to_json()
    out["hours"] = round(sum(float(c.get("duration") or 0) for c in cuts) / 3600, 2)
    out["flagged_hours"] = round(sum(float(c.get("duration") or 0) for c in cuts if rep.is_flagged(c.get("id"))) / 3600, 2)
    out["flagged_cuts_incl_perturbed"] = sum(1 for c in cuts if rep.is_flagged(c.get("id")))
    if rep_only is not None:
        out["report_only_twins"] = dict(rep_only.twin_hits)

    if write_rx:
        dest = MAN / f"{source}_rx_cuts_fbank.jsonl.gz"
        kept = [c for c in cuts if not rep.is_flagged(c.get("id"))]
        tmp = dest.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        tmp.replace(dest)
        recheck = scan_cuts(f"{source}_rx", kept, twins=enforce)
        out["rx"] = {"path": str(dest), "n_cuts": len(kept), "n_flagged_after": recheck.n_flagged}
    (OUT / "leak_audit").mkdir(parents=True, exist_ok=True)
    (OUT / "leak_audit" / f"{source}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    vol.commit()
    return out


# --------------------------------------------------------------------------- QUA fingerprint


def _fbank_sim(a_wav: str, b_wav: str, max_lag: int = 30) -> float:
    """Best-lag mean frame cosine of mean-normalised fbank; ~1 for the same recording."""
    import numpy as np
    import soundfile as sf

    from shared.fbank import compute_fbank

    def feats(p):
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            import torch
            import torchaudio

            x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, 16000).numpy()
        f = compute_fbank(x, sr=16000)
        f = f - f.mean(0, keepdims=True)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

    a, b = feats(a_wav), feats(b_wav)
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        n = min(len(aa), len(bb))
        if n < 50:
            continue
        best = max(best, float((aa[:n] * bb[:n]).sum(1).mean()))
    return best


@app.function(cpu=4, memory=8192, timeout=3 * 3600, **_KW)
def qua_fingerprint(tol_s: float = 0.06, n_controls: int = 150) -> dict:
    """Audio check of QUA clips that are (surah, ayah, duration) twins of held-out clips."""
    _boot()
    import random
    from collections import Counter, defaultdict

    clips = [r for r in json.loads((OUT / "qlab_clips.json").read_text(encoding="utf-8"))
             if r["source"] in ("everyayah_heldout", "qul_alnufais")]
    by_sa = defaultdict(list)
    for r in clips:
        by_sa[(r["surah"], r["ayah"])].append(r)
    qua_by_sa = defaultdict(list)
    for c in _iter_jsonl_gz(MAN / "qua_cuts.jsonl.gz"):
        sup = c["supervisions"][0]
        cu = sup["custom"]
        key = (int(cu["surah"]), int(cu["ayah"]))
        if key in by_sa:
            qua_by_sa[key].append((c["id"], sup.get("speaker"), float(c["duration"]), c["recording"]["sources"][0]["source"]))
    pairs, controls = [], []
    rng = random.Random(0)
    for key, hs in by_sa.items():
        for h in hs:
            cands = qua_by_sa.get(key, [])
            twins = [q for q in cands if abs(q[2] - h["duration"]) <= tol_s]
            others = [q for q in cands if abs(q[2] - h["duration"]) > 0.5]
            pairs += [(h, q) for q in twins]
            if others:
                controls.append((h, rng.choice(others)))
    rng.shuffle(controls)
    controls = controls[:n_controls]
    rows = []
    for kind, lst in (("twin", pairs), ("control", controls)):
        for h, q in lst:
            try:
                sim = _fbank_sim(h["wav"], q[3])
            except Exception as e:
                sim = None
            rows.append({"kind": kind, "heldout_source": h["source"], "qua_speaker": q[1], "sim": sim,
                         "dur_h": h["duration"], "dur_q": q[2]})
    (OUT / "qua_fingerprint.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    vol.commit()
    import numpy as np

    def q(kind, src=None):
        v = [r["sim"] for r in rows if r["kind"] == kind and r["sim"] is not None and (src is None or r["heldout_source"] == src)]
        return {"n": len(v), **({p: round(float(np.quantile(v, p)), 3) for p in (0.05, 0.5, 0.95, 1.0)} if v else {})}

    ctrl_max = max((r["sim"] for r in rows if r["kind"] == "control" and r["sim"] is not None), default=0.0)
    hits = [r for r in rows if r["kind"] == "twin" and r["sim"] is not None and r["sim"] > max(0.9, ctrl_max)]
    return {
        "twins": q("twin"), "twins_ea": q("twin", "everyayah_heldout"), "twins_nufais": q("twin", "qul_alnufais"),
        "controls": q("control"), "control_max": round(ctrl_max, 3),
        "same_recording_hits": len(hits),
        "hit_speakers": dict(Counter((r["heldout_source"], r["qua_speaker"]) for r in hits).most_common(10)),
    }


# --------------------------------------------------------------------------- label variants

WAQF_MADD = ("ۥ", "ۦ", "ا")


_WAQF_RE = __import__("re").compile(r"([ۥۦا])\1{3}(?!\1)([^ۥۦا]{0,2})$")


def waqf2(ayah_ph: str) -> str:
    """Ayah-final 4-beat madd (optionally before <=2 closing chars) -> 2 beats.

    Matches the q-lab v1.1 reference gold on 406/487 shared clips vs 228 before;
    the reference table itself keeps 4 beats on most of the rest.
    """
    m = _WAQF_RE.search(ayah_ph)
    if m and not ayah_ph[: m.start()].endswith(m.group(1)):
        return ayah_ph[: m.start()] + m.group(1) * 2 + m.group(2)
    return ayah_ph


@app.function(cpu=2, memory=8192, timeout=3 * 3600, **_KW)
def relabel_source(src: str, dst: str, repeat: int = 1, relabel_json: str = "", only_relabelled: bool = False,
                   waqf: bool = True) -> dict:
    """Write `<dst>_cuts_fbank` from `<src>_cuts_fbank` with waqf-2 labels, optional
    per-clip ayah relabels ({base_id: "s:a"}) and whole-manifest repeats."""
    _boot()
    import gzip

    from shared.leak_guard import base_cut_id
    from shared.phoneme_labels import PhonemeCorpus

    corpus = PhonemeCorpus("/app/data/zipformer/quran.json")
    relabel = json.loads(Path(relabel_json).read_text()) if relabel_json else {}
    n = changed = mismatch = relabelled = 0
    out = MAN / f"{dst}_cuts_fbank.jsonl.gz"
    tmp = out.with_suffix(".tmp")
    rows = []
    for c in _iter_jsonl_gz(MAN / f"{src}_cuts_fbank.jsonl.gz"):
        n += 1
        sup = c["supervisions"][0]
        cu = sup["custom"]
        s, a, e = int(cu["surah"]), int(cu["ayah"]), int(cu.get("ayah_end") or cu["ayah"])
        new_sa = relabel.get(base_cut_id(c["id"]))
        if only_relabelled and not new_sa:
            n -= 1
            continue
        if new_sa:
            s, a = map(int, new_sa.split(":"))
            e = a
            cu.update(surah=s, ayah=a, ayah_end=a, relabelled=True)
            relabelled += 1
        try:
            parts = [corpus.ayah_phonemes(s, x) for x in range(a, e + 1)]
        except ValueError:
            parts = None
        if parts is None or (not new_sa and "".join(parts) != sup["text"]):
            mismatch += 1
            text = waqf2(sup["text"]) if waqf else sup["text"]  # end-only fallback
        else:
            text = "".join(waqf2(p) if waqf else p for p in parts)
        changed += text != sup["text"]
        sup["text"] = text
        rows.append(c)
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in range(repeat):
            for c in rows:
                if r:
                    c = {**c, "id": f"{c['id']}_rep{r}", "supervisions": [{**c["supervisions"][0], "id": f"{c['supervisions'][0]['id']}_rep{r}"}]}
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(out)
    vol.commit()
    return {"src": src, "dst": dst, "cuts": n * repeat, "changed": changed, "fallback": mismatch, "relabelled": relabelled}


# --------------------------------------------------------------------------- TLOG filter

_S: dict = {}


def _scorer(model: str):
    if "m" not in _S:
        from shared.phoneme_labels import PhonemeCorpus, PhonemeTokenizer, load_tokens
        from shared.zipformer_score import StreamingZipformer

        _S["m"] = StreamingZipformer(MODELS[model], threads=1)
        _S["tokens"] = load_tokens("/app/tokens.txt")
        _S["tok"] = PhonemeTokenizer(_S["tokens"])
        corpus = PhonemeCorpus("/app/data/zipformer/quran.json")
        _S["corpus"] = corpus
        keys, strs = [], []
        for (s, a), words in corpus._ayah_words.items():
            keys.append(f"{s}:{a}")
            strs.append("".join(chr(0x100 + i) for i in _S["tok"].encode("".join(words))))
        _S["alt_keys"], _S["alt_strs"] = keys, strs
    return _S


def _score_one(model: str, clip_id: str, wav_path: str, surah: int, ayah: int, ref_text: str) -> dict:
    import numpy as np
    import soundfile as sf
    from Levenshtein import distance
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein as RL

    from shared.fbank import compute_fbank
    from shared.zipformer_score import score_clip

    S = _scorer(model)
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(audio)), sr, 16000).numpy()
    lp = S["m"].log_probs(compute_fbank(audio, sr=16000))
    ref = S["tok"].encode(ref_text)
    sc = score_clip(lp, ref, clip_id=clip_id, surah=surah, ayah=ayah, duration=len(audio) / 16000, tokens=S["tokens"])
    if sc.per > 0.10:
        hyp_s = "".join(chr(0x100 + i) for i in S["tok"].encode(sc.hyp)) if sc.hyp else ""
        hit = process.extractOne(hyp_s, S["alt_strs"], scorer=RL.normalized_distance)
        if hit is not None:
            _, _, idx = hit
            sc.alt_key = S["alt_keys"][idx]
            sc.alt_per = distance(S["alt_strs"][idx], hyp_s) / max(len(S["alt_strs"][idx]), 1)
    return sc.to_json()


@app.function(cpu=2, memory=4096, timeout=6 * 3600, **_KW)
def tlog_filter_shard(model: str, shard: int, n_shards: int) -> dict:
    _boot()
    import time

    out_dir = OUT / "tlog_filter" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"shard-{shard:04d}.jsonl"
    done = set()
    if dest.is_file():
        done = {json.loads(l)["id"] for l in dest.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = []
    for i, cut in enumerate(_iter_jsonl_gz(MAN / "tlog_cuts.jsonl.gz")):
        if i % n_shards != shard or cut["id"] in done:
            continue
        sup = cut["supervisions"][0]
        c = sup["custom"]
        todo.append((cut["id"], cut["recording"]["sources"][0]["source"], int(c["surah"]), int(c["ayah"]), sup["text"]))
    t0, n, errs = time.time(), 0, 0
    with dest.open("a", encoding="utf-8") as f:
        for cid, wav, s, a, text in todo:
            try:
                row = _score_one(model, cid, wav, s, a, text)
            except Exception as e:  # keep the shard going; record the failure
                errs += 1
                row = {"id": cid, "surah": s, "ayah": a, "error": f"{type(e).__name__}: {e}"[:300]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                f.flush()
                vol.commit()
    vol.commit()
    return {"shard": shard, "scored": n, "skipped_done": len(done), "errors": errs, "secs": round(time.time() - t0, 1)}


@app.function(cpu=2, memory=4096, timeout=3600, **_KW)
def tlog_holdout_score(model: str) -> dict:
    _boot()
    clips = [r for r in json.loads((OUT / "qlab_clips.json").read_text(encoding="utf-8")) if r["source"] == "tlog_holdout"]
    corpus = _scorer(model)["corpus"]
    rows = [_score_one(model, r["id"], r["wav"], r["surah"], r["ayah"], corpus.ayah_phonemes(r["surah"], r["ayah"])) for r in clips]
    dest = OUT / "tlog_filter" / model / "holdout.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    vol.commit()
    from collections import Counter

    return {"n": len(rows), "buckets": dict(Counter(r["bucket"] for r in rows))}


@app.function(cpu=2, memory=8192, timeout=3600, **_KW)
def tlog_merge(model: str, clean_max: float = 0.10, mislabel_min: float = 0.35, dev_frac: float = 0.02) -> dict:
    _boot()
    import gzip
    from collections import Counter

    from shared.leak_guard import base_cut_id
    from shared.zipformer_score import bucket

    d = OUT / "tlog_filter" / model
    rows = []
    for p in sorted(d.glob("shard-*.jsonl")):
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if "per" in r]
    for r in ok:
        r["bucket"] = bucket(r["per"], clean_max, mislabel_min)
    b = Counter(r["bucket"] for r in ok)
    hours = Counter()
    for r in ok:
        hours[r["bucket"]] += r["duration"] / 3600
    alt_better = sum(1 for r in ok if r["bucket"] != "clean" and r.get("alt_per") is not None
                     and r["alt_per"] + 0.05 < r["per"] and r.get("alt_key") != f"{r['surah']}:{r['ayah']}")
    import numpy as np

    pers = np.array([r["per"] for r in ok]) if ok else np.zeros(1)
    import hashlib

    clean_ids = {r["id"] for r in ok if r["bucket"] == "clean"}
    # TLOG has no speaker ids, so this dev slice is clip-disjoint only.
    dev_ids = {i for i in clean_ids if int(hashlib.sha1(i.encode()).hexdigest(), 16) % 1000 < int(dev_frac * 1000)}
    clean_ids -= dev_ids
    (d / "clean_ids.txt").write_text("\n".join(sorted(clean_ids)) + "\n", encoding="utf-8")
    (d / "dev_ids.txt").write_text("\n".join(sorted(dev_ids)) + "\n", encoding="utf-8")
    nonclean_dev = sorted(
        (r for r in ok if r["bucket"] != "clean"),
        key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest(),
    )[:300]
    (d / "dev_nonclean_ids.txt").write_text("\n".join(r["id"] for r in nonclean_dev) + "\n", encoding="utf-8")

    src = MAN / "tlog_rx_cuts_fbank.jsonl.gz"
    if not src.is_file():
        src = MAN / "tlog_cuts_fbank.jsonl.gz"
    dest = MAN / f"tlog_clean_{model.replace('.', '')}_cuts_fbank.jsonl.gz"
    n_out = 0
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        for c in _iter_jsonl_gz(src):
            if base_cut_id(c["id"]) in clean_ids:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_out += 1
    summary = {
        "model": model,
        "n_scored": len(ok),
        "n_errors": len(rows) - len(ok),
        "thresholds": {"clean_max": clean_max, "mislabel_min": mislabel_min},
        "buckets": dict(b),
        "bucket_hours": {k: round(v, 2) for k, v in hours.items()},
        "per_quantiles": {q: round(float(np.quantile(pers, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
        "non_clean_closer_to_other_ayah": alt_better,
        "clean_manifest": {"path": str(dest), "from": str(src), "n_cuts": n_out},
        "dev": {"clean_clips": len(dev_ids), "nonclean_clips": len(nonclean_dev)},
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    return summary


@app.local_entrypoint()
def main(action: str, model: str = "v3", shards: int = 64, write_rx: bool = False,
         clean_max: float = 0.10, mislabel_min: float = 0.35, sources: str = ",".join(AUDIT_SOURCES)):
    if model not in MODELS:
        raise SystemExit(f"--model must be one of {sorted(MODELS)}")
    if action == "qlab-clips":
        print(qlab_clips.remote())
    elif action == "leak-audit":
        srcs = [s.strip() for s in sources.split(",") if s.strip()]
        res = list(audit_source.map(srcs, kwargs={"write_rx": write_rx}))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif action == "tlog-filter":
        hold = tlog_holdout_score.spawn(model)
        res = list(tlog_filter_shard.starmap([(model, i, shards) for i in range(shards)]))
        tot = sum(r["scored"] for r in res)
        print(f"scored {tot} clips in {shards} shards; errors={sum(r['errors'] for r in res)}; "
              f"max shard secs={max(r['secs'] for r in res)}")
        print("holdout:", hold.get())
    elif action == "relabel":
        # --sources src:dst[:repeat][:relabel_json][:only|all][:w2|raw],...
        specs = []
        for spec in sources.split(","):
            parts = spec.split(":") + [""] * 6
            specs.append((parts[0], parts[1], int(parts[2] or 1), parts[3], parts[4] == "only", parts[5] != "raw"))
        for r in relabel_source.starmap(specs):
            print(r)
    elif action == "qua-fingerprint":
        print(json.dumps(qua_fingerprint.remote(), indent=2, ensure_ascii=False))
    elif action == "tlog-holdout":
        print(tlog_holdout_score.remote(model))
    elif action == "tlog-merge":
        print(json.dumps(tlog_merge.remote(model, clean_max, mislabel_min), indent=2))
    else:
        raise SystemExit(f"unknown --action {action}")
