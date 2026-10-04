"""PER eval sets on Modal, so q-lab benchmark audio never leaves the volume.

Sets (all audio on volume `zipformer-ctc-training`):
- ``qlab``: the 600 q-lab clips (`/vol/phase0/qlab`), headline/ordered/text golds.
- ``heldout_multi``: 2–4-ayah windows of the three q-lab EA reciters (test gate).
- ``dev_everyayah``: 6 reciters outside every training manifest (dev).
- ``tlog_dev`` / ``tlog_dev_nonclean``: TLOG clips held out of tlog_clean_v3
  (clip-disjoint only; TLOG has no speaker ids).

  modal run scripts/eval_modal.py --model-name v3 --onnx /vol/reference/zipformer_p_arabic_v3.onnx
  modal run scripts/eval_modal.py --upload-corpus /tmp/phase0/dev_everyayah --corpus-name dev_everyayah

Per-clip JSONL + summary.json land in /vol/phase0/eval/<model-name>/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

LAB = Path(__file__).resolve().parent.parent
for _p in (LAB, LAB / "scripts", Path("/app"), Path("/app/scripts")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SETS = ("qlab", "heldout_multi", "dev_everyayah", "tlog_dev", "tlog_dev_nonclean")
OUT = Path("/vol/phase0/eval")
CORPORA = Path("/vol/phase0/corpora")
REF = Path("/vol/reference")

vol = modal.Volume.from_name("zipformer-ctc-training")
app = modal.App("tilawa-eval")


def _data(rel: str) -> str:
    from shared.paths import resolve_data_file

    return str(resolve_data_file(rel))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("numpy<2", "onnxruntime==1.20.1", "soundfile", "librosa", "python-Levenshtein")
    .pip_install("torch", "torchaudio", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"OMP_NUM_THREADS": "1", "TILAWA_DATA_ROOT": "/app/data"})
    .add_local_file(str(LAB / "shared" / "paths.py"), "/app/shared/paths.py")
    .add_local_file(str(LAB / "shared" / "audio.py"), "/app/shared/audio.py")
    .add_local_file(str(LAB / "shared" / "fbank.py"), "/app/shared/fbank.py")
    .add_local_file(str(LAB / "shared" / "normalizer.py"), "/app/shared/normalizer.py")
    .add_local_file(str(LAB / "shared" / "quran_db.py"), "/app/shared/quran_db.py")
    .add_local_file(str(LAB / "shared" / "phoneme_labels.py"), "/app/shared/phoneme_labels.py")
    .add_local_file(str(LAB / "shared" / "zipformer_score.py"), "/app/shared/zipformer_score.py")
    .add_local_file(str(LAB / "scripts" / "qlab_per_eval.py"), "/app/scripts/qlab_per_eval.py")
    .add_local_file(str(LAB / "experiments" / "zipformer-ctc" / "tokens.txt"), "/app/experiments/zipformer-ctc/tokens.txt")
    .add_local_file(str(LAB / "data" / "quran.json"), "/app/data/quran.json")
    .add_local_file(_data("zipformer/quran.json"), "/app/data/zipformer/quran.json")
    .add_local_file(str(LAB / "benchmark" / "test_corpus_qlab" / "manifest.json"),
                    "/app/benchmark/test_corpus_qlab/manifest.json")
)
_KW = dict(image=image, volumes={"/vol": vol})


def _tok():
    from shared.phoneme_labels import PhonemeTokenizer, load_tokens

    return PhonemeTokenizer(load_tokens("/app/experiments/zipformer-ctc/tokens.txt"))


def _tlog_rows(nonclean: bool) -> list[dict]:
    import gzip

    ids_file = Path("/vol/phase0/tlog_filter/v3") / ("dev_nonclean_ids.txt" if nonclean else "dev_ids.txt")
    want = {l.strip() for l in ids_file.read_text().splitlines() if l.strip()}
    ordered = json.loads((REF / "ordered_quran_phonemes.json").read_text(encoding="utf-8"))
    tok = _tok()
    rows = []
    with gzip.open("/vol/manifests/tlog_cuts.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["id"] not in want:
                continue
            cu = c["supervisions"][0]["custom"]
            e = ordered[f"{cu['surah']}:{cu['ayah']}"]
            rows.append({"id": c["id"], "source": "tlog_dev_nonclean" if nonclean else "tlog_dev",
                         "wav": c["recording"]["sources"][0]["source"], "surah": cu["surah"], "ayah": cu["ayah"],
                         "gold_ordered": tok.encode(e["aya_phoneme"].replace(" ", "")), "gold_text": None})
    return rows


def _rows(set_name: str) -> list[dict]:
    import qlab_per_eval as q

    if set_name == "qlab":
        return q.build_rows(Path("/vol/phase0/qlab"), Path("/app/benchmark/test_corpus_qlab/manifest.json"), REF, _tok())
    if set_name.startswith("tlog_dev"):
        return _tlog_rows(set_name.endswith("nonclean"))
    return q.build_corpus_rows(CORPORA / set_name, REF, _tok())


@app.function(cpu=2, memory=4096, timeout=3600, **_KW)
def score_rows(onnx: str, rows: list[dict]) -> list[dict]:
    import qlab_per_eval as q

    q._init(onnx)
    return [q._run(r) for r in rows]


@app.function(cpu=2, memory=8192, timeout=6 * 3600, **_KW)
def run_eval(model_name: str, onnx: str, sets: list[str], shard: int = 40) -> dict:
    import qlab_per_eval as q

    out = OUT / model_name
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for s in sets:
        rows = _rows(s)
        chunks = [rows[i : i + shard] for i in range(0, len(rows), shard)]
        per_clip = [r for part in score_rows.map([onnx] * len(chunks), chunks) for r in part]
        (out / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_clip), encoding="utf-8")
        summaries[s] = q.summarize(per_clip)
        vol.commit()
    (out / "summary.json").write_text(json.dumps({"onnx": onnx, "sets": summaries}, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    return summaries


@app.local_entrypoint()
def main(model_name: str = "", onnx: str = "", sets: str = ",".join(SETS), upload_corpus: str = "", corpus_name: str = ""):
    if upload_corpus:
        with vol.batch_upload(force=True) as b:
            b.put_directory(upload_corpus, f"/phase0/corpora/{corpus_name}")
        print(f"uploaded {upload_corpus} -> /vol/phase0/corpora/{corpus_name}")
        return
    res = run_eval.remote(model_name, onnx, [s.strip() for s in sets.split(",") if s.strip()])
    for s, summ in res.items():
        for g in ("headline", "ordered", "text"):
            if summ.get(g):
                line = "  ".join(f"{k}={v['per']:.2f}(S{v['sub']:.2f}/I{v['ins']:.2f}/D{v['del']:.2f},n{v['clips']})"
                                 for k, v in summ[g].items())
                print(f"[{model_name}] {s} {g}: {line}")
        if "ayah_dropped" in summ:
            print(f"[{model_name}] {s} ayah_dropped: {summ['ayah_dropped']}")
