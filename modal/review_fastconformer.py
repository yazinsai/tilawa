"""NVIDIA Arabic FastConformer as an independent audio opinion and data check (.scratch/lab/issues/12-review-asr-calibration.md).

    modal run modal/review_fastconformer.py::smoke                 # one window outside the queue, vocab and versions
    modal run modal/review_fastconformer.py::main                  # arm 1: label-blind transcription, RNNT and CTC greedy
    modal run modal/review_fastconformer.py::likelihood_main       # arm 2: CTC pairwise likelihood, after arm 1
    modal run modal/review_fastconformer.py::check_main            # every planted edit of lab-v1.1: said vs original by CTC
    modal run modal/review_fastconformer.py::check_main --context 2  # exploratory: ~2 s of shared context words each side

nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 (NeMo hybrid Transducer + CTC, ~115M, CC-BY-4.0, writes Arabic with
harakat). Unrelated to the teacher (obadx/muaalem-model-v3_2) and the student. Trained on tarteel-ai/everyayah among
others; the held-out lab reciters exclude everyayah reciters (docs/benchmarks/lab-reciters-v1.md).

- Arm 1 hears only the item id and the 16 kHz window (same bytes and sha256 as modal/review_asr.py) and is compared with
  the labelled text by core.compare, with the same controls. Decision decoder: RNNT (the model's default); CTC greedy
  and the two combined (core.combine) are reported too. Records: /lab/reviews/<testSet>/asr-fastconformer-{rnnt,ctc}.json,
  summary asr-calibration.fastconformer.json.
- Arm 2 scores given texts (labelled, one-edit control, shifted control) by the CTC head's forced-alignment
  log-likelihood. Decision reading: diacritised candidates (core.diacritised_candidates against the window's own
  label-blind CTC greedy text and RNNT transcript), max over spellings, core.prefer with core.LIK_MARGIN. Plain (undiacritised) candidates and near-best
  against the CTC greedy path are reported too. Records: asr-fastconformer-likelihood.json, asr-calibration.fastconformer-likelihood.json.
- The data check scores, for every planted edit, the said text against the original over the teacher check's window
  (core.edit_texts); pass = said wins by core.CHECK_LLR_MIN nats with diacritised candidates. Writes
  /lab/data/<testSet>/checks-fastconformer-v1.json next to the teacher's checks-v1.json.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), "/root"]

VOL, CACHE = "/lab", "/cache"
DATA, REVIEWS = Path(VOL) / "data", Path(VOL) / "reviews"
FC, NEMO = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", "2.7.3"
FC_CONFIG = {"engine": "nemo", "nemoToolkit": NEMO, "dtype": "float32", "decoding": "greedy (model defaults)", "batchSize": 16,
             "ctcScore": "torch ctc_loss over the CTC head's log-probs, blank last, summed over alignments"}
DECODERS = ("rnnt", "ctc")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "build-essential", "git")
    .pip_install(f"nemo_toolkit[asr]=={NEMO}", "soundfile", "huggingface_hub[hf_xet]")
    .env({"HF_HOME": f"{CACHE}/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_file(ROOT / "tools/review_asr_core.py", "/root/review_asr_core.py")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_file(ROOT / "tools/review_asr_core.py", "/root/review_asr_core.py")
)
app = modal.App("tilawa-review-fastconformer")
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip_bytes(clip: dict) -> bytes:
    """The window as a 16 kHz mono PCM16 WAV: the same bytes modal/review_asr.py gives its models."""
    import io
    import wave

    import lab_review_core as lrc

    with open(DATA / lrc.safe_rel(clip["path"]), "rb") as f:
        b = lrc.wav_slice(f, clip["startS"], clip["endS"])
    with wave.open(io.BytesIO(b)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), clip["path"]
    return b


def _write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.replace(p)


@app.cls(image=image, gpu="T4", volumes={VOL: vol, CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 60,
         scaledown_window=5 * 60, max_containers=1)
class FastConformer:
    """transcribe() gets only audio windows; score() is the likelihood side and gets candidate texts."""

    @modal.enter()
    def load(self):
        import nemo.collections.asr as nemo_asr
        import torch
        from huggingface_hub import snapshot_download

        t0 = time.time()
        path = snapshot_download(FC)
        self.rev = Path(path).name
        cache_vol.commit()
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(str(next(Path(path).glob("*.nemo"))), map_location="cuda").eval()
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        tok = self.model.tokenizer
        self.pieces = list(tok.vocab)
        self.charset = set("".join(p for p in self.pieces if p != "<unk>").replace("▁", "")) | {" "}
        self.unk = tok.tokenizer.unk_id()
        self.dev = torch.device("cuda")
        self.load_s = round(time.time() - t0, 1)

    def _meta(self) -> dict:
        import importlib.metadata

        import torch

        return {"id": FC, "revision": self.rev, "implementation": "nemo", "packageVersion": importlib.metadata.version("nemo_toolkit"),
                "torch": str(torch.__version__), "device": torch.cuda.get_device_name(0), "config": FC_CONFIG, "prompt": None,
                "vocabSize": len(self.pieces), "loadSeconds": self.load_s}

    @staticmethod
    def _audio(data: bytes):
        import io

        import soundfile as sf

        a, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 16000 and a.ndim == 1
        return a

    def _decode(self, audios: list, decoder: str) -> list[str]:
        import torch

        self.model.change_decoding_strategy(decoder_type=decoder, verbose=False)
        with torch.no_grad():
            out = self.model.transcribe(audios, batch_size=FC_CONFIG["batchSize"], verbose=False)
        self.model.eval()  # nemo 2.7.3 transcribe() leaves the encoder and CTC head in train mode (dropout on)
        out = out[0] if isinstance(out, tuple) else out
        return [h.text if hasattr(h, "text") else str(h) for h in out]

    def _logprobs(self, audio):
        import torch

        assert not any(m.training for m in self.model.modules())
        with torch.no_grad():
            x = torch.tensor(audio, device=self.dev)[None]
            feats, fl = self.model.preprocessor(input_signal=x, length=torch.tensor([len(audio)], device=self.dev))
            enc, el = self.model.encoder(audio_signal=feats, length=fl)
            lp = self.model.ctc_decoder(encoder_output=enc)
        return lp[0, : int(el[0])].float().log_softmax(-1)

    def _ctc_logp(self, lp, ids: list[int]) -> float | None:
        import torch
        import torch.nn.functional as F

        if not ids:
            return None
        loss = F.ctc_loss(lp[:, None, :], torch.tensor([ids], device=lp.device), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                          blank=lp.shape[1] - 1, reduction="sum")
        return None if not torch.isfinite(loss) else -float(loss)

    def _greedy(self, lp) -> dict:
        blank, best = lp.shape[1] - 1, lp.argmax(-1).tolist()
        ids = [t for k, t in enumerate(best) if t != blank and (k == 0 or t != best[k - 1])]
        return {"text": self.model.tokenizer.ids_to_text(ids), "ids": ids, "logp": self._ctc_logp(lp, ids), "frames": lp.shape[0]}

    def _score_text(self, lp, s: str) -> dict:
        ids = self.model.tokenizer.text_to_ids(s)
        return {"text": s, "ctc": self._ctc_logp(lp, ids), "pieces": len(ids), "unk": ids.count(self.unk)}

    @modal.method()
    def info(self) -> dict:
        return json.loads(json.dumps({**self._meta(), "pieces": self.pieces, "charset": "".join(sorted(self.charset)),
                "decodingCfg": dict(self.model.cfg.decoding)}, default=str))

    @modal.method()
    def transcribe(self, clips: list[dict]) -> dict:
        """clips: [{id, path, startS, endS}] only. Both decoders, greedy, as the model card's two inference modes."""
        t0 = time.time()
        datas = [clip_bytes(c) for c in clips]
        audios = [self._audio(d) for d in datas]
        texts = {dec: self._decode(audios, dec) for dec in DECODERS}
        out = []
        for k, c in enumerate(clips):
            g = self._greedy(self._logprobs(audios[k]))
            out.append({"id": c["id"], "audioSha256": hashlib.sha256(datas[k]).hexdigest(), "audioS": round(len(audios[k]) / 16000, 2),
                        **{dec: texts[dec][k] for dec in DECODERS}, "ctcGreedy": g, "ctcGreedyMatchesDecoder": g["text"].strip() == texts["ctc"][k].strip()})
        return json.loads(json.dumps({**self._meta(), "gpuSeconds": round(time.time() - t0, 1), "results": out}, default=str))

    @modal.method()
    def score(self, jobs: list[dict]) -> dict:
        """jobs: [{id, path, startS, endS, texts: {name: Uthmani text}}]. Per window, the label-blind RNNT transcript and
        CTC greedy text first (used only for spelling), then per text and spelling the CTC log-likelihood.
        diac = core.diacritised_candidates, plain = core.plain_candidates, each against both of those transcripts;
        each reading keeps its best spelling."""
        import review_asr_core as core

        t0 = time.time()
        datas = [clip_bytes(j) for j in jobs]
        audios = [self._audio(d) for d in datas]
        heard = self._decode(audios, "rnnt")
        out = []
        for k, j in enumerate(jobs):
            lp = self._logprobs(audios[k])
            g = self._greedy(lp)
            res = {}
            for name, text in j["texts"].items():
                spell = {"diac": [s for h in (g["text"], heard[k]) for s in core.diacritised_candidates(text, h, self.charset)],
                         "plain": [s for h in (g["text"], heard[k]) for s in core.plain_candidates(text, h)]}
                read = {r: [self._score_text(lp, s) for s in dict.fromkeys(v)] for r, v in spell.items()}
                best = lambda r: max((p["ctc"] for p in read[r] if p["ctc"] is not None), default=None)
                res[name] = {"diac": best("diac"), "plain": best("plain"), "spellings": read}
            out.append({"id": j["id"], "audioSha256": hashlib.sha256(datas[k]).hexdigest(), "transcript": heard[k], "scores": res, "ctcGreedy": g})
        return json.loads(json.dumps({**self._meta(), "gpuSeconds": round(time.time() - t0, 1), "results": out}, default=str))


def _load(test_set: str, queue_rel: str, reviewer: str):
    import review_asr_core as core

    queue = json.loads((DATA / queue_rel).read_text())
    verdicts = json.loads((REVIEWS / test_set / f"{reviewer}.json").read_text())["items"]
    sessions = {}
    for it in queue["items"]:
        sid = core.session_id(it)
        if sid not in sessions and (p := DATA / test_set / "sessions" / f"{sid}.json").is_file():
            sessions[sid] = json.loads(p.read_text())
    return queue, verdicts, sessions, core.controls(queue, sessions)


def _rows(items: list[dict], verdicts: dict, ctrl: dict) -> dict:
    return {it["id"]: {"kind": it["kind"], "label": it["label"]["type"], "human": verdicts.get(it["id"], {}).get("verdict"),
                       "oneEdit": ctrl[it["id"]]["oneEdit"]["op"], "oneEditEdit": ctrl[it["id"]]["oneEdit"]["edit"],
                       "oneEditWords": ctrl[it["id"]]["oneEdit"].get("words", 1)} for it in items}


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=60 * 60)
def calibrate(test_set: str, queue_rel: str, reviewer: str, force: bool) -> dict:
    """Arm 1. Transcripts are reused from asr-fastconformer-rnnt.json when the revision and every audio sha256 match."""
    import review_asr_core as core

    queue, verdicts, sessions, ctrl = _load(test_set, queue_rel, reviewer)
    items = queue["items"]
    shas = {it["id"]: hashlib.sha256(clip_bytes(it["audio"])).hexdigest() for it in items}
    own = REVIEWS / test_set / "asr-fastconformer-rnnt.json"
    res = None
    if own.is_file() and not force:
        doc = json.loads(own.read_text())
        if all(doc["items"].get(i, {}).get("audioSha256") == s for i, s in shas.items()):
            res = {**doc["model"], "results": [{"id": i, **doc["items"][i]["raw"]} for i in shas]}
    res = res or FastConformer().transcribe.remote([{"id": it["id"], **it["audio"]} for it in items])
    by = {r["id"]: r for r in res["results"]}
    meta = {k: v for k, v in res.items() if k != "results"}
    table = _rows(items, verdicts, ctrl)
    judge = lambda text, heard: {k: v for k, v in core.compare(text, heard).items() if k in ("decision", "reasons", "ops")}
    for dec in DECODERS:
        recs = {}
        for it in items:
            r, c = by[it["id"]], ctrl[it["id"]]
            assert r["audioSha256"] == shas[it["id"]]
            recs[it["id"]] = {"audio": it["audio"], "audioSha256": r["audioSha256"], "raw": {k: v for k, v in r.items() if k != "id"},
                              "transcript": r[dec], "text": it["text"], "label": it["label"], "human": table[it["id"]]["human"],
                              "labelled": judge(it["text"], r[dec]),
                              "controls": {k: {**c[k], **judge(c[k]["text"], r[dec])} for k in ("oneEdit", "shifted")}}
            table[it["id"]][dec] = {"transcript": r[dec], "labelled": recs[it["id"]]["labelled"]["decision"],
                                    "oneEdit": recs[it["id"]]["controls"]["oneEdit"]["decision"],
                                    "shifted": recs[it["id"]]["controls"]["shifted"]["decision"],
                                    "reasons": recs[it["id"]]["labelled"]["reasons"], "oneEditReasons": recs[it["id"]]["controls"]["oneEdit"]["reasons"]}
        _write(REVIEWS / test_set / f"asr-fastconformer-{dec}.json",
               {"version": 1, "testSet": test_set, "queue": queue_rel, "createdAt": now(), "model": {**meta, "decoder": dec},
                "rule": {"source": "tools/review_asr_core.py compare", "okSim": core.OK_SIM, "wrongSim": core.WRONG_SIM}, "items": recs})
    for i in table:
        table[i]["combined"] = {k: core.combine([table[i][d][k] for d in DECODERS]) for k in ("labelled", "oneEdit", "shifted")}
    stats = {who: core.arm_stats(table, lambda i, k, w=who: table[i][w][k]) for who in (*DECODERS, "combined")}
    for s in stats.values():
        s["passesBar"] = core.passes_bar(s)
    summary = {"version": 1, "testSet": test_set, "queue": queue_rel, "reviewer": reviewer, "createdAt": now(), "model": meta,
               "humanVerdicts": sum(r["human"] == "ok" for r in table.values()), "items": len(items), "stats": stats, "table": table}
    _write(REVIEWS / test_set / "asr-calibration.fastconformer.json", summary)
    vol.commit()
    return summary


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=60 * 60)
def likelihood(test_set: str, queue_rel: str, reviewer: str) -> dict:
    """Arm 2, only after arm 1 is saved. Pairwise: the claimed text against the alternative (labelled vs one-edit either
    way round; shifted vs labelled), margin core.LIK_MARGIN. Near-best: the claimed text against the CTC greedy path."""
    import review_asr_core as core

    queue, verdicts, sessions, ctrl = _load(test_set, queue_rel, reviewer)
    items = queue["items"]
    arm1 = json.loads((REVIEWS / test_set / "asr-fastconformer-rnnt.json").read_text())
    texts = {it["id"]: {"labelled": it["text"], "oneEdit": ctrl[it["id"]]["oneEdit"]["text"], "shifted": ctrl[it["id"]]["shifted"]["text"]} for it in items}
    res = FastConformer().score.remote([{"id": it["id"], **it["audio"], "texts": texts[it["id"]]} for it in items])
    if res["revision"] != arm1["model"]["revision"]:
        raise RuntimeError(f"scores from revision {res['revision']}, transcripts from {arm1['model']['revision']}")
    by = {r["id"]: r for r in res["results"]}

    def decide(r: dict, read: str, claimed: str) -> dict:
        s = r["scores"]
        alt = "oneEdit" if claimed == "labelled" else "labelled"
        c, a, best = s[claimed][read], s[alt][read], r["ctcGreedy"]["logp"]
        pair = core.prefer(c, a) if None not in (c, a) else "mismatch" if c is None and a is not None else "match" if a is None and c is not None else "unsure"
        return {"pairwise": pair, "nearBest": core.near_best(c, best) if None not in (c, best) else "unsure", "claimed": c, "alternative": a,
                "best": best, "diff": None if None in (c, a) else round(c - a, 2)}

    table, recs = _rows(items, verdicts, ctrl), {}
    for it in items:
        r = by[it["id"]]
        assert r["audioSha256"] == arm1["items"][it["id"]]["audioSha256"]
        for read in ("diac", "plain"):
            table[it["id"]][read] = {k: decide(r, read, k) for k in ("labelled", "oneEdit", "shifted")}
        recs[it["id"]] = {"audio": it["audio"], "audioSha256": r["audioSha256"], "exposed": texts[it["id"]], "transcript": r["transcript"],
                          "transcriptMatchesArm1": r["transcript"] == arm1["items"][it["id"]]["transcript"], "candidates": r["scores"],
                          "ctcGreedy": r["ctcGreedy"], "decisions": {read: table[it["id"]][read] for read in ("diac", "plain")}}
    stats = {}
    for read in ("diac", "plain"):
        for mode in ("pairwise", "nearBest"):
            stats[f"{read}-{mode}"] = core.arm_stats(table, lambda i, k, rd=read, m=mode: table[i][rd][k][m])
            stats[f"{read}-{mode}"]["passesBar"] = core.passes_bar(stats[f"{read}-{mode}"])
    meta = {k: v for k, v in res.items() if k != "results"}
    arm = {"createdAt": now(), "model": meta, "margin": core.LIK_MARGIN, "decision": "diac-pairwise",
           "exposed": "per window: the labelled text, its one-edit control and its shifted control; never the label type",
           "rule": {"pairwise": "claimed - alternative >= margin -> match, <= -margin -> mismatch, else unsure; no CTC path for one side "
                                "decides against it; the labelled text and its one-edit control are each other's alternative; shifted vs labelled",
                    "nearBest": "claimed >= CTC greedy path - margin -> match, else mismatch",
                    "spellings": "diac: core.diacritised_candidates against the window's CTC greedy text and RNNT transcript (harakat kept, Quranic marks dropped, "
                                 "characters outside the tokenizer dropped); plain: core.plain_candidates; max over spellings"}}
    _write(REVIEWS / test_set / "asr-fastconformer-likelihood.json",
           {"version": 1, "testSet": test_set, "queue": queue_rel, **arm, "items": recs})
    summary = {"version": 1, "testSet": test_set, "queue": queue_rel, "reviewer": reviewer, **arm, "items": len(items),
               "edited": sum(r["label"] != "clean" for r in table.values()), "stats": stats, "table": table}
    _write(REVIEWS / test_set / "asr-calibration.fastconformer-likelihood.json", summary)
    vol.commit()
    return summary


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=60 * 60)
def data_check(test_set: str, variants: list[str], context_s: float = 0.0) -> dict:
    """Every planted edit (splices, repeats, restarts): does the CTC head prefer the said text over the original at that spot?
    context_s > 0 (exploratory) widens the window by whole words shared by both texts and writes checks-fastconformer-v1.ctx<s>.json."""
    import review_asr_core as core

    base = DATA / test_set
    ids = [e["id"] for e in json.loads((base / "manifest-v1.json").read_text())["sessions"]]
    sessions = {i: json.loads((base / f"sessions/{i}.json").read_text()) for i in ids}
    teacher_doc = json.loads((base / "checks-v1.json").read_text()) if (base / "checks-v1.json").is_file() else None
    teacher = {(e["id"], v): e for s in (teacher_doc or {}).get("sessions", []) for v, r in s["variants"].items() for e in r["edits"]}
    edits, skipped = [], []
    for i in ids:
        for x in sessions[i]["splices"]:
            e = core.edit_texts(sessions[i], x, context_s)
            if e:
                edits.append(e)
            else:
                skipped.append(f"{i}-{x['id']}")
    jobs = [{"id": f"{e['id']}|{v}", "path": f"{test_set}/sessions/{e['session']}.{v}.wav", "startS": e["startS"], "endS": e["endS"],
             "texts": {"said": e["said"], "original": e["original"]}} for v in variants for e in edits]
    res = FastConformer().score.remote(jobs)
    by = {r["id"]: r for r in res["results"]}
    gap = lambda heard, text: core.lev("".join(core.norm_words(heard)), "".join(core.norm_words(text)))
    out = {v: [] for v in variants}
    for v in variants:
        for e in edits:
            r = by[f"{e['id']}|{v}"]
            s = r["scores"]
            verdict = core.check_verdict(s["said"]["diac"], s["original"]["diac"])
            plain = core.check_verdict(s["said"]["plain"], s["original"]["plain"])
            t = teacher.get((e["id"], v))
            out[v].append({**e, **verdict, "plainPass": plain["pass"], "plainLlr": plain["llr"], "positive": verdict["llr"] is not None and verdict["llr"] > 0,
                           "transcript": r["transcript"], "transcriptCloser": gap(r["transcript"], e["said"]) < gap(r["transcript"], e["original"]),
                           "audioSha256": r["audioSha256"], "scores": s, "ctcGreedy": r["ctcGreedy"]["text"],
                           "teacher": t and {"pass": t["pass"], "llr": t.get("llr"), "reasons": t.get("reasons")}})
    summary = {}
    for v in variants:
        rs = out[v]
        both = [r for r in rs if r["teacher"]]
        summary[v] = {"decision": core.check_summary(rs), "plain": core.check_summary(rs, "plainPass"), "positive": core.check_summary(rs, "positive"),
                      "transcriptCloser": core.check_summary(rs, "transcriptCloser"),
                      "vsTeacher": {"bothPass": sum(r["pass"] and r["teacher"]["pass"] for r in both),
                                    "onlyTeacherPasses": sum(not r["pass"] and r["teacher"]["pass"] for r in both),
                                    "onlyFastConformerPasses": sum(r["pass"] and not r["teacher"]["pass"] for r in both),
                                    "bothFail": sum(not r["pass"] and not r["teacher"]["pass"] for r in both), "n": len(both)}}
    doc = {"version": 1, "kind": "lab-checks-independent", "testSet": test_set, "checkedAt": now(), "model": {k: v for k, v in res.items() if k != "results"},
           "teacherChecks": teacher_doc and {"path": f"{test_set}/checks-v1.json", "checkedAt": teacher_doc["checkedAt"], "teacher": teacher_doc["teacher"]},
           "rule": {"window": "word before to word after the edit, labelled times from the session JSON, padded "
                              f"{core.CHECK_PAD_S} s (tools/review_asr_core.py edit_texts)",
                    "pass": f"CTC log-likelihood of the said text minus the original >= {core.CHECK_LLR_MIN} nats, diacritised candidates against "
                            "the window's own label-blind CTC greedy text and RNNT transcript, max over spellings; an original with no CTC path counts as infinitely worse",
                    "secondary": "plain (undiacritised) candidates at the same margin; llr > 0; the RNNT transcript closer to the said text than "
                                 "to the original (letter edit distance after normalisation)"},
           "contextS": context_s, "skipped": skipped, "summary": summary,
           "failed": {v: [{k: r[k] for k in ("id", "class", "said", "original", "llr", "reason", "plainLlr", "transcript", "teacher")}
                          for r in out[v] if not r["pass"]] for v in variants},
           "edits": out}
    _write(base / f"checks-fastconformer-v1{f'.ctx{context_s:g}' if context_s else ''}.json", doc)
    vol.commit()
    return {k: doc[k] for k in ("model", "skipped", "summary", "failed")}


def _print_stats(stats: dict) -> None:
    for who, st in stats.items():
        a, o, w, sh = st["agreement"], st["oneEdit"], st["oneWord"], st["shifted"]
        print(f"{who:18} agree {a['k']}/{a['n']} {a['ci95']}  one-edit flagged {o['flagged']['k']}/{o['flagged']['n']} {o['flagged']['ci95']}  "
              f"one-word flagged {w['flagged']['k']}/{w['flagged']['n']} {w['flagged']['ci95']}  shifted {sh['flagged']['k']}/{sh['flagged']['n']}  "
              f"repeat misses {st['repeatMisses']}  bar {'PASS' if st['passesBar'] else 'fail'}")


@app.local_entrypoint()
def smoke(path: str = "lab-v1.1/sessions/lab-v1.1-e20.studio.wav", start: float = 0.0, end: float = 8.0):
    """One window outside any review queue: versions, vocabulary, both decoders and a CTC score."""
    fc = FastConformer()
    info = fc.info.remote()
    print(json.dumps({k: v for k, v in info.items() if k != "pieces"}, ensure_ascii=False)[:3000])
    print("pieces:", info["pieces"][:80])
    r = fc.transcribe.remote([{"id": "smoke", "path": path, "startS": start, "endS": end}])
    print(json.dumps(r["results"][0], ensure_ascii=False)[:1500])
    x = r["results"][0]
    s = fc.score.remote([{"id": "smoke", "path": path, "startS": start, "endS": end, "texts": {"own": x["rnnt"], "less": " ".join(x["rnnt"].split()[1:])}}])
    print(json.dumps(s["results"][0]["scores"], ensure_ascii=False)[:2500])


@app.local_entrypoint()
def main(test_set: str = "lab-v1.1", queue: str = "lab-v1.1/review/v1-check.json", reviewer: str = "me", force: bool = False,
         out: str = "docs/benchmarks/lab-review-asr-v1-fastconformer.json"):
    t0 = time.time()
    s = calibrate.remote(test_set, queue, reviewer, force)
    (ROOT / out).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    _print_stats(s["stats"])
    print(f"wrote {out} and /lab/reviews/{test_set}/asr-fastconformer-*.json; gpu {s['model'].get('gpuSeconds')} s, wall {time.time() - t0:.0f} s")


@app.local_entrypoint()
def likelihood_main(test_set: str = "lab-v1.1", queue: str = "lab-v1.1/review/v1-check.json", reviewer: str = "me",
                    out: str = "docs/benchmarks/lab-review-asr-v1-fastconformer-likelihood.json"):
    t0 = time.time()
    s = likelihood.remote(test_set, queue, reviewer)
    (ROOT / out).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    _print_stats(s["stats"])
    print(f"wrote {out}; gpu {s['model'].get('gpuSeconds')} s, wall {time.time() - t0:.0f} s")


@app.local_entrypoint()
def check_main(test_set: str = "lab-v1.1", variants: str = "studio,phone", context: float = 0.0, out: str = ""):
    t0 = time.time()
    r = data_check.remote(test_set, variants.split(","), context)
    out = out or f"docs/benchmarks/lab-checks-fastconformer-v1{f'-ctx{context:g}' if context else ''}.json"
    (ROOT / out).write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(r["summary"], indent=1))
    for v, fs in r["failed"].items():
        print(f"{v}: {len(fs)} failed")
        for f in fs:
            print(f"  {f['id']} {f['class']}: llr {f['llr']} (plain {f['plainLlr']}) teacher {f['teacher'] and f['teacher']['pass']} | {f['transcript']}")
    print(f"skipped {r['skipped']}; gpu {r['model'].get('gpuSeconds')} s, wall {time.time() - t0:.0f} s")
