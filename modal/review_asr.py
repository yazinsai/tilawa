"""Independent audio opinion for lab review queues (.scratch/lab/issues/12-review-asr-calibration.md, docs/agents/audio-review.md).

    modal run modal/review_asr.py::main                                              # calibrate on lab-v1.1 v1-check + me.json
    modal run modal/review_asr.py::main --test-set lab-v1.1 --queue lab-v1.1/review/v1-check.json --reviewer me --force
    modal run modal/review_asr.py::main --models nemotron-3.5-transcribe --name nemotron-3.5-transcribe [--pad 1]
    modal run modal/review_asr.py::likelihood_main                                   # Nemotron likelihood arm, after its transcripts
    modal run modal/review_asr.py::prescreen --queue lab-v1.2/review/v1-check.json   # the subset the listener must hear

Literal ASR models, none related to the teacher (obadx/muaalem-model-v3_2) or the student: Qwen/Qwen3-ASR-1.7B and
Whisper large-v3 (faster-whisper, Systran/faster-whisper-large-v3) on an L4, and NVIDIA Nemotron 3.5 Transcribe
(30B-A4B, Transformers, bf16) on an H100 with a fixed instruction (NEMOTRON_PROMPTS). The transcribers get only the item
id and audio window, never the text or label. tools/review_asr_core.py then compares each transcript with
the labelled text (match / mismatch / unsure) and builds the negative controls: the same audio with its planted edit undone
in the text, a one-word change planted in the text of clean windows, and the text of another item.

Per model, /lab/reviews/<testSet>/asr-<model>.json keeps the model id and revision, config, and per item the audio
sha256, raw output, transcript and every comparison. Transcripts are reused by audio sha256 and model revision unless
--force. Calibration writes /lab/reviews/<testSet>/asr-calibration[.<name>].json and docs/benchmarks/lab-review-asr-v1[-<name>].json.
The likelihood arm (Nemotron scores given candidate texts, so it is not label-blind) adds its scores to
asr-nemotron-3.5-transcribe.json. Pre-screen (default: the pairwise likelihood mode, the only one that passed) writes
/lab/data/<testSet>/review/<queue>.prescreen.json in the ticket 06 queue format.
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
GPUS = ["L4", "A10", "L40S"]
QWEN, WHISPER = "Qwen/Qwen3-ASR-1.7B", "Systran/faster-whisper-large-v3"
WHISPER_CONFIG = {"language": "ar", "beam_size": 5, "condition_on_previous_text": False, "vad_filter": False, "word_timestamps": True}
QWEN_CONFIG = {"language": "Arabic", "context": None, "dtype": "bfloat16", "max_new_tokens": 512}
NEMOTRON, TRANSFORMERS_REV = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", "144ecf007b8b4f2910da104c84a79801db6985a6"
NEMOTRON_PROMPTS = {
    # the model card's verbatim ASR prompt with the language named; the decision prompt
    "verbatim": "Audio transcription. Language: Arabic. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.",
    # exploratory
    "recitation": "Transcribe this Quran recitation verbatim, exactly as recited, including repetitions and mistakes. Language: Arabic. Output: transcript only.",
    # round 2 ablation, fixed before any of them ran: a 2 x 2 of the word "Quran" and the clause "including repetitions and mistakes"
    "mistakes": "Transcribe this recitation verbatim, exactly as recited, including repetitions and mistakes. Language: Arabic. Output: transcript only.",
    "quran": "Transcribe this Quran recitation verbatim, exactly as recited. Language: Arabic. Output: transcript only.",
    "plain": "Transcribe this recitation verbatim, exactly as recited. Language: Arabic. Output: transcript only.",
}
ROVER_PERTURB = [(False, 1.0), (False, 0.9), (False, 1.1), (True, 1.0), (True, 0.9), (True, 1.1)]  # (denoised, speed); unperturbed first
ROVER_PROMPTS = ["verbatim", "recitation", "mistakes"]  # the A/B prompts; the best one by core.pick_prompt is perturbed
NEMOTRON_CONFIG = {"engine": "transformers", "transformersRevision": TRANSFORMERS_REV, "dtype": "bfloat16", "decoding": "greedy",
                   "enableThinking": False, "systemPrompt": None, "maxNewTokens": 512, "loopGuardrail": "model card cleanup_generation_loops"}

base = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").env({"HF_HOME": f"{CACHE}/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
qwen_image = base.pip_install("qwen-asr==0.0.6", "soundfile").add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
whisper_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.2.1", "soundfile")
    .env({"HF_HOME": f"{CACHE}/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
)
nemotron_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch", "torchaudio", "accelerate", "soundfile", "scipy", "librosa", f"transformers @ git+https://github.com/huggingface/transformers@{TRANSFORMERS_REV}")
    .env({"HF_HOME": f"{CACHE}/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_file(ROOT / "tools/review_asr_core.py", "/root/review_asr_core.py")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
    .add_local_file(ROOT / "tools/review_asr_core.py", "/root/review_asr_core.py")
)
denoise_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.1.2", "torchaudio==2.1.2", "numpy<2", "soundfile", "deepfilternet==0.5.6")
    .add_local_file(ROOT / "tools/lab_review_core.py", "/root/lab_review_core.py")
)
app = modal.App("tilawa-review-asr")
vol = modal.Volume.from_name("tilawa-lab")
cache_vol = modal.Volume.from_name("tilawa-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip_bytes(clip: dict) -> bytes:
    """The review window as a 16 kHz mono PCM16 WAV, exactly the bytes the models hear (and whose sha256 is saved)."""
    import io
    import subprocess
    import wave

    import lab_review_core as lrc

    p = DATA / lrc.safe_rel(clip["path"])
    if p.suffix.lower() == ".wav":
        with open(p, "rb") as f:
            b = lrc.wav_slice(f, clip["startS"], clip["endS"])
        with wave.open(io.BytesIO(b)) as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2):
                return b
    a, e = clip["startS"], clip["endS"]
    cmd = ["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(e - a), "-i", str(p), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "-"]
    return lrc.wav_slice(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout))


def _ffmpeg(data: bytes, *args: str, rate: int = 16000) -> bytes:
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-i", "-", *args, "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-f", "wav", "-"]
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def audio_bytes(clip: dict) -> bytes:
    """clip_bytes, optionally perturbed (round 2 ROVER): clip['denoised'] is the DeepFilterNet copy of the same window
    (written by denoise()), clip['speed'] a pitch-preserving ffmpeg atempo change. Without either it is clip_bytes."""
    import io

    import lab_review_core as lrc

    data = (REVIEWS / clip["denoised"]).read_bytes() if clip.get("denoised") else clip_bytes(clip)
    if clip.get("speed", 1.0) != 1.0:
        data = lrc.wav_slice(io.BytesIO(_ffmpeg(data, "-filter:a", f"atempo={clip['speed']}")))
    return data


def plain(x):
    """JSON-safe copy (numpy scalars and arrays from the models become floats and lists)."""
    return json.loads(json.dumps(x, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


def _snapshot(repo: str) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo)
    return path, Path(path).name


@app.function(image=base.pip_install("huggingface_hub[hf_xet]"), volumes={CACHE: cache_vol}, secrets=[hf_secret], timeout=2 * 60 * 60, cpu=4, memory=8192)
def fetch(repo: str) -> str:
    """Download a large checkpoint into the HF cache volume on CPU, so the GPU container only loads it."""
    path, rev = _snapshot(repo)
    cache_vol.commit()
    return f"{repo}@{rev} {sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) / 1e9:.1f} GB"


@app.function(image=qwen_image, gpu=GPUS, volumes={VOL: vol, CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 60, max_containers=1)
def qwen(clips: list[dict]) -> dict:
    """clips: [{id, path, startS, endS}] only."""
    import dataclasses
    import importlib.metadata
    import tempfile

    import torch
    from qwen_asr import Qwen3ASRModel

    path, rev = _snapshot(QWEN)
    model = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=1,
                                          max_new_tokens=QWEN_CONFIG["max_new_tokens"])
    out = []
    for c in clips:
        data = clip_bytes(c)
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(data)
            f.flush()
            r = model.transcribe(audio=f.name, language=QWEN_CONFIG["language"])[0]
        raw = dataclasses.asdict(r) if dataclasses.is_dataclass(r) else {k: v for k, v in vars(r).items() if isinstance(v, (str, int, float, list, dict, type(None)))}
        out.append({"id": c["id"], "audioSha256": hashlib.sha256(data).hexdigest(), "transcript": r.text, "raw": raw, "seconds": round(time.time() - t0, 2)})
    cache_vol.commit()
    return plain({"id": QWEN, "revision": rev, "implementation": "qwen-asr", "packageVersion": importlib.metadata.version("qwen-asr"),
            "device": torch.cuda.get_device_name(0), "config": QWEN_CONFIG, "prompt": None, "results": out})


@app.function(image=whisper_image, gpu=GPUS, volumes={VOL: vol, CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 60, max_containers=1)
def whisper(clips: list[dict]) -> dict:
    """clips: [{id, path, startS, endS}] only."""
    import importlib.metadata
    import io

    import ctranslate2
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    path, rev = _snapshot(WHISPER)
    model = WhisperModel(path, device="cuda", compute_type="float16")
    out = []
    for c in clips:
        data = clip_bytes(c)
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        t0 = time.time()
        segments, info = model.transcribe(np.asarray(audio), **WHISPER_CONFIG)
        segs = [{"start": s.start, "end": s.end, "text": s.text, "avgLogprob": s.avg_logprob, "noSpeechProb": s.no_speech_prob,
                 "compressionRatio": s.compression_ratio, "temperature": s.temperature,
                 "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in s.words or []]}
                for s in segments]
        out.append({"id": c["id"], "audioSha256": hashlib.sha256(data).hexdigest(), "transcript": " ".join(s["text"].strip() for s in segs),
                    "raw": {"language": info.language, "languageProbability": info.language_probability, "segments": segs},
                    "seconds": round(time.time() - t0, 2)})
    cache_vol.commit()
    return plain({"id": WHISPER, "revision": rev, "implementation": "faster-whisper", "packageVersion": importlib.metadata.version("faster-whisper"),
            "ctranslate2": ctranslate2.__version__, "device": "cuda float16", "config": WHISPER_CONFIG, "prompt": None, "results": out})


@app.cls(image=nemotron_image, gpu=["H100", "A100-80GB"], volumes={VOL: vol, CACHE: cache_vol}, secrets=[hf_secret], timeout=60 * 60,
         scaledown_window=10 * 60, max_containers=1, memory=96 * 1024)
class Nemotron:
    """nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B in-process with Transformers (the model card's in-process path), bf16.
    transcribe() gets only the audio windows and a fixed instruction; score() is the likelihood arm and gets candidate texts."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        t0 = time.time()
        self.path, self.rev = _snapshot(NEMOTRON)
        self.processor = AutoProcessor.from_pretrained(self.path)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.path, dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.load_s = round(time.time() - t0, 1)

    def _meta(self, prompt: str) -> dict:
        import importlib.metadata

        import torch

        return {"id": NEMOTRON, "revision": self.rev, "implementation": "transformers", "packageVersion": importlib.metadata.version("transformers"),
                "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "config": NEMOTRON_CONFIG, "promptKey": prompt,
                "prompt": NEMOTRON_PROMPTS[prompt], "loadSeconds": self.load_s}

    def _inputs(self, data: bytes, prompt: str):
        import io

        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        return self.processor.apply_transcription_request(audio, prompt=NEMOTRON_PROMPTS[prompt]).to(self.model.device)

    def _ctc_frames(self, logits, mask):
        """Log-probabilities of the CTC alignment head over the valid frames (the last class is the blank)."""
        n_enc = int(self.processor._get_audio_token_length(mask[0].sum()))
        up = logits.shape[1] // int(self.processor._get_audio_token_length(mask.shape[-1]))
        return logits[0, : n_enc * up].float().log_softmax(-1)

    def _ctc_logp(self, lp, ids: list[int]) -> float | None:
        import torch
        import torch.nn.functional as F

        if not ids:
            return None
        loss = F.ctc_loss(lp[:, None, :], torch.tensor([ids], device=lp.device), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                          blank=lp.shape[1] - 1, reduction="sum")
        return None if not torch.isfinite(loss) else -float(loss)

    def _ctc_greedy(self, lp) -> dict:
        blank, best = lp.shape[1] - 1, lp.argmax(-1).tolist()
        ids = [t for k, t in enumerate(best) if t != blank and (k == 0 or t != best[k - 1])]
        return {"text": self.processor.alignment_tokenizer.decode(ids), "ids": ids, "logp": self._ctc_logp(lp, ids), "frames": lp.shape[0]}

    @modal.method()
    def transcribe(self, clips: list[dict], prompt: str = "verbatim") -> dict:
        """clips: [{id, path, startS, endS}] only. Greedy, reasoning off, as the model card's transcription preset."""
        import torch

        import review_asr_core as core

        if any(c.get("denoised") for c in clips):
            vol.reload()
        out = []
        for c in clips:
            data = audio_bytes(c)
            t0 = time.time()
            inputs = self._inputs(data, prompt)
            n = inputs["input_ids"].shape[1]
            with torch.no_grad():
                g = self.model.generate(**inputs, max_new_tokens=NEMOTRON_CONFIG["maxNewTokens"], do_sample=False, return_word_timestamp_logits=True)
            ids = g.sequences[0, n:].tolist()
            text = self.processor.decode(ids, skip_special_tokens=True).strip()
            ctc = self._ctc_greedy(self._ctc_frames(g.word_timestamp_logits, inputs["input_features_mask"]))
            out.append({"id": c["id"], "audioSha256": hashlib.sha256(data).hexdigest(), "transcript": core.cleanup_generation_loops(text),
                        "raw": {"text": self.processor.decode(ids, skip_special_tokens=False), "tokenIds": ids, "beforeLoopGuardrail": text, "ctcGreedy": ctc},
                        "seconds": round(time.time() - t0, 2)})
        return plain({**self._meta(prompt), "results": out})

    @modal.method()
    def score(self, items: list[dict], prompt: str = "verbatim") -> dict:
        """items: [{id, path, startS, endS, candidates: {name: [spellings]}}]. Per candidate and spelling: the language model's
        log-likelihood of the text as the reply to the transcription request (teacher-forced, through <|im_end|>), and
        the CTC alignment head's log-likelihood of its alignment-tokenizer pieces (audio encoder only, no text prior)."""
        import torch

        tok, atok = self.processor.tokenizer, self.processor.alignment_tokenizer
        end = tok.convert_tokens_to_ids("<|im_end|>")
        out = []
        for it in items:
            data = clip_bytes(it)
            t0 = time.time()
            inputs = self._inputs(data, prompt)
            n = inputs["input_ids"].shape[1]
            with torch.no_grad():
                audio = self.model.get_audio_features(inputs["input_features"], inputs["input_features_mask"])
                lp = self._ctc_frames(self.model.get_word_timestamp_logits(audio.asr_last_hidden_state, inputs["input_features_mask"]),
                                      inputs["input_features_mask"])
                res = {}
                for name, spellings in it["candidates"].items():
                    per = []
                    for s in spellings:
                        cand = tok.encode(s, add_special_tokens=False) + [end]
                        ids = torch.cat([inputs["input_ids"], torch.tensor([cand], device=self.model.device)], 1)
                        logits = self.model(input_ids=ids, input_features=inputs["input_features"], input_features_mask=inputs["input_features_mask"],
                                            attention_mask=torch.ones_like(ids), use_cache=False).logits
                        t = logits[0, n - 1:-1].float().log_softmax(-1).gather(1, torch.tensor(cand, device=logits.device)[:, None])[:, 0]
                        pieces = [p for w in s.split() for p in atok.encode(" " + w, add_special_tokens=False)]
                        per.append({"text": s, "llm": float(t.sum()), "llmTokens": len(cand), "llmMinToken": float(t.min()),
                                    "ctc": self._ctc_logp(lp, pieces), "ctcPieces": len(pieces), "ctcUnk": pieces.count(atok.unk_token_id)})
                    best = lambda k: max((p[k] for p in per if p[k] is not None), default=None)
                    res[name] = {"llm": best("llm"), "ctc": best("ctc"), "spellings": per}
            out.append({"id": it["id"], "audioSha256": hashlib.sha256(data).hexdigest(), "scores": res, "ctcGreedy": self._ctc_greedy(lp),
                        "seconds": round(time.time() - t0, 2)})
        return plain({**self._meta(prompt), "results": out})


MODELS = {"qwen3-asr-1.7b": (qwen, {}), "whisper-large-v3": (whisper, {}),
          "nemotron-3.5-transcribe": (Nemotron().transcribe, {"prompt": "verbatim"}),
          "nemotron-recitation-prompt": (Nemotron().transcribe, {"prompt": "recitation"}),
          **{f"nemotron-{p}-prompt": (Nemotron().transcribe, {"prompt": p}) for p in ("mistakes", "quran", "plain")}}
PROMPT_RECORDS = {"verbatim": "nemotron-3.5-transcribe", "recitation": "nemotron-recitation-prompt",
                  **{p: f"nemotron-{p}-prompt" for p in ("mistakes", "quran", "plain")}}


@app.function(image=denoise_image, volumes={VOL: vol}, timeout=60 * 60, cpu=8, memory=16 * 1024)
def denoise(test_set: str, clips: list[dict]) -> dict:
    """DeepFilterNet3 speech enhancement of each review window (16 kHz -> 48 kHz -> enhanced -> 16 kHz PCM16), written to
    /lab/reviews/<testSet>/rover/denoised/<id>.wav for audio_bytes. clips: [{id, path, startS, endS}]."""
    import importlib.metadata
    import io

    import lab_review_core as lrc
    import soundfile as sf
    import torch
    from df.enhance import enhance, init_df

    model, state = init_df()[:2]
    out = {}
    for c in clips:
        data = clip_bytes(c)
        t0 = time.time()
        x, sr = sf.read(io.BytesIO(_ffmpeg(data, rate=48000)), dtype="float32")
        assert sr == state.sr() and x.ndim == 1
        y = enhance(model, state, torch.from_numpy(x)[None]).squeeze(0).numpy()
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="FLOAT")
        clean = lrc.wav_slice(io.BytesIO(_ffmpeg(buf.getvalue())))
        rel = f"{test_set}/rover/denoised/{c['id']}.wav"
        (REVIEWS / rel).parent.mkdir(parents=True, exist_ok=True)
        (REVIEWS / rel).write_bytes(clean)
        out[c["id"]] = {"denoised": rel, "inSha256": hashlib.sha256(data).hexdigest(), "sha256": hashlib.sha256(clean).hexdigest(),
                        "seconds": round(time.time() - t0, 2)}
    vol.commit()
    return {"model": "DeepFilterNet3 (init_df default)", "package": importlib.metadata.version("deepfilternet"), "torch": str(torch.__version__),
            "items": out}


def _transcribe(test_set: str, items: list[dict], models: list[str], force: bool, record: str) -> dict[str, dict]:
    """Per model: {meta, results by item id}. Unless force, reuses any transcript of the same audio (sha256) saved for
    this test set by the same model, preferring `record`'s own file and its revision."""
    shas = {it["id"]: hashlib.sha256(clip_bytes(it["audio"])).hexdigest() for it in items}
    calls = {}
    for name in models:
        own = REVIEWS / test_set / record.format(model=name)
        files = [] if force else sorted((REVIEWS / test_set).glob(f"asr-{name}*.json"), key=lambda p: p != own)
        meta, by_sha = None, {}
        for p in files:
            doc = json.loads(p.read_text())
            meta = meta or doc.get("model")
            if all(doc.get("model", {}).get(k) == meta.get(k) for k in ("revision", "prompt")):
                for r in doc.get("items", {}).values():
                    if "raw" in r:
                        by_sha.setdefault(r["audioSha256"], r)
        have = {i: {**by_sha[s], "audioSha256": s} for i, s in shas.items() if s in by_sha}
        need = [{"id": it["id"], **it["audio"]} for it in items if it["id"] not in have]
        fn, kw = MODELS[name]
        calls[name] = (meta, have, fn.spawn(need, **kw) if need else None)
    out = {}
    for name, (meta, have, call) in calls.items():
        res = call.get() if call else None
        if res:
            if meta and meta.get("revision") != res["revision"]:
                raise RuntimeError(f"{name}: saved transcripts are from revision {meta.get('revision')}, now {res['revision']}; rerun with --force")
            meta = {k: v for k, v in res.items() if k != "results"}
            have = {**have, **{r["id"]: {k: v for k, v in r.items() if k != "id"} for r in res["results"]}}
        out[name] = {"model": meta, "items": have}
    return out


def _write(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    tmp.replace(p)


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=2 * 60 * 60)
def calibrate(test_set: str, queue_rel: str, reviewer: str, models: list[str], force: bool, pad: float = 0.0, label: str = "") -> dict:
    """pad > 0 (exploratory): the models hear the window widened by `pad` seconds each side, and the session's words in
    the widened part are given to the comparison as context that never counts."""
    import review_asr_core as core

    queue = json.loads((DATA / queue_rel).read_text())
    verdicts = json.loads((REVIEWS / test_set / f"{reviewer}.json").read_text())["items"]
    items = queue["items"]
    sessions = {}
    for it in items:
        sid = core.session_id(it)
        if sid not in sessions and (p := DATA / test_set / "sessions" / f"{sid}.json").is_file():
            sessions[sid] = json.loads(p.read_text())
    ctrl = core.controls(queue, sessions)
    tag = f".pad{pad:g}" if pad else ""
    ctx, heard = {}, []
    for it in items:
        s, a, b = sessions.get(core.session_id(it)), it["audio"]["startS"], it["audio"]["endS"]
        if pad and s:
            lo, hi = max(0.0, a - pad), min(s["durationS"], b + pad)
            ctx[it["id"]] = (" ".join(w["text"] for w in s["said"] if lo < w["endS"] <= a), " ".join(w["text"] for w in s["said"] if b <= w["startS"] < hi))
            heard.append({**it, "audio": {**it["audio"], "startS": round(lo, 3), "endS": round(hi, 3)}})
        else:
            ctx[it["id"]] = ("", "")
            heard.append(it)
    asr = _transcribe(test_set, heard, models, force, "asr-{model}" + tag + ".json")
    heard = {it["id"]: it["audio"] for it in heard}
    cur = {}

    def judge(text: str, transcript: str) -> dict:
        r = core.compare(text, transcript, *ctx[cur["id"]])
        return {"decision": r["decision"], "reasons": r["reasons"], "ops": r["ops"]}

    per, table = {}, {it["id"]: {"kind": it["kind"], "label": it["label"]["type"], "human": verdicts.get(it["id"], {}).get("verdict"),
                                 "humanNote": verdicts.get(it["id"], {}).get("note", ""), "oneEdit": ctrl[it["id"]]["oneEdit"]["op"],
                                 "oneEditEdit": ctrl[it["id"]]["oneEdit"]["edit"], "oneEditWords": ctrl[it["id"]]["oneEdit"].get("words", 1)} for it in items}
    for name in models:
        recs = {}
        for it in items:
            cur["id"] = it["id"]
            r = asr[name]["items"][it["id"]]
            c = ctrl[it["id"]]
            recs[it["id"]] = {"audio": heard[it["id"]], "window": it["audio"], "context": ctx[it["id"]], "audioSha256": r["audioSha256"], "raw": r["raw"], "transcript": r["transcript"],
                              "seconds": r.get("seconds"), "text": it["text"], "label": it["label"],
                              "human": verdicts.get(it["id"], {}).get("verdict"),
                              "labelled": judge(it["text"], r["transcript"]),
                              "controls": {k: {**c[k], **judge(c[k]["text"], r["transcript"])} for k in ("oneEdit", "shifted")}}
            table[it["id"]][name] = {"transcript": r["transcript"], "labelled": recs[it["id"]]["labelled"]["decision"],
                                     "oneEdit": recs[it["id"]]["controls"]["oneEdit"]["decision"], "shifted": recs[it["id"]]["controls"]["shifted"]["decision"],
                                     "reasons": recs[it["id"]]["labelled"]["reasons"], "oneEditReasons": recs[it["id"]]["controls"]["oneEdit"]["reasons"]}
        per[name] = recs
        _write(REVIEWS / test_set / f"asr-{name}{tag}.json", {"version": 1, "testSet": test_set, "queue": queue_rel, "createdAt": now(),
                                                            "model": asr[name]["model"], "rule": rule(core), "padS": pad, "items": recs})
    for it in items:
        t = table[it["id"]]
        t["combined"] = {k: core.combine([t[m][k] for m in models]) for k in ("labelled", "oneEdit", "shifted")}
        full = " ".join(x for x in (ctx[it["id"]][0], it["text"], ctx[it["id"]][1]) if x)
        t["oracle"] = {k: core.compare(v, full, *ctx[it["id"]])["decision"]
                       for k, v in (("labelled", it["text"]), ("oneEdit", ctrl[it["id"]]["oneEdit"]["text"]), ("shifted", ctrl[it["id"]]["shifted"]["text"]))}

    ok = [i for i in table if table[i]["human"] == "ok"]
    single = [i for i in table if table[i]["oneEditWords"] == 1 and table[i]["oneEditEdit"] not in ("repeat", "restart", "skipped_passage")]
    stats = {}
    for who in [*models, "combined", "oracle"]:
        get = (lambda i, k: table[i][who][k]) if who in ("combined", "oracle") else (lambda i, k: table[i][who][k])
        dec = {**{f"{i}|pos": get(i, "labelled") for i in ok}, **{f"{i}|one": get(i, "oneEdit") for i in table}}
        truth = {**{f"{i}|pos": True for i in ok}, **{f"{i}|one": False for i in table}}
        s = core.score(dec, truth)
        sh = core.score({f"{i}|sh": get(i, "shifted") for i in table}, {f"{i}|sh": False for i in table})
        one_word = core.score({f"{i}|one": get(i, "oneEdit") for i in single}, {f"{i}|one": False for i in single})
        stats[who] = {"agreement": s["specificity"], "disagreement": s["falseAlarm"], "positives": s["positives"],
                      "oneEdit": {"sensitivity": s["sensitivity"], "flagged": s["flagged"], "counts": s["negatives"]},
                      "oneWord": {"sensitivity": one_word["sensitivity"], "flagged": one_word["flagged"], "counts": one_word["negatives"]},
                      "shifted": {"sensitivity": sh["sensitivity"], "flagged": sh["flagged"], "counts": sh["negatives"]}}
    summary = {"version": 1, "testSet": test_set, "queue": queue_rel, "reviewer": reviewer, "createdAt": now(), "padS": pad,
               "models": {m: asr[m]["model"] for m in models}, "rule": rule(core), "humanVerdicts": len(ok), "items": len(items),
               "stats": stats, "table": table}
    _write(REVIEWS / test_set / f"asr-calibration{'.' + label if label else ''}{tag}.json", summary)
    vol.commit()
    return summary


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=2 * 60 * 60)
def likelihood(test_set: str, queue_rel: str, reviewer: str, record: str = "nemotron-3.5-transcribe") -> dict:
    """The likelihood arm, run only after the label-blind transcripts are saved in asr-<record>.json. Nemotron scores,
    per window, the labelled text, its one-edit and shifted controls, and its own transcript (all respelled by
    core.plain_candidates). Pairwise: the claimed text against the alternative (labelled vs one-edit, either way round;
    shifted vs labelled). Near-best: the claimed text against the model's own best hypothesis (its transcript for the
    language model, its greedy CTC path for the CTC head). Writes the scores into asr-<record>.json and a summary to
    asr-calibration.<record>-likelihood.json."""
    import review_asr_core as core

    queue = json.loads((DATA / queue_rel).read_text())
    verdicts = json.loads((REVIEWS / test_set / f"{reviewer}.json").read_text())["items"]
    items = queue["items"]
    sessions = {}
    for it in items:
        sid = core.session_id(it)
        if sid not in sessions and (p := DATA / test_set / "sessions" / f"{sid}.json").is_file():
            sessions[sid] = json.loads(p.read_text())
    ctrl = core.controls(queue, sessions)
    rec_path = REVIEWS / test_set / f"asr-{record}.json"
    rec = json.loads(rec_path.read_text())
    texts = {it["id"]: {"labelled": it["text"], "oneEdit": ctrl[it["id"]]["oneEdit"]["text"], "shifted": ctrl[it["id"]]["shifted"]["text"]} for it in items}
    jobs = []
    for it in items:
        heard = rec["items"][it["id"]]["transcript"]
        cands = {k: core.plain_candidates(v, heard) for k, v in texts[it["id"]].items()}
        cands["own"] = [core.plain_transcript(heard)] if core.plain_transcript(heard) else []
        jobs.append({"id": it["id"], **it["audio"], "candidates": {k: v for k, v in cands.items() if v}})
    res = Nemotron().score.remote(jobs, rec["model"]["promptKey"])
    if res["revision"] != rec["model"]["revision"]:
        raise RuntimeError(f"scores from revision {res['revision']}, transcripts from {rec['model']['revision']}")
    by_id = {r["id"]: r for r in res["results"]}

    def decide(r: dict, src: str, claimed: str) -> dict:
        s = r["scores"]
        val = lambda k: s.get(k, {}).get(src)
        alt = "oneEdit" if claimed == "labelled" else "labelled"
        best = (s.get("own", {}).get("llm") if src == "llm" else r["ctcGreedy"]["logp"])
        pair = core.prefer(val(claimed), val(alt)) if None not in (val(claimed), val(alt)) else "unsure"
        near = core.near_best(val(claimed), best) if None not in (val(claimed), best) else "unsure"
        return {"pairwise": pair, "nearBest": near, "claimed": val(claimed), "alternative": val(alt), "best": best}

    table, stats = {}, {}
    for it in items:
        r = by_id[it["id"]]
        assert r["audioSha256"] == rec["items"][it["id"]]["audioSha256"]
        table[it["id"]] = {"kind": it["kind"], "label": it["label"]["type"], "human": verdicts.get(it["id"], {}).get("verdict"),
                           "oneEditEdit": ctrl[it["id"]]["oneEdit"]["edit"], "oneEditWords": ctrl[it["id"]]["oneEdit"].get("words", 1),
                           **{src: {k: decide(r, src, k) for k in ("labelled", "oneEdit", "shifted")} for src in ("llm", "ctc")}}
        rec["items"][it["id"]]["likelihood"] = {"exposed": texts[it["id"]], "candidates": r["scores"], "ctcGreedy": r["ctcGreedy"],
                                                "seconds": r["seconds"], "decisions": {src: table[it["id"]][src] for src in ("llm", "ctc")}}
    ok = [i for i in table if table[i]["human"] == "ok"]
    single = [i for i in table if table[i]["oneEditWords"] == 1 and table[i]["oneEditEdit"] not in ("repeat", "restart", "skipped_passage")]
    edited = [i for i in table if table[i]["label"] != "clean"]
    for src in ("llm", "ctc"):
        for mode in ("pairwise", "nearBest"):
            get = lambda i, k: table[i][src][k][mode]
            pos = core.score({i: get(i, "labelled") for i in ok}, {i: True for i in ok})
            sub = lambda ids, k: core.score({i: get(i, k) for i in ids}, {i: False for i in ids})
            one, word, sh = sub(list(table), "oneEdit"), sub(single, "oneEdit"), sub(list(table), "shifted")
            stats[f"{src}-{mode}"] = {"agreement": pos["specificity"], "positives": pos["positives"],
                                      "agreementEdited": core.score({i: get(i, "labelled") for i in ok if i in edited}, {i: True for i in ok if i in edited})["specificity"],
                                      "oneEdit": {"sensitivity": one["sensitivity"], "flagged": one["flagged"], "counts": one["negatives"]},
                                      "oneEditEdited": sub(edited, "oneEdit")["flagged"],
                                      "oneWord": {"sensitivity": word["sensitivity"], "flagged": word["flagged"], "counts": word["negatives"]},
                                      "shifted": {"sensitivity": sh["sensitivity"], "flagged": sh["flagged"], "counts": sh["negatives"]},
                                      "repeatMisses": [i for i in table if table[i]["oneEditEdit"] in ("repeat", "restart") and get(i, "oneEdit") == "match"]}
    arm = {"createdAt": now(), "model": {k: v for k, v in res.items() if k != "results"}, "margin": core.LIK_MARGIN,
           "exposed": "per window: the labelled text, its one-edit control and its shifted control, respelled without diacritics "
                      "(words shared with the model's own transcript take its spelling), plus the model's own transcript; never the label type",
           "rule": {"pairwise": "claimed - alternative >= margin -> match, <= -margin -> mismatch, else unsure; the alternative of the labelled text "
                                "is its one-edit control and vice versa; shifted is compared with the labelled text",
                    "nearBest": "claimed >= best - margin -> match, else mismatch; best = the model's own transcript (llm) or greedy CTC path (ctc)",
                    "spellings": "max over spellings", "source": "tools/review_asr_core.py prefer / near_best / plain_candidates"}}
    rec["likelihoodArm"] = arm
    _write(rec_path, rec)
    summary = {"version": 1, "testSet": test_set, "queue": queue_rel, "reviewer": reviewer, **arm, "humanVerdicts": len(ok), "items": len(items),
               "edited": len(edited), "stats": stats, "table": table}
    _write(REVIEWS / test_set / f"asr-calibration.{record}-likelihood.json", summary)
    vol.commit()
    return summary


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=3 * 60 * 60)
def round2(test_set: str, queue_rel: str, reviewer: str, perturb: bool, force: bool = False) -> dict:
    """Round 2 (ticket 12). C1: every saved transcript re-judged by the rasm rule (core.judge_rasm with the session's edge
    words), next to the round-1 rule. C2 (perturb): the prompt chosen by core.pick_prompt among ROVER_PROMPTS (on the
    round-1 rule) is run on each window under ROVER_PERTURB; core.rover votes the six transcripts into one, also with
    the Qwen3 and Whisper transcripts as two extra voters. Every transcript source is judged by every rule on the
    labelled text and its controls. Writes asr-nemotron-rover.json and asr-calibration.round2.json."""
    import review_asr_core as core

    queue = json.loads((DATA / queue_rel).read_text())
    verdicts = json.loads((REVIEWS / test_set / f"{reviewer}.json").read_text())["items"]
    items = queue["items"]
    sessions = {}
    for it in items:
        sid = core.session_id(it)
        if sid not in sessions and (p := DATA / test_set / "sessions" / f"{sid}.json").is_file():
            sessions[sid] = json.loads(p.read_text())
    ctrl = core.controls(queue, sessions)
    edge = {it["id"]: core.edge_context(it, sessions.get(core.session_id(it))) for it in items}
    texts = {it["id"]: {"labelled": it["text"], "oneEdit": ctrl[it["id"]]["oneEdit"]["text"], "shifted": ctrl[it["id"]]["shifted"]["text"]} for it in items}
    ok = [it["id"] for it in items if verdicts.get(it["id"], {}).get("verdict") == "ok"]
    rules = {"round1": lambda t, h, i: core.compare(t, h),
             "rasm": lambda t, h, i: core.judge_rasm(t, h, edge[i]),
             "rasm-no-edge": lambda t, h, i: core.compare_rasm(t, h),
             "rasm-no-hamza": lambda t, h, i: core.judge_rasm(t, h, edge[i], hamza=False),
             "rasm-phonetic": lambda t, h, i: core.judge_rasm(t, h, edge[i], phonetic=True)}

    def evaluate(heard: dict[str, str]) -> dict:
        res = {}
        for name, fn in rules.items():
            table = {}
            for it in items:
                i = it["id"]
                d = {k: fn(texts[i][k], heard[i], i) for k in texts[i]}
                table[i] = {**{k: v["decision"] for k, v in d.items()}, "reasons": d["labelled"]["reasons"],
                            "oneEditEdit": ctrl[i]["oneEdit"]["edit"], "oneEditWords": ctrl[i]["oneEdit"].get("words", 1)}
            st = core.calib_stats(table, ok)
            res[name] = {"stats": st, "passes": core.passes_bar(st), "table": table}
        return res

    recs = {}
    for name in ["qwen3-asr-1.7b", "whisper-large-v3", *PROMPT_RECORDS.values()]:
        if (p := REVIEWS / test_set / f"asr-{name}.json").is_file():
            recs[name] = json.loads(p.read_text())
    sources = {name: {"transcripts": {i: r["items"][i]["transcript"] for i in texts}} for name, r in recs.items()}
    for name, s in sources.items():
        s["rules"] = evaluate(s["transcripts"])
    pick = core.pick_prompt({p: sources[PROMPT_RECORDS[p]]["rules"]["round1"]["stats"] for p in ROVER_PROMPTS if PROMPT_RECORDS[p] in sources})
    out = {"version": 1, "testSet": test_set, "queue": queue_rel, "reviewer": reviewer, "createdAt": now(), "humanVerdicts": len(ok),
           "rule": {**rule(core), "edgeTol": core.EDGE_TOL, "rasmGroup": core.RASM_GROUP, "rasmMergeSim": core.RASM_MERGE_SIM,
                    "source": "tools/review_asr_core.py compare / judge_rasm / rover / calib_stats / passes_bar / pick_prompt"},
           "edgeContext": edge, "roverPrompt": pick}
    if perturb:
        rec_path = REVIEWS / test_set / "asr-nemotron-rover.json"
        old = json.loads(rec_path.read_text()) if rec_path.is_file() and not force else {"items": {}}
        den = denoise.remote(test_set, [{"id": it["id"], **it["audio"]} for it in items])
        vol.reload()
        base = recs[PROMPT_RECORDS[pick]]
        clips, have = [], {}
        for it in items:
            for dn, speed in ROVER_PERTURB:
                key = f"{it['id']}|{'denoised' if dn else 'raw'}|{speed:g}"
                clip = {"id": key, **it["audio"], "speed": speed, **({"denoised": den["items"][it["id"]]["denoised"]} if dn else {})}
                sha = hashlib.sha256(audio_bytes(clip)).hexdigest()
                prev = old["items"].get(key) or (base["items"][it["id"]] if (dn, speed) == (False, 1.0) else None)
                if prev and prev.get("audioSha256") == sha and old.get("promptKey", pick) == pick:
                    have[key] = {k: v for k, v in prev.items() if k in ("audioSha256", "transcript", "raw", "seconds")}
                else:
                    clips.append(clip)
        res = Nemotron().transcribe.remote(clips, pick) if clips else None
        if res and res["revision"] != base["model"]["revision"]:
            raise RuntimeError(f"perturbed transcripts from {res['revision']}, base from {base['model']['revision']}")
        for r in (res or {}).get("results", []):
            have[r["id"]] = {k: v for k, v in r.items() if k != "id"}
        hyps, rv, rv8 = {}, {}, {}
        for it in items:
            i = it["id"]
            hyps[i] = [have[f"{i}|{'denoised' if dn else 'raw'}|{speed:g}"]["transcript"] for dn, speed in ROVER_PERTURB]
            rv[i] = core.rover(hyps[i])
            rv8[i] = core.rover(hyps[i] + [sources["qwen3-asr-1.7b"]["transcripts"][i], sources["whisper-large-v3"]["transcripts"][i]])
        for k, (dn, speed) in enumerate(ROVER_PERTURB):
            sources[f"perturbed {'denoised' if dn else 'raw'} {speed:g}x"] = {"transcripts": {i: hyps[i][k] for i in hyps}}
        sources["rover"] = {"transcripts": {i: rv[i]["transcript"] for i in rv}}
        sources["rover+qwen+whisper"] = {"transcripts": {i: rv8[i]["transcript"] for i in rv8}}
        for name in [n for n in sources if "rules" not in sources[n]]:
            sources[name]["rules"] = evaluate(sources[name]["transcripts"])
        meta = {k: v for k, v in (res or {}).items() if k != "results"} or old.get("model") or base["model"]
        _write(rec_path, {"version": 1, "testSet": test_set, "queue": queue_rel, "createdAt": now(), "model": meta, "promptKey": pick,
                          "prompt": NEMOTRON_PROMPTS[pick], "perturbations": [{"denoised": dn, "speed": s} for dn, s in ROVER_PERTURB],
                          "denoiser": {k: v for k, v in den.items() if k != "items"}, "denoised": den["items"],
                          "items": have, "rover": {i: {"hyps": hyps[i], **rv[i]} for i in rv},
                          "roverWithQwenWhisper": {i: {"transcript": rv8[i]["transcript"], "pivot": rv8[i]["pivot"]} for i in rv8},
                          "decisions": {n: {r: v["table"] for r, v in sources[n]["rules"].items()} for n in sources if n.startswith(("perturbed", "rover"))}})
    out["sources"] = {n: {"transcripts": s["transcripts"], "rules": s["rules"]} for n, s in sources.items()}
    _write(REVIEWS / test_set / "asr-calibration.round2.json", out)
    vol.commit()
    return out


def rule(core) -> dict:
    return {"okSim": core.OK_SIM, "wrongSim": core.WRONG_SIM, "mergeSim": core.MERGE_SIM, "edgeFree": core.EDGE_FREE,
            "combine": "all match -> match; any mismatch and none match -> mismatch; else unsure", "source": "tools/review_asr_core.py"}


def _likelihood_decisions(queue: dict, queue_rel: str, force: bool) -> tuple[dict[str, str], str, dict[str, str]]:
    """The pre-screen that passed calibration (pairwise likelihood): Nemotron first transcribes every window label-blind
    (for spelling only); then, for windows with a planted edit, it scores the labelled text against the original text
    (the edit undone), and the labelled text must win by core.LIK_MARGIN. Clean windows have no alternative text and
    stay unsure, so the listener hears them. Writes /lab/reviews/<testSet>/asr-nemotron-likelihood.<queue>.json."""
    import review_asr_core as core

    test_set, stem, items, name = queue["testSet"], Path(queue_rel).stem, queue["items"], "nemotron-3.5-transcribe"
    asr = _transcribe(test_set, items, [name], force, "asr-{model}." + stem + ".json")[name]
    sessions = {}
    for it in items:
        sid = core.session_id(it)
        if sid not in sessions and (p := DATA / test_set / "sessions" / f"{sid}.json").is_file():
            sessions[sid] = json.loads(p.read_text())
    alts, jobs = {}, []
    for it in items:
        s = sessions.get(core.session_id(it))
        if s and it["label"]["type"] != "clean" and (alt := core.revert_text(it, s)):
            heard = asr["items"][it["id"]]["transcript"]
            alts[it["id"]] = alt["text"]
            jobs.append({"id": it["id"], **it["audio"], "candidates": {"labelled": core.plain_candidates(it["text"], heard),
                                                                       "original": core.plain_candidates(alt["text"], heard)}})
    res = Nemotron().score.remote(jobs, asr["model"]["promptKey"]) if jobs else {"results": []}
    by = {r["id"]: r for r in res["results"]}
    dec, recs = {}, {}
    for it in items:
        r, a = by.get(it["id"]), asr["items"][it["id"]]
        dec[it["id"]] = core.prefer(r["scores"]["labelled"]["llm"], r["scores"]["original"]["llm"]) if r else "unsure"
        recs[it["id"]] = {"audio": it["audio"], "audioSha256": a["audioSha256"], "raw": a["raw"], "transcript": a["transcript"], "text": it["text"],
                          "label": it["label"], "original": alts.get(it["id"]), "scores": r and r["scores"], "decision": dec[it["id"]],
                          "why": None if r else "no alternative text: clean window"}
    _write(REVIEWS / test_set / f"asr-nemotron-likelihood.{stem}.json",
           {"version": 1, "testSet": test_set, "queue": queue_rel, "createdAt": now(), "model": asr["model"], "margin": core.LIK_MARGIN,
            "exposed": "the labelled text and the original text (planted edit undone), respelled without diacritics", "items": recs})
    return (dec, f"{asr['model']['id']}@{asr['model']['revision']} pairwise likelihood, margin {core.LIK_MARGIN} nats",
            {i: r["why"] for i, r in recs.items() if r["why"]})


@app.function(image=cpu_image, volumes={VOL: vol}, timeout=2 * 60 * 60)
def prescreen_queue(queue_rel: str, models: list[str], spot: float, seed: int, seams: str, force: bool) -> dict:
    import review_asr_core as core

    queue = json.loads((DATA / queue_rel).read_text())
    test_set, stem = queue["testSet"], Path(queue_rel).stem
    if models == ["nemotron-likelihood"]:
        combined, used, notes = _likelihood_decisions(queue, queue_rel, force)
        out = core.prescreen(queue, combined, spot=spot, seed=seed, seams=seams, notes=notes)
        out["prescreen"] |= {"queue": queue_rel, "models": {"nemotron-likelihood": used}, "decisions": combined, "createdAt": now()}
        dest = f"{Path(queue_rel).parent}/{stem}.prescreen.json"
        _write(DATA / dest, out)
        vol.commit()
        return {**out["prescreen"], "wrote": dest}
    asr = _transcribe(test_set, queue["items"], models, force, "asr-{model}." + stem + ".json")
    dec = {}
    for name in models:
        recs = {}
        for it in queue["items"]:
            r = asr[name]["items"][it["id"]]
            c = core.compare(it["text"], r["transcript"])
            recs[it["id"]] = {"audio": it["audio"], "audioSha256": r["audioSha256"], "raw": r["raw"], "transcript": r["transcript"],
                              "text": it["text"], "label": it["label"], "labelled": {k: c[k] for k in ("decision", "reasons", "ops")}}
        _write(REVIEWS / test_set / f"asr-{name}.{stem}.json", {"version": 1, "testSet": test_set, "queue": queue_rel, "createdAt": now(),
                                                              "model": asr[name]["model"], "rule": rule(core), "items": recs})
        dec[name] = {i: r["labelled"]["decision"] for i, r in recs.items()}
    combined = {it["id"]: core.combine([dec[m][it["id"]] for m in models]) for it in queue["items"]}
    out = core.prescreen(queue, combined, spot=spot, seed=seed, seams=seams)
    out["prescreen"] |= {"queue": queue_rel, "models": {m: asr[m]["model"]["id"] + "@" + asr[m]["model"]["revision"] for m in models},
                         "decisions": combined, "createdAt": now()}
    dest = f"{Path(queue_rel).parent}/{stem}.prescreen.json"
    _write(DATA / dest, out)
    vol.commit()
    return {**out["prescreen"], "wrote": dest}


@app.local_entrypoint()
def main(test_set: str = "lab-v1.1", queue: str = "lab-v1.1/review/v1-check.json", reviewer: str = "me",
         models: str = "qwen3-asr-1.7b,whisper-large-v3", force: bool = False, pad: float = 0.0, out: str = "", name: str = ""):
    s = calibrate.remote(test_set, queue, reviewer, models.split(","), force, pad, name)
    out = out or f"docs/benchmarks/lab-review-asr-v1{f'-{name}' if name else ''}{f'-pad{pad:g}' if pad else ''}.json"
    (ROOT / out).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    for who, st in s["stats"].items():
        a, o, w, sh = st["agreement"], st["oneEdit"], st["oneWord"], st["shifted"]
        print(f"{who:18} agree {a['k']}/{a['n']} {a['ci95']}  positives {st['positives']}  one-edit flagged {o['flagged']['k']}/{o['flagged']['n']} "
              f"(mismatch {o['sensitivity']['k']})  one-word flagged {w['flagged']['k']}/{w['flagged']['n']}  shifted flagged {sh['flagged']['k']}/{sh['flagged']['n']}")
    print(f"wrote {out} and /lab/reviews/{test_set}/asr-*.json")


@app.local_entrypoint()
def likelihood_main(test_set: str = "lab-v1.1", queue: str = "lab-v1.1/review/v1-check.json", reviewer: str = "me",
                    record: str = "nemotron-3.5-transcribe", out: str = ""):
    s = likelihood.remote(test_set, queue, reviewer, record)
    out = out or f"docs/benchmarks/lab-review-asr-v1-{record}-likelihood.json"
    (ROOT / out).write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n")
    for who, st in s["stats"].items():
        a, o, w, sh = st["agreement"], st["oneEdit"], st["oneWord"], st["shifted"]
        print(f"{who:14} agree {a['k']}/{a['n']} {a['ci95']}  one-edit flagged {o['flagged']['k']}/{o['flagged']['n']}  one-word flagged "
              f"{w['flagged']['k']}/{w['flagged']['n']}  shifted flagged {sh['flagged']['k']}/{sh['flagged']['n']}  repeat misses {st['repeatMisses']}")
    print(f"wrote {out} and /lab/reviews/{test_set}/asr-{record}.json (likelihood)")


@app.local_entrypoint()
def round2_main(test_set: str = "lab-v1.1", queue: str = "lab-v1.1/review/v1-check.json", reviewer: str = "me", perturb: bool = False,
                force: bool = False, out: str = "docs/benchmarks/lab-review-asr-v1-round2.json"):
    s = round2.remote(test_set, queue, reviewer, perturb, force)
    slim = {**s, "sources": {n: {r: {k: v for k, v in x.items() if k != "table"} | {"decisions": {i: {k: t[k] for k in ("labelled", "oneEdit", "shifted")}
                                                                                                  for i, t in x["table"].items()}}
                                     for r, x in src["rules"].items()} for n, src in s["sources"].items()}}
    (ROOT / out).write_text(json.dumps(slim, ensure_ascii=False, indent=1) + "\n")
    print("rover prompt:", s["roverPrompt"])
    for n, src in s["sources"].items():
        for r, x in src["rules"].items():
            st = x["stats"]
            print(f"{n:34} {r:14} agree {st['agreement']['k']:2}/{st['agreement']['n']}  one-edit {st['oneEdit']['k']}/{st['oneEdit']['n']}  "
                  f"one-word {st['oneWord']['k']}/{st['oneWord']['n']}  shifted {st['shifted']['k']}/{st['shifted']['n']}  "
                  f"repeat misses {st['repeatMisses']}{'  PASSES' if x['passes'] else ''}")
    print(f"wrote {out} and /lab/reviews/{test_set}/asr-calibration.round2.json")


@app.local_entrypoint()
def denoise_smoke(path: str = "lab-v1.1/sessions/lab-v1.1-e20.phone.wav", start: float = 0.0, end: float = 8.0):
    """The denoiser on one window outside any review queue."""
    r = denoise.remote("smoke", [{"id": "smoke", "path": path, "startS": start, "endS": end}])
    print(json.dumps(r, ensure_ascii=False))


@app.function(image=nemotron_image, timeout=20 * 60)
def nemotron_env() -> dict:
    import importlib.metadata

    import torch
    import transformers

    env = {"transformers": transformers.__version__, "torch": torch.__version__, "cls": transformers.AutoModelForMultimodalLM.__name__,
           "processor": getattr(transformers, "Nemotron3_5TranscribeProcessor").__name__, "accelerate": importlib.metadata.version("accelerate")}
    print(env)
    return env


@app.local_entrypoint()
def smoke(path: str = "lab-v1.1/sessions/lab-v1.1-e20.studio.wav", start: float = 0.0, end: float = 8.0):
    """One window outside any review queue, to check the Nemotron container end to end."""
    r = Nemotron().transcribe.remote([{"id": "smoke", "path": path, "startS": start, "endS": end}])
    print(json.dumps({k: v for k, v in r.items() if k != "results"}, ensure_ascii=False))
    x = r["results"][0]
    print(x["transcript"], "|", x["raw"]["text"], "|", x["raw"]["tokenIds"][:8], "| ctc:", x["raw"]["ctcGreedy"]["text"], x["seconds"])
    s = Nemotron().score.remote([{"id": "smoke", "path": path, "startS": start, "endS": end,
                                  "candidates": {"own": [x["transcript"]], "plain": [" ".join(x["transcript"].split()[1:])]}}])
    print(json.dumps(s["results"][0], ensure_ascii=False)[:1500])


@app.local_entrypoint()
def prescreen(queue: str, models: str = "nemotron-likelihood", spot: float = 0.10, seed: int = 12, seams: str = "listen", force: bool = False):
    r = prescreen_queue.remote(queue, models.split(","), spot, seed, seams, force)
    print(json.dumps({k: v for k, v in r.items() if k != "decisions"}, ensure_ascii=False, indent=1))
