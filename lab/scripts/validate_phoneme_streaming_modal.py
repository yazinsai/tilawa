"""Validate phoneme ONNX model with streaming simulation on Modal.

Simulates the browser streaming pipeline:
- Audio fed in 300ms chunks (4800 samples at 16kHz)
- Discovery mode: inference every 2s, 10s max window
- Tracking mode: inference every 0.5s, 5s max window
- Same verse matching (Levenshtein + continuation bonuses)
- Same word alignment + verse transitions

Usage:
  modal run scripts/validate_phoneme_streaming_modal.py
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import modal

app = modal.App("fastconformer-phoneme-streaming-validate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "onnxruntime>=1.16",
        "librosa",
        "rapidfuzz",
    )
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


def get_export_paths(output_name: str) -> tuple[str, str]:
    export_dir = Path(f"/vol/{output_name}/export")
    return (
        str(export_dir / "fastconformer_phoneme_q8.onnx"),
        str(export_dir / "phoneme_vocab.json"),
    )

# --- Constants matching web app exactly ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 4800  # 300ms at 16kHz (AudioWorklet sends this)
TRIGGER_SAMPLES = 32000  # 2s — discovery mode inference interval
MAX_WINDOW_SAMPLES = 160000  # 10s — max audio window in discovery
TRACKING_TRIGGER_SAMPLES = 8000  # 0.5s — tracking mode inference interval
TRACKING_MAX_WINDOW_SAMPLES = 80000  # 5s — max audio window in tracking
TRACKING_SILENCE_SAMPLES = 64000  # 4s silence → exit tracking
SILENCE_RMS_THRESHOLD = 0.005
STALE_CYCLE_LIMIT = 4
LOOKAHEAD = 5
FIRST_MATCH_THRESHOLD = 0.75
VERSE_MATCH_THRESHOLD = 0.45
RAW_TRANSCRIPT_THRESHOLD = 0.25


def _install_kaldialign_fallback():
    import sys
    try:
        import kaldialign
        return
    except Exception:
        pass
    def align(ref, hyp, eps="<eps>"):
        ref, hyp = list(ref), list(hyp)
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1): dp[i][0] = i
        for j in range(1, m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                dp[i][j] = min(dp[i-1][j-1]+cost, dp[i][j-1]+1, dp[i-1][j]+1)
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            if i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+(0 if ref[i-1]==hyp[j-1] else 1):
                out.append((ref[i-1], hyp[j-1])); i-=1; j-=1
            elif j>0 and dp[i][j]==dp[i][j-1]+1:
                out.append((eps, hyp[j-1])); j-=1
            else:
                out.append((ref[i-1], eps)); i-=1
        out.reverse()
        return out
    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


# --- Levenshtein (using rapidfuzz for C-speed) ---
def _levenshtein_ratio(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return (len(a) + len(b) - dist) / (len(a) + len(b))


def _partial_ratio(short: str, long: str) -> float:
    if not short or not long:
        return 0.0
    if len(short) > len(long):
        short, long = long, short
    window = len(short)
    best = 0.0
    for i in range(max(0, len(long) - window) + 1):
        r = _levenshtein_ratio(short, long[i:i+window])
        if r > best:
            best = r
            if best == 1.0:
                break
    return best


def _suffix_prefix_score(text: str, verse_text: str) -> float:
    words_t = text.split(" ")
    words_v = verse_text.split(" ")
    if len(words_t) < 2 or len(words_v) < 2:
        return 0.0
    best = 0.0
    max_trim = min(len(words_t) // 2, 4)
    for trim in range(1, max_trim + 1):
        suffix = " ".join(words_t[trim:])
        n = len(words_t) - trim
        prefix = " ".join(words_v[:min(n, len(words_v))])
        best = max(best, _levenshtein_ratio(suffix, prefix))
    return best


# --- CTC decode ---
def _join_phonemes(raw_tokens: list[str]) -> str:
    words = []
    current = []
    for tok in raw_tokens:
        if tok == "|":
            if current:
                words.append("".join(current))
            current = []
        else:
            current.append(tok)
    if current:
        words.append("".join(current))
    return " ".join(words)


def _ctc_decode(logprobs, vocab, blank_id):
    import numpy as np
    ids = np.argmax(logprobs, axis=-1)
    tokens = []
    prev = -1
    for idx in ids:
        if idx != prev and idx != blank_id:
            tokens.append(vocab.get(str(idx), "?"))
        prev = idx
    raw_phonemes = " ".join(tokens)
    joined = _join_phonemes(tokens)
    return raw_phonemes, joined


# --- QuranDB (Python port of quran-db.ts) ---
class QuranDB:
    BSM = "bismi allahi arraHmaani arraHiimi"

    def __init__(self, verses: list[dict]):
        self.verses = verses
        self._by_ref: dict[str, dict] = {}
        self._by_surah: dict[int, list[dict]] = {}
        for v in verses:
            key = f"{v['surah']}:{v['ayah']}"
            self._by_ref[key] = v
            self._by_surah.setdefault(v["surah"], []).append(v)
            # Pre-compute no-bsm
            if (v["ayah"] == 1 and v["surah"] not in (1, 9)
                    and v["phonemes_joined"].startswith(self.BSM)):
                stripped = v["phonemes_joined"][len(self.BSM):].strip()
                v["phonemes_joined_no_bsm"] = stripped or None
            else:
                v["phonemes_joined_no_bsm"] = None

    def get_verse(self, surah: int, ayah: int) -> dict | None:
        return self._by_ref.get(f"{surah}:{ayah}")

    def get_next_verse(self, surah: int, ayah: int) -> dict | None:
        verses = self._by_surah.get(surah, [])
        for i, v in enumerate(verses):
            if v["ayah"] == ayah:
                if i + 1 < len(verses):
                    return verses[i + 1]
                next_surah = self._by_surah.get(surah + 1, [])
                return next_surah[0] if next_surah else None
        return None

    def _continuation_bonuses(self, hint: tuple[int, int] | None) -> dict[str, float]:
        bonuses = {}
        if not hint:
            return bonuses
        h_surah, h_ayah = hint
        if f"{h_surah}:{h_ayah+1}" in self._by_ref:
            bonuses[f"{h_surah}:{h_ayah+1}"] = 0.22
            if f"{h_surah}:{h_ayah+2}" in self._by_ref:
                bonuses[f"{h_surah}:{h_ayah+2}"] = 0.12
            if f"{h_surah}:{h_ayah+3}" in self._by_ref:
                bonuses[f"{h_surah}:{h_ayah+3}"] = 0.06
        else:
            next_verses = self._by_surah.get(h_surah + 1, [])
            bonus_values = [0.22, 0.12, 0.06]
            for i in range(min(len(next_verses), 3)):
                bonuses[f"{next_verses[i]['surah']}:{next_verses[i]['ayah']}"] = bonus_values[i]
        return bonuses

    def match_verse(self, text: str, threshold: float = 0.3,
                    max_span: int = 3, hint: tuple[int, int] | None = None) -> dict | None:
        if not text.strip():
            return None
        bonuses = self._continuation_bonuses(hint)

        # Score all single verses
        scored = []
        for v in self.verses:
            raw = _levenshtein_ratio(text, v["phonemes_joined"])
            if v.get("phonemes_joined_no_bsm"):
                raw = max(raw, _levenshtein_ratio(text, v["phonemes_joined_no_bsm"]))
            bonus = bonuses.get(f"{v['surah']}:{v['ayah']}", 0.0)
            if bonus > 0:
                sp = _suffix_prefix_score(text, v["phonemes_joined"])
                raw = max(raw, sp)
            scored.append((v, raw, bonus, min(raw + bonus, 1.0)))
        scored.sort(key=lambda x: -x[3])

        best_v, best_raw, best_bonus, best_score = scored[0]
        best = {
            "surah": best_v["surah"],
            "ayah": best_v["ayah"],
            "phonemes_joined": best_v["phonemes_joined"],
            "score": best_score,
            "raw_score": best_raw,
            "bonus": best_bonus,
        }

        # Multi-ayah spans
        seen_surahs = set()
        for idx in range(min(len(scored), 20)):
            v = scored[idx][0]
            s = v["surah"]
            if s in seen_surahs:
                continue
            seen_surahs.add(s)
            verses = self._by_surah[s]
            for i in range(len(verses)):
                for span in range(2, max_span + 1):
                    if i + span > len(verses):
                        break
                    chunk = verses[i:i+span]
                    first_text = chunk[0].get("phonemes_joined_no_bsm") or chunk[0]["phonemes_joined"]
                    combined = " ".join([first_text] + [c["phonemes_joined"] for c in chunk[1:]])
                    raw = _levenshtein_ratio(text, combined)
                    bonus = bonuses.get(f"{chunk[0]['surah']}:{chunk[0]['ayah']}", 0.0)
                    score = min(raw + bonus, 1.0)
                    if score > best_score:
                        best_score = score
                        best = {
                            "surah": s,
                            "ayah": chunk[0]["ayah"],
                            "ayah_end": chunk[-1]["ayah"],
                            "phonemes_joined": combined,
                            "score": score,
                            "raw_score": raw,
                            "bonus": bonus,
                        }

        if best_score >= threshold:
            return best
        return None


# --- Streaming Tracker (Python port of tracker.ts) ---
class StreamingTracker:
    def __init__(self, db: QuranDB, transcribe_fn):
        self.db = db
        self.transcribe = transcribe_fn
        self.full_audio = None  # numpy array
        self.new_audio_count = 0
        self.last_emitted_ref: tuple[int, int] | None = None
        self.last_emitted_text = ""
        self.prev_emitted_ref: tuple[int, int] | None = None
        self.prev_emitted_text = ""

        # Tracking state
        self.tracking_verse: dict | None = None
        self.tracking_verse_words: list[str] = []
        self.tracking_last_word_idx = -1
        self.silence_samples = 0
        self.stale_cycles = 0

        # Results
        self.discovered_verses: list[tuple[int, int]] = []
        self.events: list[dict] = []

    def feed(self, samples):
        import numpy as np
        if self.full_audio is None:
            self.full_audio = samples.copy()
        else:
            self.full_audio = np.concatenate([self.full_audio, samples])
        self.new_audio_count += len(samples)

        # Trim to max window
        max_samples = TRACKING_MAX_WINDOW_SAMPLES if self.tracking_verse else MAX_WINDOW_SAMPLES
        if len(self.full_audio) > max_samples:
            self.full_audio = self.full_audio[-max_samples:]

        if self.tracking_verse is not None:
            self._handle_tracking(samples)
        else:
            self._handle_discovery()

    def _is_silence(self, audio) -> bool:
        import numpy as np
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < SILENCE_RMS_THRESHOLD

    def _handle_discovery(self):
        if self.new_audio_count < TRIGGER_SAMPLES:
            return
        self.new_audio_count = 0

        # Skip silence
        tail = self.full_audio[-TRIGGER_SAMPLES:]
        if self._is_silence(tail):
            return

        text = self.transcribe(self.full_audio.copy())
        if not text or len(text.strip()) < 5:
            return

        # Skip residual from last verse
        if self.last_emitted_text:
            residual = _partial_ratio(text, self.last_emitted_text)
            if residual > 0.7:
                return

        match = self.db.match_verse(
            text,
            threshold=RAW_TRANSCRIPT_THRESHOLD,
            max_span=4,
            hint=self.last_emitted_ref,
        )

        effective_threshold = FIRST_MATCH_THRESHOLD if self.last_emitted_ref is None else VERSE_MATCH_THRESHOLD

        # Diagnostic: log discovery attempts
        audio_secs = len(self.full_audio) / SAMPLE_RATE
        if match:
            self.events.append({
                "type": "discovery_attempt",
                "audio_secs": round(audio_secs, 1),
                "text": text[:60],
                "best_match": f"{match['surah']}:{match['ayah']}",
                "score": round(match["score"], 3),
                "raw_score": round(match.get("raw_score", match["score"]), 3),
                "threshold": effective_threshold,
                "passed": match["score"] >= effective_threshold,
            })

        if match and match["score"] >= effective_threshold:
            ref = (match["surah"], match["ayah"])

            # Dedup
            if self.last_emitted_ref and self.last_emitted_ref == ref:
                return

            verse = self.db.get_verse(match["surah"], match["ayah"])
            ayah_end = match.get("ayah_end")
            effective_ref = (match["surah"], ayah_end) if ayah_end else ref

            self.prev_emitted_ref = self.last_emitted_ref
            self.prev_emitted_text = self.last_emitted_text
            self.last_emitted_ref = effective_ref
            self.last_emitted_text = match.get("phonemes_joined", verse["phonemes_joined"] if verse else "")

            self.discovered_verses.append(ref)
            self.events.append({
                "type": "verse_match",
                "surah": match["surah"],
                "ayah": match["ayah"],
                "score": match["score"],
                "text": text[:60],
            })

            if verse:
                self._enter_tracking(verse)

    def _handle_tracking(self, samples):
        import numpy as np

        # Silence check
        chunk_rms = np.sqrt(np.mean(samples ** 2))
        if chunk_rms < SILENCE_RMS_THRESHOLD:
            self.silence_samples += len(samples)
            if self.silence_samples >= TRACKING_SILENCE_SAMPLES:
                self._exit_tracking("extended silence")
                self.new_audio_count = 0
                return
        else:
            self.silence_samples = 0

        if self.new_audio_count < TRACKING_TRIGGER_SAMPLES:
            return
        self.new_audio_count = 0

        text = self.transcribe(self.full_audio.copy())
        if not text or len(text.strip()) < 3:
            return

        recognized_words = text.split(" ")
        resume_from = max(self.tracking_last_word_idx, 0)
        matched_indices = self._align_position(recognized_words, self.tracking_verse_words, resume_from)

        advanced = (len(matched_indices) > 0 and
                    matched_indices[-1] > self.tracking_last_word_idx)

        if not advanced:
            self.stale_cycles += 1
            if self.stale_cycles >= STALE_CYCLE_LIMIT:
                self._exit_tracking(f"stale ({self.stale_cycles} cycles)")
                self.new_audio_count = 0
                return
        else:
            self.stale_cycles = 0

        if advanced:
            self.tracking_last_word_idx = matched_indices[-1]
            self.events.append({
                "type": "word_progress",
                "surah": self.tracking_verse["surah"],
                "ayah": self.tracking_verse["ayah"],
                "word_index": self.tracking_last_word_idx + 1,
                "total_words": len(self.tracking_verse_words),
            })

        # Check verse completion
        if matched_indices:
            coverage = len(matched_indices) / len(self.tracking_verse_words)
            near_end = matched_indices[-1] >= len(self.tracking_verse_words) - 2

            if coverage >= 0.8 and near_end:
                cur_ref = (self.tracking_verse["surah"], self.tracking_verse["ayah"])
                self.last_emitted_ref = cur_ref
                self.last_emitted_text = self.tracking_verse["phonemes_joined"]
                next_v = self.db.get_next_verse(cur_ref[0], cur_ref[1])
                self._exit_tracking("verse complete")

                if next_v:
                    next_ref = (next_v["surah"], next_v["ayah"])
                    self.prev_emitted_ref = self.last_emitted_ref
                    self.prev_emitted_text = self.last_emitted_text
                    self.last_emitted_ref = next_ref
                    self.last_emitted_text = next_v["phonemes_joined"]
                    self.discovered_verses.append(next_ref)
                    self.events.append({
                        "type": "verse_match",
                        "surah": next_v["surah"],
                        "ayah": next_v["ayah"],
                        "score": 0.99,
                        "text": "(auto-advance)",
                    })
                    self._enter_tracking(next_v)

                # Keep last 2s
                keep = min(len(self.full_audio), TRIGGER_SAMPLES)
                self.full_audio = self.full_audio[-keep:]

    def _align_position(self, recognized: list[str], verse_words: list[str], start_from: int = 0) -> list[int]:
        matched = []
        verse_ptr = start_from
        for rec in recognized:
            if verse_ptr >= len(verse_words):
                break
            limit = min(verse_ptr + LOOKAHEAD, len(verse_words))
            for j in range(verse_ptr, limit):
                if self._words_match(rec, verse_words[j]):
                    matched.append(j)
                    verse_ptr = j + 1
                    break
        return matched

    @staticmethod
    def _words_match(w1: str, w2: str, threshold: float = 0.7) -> bool:
        if w1 == w2:
            return True
        if len(w1) <= 2 or len(w2) <= 2:
            return w1 == w2
        return _levenshtein_ratio(w1, w2) >= threshold

    def _enter_tracking(self, verse: dict):
        self.tracking_verse = verse
        self.tracking_verse_words = verse.get("phoneme_words", verse["phonemes_joined"].split(" "))
        self.tracking_last_word_idx = -1
        self.silence_samples = 0
        self.stale_cycles = 0

    def _exit_tracking(self, reason: str):
        verse_len = len(self.tracking_verse_words)
        progress = (self.tracking_last_word_idx + 1) / verse_len if verse_len > 0 else 0

        if reason == "verse complete":
            pass  # caller already updated
        elif reason.startswith("stale") and progress < 0.5:
            self.last_emitted_ref = self.prev_emitted_ref
            self.last_emitted_text = self.prev_emitted_text
        elif reason.startswith("stale") and self.tracking_last_word_idx >= 0:
            self.last_emitted_text = " ".join(self.tracking_verse_words[:self.tracking_last_word_idx + 1])

        self.tracking_verse = None
        self.tracking_verse_words = []
        self.tracking_last_word_idx = -1
        self.silence_samples = 0
        self.stale_cycles = 0


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "test_corpus"


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/vol": vol},
    timeout=600,
)
def upload_corpus(manifest_json: str, quran_phonemes_json: str, audio_files: dict[str, bytes]):
    """Upload test corpus to volume (reused from non-streaming script)."""
    corpus_dir = Path("/vol/test_corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with open(corpus_dir / "manifest.json", "w") as f:
        f.write(manifest_json)
    with open("/vol/quran_phonemes.json", "w") as f:
        f.write(quran_phonemes_json)
    for fname, data in audio_files.items():
        fpath = corpus_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "wb") as f:
            f.write(data)
    vol.commit()
    print(f"Uploaded {len(audio_files)} audio files + manifest + quran_phonemes")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vol": vol},
)
def validate_streaming(
    first_match_threshold: float = FIRST_MATCH_THRESHOLD,
    output_name: str = "fastconformer-phoneme-v2",
):
    """Run streaming validation: simulate browser audio pipeline."""
    import numpy as np
    import onnxruntime as ort
    import torch
    import librosa

    _install_kaldialign_fallback()
    import nemo.collections.asr as nemo_asr

    # Allow threshold override for experimentation
    global FIRST_MATCH_THRESHOLD
    FIRST_MATCH_THRESHOLD = first_match_threshold
    print(f"Using FIRST_MATCH_THRESHOLD={FIRST_MATCH_THRESHOLD}")

    vol.reload()

    # Load data
    with open("/vol/test_corpus/manifest.json") as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    with open("/vol/quran_phonemes.json") as f:
        quran_phonemes = json.load(f)

    onnx_path, vocab_path = get_export_paths(output_name)
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"Missing ONNX export: {onnx_path}")
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Missing vocab export: {vocab_path}")

    with open(vocab_path) as f:
        vocab = json.load(f)
    blank_id = max(int(k) for k in vocab.keys())

    # Build QuranDB
    db = QuranDB(quran_phonemes)

    # Load NeMo base model for preprocessor
    print("Loading base model for preprocessor...")
    base_model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)
    base_model = base_model.cpu()
    base_model.eval()

    # Load ONNX session
    print(f"Loading ONNX model for {output_name}: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    def transcribe(audio: np.ndarray) -> str:
        """Run mel + ONNX inference + CTC decode, return joined phoneme text."""
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]])
        with torch.no_grad():
            processed, proc_len = base_model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )
        ort_inputs = {
            input_names[0]: processed.numpy(),
            input_names[1]: proc_len.numpy(),
        }
        logprobs = sess.run(None, ort_inputs)[0]
        _, joined = _ctc_decode(logprobs[0], vocab, blank_id)
        return joined

    # Run streaming validation
    results = []

    for sample in samples:
        sample_id = sample["id"]
        expected_verses = sample.get("expected_verses",
                                      [{"surah": sample["surah"], "ayah": sample["ayah"]}])
        audio_path = f"/vol/test_corpus/{sample['file']}"

        # Load full audio
        audio_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Create tracker for this sample
        tracker = StreamingTracker(db, transcribe)

        # Feed audio in 300ms chunks (simulating AudioWorklet)
        total_samples = len(audio_data)
        offset = 0
        while offset < total_samples:
            end = min(offset + CHUNK_SIZE, total_samples)
            chunk = audio_data[offset:end]
            tracker.feed(chunk)
            offset = end

        # In the real browser, audio keeps flowing after the recitation
        # (user hasn't stopped recording). Simulate by feeding low-level
        # noise (not pure silence, so the silence detector doesn't skip it)
        # for one more inference cycle.
        if not tracker.discovered_verses and not tracker.tracking_verse:
            noise = np.random.normal(0, SILENCE_RMS_THRESHOLD * 2,
                                     TRIGGER_SAMPLES + CHUNK_SIZE).astype(np.float32)
            noffset = 0
            while noffset < len(noise):
                nend = min(noffset + CHUNK_SIZE, len(noise))
                tracker.feed(noise[noffset:nend])
                noffset = nend

        # Evaluate: did the tracker discover the expected verses?
        expected_set = {(v["surah"], v["ayah"]) for v in expected_verses}
        discovered_set = set(tracker.discovered_verses)

        # For single-verse: check if the first expected verse was discovered
        first_expected = (expected_verses[0]["surah"], expected_verses[0]["ayah"])
        first_found = first_expected in discovered_set

        # For multi-verse: check how many expected verses were found
        found_count = len(expected_set & discovered_set)
        total_expected = len(expected_set)

        result = {
            "id": sample_id,
            "expected": [f"{v['surah']}:{v['ayah']}" for v in expected_verses],
            "discovered": [f"{s}:{a}" for s, a in tracker.discovered_verses],
            "first_found": first_found,
            "found_count": found_count,
            "total_expected": total_expected,
            "events": tracker.events,
        }
        results.append(result)

        # Print status
        if total_expected == 1:
            status = "✓" if first_found else "✗"
            disc_str = ", ".join(f"{s}:{a}" for s, a in tracker.discovered_verses) or "NONE"
            print(f"  {status} {sample_id}: discovered [{disc_str}] (expected {first_expected[0]}:{first_expected[1]})")
        else:
            status = "✓" if found_count == total_expected else f"~{found_count}/{total_expected}"
            disc_str = ", ".join(f"{s}:{a}" for s, a in tracker.discovered_verses) or "NONE"
            exp_str = ", ".join(f"{v['surah']}:{v['ayah']}" for v in expected_verses)
            print(f"  {status} {sample_id}: discovered [{disc_str}] (expected [{exp_str}])")

    # Compute stats
    single_samples = [r for r in results if r["total_expected"] == 1]
    multi_samples = [r for r in results if r["total_expected"] > 1]

    single_correct = sum(1 for r in single_samples if r["first_found"])
    multi_all_found = sum(1 for r in multi_samples if r["found_count"] == r["total_expected"])
    multi_first_found = sum(1 for r in multi_samples if r["first_found"])
    multi_verse_total = sum(r["total_expected"] for r in multi_samples)
    multi_verse_found = sum(r["found_count"] for r in multi_samples)

    print(f"\n{'='*60}")
    print(f"SINGLE VERSE:  {single_correct}/{len(single_samples)} ({single_correct/len(single_samples)*100:.1f}%)")
    print(f"MULTI VERSE:")
    print(f"  First verse: {multi_first_found}/{len(multi_samples)} ({multi_first_found/len(multi_samples)*100:.1f}%)")
    print(f"  All verses:  {multi_all_found}/{len(multi_samples)} ({multi_all_found/len(multi_samples)*100:.1f}%)")
    print(f"  Per-verse:   {multi_verse_found}/{multi_verse_total} ({multi_verse_found/multi_verse_total*100:.1f}%)")
    total_correct = single_correct + multi_first_found
    total = len(single_samples) + len(multi_samples)
    print(f"OVERALL (first verse): {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print(f"{'='*60}")

    return {
        "single_correct": single_correct,
        "single_total": len(single_samples),
        "multi_first_found": multi_first_found,
        "multi_all_found": multi_all_found,
        "multi_total": len(multi_samples),
        "multi_verse_found": multi_verse_found,
        "multi_verse_total": multi_verse_total,
        "results": results,
    }


@app.local_entrypoint()
def main(
    threshold: float = 0.75,
    output_name: str = "fastconformer-phoneme-v2",
):
    # Upload corpus if needed
    print("=== Uploading test corpus to Modal volume ===")
    manifest_path = CORPUS_DIR / "manifest.json"
    quran_phonemes_path = PROJECT_ROOT / "web" / "frontend" / "public" / "quran_phonemes.json"

    with open(manifest_path) as f:
        manifest_json = f.read()
    manifest = json.loads(manifest_json)

    with open(quran_phonemes_path) as f:
        quran_phonemes_json = f.read()

    audio_files = {}
    for sample in manifest["samples"]:
        fpath = CORPUS_DIR / sample["file"]
        audio_files[sample["file"]] = fpath.read_bytes()
    print(f"Read {len(audio_files)} audio files locally")

    upload_corpus.remote(manifest_json, quran_phonemes_json, audio_files)

    # Run streaming validation
    print(
        f"\n=== Running streaming validation "
        f"(threshold={threshold}, output_name={output_name}) ===\n"
    )
    result = validate_streaming.remote(
        first_match_threshold=threshold,
        output_name=output_name,
    )

    print(f"\n{'='*60}")
    print(f"SINGLE VERSE:  {result['single_correct']}/{result['single_total']}")
    print(f"MULTI VERSE (first): {result['multi_first_found']}/{result['multi_total']}")
    print(f"MULTI VERSE (all):   {result['multi_all_found']}/{result['multi_total']}")
    print(f"MULTI PER-VERSE:     {result['multi_verse_found']}/{result['multi_verse_total']}")
    print(f"{'='*60}")

    # Print failures with diagnostics
    failures = [r for r in result["results"] if not r["first_found"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            exp_str = ", ".join(f["expected"])
            disc_str = ", ".join(f["discovered"]) or "NONE"
            print(f"  {f['id']}: expected [{exp_str}], discovered [{disc_str}]")
            for ev in f["events"]:
                if ev["type"] == "verse_match":
                    print(f"    → match {ev['surah']}:{ev['ayah']} ({ev['score']:.3f}) text='{ev['text']}'")
                elif ev["type"] == "discovery_attempt":
                    status = "✓" if ev["passed"] else "✗"
                    print(f"    {status} @{ev['audio_secs']}s: {ev['best_match']} score={ev['score']} (raw={ev['raw_score']}) thresh={ev['threshold']} text='{ev['text']}'")
