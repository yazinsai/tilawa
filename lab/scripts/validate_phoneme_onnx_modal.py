"""Validate quantized phoneme ONNX model against test corpus on Modal.

Runs the quantized ONNX model on all benchmark test samples and evaluates:
1. Phoneme CTC output quality (decoded phonemes)
2. Verse matching accuracy using phoneme-based Levenshtein matching
3. Comparison against the NeMo native model (if --compare-nemo)

Usage:
  modal run scripts/validate_phoneme_onnx_modal.py
  modal run scripts/validate_phoneme_onnx_modal.py --compare-nemo
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import modal

app = modal.App("fastconformer-phoneme-validate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "onnxruntime>=1.16",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
ONNX_PATH = "/vol/export/fastconformer_phoneme_q8.onnx"
VOCAB_PATH = "/vol/export/phoneme_vocab.json"

PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]


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


def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity ratio (same as web app)."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0:
        return 0.0
    if lb == 0:
        return 0.0
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[i] = min(prev[i]+1, curr[i-1]+1, prev[i-1]+cost)
        prev = curr
    dist = prev[la]
    return (la + lb - dist) / (la + lb)


def _join_phonemes(raw_tokens: list[str]) -> str:
    """Join phoneme tokens into word representation (same as web app CTC decoder)."""
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
    """Greedy CTC decode (same as web app)."""
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "test_corpus"


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/vol": vol},
    timeout=600,
)
def upload_corpus(manifest_json: str, quran_phonemes_json: str, audio_files: dict[str, bytes]):
    """Upload test corpus and quran phonemes to the volume."""
    import os
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
        print(f"  Uploaded: {fname} ({len(data)} bytes)")

    vol.commit()
    print(f"Uploaded {len(audio_files)} audio files + manifest + quran_phonemes")


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/vol": vol},
)
def validate(compare_nemo: bool = False):
    """Validate ONNX model on test audio samples."""
    import tempfile
    import tarfile
    import numpy as np
    import onnxruntime as ort
    import torch

    _install_kaldialign_fallback()
    import nemo.collections.asr as nemo_asr

    vol.reload()

    # Load manifest and quran phonemes from volume
    with open("/vol/test_corpus/manifest.json") as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    with open("/vol/quran_phonemes.json") as f:
        quran_phonemes = json.load(f)

    # Load vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    blank_id = max(int(k) for k in vocab.keys())

    # Build phoneme verse index for matching
    BSM = "bismi allahi arraHmaani arraHiimi"
    verse_data = []
    for v in quran_phonemes:
        joined = v["phonemes_joined"]
        verse_data.append({
            "surah": v["surah"],
            "ayah": v["ayah"],
            "phonemes_joined": joined,
            "phonemes_joined_no_bsm": (
                joined[len(BSM):].strip()
                if v["ayah"] == 1 and v["surah"] not in (1, 9) and joined.startswith(BSM)
                else None
            ),
        })

    # Load base model (for preprocessor)
    print("Loading base model for preprocessor...")
    base_model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)
    base_model = base_model.cpu()
    base_model.eval()

    # Load ONNX session
    print(f"Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    # Optionally load NeMo phoneme model for comparison
    nemo_phoneme_model = None
    if compare_nemo:
        print("Loading NeMo phoneme model for comparison...")
        nemo_phoneme_model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)
        nemo_phoneme_model = nemo_phoneme_model.cpu()

        # Replace CTC head
        vocab_size = len(PHONEME_VOCAB) + 1
        dec = nemo_phoneme_model.ctc_decoder
        if hasattr(dec, "decoder_layers") and len(dec.decoder_layers) > 0:
            last = dec.decoder_layers[-1]
            inf = getattr(last, "in_channels", getattr(last, "in_features", 512))
            dec.decoder_layers[-1] = torch.nn.Conv1d(inf, vocab_size, kernel_size=1)

        # Load fine-tuned weights
        ckpt = Path("/vol/fastconformer-phoneme-v1/model/model.nemo")
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with tarfile.open(str(ckpt), "r:gz") as tar:
                    tar.extractall(tmpdir)
            except tarfile.ReadError:
                with tarfile.open(str(ckpt), "r:") as tar:
                    tar.extractall(tmpdir)
            weights_path = Path(tmpdir) / "model_weights.ckpt"
            if not weights_path.exists():
                for p in Path(tmpdir).rglob("*.ckpt"):
                    weights_path = p
                    break
            state_dict = torch.load(weights_path, map_location="cpu")
            nemo_phoneme_model.load_state_dict(state_dict, strict=False)
        nemo_phoneme_model.eval()

    def match_verse(text: str) -> dict | None:
        """Match phoneme text against all verses using Levenshtein ratio."""
        if not text.strip():
            return None
        best_score = 0.0
        best_verse = None
        for v in verse_data:
            score = _levenshtein_ratio(text, v["phonemes_joined"])
            if v["phonemes_joined_no_bsm"]:
                score = max(score, _levenshtein_ratio(text, v["phonemes_joined_no_bsm"]))
            if score > best_score:
                best_score = score
                best_verse = v
        if best_verse and best_score >= 0.3:
            return {"surah": best_verse["surah"], "ayah": best_verse["ayah"], "score": best_score}
        return None

    # Run validation
    results = []
    correct = 0
    total = 0

    # Build sample list with expected verses
    audio_data_list = []
    for sample in samples:
        expected = sample.get("expected_verses", [{"surah": sample["surah"], "ayah": sample["ayah"]}])
        audio_data_list.append((
            sample["id"],
            f"/vol/test_corpus/{sample['file']}",
            expected[0]["surah"],
            expected[0]["ayah"],
        ))

    for sample_id, audio_path, exp_surah, exp_ayah in audio_data_list:
        import librosa
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Preprocess with NeMo's preprocessor
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]])
        with torch.no_grad():
            processed, proc_len = base_model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )

        # ONNX inference
        ort_inputs = {
            input_names[0]: processed.numpy(),
            input_names[1]: proc_len.numpy(),
        }
        logprobs = sess.run(None, ort_inputs)[0]
        raw_phonemes, joined_text = _ctc_decode(logprobs[0], vocab, blank_id)

        # Match verse
        match = match_verse(joined_text)
        matched = match and match["surah"] == exp_surah and match["ayah"] == exp_ayah
        if matched:
            correct += 1
        total += 1

        result = {
            "id": sample_id,
            "expected": f"{exp_surah}:{exp_ayah}",
            "predicted_phonemes": joined_text[:80],
            "match": f"{match['surah']}:{match['ayah']} ({match['score']:.3f})" if match else "NONE",
            "correct": matched,
        }

        # NeMo comparison
        if nemo_phoneme_model:
            with torch.no_grad():
                nemo_processed, nemo_len = nemo_phoneme_model.preprocessor(
                    input_signal=audio_tensor, length=audio_length
                )
                nemo_encoded, nemo_enc_len = nemo_phoneme_model.encoder(
                    audio_signal=nemo_processed, length=nemo_len
                )
                nemo_logprobs = nemo_phoneme_model.ctc_decoder(encoder_output=nemo_encoded)
                nemo_lp = nemo_logprobs[0].numpy()
            nemo_raw, nemo_joined = _ctc_decode(nemo_lp, vocab, blank_id)
            result["nemo_phonemes"] = nemo_joined[:80]
            result["match_nemo"] = "SAME" if joined_text == nemo_joined else "DIFFER"

        results.append(result)
        status = "✓" if matched else "✗"
        print(f"  {status} {sample_id}: {result['match']} (expected {exp_surah}:{exp_ayah})")

    accuracy = correct / total if total else 0
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"{'='*60}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


@app.local_entrypoint()
def main(compare_nemo: bool = False):
    # First, upload test corpus to the volume
    print("=== Uploading test corpus to Modal volume ===")

    manifest_path = CORPUS_DIR / "manifest.json"
    quran_phonemes_path = PROJECT_ROOT / "web" / "frontend" / "public" / "quran_phonemes.json"

    with open(manifest_path) as f:
        manifest_json = f.read()
    manifest = json.loads(manifest_json)

    with open(quran_phonemes_path) as f:
        quran_phonemes_json = f.read()

    # Read all audio files
    audio_files = {}
    for sample in manifest["samples"]:
        fpath = CORPUS_DIR / sample["file"]
        audio_files[sample["file"]] = fpath.read_bytes()
    print(f"Read {len(audio_files)} audio files locally")

    upload_corpus.remote(manifest_json, quran_phonemes_json, audio_files)

    # Now run validation
    print(f"\n=== Running validation ===")
    print(f"Compare with NeMo: {compare_nemo}")
    print()

    result = validate.remote(compare_nemo=compare_nemo)

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
    print(f"{'='*60}")

    # Print failures
    failures = [r for r in result["results"] if not r["correct"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  {f['id']}: expected {f['expected']}, got {f['match']}")
            print(f"    phonemes: {f['predicted_phonemes']}")
