"""Relabel training manifests using wav2vec2 teacher model (hetchyy/r7 ONNX int8).

Runs the quantized ONNX teacher over all training audio in parallel (N workers)
and writes pseudo-labels to a new manifest. IPA phonemes are converted to
Buckwalter encoding to match the FastConformer training vocabulary.

The ONNX model is embedded in the Modal image from the local HuggingFace cache.

Usage:
  modal run --detach scripts/teacher_relabel_modal.py --output-name v6-augmented
  modal run --detach scripts/teacher_relabel_modal.py --output-name v6-augmented --num-workers 16
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("teacher-relabel")

HF_CACHE_SNAP = Path.home() / ".cache/huggingface/hub/models--hetchyy--r7_onnx_int8/snapshots/2d41a64d55c0db61a057ba94e7d80c8d047552c6"
MODAL_MODEL_DIR = "/model"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "onnxruntime>=1.18",
        "soundfile",
        "librosa",
        "Levenshtein",
        "numpy",
    )
    .add_local_file(str(HF_CACHE_SNAP / "model_quantized.onnx"), f"{MODAL_MODEL_DIR}/model_quantized.onnx")
    .add_local_file(str(HF_CACHE_SNAP / "model_quantized.onnx.data"), f"{MODAL_MODEL_DIR}/model_quantized.onnx.data")
    .add_local_file(str(HF_CACHE_SNAP / "vocab.json"), f"{MODAL_MODEL_DIR}/vocab.json")
    .add_local_file(str(HF_CACHE_SNAP / "preprocessor_config.json"), f"{MODAL_MODEL_DIR}/preprocessor_config.json")
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

# ---------------------------------------------------------------------------
# IPA -> Buckwalter mapping (matched to hetchyy/r7_onnx_int8 vocab)
# ---------------------------------------------------------------------------
IPA_TO_BUCKWALTER = {
    "b": "b", "t": "t", "θ": "^", "ħ": "H", "x": "x", "d": "d",
    "ð": "*", "r": "r", "z": "z", "s": "s", "ʃ": "$", "sˤ": "S",
    "dˤ": "D", "tˤ": "T", "ðˤ": "Z", "ʕ": "E", "ɣ": "g", "f": "f",
    "q": "q", "k": "k", "l": "l", "m": "m", "n": "n", "h": "h",
    "w": "w", "j": "y", "ʔ": "<", "ʒ": "j", "rˤ": "r",
    # Short vowels
    "a": "a", "u": "u", "i": "i",
    # Long vowels (colon notation)
    "a:": "A", "u:": "U", "i:": "I",
    # Emphatic vowels
    "aˤ": "a", "aˤ:": "A",
    # Geminated
    "bb": "bb", "tt": "tt", "θθ": "^^", "ħħ": "HH", "xx": "xx",
    "dd": "dd", "ðð": "**", "rr": "rr", "zz": "zz", "ss": "ss",
    "ʃʃ": "$$", "sˤsˤ": "SS", "dˤdˤ": "DD", "tˤtˤ": "TT",
    "ðˤðˤ": "ZZ", "ʕʕ": "EE", "ff": "ff", "qq": "qq", "kk": "kk",
    "ll": "ll", "mm": "mm", "hh": "hh", "ww": "ww", "jj": "yy",
    "ʒʒ": "jj", "rˤrˤ": "rr", "lˤlˤ": "ll",
    # Nasals
    "ñ": "n", "ŋ": "n", "m̃": "m", "w̃": "w", "j̃": "y",
    # Special
    "Q": "", "|": "|",
}


def convert_ipa_to_buckwalter(ipa_tokens: list[str]) -> str:
    bw = []
    for tok in ipa_tokens:
        if tok in IPA_TO_BUCKWALTER:
            mapped = IPA_TO_BUCKWALTER[tok]
            if mapped:
                bw.extend(mapped.split())
    return " ".join(bw)


def ctc_collapse(pred_ids: list[int], id_to_token: dict[int, str], pad_id: int = 0) -> list[str]:
    collapsed = []
    prev_id = None
    for idx in pred_ids:
        if idx == pad_id or idx == prev_id:
            prev_id = idx
            continue
        tok = id_to_token.get(idx, "")
        if tok == "|":
            prev_id = idx
            continue
        if tok and tok not in ("[PAD]", "[UNK]"):
            collapsed.append(tok)
        prev_id = idx
    return collapsed


@app.function(
    image=image,
    volumes={"/training": vol},
    timeout=6 * 3600,
    cpu=8,
    memory=16384,
)
def relabel_chunk(chunk_entries: list[dict], chunk_id: int, output_name: str = "v6-augmented") -> int:
    """Process a chunk and write results directly to volume."""
    import librosa
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    model_dir = Path(MODAL_MODEL_DIR)
    onnx_path = model_dir / "model_quantized.onnx"
    vocab = json.loads((model_dir / "vocab.json").read_text())
    id_to_token = {v: k for k, v in vocab.items()}
    pad_id = vocab.get("[PAD]", 0)
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())
    target_sr = preproc.get("sampling_rate", 16000)

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 2
    sess_opts.intra_op_num_threads = 6
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    chunk_path = Path(f"/training/{output_name}/manifests/relabel_chunk_{chunk_id:03d}.jsonl")

    print(f"[Worker {chunk_id}] Processing {len(chunk_entries)} entries...")
    t_start = time.time()
    results = []
    num_ok = 0

    for i, entry in enumerate(chunk_entries):
        try:
            audio, sr = sf.read(entry["audio_filepath"], dtype="float32")
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            logits = session.run([out_name], {input_name: audio.reshape(1, -1).astype(np.float32)})[0]
            pred_ids = np.argmax(logits[0], axis=-1).tolist()
            ipa_tokens = ctc_collapse(pred_ids, id_to_token, pad_id)
            teacher_text = convert_ipa_to_buckwalter(ipa_tokens)

            results.append({
                "audio_filepath": entry["audio_filepath"],
                "duration": entry["duration"],
                "text": teacher_text,
            })
            num_ok += 1
        except Exception:
            results.append(entry)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [Worker {chunk_id}] {i+1}/{len(chunk_entries)} ({rate:.1f}/s)")

    # Write chunk to volume
    with open(chunk_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()

    elapsed = time.time() - t_start
    rate = len(chunk_entries) / elapsed if elapsed > 0 else 0
    print(f"[Worker {chunk_id}] Done: {num_ok}/{len(chunk_entries)} in {elapsed:.0f}s ({rate:.1f}/s), wrote {chunk_path}")
    return num_ok


@app.function(
    image=image,
    volumes={"/training": vol},
    timeout=600,
    cpu=2,
    memory=4096,
)
def merge_chunks(output_name: str, num_chunks: int) -> int:
    """Merge chunk files from volume into final relabeled manifest."""
    from Levenshtein import ratio as lev_ratio

    vol.reload()
    manifests_dir = Path(f"/training/{output_name}/manifests")
    relabeled_path = manifests_dir / "train_manifest_relabeled.jsonl"

    merged = []
    missing = []
    for i in range(num_chunks):
        chunk_path = manifests_dir / f"relabel_chunk_{i:03d}.jsonl"
        if chunk_path.exists():
            entries = [json.loads(l) for l in chunk_path.read_text().splitlines() if l.strip()]
            merged.extend(entries)
            print(f"  Chunk {i}: {len(entries)} entries")
        else:
            missing.append(i)
            print(f"  Chunk {i}: MISSING")

    # For missing chunks, use original entries
    if missing:
        orig_path = manifests_dir / "train_manifest_original.jsonl"
        if orig_path.exists():
            all_orig = [json.loads(l) for l in orig_path.read_text().splitlines() if l.strip()]
            chunk_size = (len(all_orig) + num_chunks - 1) // num_chunks
            for ci in missing:
                start = ci * chunk_size
                end = min(start + chunk_size, len(all_orig))
                merged.extend(all_orig[start:end])
                print(f"  Chunk {ci}: used {end - start} original entries as fallback")

    with open(relabeled_path, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    vol.commit()

    print(f"\nWrote {len(merged)} entries to {relabeled_path}")
    if missing:
        print(f"Missing chunks {missing} used original labels as fallback")

    # Sample comparisons
    orig_path = manifests_dir / "train_manifest_original.jsonl"
    if orig_path.exists():
        originals = [json.loads(l) for l in orig_path.read_text().splitlines()[:5] if l.strip()]
        print("\n--- Sample comparisons ---")
        for i, (orig, relab) in enumerate(zip(originals, merged[:5])):
            sim = lev_ratio(orig.get("text", ""), relab.get("text", ""))
            print(f"[{i}] sim={sim:.3f}  orig: {orig.get('text','')[:60]}  →  teacher: {relab.get('text','')[:60]}")

    return len(merged)


@app.local_entrypoint()
def main(output_name: str = "v6-augmented", num_workers: int = 12, merge_only: bool = False):
    if merge_only:
        print(f"Merging {num_workers} chunks...")
        total = merge_chunks.remote(output_name, num_workers)
        print(f"Done! {total} entries merged.")
        return

    print(f"Reading manifest for {output_name}...")
    entries = read_manifest.remote(output_name)
    print(f"Loaded {len(entries)} entries, splitting into {num_workers} chunks")

    chunk_size = (len(entries) + num_workers - 1) // num_workers
    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    print(f"Chunk sizes: {[len(c) for c in chunks]}")

    chunk_args = [(chunk, i, output_name) for i, chunk in enumerate(chunks)]
    results = list(relabel_chunk.starmap(chunk_args))
    total_ok = sum(results)
    print(f"All workers done. {total_ok} entries relabeled.")

    print("Merging chunks...")
    total = merge_chunks.remote(output_name, num_workers)
    print(f"Done! {total} entries merged.")


@app.function(
    image=image,
    volumes={"/training": vol},
    timeout=300,
    cpu=1,
    memory=2048,
)
def read_manifest(output_name: str) -> list[dict]:
    """Read manifest from volume and return entries."""
    vol.reload()
    manifest_path = Path(f"/training/{output_name}/manifests/train_manifest.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")

    # Backup if needed
    backup = manifest_path.parent / "train_manifest_original.jsonl"
    if not backup.exists():
        import shutil
        shutil.copy2(manifest_path, backup)
        vol.commit()
        print(f"Backed up to {backup}")

    entries = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    print(f"Read {len(entries)} entries from {manifest_path}")
    return entries
