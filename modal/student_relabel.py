"""Audit natural teacher speech omitted by the student's spliced CTC target.

Uses the cached, unconstrained teacher pass; reference text never enters teacher
inference. Mining reads training data only and preserves the gold exclusion.
"""

import json
import os
from collections import Counter
from pathlib import Path


def speech_insertions(ref, hyp):
    """Interior teacher insertions between exact anchors, after length collapse."""
    from per_breakdown import collapse
    from rapidfuzz.distance import Levenshtein
    from student import spliced_target

    target, heard = spliced_target(ref, hyp), collapse(hyp)
    ops = Levenshtein.opcodes(target, heard)
    found = []
    for i, op in enumerate(ops):
        if op.tag != "insert" or i == 0 or i + 1 == len(ops):
            continue
        left, right = ops[i - 1], ops[i + 1]
        if any(x.tag != "equal" or x.src_end - x.src_start < 3 for x in (left, right)):
            continue
        text = heard[op.dest_start:op.dest_end]
        # Inventory IDs 1..28 are the base consonants; exclude vowel, madd,
        # nasal-length and other recitation annotations from the shape count.
        from student import TOKENS
        if len(text) >= 3 and sum(c in TOKENS[:28] for c in text) >= 2:
            found.append(dict(at=op.src_start, start=op.dest_start, end=op.dest_end, phones=text))
    return target, heard, found


def free_tokens(ids, probs):
    """Run-collapsed free teacher tokens with all contributing peak frames."""
    tokens = []
    for frame, (token, prob) in enumerate(zip(ids[:, 0], probs[:, 0])):
        token = int(token)
        if token == 0:
            continue
        if tokens and tokens[-1]["id"] == token:
            tokens[-1]["frames"].append(frame)
            tokens[-1]["confidence"] = max(tokens[-1]["confidence"], float(prob))
        else:
            tokens.append(dict(id=token, frames=[frame], confidence=float(prob)))
    return tokens


def canonical_path(ids, probs, target):
    """CTC Viterbi path on cached top-five probabilities, without a scorer import."""
    import numpy as np
    import torch
    import torchaudio
    from student import VOCAB, phoneme_ids

    lp = np.full((len(ids), VOCAB), np.log(1e-8), dtype=np.float32)
    np.put_along_axis(lp, ids.astype(np.int64), np.log(np.maximum(probs, 1e-8)), axis=1)
    with torch.no_grad():
        path, _ = torchaudio.functional.forced_align(
            torch.from_numpy(lp)[None], torch.tensor([phoneme_ids(target)]), blank=0)
    return path[0].numpy()


def relabel(ref, hyp, ids, probs, speech):
    """Return a corrected target and uncertain blank frames; keep silence intact.

    ``speech`` is a VAD decision at the cached teacher's 25 Hz frame rate.
    Confidence thresholds are fixed before practice evaluation.
    """
    from per_breakdown import collapse
    from student import TOKENS, ids_phonemes

    target, heard, candidates = speech_insertions(ref, hyp)
    tokens = free_tokens(ids, probs)
    if ids_phonemes([t["id"] for t in tokens]) != heard:
        return dict(target=target, mask=[], spans=[], mismatch=True)
    if not candidates:
        return dict(target=target, mask=[], spans=[], mismatch=False)
    path = canonical_path(ids, probs, target)
    mask, spans, inserts = set(), [], []
    for candidate in candidates:
        ts = tokens[candidate["start"]:candidate["end"]]
        lo, hi = ts[0]["frames"][0], ts[-1]["frames"][-1] + 1
        consonants = [t for t in ts if TOKENS[t["id"] - 1] in TOKENS[:28]]
        # Require two separate consonants in VAD speech left blank by the
        # canonical path. A held vowel or breath does not qualify.
        blank_consonants = [t for t in consonants if any(path[f] == 0 and speech[f] for f in t["frames"])]
        if len(blank_consonants) < 2:
            continue
        confident = (all(t["confidence"] >= .9 for t in ts)
                     and sum(any(path[f] == 0 and speech[f] and probs[f, 0] >= .9
                                 for f in t["frames"]) for t in blank_consonants) >= 2)
        if confident:
            inserts.append((candidate["at"], candidate["phones"]))
        else:
            # Do not mask any frames consumed by canonical phones.
            mask.update(f for f in range(lo, hi) if path[f] == 0 and speech[f])
        spans.append(dict(**candidate, lo=lo, hi=hi, confident=confident))
    for at, phones in reversed(inserts):
        target = target[:at] + phones + target[at:]
    return dict(target=collapse(target), mask=sorted(mask), spans=spans, mismatch=False)


def masked_ctc_log_probs(log_probs, masked):
    """Replace uncertain alignment-blank frames with a constant blank bridge.

    This preserves time and separates repeated labels, while both the blank
    and phone logits receive zero CTC gradient at uncertain frames. Detaching
    the original logits would still penalise off-text speech in the CTC path.
    """
    import torch

    blank = torch.full_like(log_probs, -1e4)
    blank[..., 0] = 0
    return torch.where(masked[..., None], blank, log_probs)


def apply_labels(clips, path):
    """Load a training-only overlay without changing cached source data."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    by_key = {r["key"]: r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate relabel key")
    clips["loss_mask"] = [[] for _ in clips["key"]]
    changed = masked = 0
    for i, key in enumerate(clips["key"]):
        r = by_key.get(key)
        if r is None:
            continue
        if r["base_target"] != clips["target"][i] or r["n_frames"] != int(clips["n_frames"][i]):
            raise ValueError(f"Stale relabel for {key}")
        if any(not isinstance(f, int) or not 0 <= f < r["n_frames"] for f in r["mask"]):
            raise ValueError(f"Invalid mask for {key}")
        changed += r["target"] != clips["target"][i]
        masked += bool(r["mask"])
        clips["target"][i], clips["loss_mask"][i] = r["target"], r["mask"]
    return dict(changed=changed, masked=masked, overlay_rows=len(rows))


def mine():
    """VAD-verify the cached free teacher pass on gold-excluded training clips."""
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    from student import exclusions, held_out

    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    vad = load_silero_vad()
    gold = exclusions(Path("/app/docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    shard, shards = int(os.environ.get("LAB_SHARD", 0)), int(os.environ.get("LAB_SHARDS", 1))
    counts, rows = Counter(), []
    for file in sorted(Path("/student/s1").glob("*.parquet"))[shard::shards]:
        columns = ["shard", "row", "bucket", "split", "n_samples", "ref", "hyp", "top5_ids", "top5_p"]
        data = pq.read_table(file, columns=columns).to_pylist()
        selected = []
        for i, r in enumerate(data):
            if r["bucket"] != "clean-match" or r["split"] != "train" or r["n_samples"] > 30 * 16000:
                continue
            if held_out(gold, r["shard"], r["row"], "train"):
                continue
            counts["training_clips"] += 1
            if speech_insertions(r["ref"], r["hyp"])[2]:
                selected.append(i)
        if not selected:
            continue
        audio = pq.read_table(file, columns=["flac"]).take(selected).column("flac").to_pylist()
        for i, flac in zip(selected, audio):
            r = data[i]
            ids = np.frombuffer(r["top5_ids"], np.uint8).reshape(-1, 5)
            probs = np.frombuffer(r["top5_p"], np.float16).reshape(-1, 5).astype(np.float32)
            wave, sr = sf.read(io.BytesIO(flac), dtype="float32")
            if sr != 16000 or wave.ndim != 1:
                raise ValueError("Expected mono 16 kHz source")
            speech = np.zeros(len(ids), dtype=bool)
            centers = (np.arange(len(ids)) + .5) * 640
            for span in get_speech_timestamps(torch.from_numpy(wave), vad, sampling_rate=sr, threshold=.5):
                speech |= (centers >= span["start"]) & (centers < span["end"])
            result = relabel(r["ref"], r["hyp"], ids, probs, speech)
            counts["cache_transcript_mismatch"] += result["mismatch"]
            if not result["spans"]:
                continue
            base_target = speech_insertions(r["ref"], r["hyp"])[0]
            counts["relabelled_clips"] += result["target"] != base_target
            counts["masked_clips"] += bool(result["mask"])
            counts["confident_spans"] += sum(s["confident"] for s in result["spans"])
            counts["uncertain_spans"] += sum(not s["confident"] for s in result["spans"])
            rows.append(dict(key=f'{r["shard"]}:{r["row"]}', base_target=base_target, n_frames=len(ids), **result))
        print(json.dumps(dict(file=file.name, counts=counts)), flush=True)
    out = Path(os.environ["LAB_OUT"])
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "mining.json").write_text(json.dumps(dict(counts=counts, vad="silero-vad-6.2.0", threshold=.5), indent=2))
    print(json.dumps(counts), flush=True)


def audit():
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from student import TOKENS, exclusions, held_out, ids_phonemes

    torch.set_num_threads(2)
    gold = exclusions(Path("/app/docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    shard, shards = int(os.environ.get("LAB_SHARD", 0)), int(os.environ.get("LAB_SHARDS", 1))
    counts, examples, mismatches = Counter(), [], []
    columns = ["shard", "row", "bucket", "split", "n_samples", "ref", "hyp", "top5_ids", "top5_p"]
    files = sorted(Path("/student/s1").glob("*.parquet"))[shard::shards]
    for file in files:
        for r in pq.read_table(file, columns=columns).to_pylist():
            if r["bucket"] != "clean-match" or r["split"] != "train" or r["n_samples"] > 30 * 16000:
                continue
            if held_out(gold, r["shard"], r["row"], "train"):
                counts["gold_excluded"] += 1
                continue
            counts["training_clips"] += 1
            target, heard, found = speech_insertions(r["ref"], r["hyp"])
            if not found:
                continue
            counts["clips_with_insertions"] += 1
            ids = np.frombuffer(r["top5_ids"], np.uint8).reshape(-1, 5)
            probs = np.frombuffer(r["top5_p"], np.float16).reshape(-1, 5).astype(np.float32)
            tokens = free_tokens(ids, probs)
            if ids_phonemes([t["id"] for t in tokens]) != heard:
                counts["cache_transcript_mismatch"] += 1
                if len(mismatches) < 10:
                    mismatches.append(dict(key=f'{r["shard"]}:{r["row"]}', heard=heard,
                                           decoded=ids_phonemes([t["id"] for t in tokens])))
                continue
            path = canonical_path(ids, probs, target)
            for candidate in found:
                ts = tokens[candidate["start"]:candidate["end"]]
                cs = [t for t in ts if TOKENS[t["id"] - 1] in TOKENS[:28]]
                counts["word_shaped_insertions"] += 1
                confident = sum(t["confidence"] >= 0.9 for t in cs) >= 2
                blank = sum(any(path[f] == 0 and probs[f, 0] >= 0.9 for f in t["frames"]) for t in cs) >= 2
                counts["two_confident_consonants"] += confident
                counts["two_confident_blank_consonants"] += blank
                item = dict(key=f'{r["shard"]}:{r["row"]}', target=target, heard=heard, **candidate,
                            start_s=ts[0]["frames"][0] * .04, end_s=(ts[-1]["frames"][-1] + 1) * .04,
                            confidence=[t["confidence"] for t in ts], canonical_blank=blank)
                if len(examples) < 50:
                    examples.append(item)
        print(json.dumps(dict(file=file.name, counts=counts)), flush=True)
    out = Path(os.environ["LAB_OUT"])
    (out / "audit.json").write_text(json.dumps(dict(counts=counts, examples=examples, mismatches=mismatches), ensure_ascii=False, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    import sys
    mine() if "--mine" in sys.argv else audit()
