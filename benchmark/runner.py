"""
Benchmark runner for all experiments — streaming sequence evaluation.

Usage:
    python -m benchmark.runner                           # all experiments
    python -m benchmark.runner --experiment whisper-lora  # one experiment
    python -m benchmark.runner --category short           # filter by category
"""

import sys
import json
import time
import importlib.util
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",

    "embedding-search": EXPERIMENTS_DIR / "embedding-search" / "run.py",
    "contrastive": EXPERIMENTS_DIR / "contrastive" / "run.py",
    "streaming-asr": EXPERIMENTS_DIR / "streaming-asr" / "run.py",
    "ctc-alignment": EXPERIMENTS_DIR / "ctc-alignment" / "run.py",
    "tarteel-whisper-base": EXPERIMENTS_DIR / "tarteel-whisper-base" / "run.py",
    "two-stage": EXPERIMENTS_DIR / "two-stage" / "run.py",
    "two-stage-faster-whisper-pruned": EXPERIMENTS_DIR / "two-stage-faster-whisper-pruned" / "run.py",
    "distilled-ctc": EXPERIMENTS_DIR / "distilled-ctc" / "run.py",
    "rabah-pruned-ctc": EXPERIMENTS_DIR / "rabah-pruned-ctc" / "run.py",
    "nvidia-fastconformer": EXPERIMENTS_DIR / "nvidia-fastconformer" / "run.py",
    "fastconformer-quran-lm-fusion": EXPERIMENTS_DIR / "fastconformer-quran-lm-fusion" / "run.py",
    "fastconformer-ctc-rescore": EXPERIMENTS_DIR / "fastconformer-ctc-rescore" / "run.py",
    "fastconformer-nbest-bruteforce": EXPERIMENTS_DIR / "fastconformer-nbest-bruteforce" / "run.py",
    "contrastive-v2": EXPERIMENTS_DIR / "contrastive-v2" / "run.py",
    "fastconformer-phoneme": EXPERIMENTS_DIR / "fastconformer-phoneme" / "run.py",
    "w2v-phonemes": EXPERIMENTS_DIR / "w2v-phonemes" / "run.py",
}

NEW_MODELS_PATH = EXPERIMENTS_DIR / "new-models" / "run.py"


def _load_module(name: str, file_path: Path):
    """Load a Python module from a file path (handles hyphenated dirs)."""
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_manifest() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return data["samples"]


def score_sequence(expected: list[dict], predicted: list[dict]) -> dict:
    """Score a predicted verse sequence against expected.

    Uses ordered subsequence matching: a predicted verse counts as a
    recall hit if it matches an expected verse and appears in the
    correct relative order.

    Args:
        expected: [{"surah": int, "ayah": int}, ...]
        predicted: [{"surah": int, "ayah": int, ...}, ...]

    Returns:
        {"recall": float, "precision": float, "sequence_accuracy": float}
    """
    if not expected:
        return {"recall": 1.0, "precision": 1.0, "sequence_accuracy": 1.0}

    if not predicted:
        return {"recall": 0.0, "precision": 0.0, "sequence_accuracy": 0.0}

    # Greedy ordered match: walk through expected, find each in predicted (in order)
    pred_tuples = [(p["surah"], p["ayah"]) for p in predicted]
    exp_tuples = [(e["surah"], e["ayah"]) for e in expected]

    matched = 0
    pred_idx = 0
    matched_pred_indices = set()
    for exp in exp_tuples:
        for j in range(pred_idx, len(pred_tuples)):
            if pred_tuples[j] == exp:
                matched += 1
                matched_pred_indices.add(j)
                pred_idx = j + 1
                break

    recall = matched / len(exp_tuples)
    precision = len(matched_pred_indices) / len(pred_tuples)
    seq_acc = 1.0 if pred_tuples == exp_tuples else 0.0

    return {"recall": recall, "precision": precision, "sequence_accuracy": seq_acc}


def discover_experiments(filter_name: str | None = None) -> list[dict]:
    """Return list of {name, run_path, model_name (optional)}."""
    experiments = []

    for name, run_path in EXPERIMENT_REGISTRY.items():
        if filter_name and filter_name != name and not filter_name.startswith(f"{name}/"):
            continue
        if not run_path.exists():
            print(f"Warning: {name} run.py not found at {run_path}")
            continue
        try:
            mod = _load_module(name.replace("/", "_").replace("-", "_"), run_path)
            if hasattr(mod, "list_models"):
                for model_name in mod.list_models():
                    entry_name = f"{name}/{model_name}"
                    if filter_name and filter_name != entry_name and filter_name != name:
                        continue
                    experiments.append({
                        "name": entry_name,
                        "run_path": run_path,
                        "model_name": model_name,
                    })
                continue
        except Exception as e:
            # Fall back to treating this as a single experiment; the runtime
            # will surface import/dependency errors per sample as needed.
            print(f"Warning: could not expand models for {name}: {e}")

        experiments.append({"name": name, "run_path": run_path, "model_name": None})

    # Expand new-models
    if not filter_name or filter_name.startswith("new-models"):
        try:
            mod = _load_module("new_models_run", NEW_MODELS_PATH)
            for model_name in mod.list_models():
                entry_name = f"new-models/{model_name}"
                if filter_name and filter_name != entry_name and filter_name != "new-models":
                    continue
                experiments.append({
                    "name": entry_name,
                    "run_path": NEW_MODELS_PATH,
                    "model_name": model_name,
                })
        except Exception as e:
            print(f"Warning: could not load new-models: {e}")

    return experiments


def _predict_to_emissions(predict_result: dict) -> list[dict]:
    """Convert a predict() result dict to a list of verse emissions.

    predict() returns: {surah, ayah, ayah_end, score, transcript}
    If ayah_end is set, expand to individual verse emissions.
    """
    if not predict_result or predict_result.get("surah", 0) == 0:
        return []

    surah = predict_result["surah"]
    ayah_start = predict_result["ayah"]
    ayah_end = predict_result.get("ayah_end") or ayah_start
    score = predict_result.get("score", 0.0)

    emissions = []
    for ayah in range(ayah_start, ayah_end + 1):
        emissions.append({"surah": surah, "ayah": ayah, "score": score})
    return emissions


def run_experiment(
    exp: dict,
    samples: list[dict],
    pipeline: StreamingPipeline,
    mode: str = "full",
    chunk_seconds: float = 3.0,
) -> dict | None:
    """Run one experiment against all samples.

    Uses predict() directly when available (e.g. CTC alignment does its own
    candidate scoring). Falls back to transcribe() + StreamingPipeline for
    ASR-based experiments.

    Args:
        mode: "full" for whole-file transcription, "streaming" for chunked.
        chunk_seconds: Chunk duration for streaming mode.
    """
    mod = _load_module(exp["name"].replace("/", "_").replace("-", "_"), exp["run_path"])

    # Prefer transcribe() + streaming pipeline over predict().
    # predict() does a single match_verse() call which can't handle multi-verse
    # recordings. Only fall back to predict() for experiments without transcribe().
    use_predict = hasattr(mod, "predict") and not hasattr(mod, "transcribe") and mode == "full"

    if not use_predict and not hasattr(mod, "transcribe"):
        print(f"  Skipping {exp['name']} — no transcribe() or predict() function")
        return None

    if use_predict:
        predict_fn = mod.predict
        if exp["model_name"]:
            base_fn = predict_fn
            predict_fn = lambda path, _mn=exp["model_name"]: base_fn(path, model_name=_mn)
    else:
        transcribe_fn = mod.transcribe
        if exp["model_name"]:
            base_fn = transcribe_fn
            transcribe_fn = lambda path, _mn=exp["model_name"]: base_fn(path, model_name=_mn)

    # Warmup
    warmup_sample = samples[0]
    audio_path = str(CORPUS_DIR / warmup_sample["file"])
    try:
        if use_predict:
            predict_fn(audio_path)
        else:
            transcribe_fn(audio_path)
    except Exception as e:
        print(f"  Warmup failed for {exp['name']}: {e}")

    # Get model size
    try:
        if exp["model_name"]:
            size = mod.model_size(model_name=exp["model_name"])
        else:
            size = mod.model_size()
    except Exception:
        size = 0

    total_recall = 0.0
    total_precision = 0.0
    total_seq_acc = 0.0
    latencies = []
    per_sample = []

    for sample in samples:
        audio_path = str(CORPUS_DIR / sample["file"])
        expected = sample.get("expected_verses", [{"surah": sample["surah"], "ayah": sample["ayah"]}])

        result = None
        try:
            start = time.perf_counter()
            if use_predict:
                result = predict_fn(audio_path)
                emissions = _predict_to_emissions(result)
            elif mode == "streaming":
                emissions = pipeline.run_on_audio_chunked(
                    audio_path, transcribe_fn, chunk_seconds=chunk_seconds,
                )
            else:
                emissions = pipeline.run_on_full_transcript(audio_path, transcribe_fn)
            elapsed = time.perf_counter() - start
        except Exception as e:
            print(f"  Error on {sample['id']}: {e}")
            emissions = []
            elapsed = 0.0

        scores = score_sequence(expected, emissions)
        for alt_expected in sample.get("also_accept", []):
            alt_scores = score_sequence(alt_expected, emissions)
            if alt_scores["sequence_accuracy"] > scores["sequence_accuracy"]:
                scores = alt_scores
                expected = alt_expected
        total_recall += scores["recall"]
        total_precision += scores["precision"]
        total_seq_acc += scores["sequence_accuracy"]
        latencies.append(elapsed)

        per_sample.append({
            "id": sample["id"],
            "expected": expected,
            "predicted": emissions,
            "recall": scores["recall"],
            "precision": scores["precision"],
            "sequence_accuracy": scores["sequence_accuracy"],
            "latency": elapsed,
            "raw_predict": result if use_predict else None,
        })

    n = len(samples)
    avg_latency = sum(latencies) / n if n else 0

    exp_name = exp["name"] if mode == "full" else f"{exp['name']} (stream {chunk_seconds:.0f}s)"

    return {
        "name": exp_name,
        "recall": total_recall / n if n else 0,
        "precision": total_precision / n if n else 0,
        "sequence_accuracy": total_seq_acc / n if n else 0,
        "total": n,
        "avg_latency": avg_latency,
        "model_size": size,
        "per_sample": per_sample,
    }


def format_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.1f} GB"


def print_table(results: list[dict]):
    print()
    print(f"{'Experiment':<30} {'Recall':>8} {'Precision':>10} {'SeqAcc':>8} {'Latency':>10} {'Size':>10}")
    print("-" * 78)
    for r in results:
        rec = f"{r['recall']:.0%}"
        prec = f"{r['precision']:.0%}"
        seq = f"{r['sequence_accuracy']:.0%}"
        lat = f"{r['avg_latency']:.2f}s"
        size = format_size(r['model_size'])
        print(f"{r['name']:<30} {rec:>8} {prec:>10} {seq:>8} {lat:>10} {size:>10}")
    print()


def save_results(
    results: list[dict],
    *,
    mode: str = "full",
    category: str | None = None,
    chunk_seconds: float = 3.0,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")

    latest_path = RESULTS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            loaded_latest = json.load(f)
    else:
        loaded_latest = []

    latest = {}
    for entry in loaded_latest:
        key = (
            entry.get("name"),
            entry.get("mode", "full"),
            entry.get("category"),
            entry.get("total"),
            entry.get("chunk_seconds"),
        )
        latest[key] = entry

    for r in results:
        effective_chunk = chunk_seconds if mode == "streaming" else None
        summary = {
            "name": r["name"],
            "recall": r["recall"],
            "precision": r["precision"],
            "sequence_accuracy": r["sequence_accuracy"],
            "total": r["total"],
            "avg_latency": r["avg_latency"],
            "model_size": r["model_size"],
            "timestamp": timestamp,
            "mode": mode,
            "category": category,
            "chunk_seconds": effective_chunk,
            "source_file": path.name,
        }

        key = (
            summary["name"],
            summary["mode"],
            summary["category"],
            summary["total"],
            summary["chunk_seconds"],
        )
        prev = latest.get(key)
        if prev is None or r["sequence_accuracy"] > prev.get("sequence_accuracy", 0) or (
            r["sequence_accuracy"] == prev.get("sequence_accuracy", 0)
            and r["avg_latency"] < prev.get("avg_latency", float("inf"))
        ):
            latest[key] = summary

    with open(latest_path, "w") as f:
        json.dump(
            sorted(
                latest.values(),
                key=lambda x: (
                    x.get("name", ""),
                    x.get("mode", "full"),
                    x.get("category") or "",
                    x.get("total", 0),
                    x.get("chunk_seconds") or 0,
                ),
            ),
            f,
            indent=2,
            default=str,
        )
    print(f"Updated {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all experiments (streaming)")
    parser.add_argument("--experiment", type=str, help="Run only this experiment")
    parser.add_argument("--category", type=str, help="Filter samples by category")
    parser.add_argument("--corpus", type=str, default="test_corpus",
                        help="Corpus directory name under benchmark/ (default: test_corpus)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "streaming"],
                        help="full = transcribe whole file; streaming = chunked audio")
    parser.add_argument("--chunk", type=float, default=3.0,
                        help="Chunk duration in seconds for streaming mode (default: 3.0)")
    args = parser.parse_args()

    global CORPUS_DIR
    CORPUS_DIR = Path(__file__).parent / args.corpus
    if not CORPUS_DIR.exists():
        print(f"Error: corpus directory not found: {CORPUS_DIR}")
        return

    samples = load_manifest()
    if args.category:
        samples = [s for s in samples if s["category"] == args.category]
        print(f"Filtered to {len(samples)} samples in category '{args.category}'")

    experiments = discover_experiments(args.experiment)
    if not experiments:
        print(f"No experiments found matching '{args.experiment}'")
        return

    db = QuranDB()
    pipeline = StreamingPipeline(db=db)

    mode_label = f"streaming ({args.chunk:.0f}s chunks)" if args.mode == "streaming" else "full transcript"
    print(f"Running {len(experiments)} experiment(s) on {len(samples)} sample(s) [{mode_label}]...")

    results = []
    for exp in experiments:
        print(f"\n>>> {exp['name']}")
        result = run_experiment(exp, samples, pipeline, mode=args.mode, chunk_seconds=args.chunk)
        if result is None:
            continue
        results.append(result)
        print(f"    Recall: {result['recall']:.0%}  Precision: {result['precision']:.0%}  SeqAcc: {result['sequence_accuracy']:.0%}")

    print_table(results)
    save_results(
        results,
        mode=args.mode,
        category=args.category,
        chunk_seconds=args.chunk,
    )


if __name__ == "__main__":
    main()
