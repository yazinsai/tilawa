"""Standalone benchmark script — avoids the double-load issue in benchmark/runner.py."""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run import predict, model_size

CORPUS_DIR = PROJECT_ROOT / "benchmark" / "test_corpus"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


def score_sequence(expected, predicted):
    if not expected:
        return {"recall": 1.0, "precision": 1.0, "sequence_accuracy": 1.0}
    if not predicted:
        return {"recall": 0.0, "precision": 0.0, "sequence_accuracy": 0.0}

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
    precision = len(matched_pred_indices) / len(pred_tuples) if pred_tuples else 0
    seq_acc = 1.0 if pred_tuples == exp_tuples else 0.0
    return {"recall": recall, "precision": precision, "sequence_accuracy": seq_acc}


def predict_to_emissions(result):
    if not result or result.get("surah", 0) == 0:
        return []
    surah = result["surah"]
    ayah_start = result["ayah"]
    ayah_end = result.get("ayah_end") or ayah_start
    score = result.get("score", 0.0)
    return [{"surah": surah, "ayah": ayah, "score": score} for ayah in range(ayah_start, ayah_end + 1)]


def main():
    with open(CORPUS_DIR / "manifest.json") as f:
        samples = json.load(f)["samples"]

    print(f"Running fastconformer-nbest-bruteforce on {len(samples)} samples...")

    # Warmup
    predict(str(CORPUS_DIR / samples[0]["file"]))

    size = model_size()
    total_recall = 0.0
    total_precision = 0.0
    total_seq_acc = 0.0
    latencies = []
    per_sample = []

    for i, sample in enumerate(samples):
        audio_path = str(CORPUS_DIR / sample["file"])
        expected = sample.get("expected_verses", [{"surah": sample["surah"], "ayah": sample["ayah"]}])

        start = time.perf_counter()
        try:
            result = predict(audio_path)
            emissions = predict_to_emissions(result)
        except Exception as e:
            print(f"  Error on {sample['id']}: {e}")
            emissions = []
            result = {}
        elapsed = time.perf_counter() - start

        scores = score_sequence(expected, emissions)
        total_recall += scores["recall"]
        total_precision += scores["precision"]
        total_seq_acc += scores["sequence_accuracy"]
        latencies.append(elapsed)

        status = "OK" if scores["sequence_accuracy"] == 1.0 else "MISS"
        exp_str = f"{expected[0]['surah']}:{expected[0]['ayah']}" + (f"-{expected[-1]['ayah']}" if len(expected) > 1 else "")
        pred_str = f"{result.get('surah', 0)}:{result.get('ayah', 0)}" if result else "?"
        if result and result.get("ayah_end"):
            pred_str += f"-{result['ayah_end']}"
        print(f"  [{i+1:2d}/{len(samples)}] {sample['id']:30s} exp={exp_str:>10s} got={pred_str:>10s} ({elapsed:.2f}s) [{status}]")

        per_sample.append({
            "id": sample["id"],
            "expected": expected,
            "predicted": emissions,
            "recall": scores["recall"],
            "precision": scores["precision"],
            "sequence_accuracy": scores["sequence_accuracy"],
            "latency": elapsed,
        })

    n = len(samples)
    avg_lat = sum(latencies) / n

    print(f"\n{'='*60}")
    print(f"fastconformer-nbest-bruteforce")
    print(f"  Recall:    {total_recall/n:.0%}")
    print(f"  Precision: {total_precision/n:.0%}")
    print(f"  SeqAcc:    {total_seq_acc/n:.0%}")
    print(f"  Latency:   {avg_lat:.2f}s")
    print(f"  Size:      {size/(1024**3):.1f} GB")
    print(f"{'='*60}")

    # Save results in the same format as benchmark/runner.py
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_data = [{
        "name": "fastconformer-nbest-bruteforce",
        "recall": total_recall / n,
        "precision": total_precision / n,
        "sequence_accuracy": total_seq_acc / n,
        "total": n,
        "avg_latency": avg_lat,
        "model_size": size,
        "per_sample": per_sample,
    }]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
