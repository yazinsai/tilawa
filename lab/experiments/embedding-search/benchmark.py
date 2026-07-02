#!/usr/bin/env python3
"""Benchmark embedding search accuracy across models and test cases."""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from search import load_search_model, load_index, search_verse

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Test cases: reference audio (same reciter) and user recordings (different speaker)
TEST_CASES = [
    # User recordings (different speaker - the real test)
    {
        "file": str(DATA_DIR / "test_audio" / "nisa_51.m4a"),
        "expected_surah": 4,
        "expected_ayah": 51,
        "description": "An-Nisa:51 (user recording)",
        "category": "user",
    },
    {
        "file": str(DATA_DIR / "test_audio" / "ikhlas_2_3.m4a"),
        "expected_surah": 112,
        "expected_ayah": 2,
        "description": "Al-Ikhlas:2-3 (user recording, multi-verse)",
        "category": "user",
    },
    # Reference audio (same reciter - should match perfectly)
    {
        "file": str(DATA_DIR / "reference_audio" / "001001.mp3"),
        "expected_surah": 1,
        "expected_ayah": 1,
        "description": "Al-Fatiha:1 (ref)",
        "category": "reference",
    },
    {
        "file": str(DATA_DIR / "reference_audio" / "001002.mp3"),
        "expected_surah": 1,
        "expected_ayah": 2,
        "description": "Al-Fatiha:2 (ref)",
        "category": "reference",
    },
    {
        "file": str(DATA_DIR / "reference_audio" / "002255.mp3"),
        "expected_surah": 2,
        "expected_ayah": 255,
        "description": "Ayat al-Kursi (ref)",
        "category": "reference",
    },
    {
        "file": str(DATA_DIR / "reference_audio" / "112001.mp3"),
        "expected_surah": 112,
        "expected_ayah": 1,
        "description": "Al-Ikhlas:1 (ref)",
        "category": "reference",
    },
    {
        "file": str(DATA_DIR / "reference_audio" / "036001.mp3"),
        "expected_surah": 36,
        "expected_ayah": 1,
        "description": "Ya-Sin:1 (ref)",
        "category": "reference",
    },
]


def check_match(results: list[dict], expected_surah: int, expected_ayah: int, top_n: int) -> bool:
    """Check if the expected verse is in the top-N results."""
    for r in results[:top_n]:
        if r["surah"] == expected_surah and r["ayah"] == expected_ayah:
            return True
    return False


def check_surah_match(results: list[dict], expected_surah: int, top_n: int) -> bool:
    """Check if at least one result in top-N is from the correct surah."""
    for r in results[:top_n]:
        if r["surah"] == expected_surah:
            return True
    return False


def run_benchmark(model_key: str, reciter: str = "alafasy"):
    """Run benchmark for a single model/reciter combo."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_key} / {reciter}")
    print(f"{'='*70}")

    # Load model and index
    t0 = time.time()
    model, extractor, device = load_search_model(model_key)
    index, metadata = load_index(model_key, reciter)
    load_time = time.time() - t0
    print(f"Model + index loaded in {load_time:.1f}s")

    results_list = []

    for tc in TEST_CASES:
        if not Path(tc["file"]).exists():
            print(f"  SKIP: {tc['description']} (file not found)")
            results_list.append({
                **tc,
                "status": "skipped",
                "results": [],
                "embed_time_ms": 0,
                "search_time_ms": 0,
            })
            continue

        try:
            results, embed_time, search_time = search_verse(
                tc["file"], index, metadata, model, extractor, device, top_k=10
            )

            top1 = check_match(results, tc["expected_surah"], tc["expected_ayah"], 1)
            top3 = check_match(results, tc["expected_surah"], tc["expected_ayah"], 3)
            top5 = check_match(results, tc["expected_surah"], tc["expected_ayah"], 5)
            top10 = check_match(results, tc["expected_surah"], tc["expected_ayah"], 10)
            surah_top1 = check_surah_match(results, tc["expected_surah"], 1)
            surah_top5 = check_surah_match(results, tc["expected_surah"], 5)

            status_icon = "OK" if top1 else ("~" if top5 else "X")
            print(f"\n  [{status_icon}] {tc['description']}")
            print(f"      Expected: {tc['expected_surah']}:{tc['expected_ayah']}")
            print(f"      Top-1: {'YES' if top1 else 'no'} | Top-3: {'YES' if top3 else 'no'} | "
                  f"Top-5: {'YES' if top5 else 'no'} | Top-10: {'YES' if top10 else 'no'}")
            print(f"      Surah match: top-1={'YES' if surah_top1 else 'no'} top-5={'YES' if surah_top5 else 'no'}")
            print(f"      Time: embed={embed_time*1000:.0f}ms search={search_time*1000:.2f}ms")
            print(f"      Top-5 results:")
            for i, r in enumerate(results[:5], 1):
                marker = " <<<" if r["surah"] == tc["expected_surah"] and r["ayah"] == tc["expected_ayah"] else ""
                print(f"        #{i}: {r['surah_name_en']} {r['surah']}:{r['ayah']} "
                      f"(score={r['score']:.4f}){marker}")

            results_list.append({
                "description": tc["description"],
                "category": tc["category"],
                "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
                "top1_match": top1,
                "top3_match": top3,
                "top5_match": top5,
                "top10_match": top10,
                "surah_top1": surah_top1,
                "surah_top5": surah_top5,
                "embed_time_ms": round(embed_time * 1000, 1),
                "search_time_ms": round(search_time * 1000, 3),
                "total_time_ms": round((embed_time + search_time) * 1000, 1),
                "top5_results": [
                    {"verse": f"{r['surah']}:{r['ayah']}", "name": r["surah_name_en"], "score": round(r["score"], 4)}
                    for r in results[:5]
                ],
                "status": "ok",
            })

        except Exception as e:
            print(f"  [ERR] {tc['description']}: {e}")
            results_list.append({
                "description": tc["description"],
                "category": tc["category"],
                "expected": f"{tc['expected_surah']}:{tc['expected_ayah']}",
                "status": "error",
                "error": str(e),
            })

    return {
        "model": model_key,
        "reciter": reciter,
        "index_size": index.ntotal,
        "load_time_s": round(load_time, 1),
        "results": results_list,
    }


def compute_summary(benchmark_result: dict) -> dict:
    """Compute aggregate metrics from benchmark results."""
    valid = [r for r in benchmark_result["results"] if r.get("status") == "ok"]
    ref = [r for r in valid if r["category"] == "reference"]
    user = [r for r in valid if r["category"] == "user"]

    def acc(items, key):
        if not items:
            return 0
        return sum(1 for r in items if r.get(key, False)) / len(items) * 100

    def avg(items, key):
        vals = [r[key] for r in items if key in r]
        return round(sum(vals) / len(vals), 1) if vals else 0

    return {
        "model": benchmark_result["model"],
        "reciter": benchmark_result["reciter"],
        "index_size": benchmark_result["index_size"],
        "all": {
            "count": len(valid),
            "top1_acc": round(acc(valid, "top1_match"), 1),
            "top3_acc": round(acc(valid, "top3_match"), 1),
            "top5_acc": round(acc(valid, "top5_match"), 1),
            "top10_acc": round(acc(valid, "top10_match"), 1),
            "surah_top1_acc": round(acc(valid, "surah_top1"), 1),
            "surah_top5_acc": round(acc(valid, "surah_top5"), 1),
            "avg_embed_ms": avg(valid, "embed_time_ms"),
            "avg_search_ms": avg(valid, "search_time_ms"),
            "avg_total_ms": avg(valid, "total_time_ms"),
        },
        "reference": {
            "count": len(ref),
            "top1_acc": round(acc(ref, "top1_match"), 1),
            "top3_acc": round(acc(ref, "top3_match"), 1),
            "top5_acc": round(acc(ref, "top5_match"), 1),
        },
        "user": {
            "count": len(user),
            "top1_acc": round(acc(user, "top1_match"), 1),
            "top3_acc": round(acc(user, "top3_match"), 1),
            "top5_acc": round(acc(user, "top5_match"), 1),
            "surah_top1_acc": round(acc(user, "surah_top1"), 1),
            "surah_top5_acc": round(acc(user, "surah_top5"), 1),
        },
    }


def print_summary_table(summaries: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Model':<20} {'Index':<8} {'Category':<12} "
          f"{'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Surah@1':<8} {'Avg ms':<8}")
    print("-" * 80)

    for s in summaries:
        model_label = f"{s['model']}"
        print(f"{model_label:<20} {s['index_size']:<8} {'All':<12} "
              f"{s['all']['top1_acc']:>5.1f}%  {s['all']['top3_acc']:>5.1f}%  "
              f"{s['all']['top5_acc']:>5.1f}%  {s['all']['surah_top1_acc']:>5.1f}%  "
              f"{s['all']['avg_total_ms']:>6.0f}")
        print(f"{'':<20} {'':<8} {'Reference':<12} "
              f"{s['reference']['top1_acc']:>5.1f}%  {s['reference']['top3_acc']:>5.1f}%  "
              f"{s['reference']['top5_acc']:>5.1f}%  {'':>8} {'':>8}")
        print(f"{'':<20} {'':<8} {'User':<12} "
              f"{s['user']['top1_acc']:>5.1f}%  {s['user']['top3_acc']:>5.1f}%  "
              f"{s['user']['top5_acc']:>5.1f}%  {s['user'].get('surah_top1_acc', 0):>5.1f}%  {'':>8}")
        print()


def main():
    models_to_test = []
    reciter = "alafasy"

    # Check which indices exist
    indices_dir = EXPERIMENT_DIR / "indices"
    if indices_dir.exists():
        for model_key in ["hubert", "wav2vec2"]:
            index_file = indices_dir / f"{model_key}_{reciter}.index"
            if index_file.exists():
                models_to_test.append(model_key)

    if not models_to_test:
        print("No indices found. Run build_index.py first.")
        sys.exit(1)

    # Override from args
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_key = sys.argv[idx + 1]
        models_to_test = [model_key]

    all_benchmarks = []
    all_summaries = []

    for model_key in models_to_test:
        result = run_benchmark(model_key, reciter)
        summary = compute_summary(result)
        all_benchmarks.append(result)
        all_summaries.append(summary)

    print_summary_table(all_summaries)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": all_benchmarks,
        "summaries": all_summaries,
    }

    results_path = EXPERIMENT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate results.md
    generate_results_md(all_summaries, all_benchmarks)


def generate_results_md(summaries: list[dict], benchmarks: list[dict]):
    """Generate a markdown results file."""
    lines = ["# Embedding Search Benchmark Results\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## Summary\n")
    lines.append("| Model | Index Size | Category | Top-1 | Top-3 | Top-5 | Surah@1 | Avg Time |")
    lines.append("|-------|-----------|----------|-------|-------|-------|---------|----------|")

    for s in summaries:
        lines.append(f"| {s['model']} | {s['index_size']} | All | "
                     f"{s['all']['top1_acc']}% | {s['all']['top3_acc']}% | "
                     f"{s['all']['top5_acc']}% | {s['all']['surah_top1_acc']}% | "
                     f"{s['all']['avg_total_ms']}ms |")
        lines.append(f"| | | Reference | {s['reference']['top1_acc']}% | "
                     f"{s['reference']['top3_acc']}% | {s['reference']['top5_acc']}% | - | - |")
        lines.append(f"| | | User | {s['user']['top1_acc']}% | "
                     f"{s['user']['top3_acc']}% | {s['user']['top5_acc']}% | "
                     f"{s['user'].get('surah_top1_acc', 0)}% | - |")

    lines.append("\n## Detailed Results\n")

    for bench in benchmarks:
        lines.append(f"### {bench['model']} / {bench['reciter']}\n")
        for r in bench["results"]:
            if r.get("status") == "skipped":
                lines.append(f"- **{r['description']}**: SKIPPED (file not found)")
                continue
            if r.get("status") == "error":
                lines.append(f"- **{r['description']}**: ERROR - {r.get('error', 'unknown')}")
                continue

            match_str = "TOP-1" if r["top1_match"] else ("TOP-3" if r["top3_match"] else
                        ("TOP-5" if r["top5_match"] else "MISS"))
            lines.append(f"- **{r['description']}** (expected {r['expected']}): {match_str}")
            lines.append(f"  - Time: {r['total_time_ms']}ms (embed: {r['embed_time_ms']}ms, search: {r['search_time_ms']}ms)")
            for i, res in enumerate(r.get("top5_results", []), 1):
                marker = " **<<<**" if res["verse"] == r["expected"] else ""
                lines.append(f"  - #{i}: {res['name']} {res['verse']} (score: {res['score']}){marker}")
        lines.append("")

    results_md_path = EXPERIMENT_DIR / "results.md"
    with open(results_md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Results markdown saved to: {results_md_path}")


if __name__ == "__main__":
    main()
