"""
Generate a clean markdown comparison table from benchmark results.

Reads benchmark_results.json and produces results.md with:
- Overall accuracy ranking
- Per-test-case breakdown
- Speed comparison
- Recommendations
"""

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"
OUTPUT_FILE = Path(__file__).parent / "results.md"


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def generate_markdown(data: list[dict]) -> str:
    lines = []
    lines.append("# New Models Benchmark Results")
    lines.append("")
    lines.append("Benchmark of ASR models for offline Quran verse identification.")
    lines.append("")

    # ── Overall Summary Table ────────────────────────────────────────────────
    lines.append("## Overall Ranking")
    lines.append("")
    lines.append(
        "| Rank | Model | Type | Size (MB) | Accuracy | Avg Score | Avg Time (s) | Load Time (s) |"
    )
    lines.append(
        "|------|-------|------|-----------|----------|-----------|--------------|---------------|"
    )

    # Compute summary stats
    summaries = []
    for entry in data:
        if entry.get("type") == "FAILED":
            summaries.append({
                "model": entry["model"],
                "type": "FAILED",
                "size_mb": "-",
                "correct": 0,
                "total": 0,
                "accuracy": "FAILED",
                "avg_score": 0,
                "avg_time": 0,
                "load_time": "-",
                "error": entry.get("error", "Unknown"),
            })
            continue

        results = entry["results"]
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        avg_time = sum(r["time_s"] for r in results) / max(total, 1)
        avg_score = sum(r["match_score"] for r in results) / max(total, 1)

        summaries.append({
            "model": entry["model"],
            "type": entry.get("type", "unknown"),
            "size_mb": entry.get("size_mb", "?"),
            "correct": correct,
            "total": total,
            "accuracy": f"{correct}/{total}",
            "avg_score": avg_score,
            "avg_time": avg_time,
            "load_time": entry.get("load_time_s", "-"),
        })

    # Sort: by correct (desc), then avg_score (desc), then avg_time (asc)
    summaries.sort(
        key=lambda x: (
            -(x["correct"] if isinstance(x["correct"], int) else -1),
            -(x["avg_score"] if isinstance(x["avg_score"], (int, float)) else 0),
            x["avg_time"] if isinstance(x["avg_time"], (int, float)) else 999,
        )
    )

    for rank, s in enumerate(summaries, 1):
        if s.get("error"):
            lines.append(
                f"| {rank} | {s['model']} | {s['type']} | - | FAILED | - | - | - |"
            )
        else:
            lines.append(
                f"| {rank} | {s['model']} | {s['type']} | {s['size_mb']} | "
                f"{s['accuracy']} | {s['avg_score']:.3f} | {s['avg_time']:.2f} | {s['load_time']} |"
            )

    lines.append("")

    # ── Per-Test-Case Breakdown ──────────────────────────────────────────────
    lines.append("## Per-Test-Case Results")
    lines.append("")

    # Collect all test case descriptions
    test_names = []
    for entry in data:
        if entry.get("type") != "FAILED":
            for r in entry["results"]:
                if r["test"] not in test_names:
                    test_names.append(r["test"])

    for test_name in test_names:
        lines.append(f"### {test_name}")
        lines.append("")
        lines.append("| Model | Correct? | Score | Time (s) | Transcription (clean) |")
        lines.append("|-------|----------|-------|----------|----------------------|")

        for entry in data:
            if entry.get("type") == "FAILED":
                lines.append(f"| {entry['model']} | FAILED | - | - | Error: {entry.get('error', '?')[:60]} |")
                continue

            for r in entry["results"]:
                if r["test"] == test_name:
                    status = "Yes" if r["correct"] else "No"
                    clean_text = r["transcription_clean"][:80]
                    # Escape pipes in markdown
                    clean_text = clean_text.replace("|", "\\|")
                    lines.append(
                        f"| {entry['model']} | {status} | {r['match_score']:.3f} | "
                        f"{r['time_s']:.2f} | {clean_text} |"
                    )

        lines.append("")

    # ── Speed Comparison ─────────────────────────────────────────────────────
    lines.append("## Speed Comparison")
    lines.append("")
    lines.append("| Model | Avg Inference (s) | Load Time (s) | Size (MB) |")
    lines.append("|-------|-------------------|---------------|-----------|")

    speed_sorted = sorted(
        [s for s in summaries if not s.get("error")],
        key=lambda x: x["avg_time"],
    )
    for s in speed_sorted:
        lines.append(
            f"| {s['model']} | {s['avg_time']:.2f} | {s['load_time']} | {s['size_mb']} |"
        )

    lines.append("")

    # ── Failed Models ────────────────────────────────────────────────────────
    failed = [s for s in summaries if s.get("error")]
    if failed:
        lines.append("## Failed Models")
        lines.append("")
        for s in failed:
            lines.append(f"- **{s['model']}**: {s['error']}")
        lines.append("")

    return "\n".join(lines)


def main():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found. Run benchmark.py first.")
        return

    data = load_results()
    md = generate_markdown(data)

    with open(OUTPUT_FILE, "w") as f:
        f.write(md)

    print(f"Results table written to {OUTPUT_FILE}")
    print()
    print(md)


if __name__ == "__main__":
    main()
