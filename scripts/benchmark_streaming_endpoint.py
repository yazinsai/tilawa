"""
Benchmark the live WebSocket streaming endpoint against the test corpus.

Usage:
    python scripts/benchmark_streaming_endpoint.py
    python scripts/benchmark_streaming_endpoint.py --sample multi_113_001_005
    python scripts/benchmark_streaming_endpoint.py --category multi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runner import score_sequence
from shared.audio import load_audio

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4800  # 300 ms, matching the frontend/web test client
TAIL_SILENCE_SECONDS = 4.0
TAIL_SILENCE_CHUNKS = int((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES)


def _load_manifest() -> list[dict]:
    manifest_path = PROJECT_ROOT / "benchmark" / "test_corpus" / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)["samples"]


async def _collect_messages(
    ws,
    idle_timeout: float,
) -> list[dict]:
    messages: list[dict] = []
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
        except asyncio.TimeoutError:
            break
        except websockets.ConnectionClosed:
            break
        messages.append(json.loads(raw))
    return messages


async def _run_sample(
    sample: dict,
    endpoint: str,
    idle_timeout: float,
) -> dict:
    audio_path = PROJECT_ROOT / "benchmark" / "test_corpus" / sample["file"]
    audio = load_audio(str(audio_path), sr=SAMPLE_RATE)
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    async with websockets.connect(
        endpoint,
        max_size=None,
        ping_interval=None,
    ) as ws:
        recv_task = asyncio.create_task(_collect_messages(ws, idle_timeout=idle_timeout))

        for offset in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[offset: offset + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            await ws.send(chunk.astype(np.float32, copy=False).tobytes())

        # The server only flushes after additional receive cycles, so pad with silence.
        for _ in range(TAIL_SILENCE_CHUNKS):
            await ws.send(silence.tobytes())

        messages = await recv_task

    predicted = [
        {"surah": msg["surah"], "ayah": msg["ayah"]}
        for msg in messages
        if msg.get("type") == "verse_match"
    ]
    expected = sample.get(
        "expected_verses",
        [{"surah": sample["surah"], "ayah": sample["ayah"]}],
    )
    metrics = score_sequence(expected, predicted)

    return {
        "id": sample["id"],
        "file": sample["file"],
        "expected": expected,
        "predicted": predicted,
        "metrics": metrics,
        "messages": messages,
    }


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--idle-timeout", type=float, default=3.0)
    args = parser.parse_args()

    samples = _load_manifest()
    if args.sample:
        samples = [s for s in samples if s["id"] == args.sample]
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if not samples:
        raise SystemExit("No samples matched the requested filters")

    results = []
    for idx, sample in enumerate(samples, start=1):
        result = await _run_sample(
            sample=sample,
            endpoint=args.endpoint,
            idle_timeout=args.idle_timeout,
        )
        results.append(result)
        metrics = result["metrics"]
        print(
            f"[{idx}/{len(samples)}] {sample['id']}: "
            f"recall={metrics['recall']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"seq={metrics['sequence_accuracy']:.3f} "
            f"pred={result['predicted']}"
        )

    summary = {
        "count": len(results),
        "recall": sum(r["metrics"]["recall"] for r in results) / len(results),
        "precision": sum(r["metrics"]["precision"] for r in results) / len(results),
        "sequence_accuracy": sum(r["metrics"]["sequence_accuracy"] for r in results) / len(results),
    }
    print("\nSummary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
