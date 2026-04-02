/**
 * Streaming validation using the exact same TypeScript code as the browser.
 *
 * Usage:
 *   tsx test/validate-streaming.ts              # streaming (300ms chunks)
 *   tsx test/validate-streaming.ts --no-streaming  # full-file baseline
 *   tsx test/validate-streaming.ts --sample ref_001001  # single sample
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);

// After audio ends, feed this many seconds of near-silence to flush discovery
const TAIL_SILENCE_SECONDS = 4.0;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
const noStreaming = args.includes("--no-streaming");
const corpusArg = args.find((a) => a.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus";
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);
const sampleFilter = args.find((a) => !a.startsWith("--"));

// ---------------------------------------------------------------------------
// Audio loading via ffmpeg
// ---------------------------------------------------------------------------
function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---------------------------------------------------------------------------
// Transcribe function (same as inference.ts)
// ---------------------------------------------------------------------------
let decoder: CTCDecoder;

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    numMels,
    timeFrames,
  );
  return {
    ...decoder.decode(logprobs, timeSteps, vocabSize),
    acoustic: {
      logprobs,
      timeSteps,
      vocabSize,
      blankId: decoder.getBlankId(),
    },
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
interface Sample {
  id: string;
  file: string;
  surah: number;
  ayah: number;
  ayah_end: number | null;
  category: string;
  source: string;
  expected_verses: { surah: number; ayah: number }[];
}

async function main() {
  console.log(noStreaming ? "=== NON-STREAMING MODE ===" : "=== STREAMING MODE ===");
  console.log();

  // 1. Load model
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  console.log("Loading ONNX model...");
  await createSession(modelPath);

  // 2. Load vocab
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  // 3. Load Quran data
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  console.log(`Loaded ${db.totalVerses} verses from ${db.surahCount} surahs`);

  // 4. Load manifest
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );

  let samples = manifest.samples;
  if (sampleFilter) {
    samples = samples.filter((s) => s.id === sampleFilter);
    if (!samples.length) {
      console.error(`Sample "${sampleFilter}" not found`);
      process.exit(1);
    }
  }

  console.log(`Testing ${samples.length} samples\n`);

  // 5. Run validation
  let correct = 0;
  let total = 0;
  let totalRecall = 0;
  let totalPrecision = 0;
  let totalSeqAcc = 0;
  const failures: string[] = [];

  for (const sample of samples) {
    total++;
    const audioPath = resolve(BENCHMARK, sample.file);
    const audio = loadAudio(audioPath);
    const durationSec = (audio.length / SAMPLE_RATE).toFixed(1);

    const discoveredVerses: { surah: number; ayah: number }[] = [];

    if (noStreaming) {
      // Full-file: transcribe entire audio, match once
      const result = await transcribe(audio);
      const match = db.matchVerse(result.text, 0.25, 4, null, 0);
      if (match && match.score >= 0.45) {
        const ayahEnd = match.ayah_end ?? match.ayah;
        for (let a = match.ayah; a <= ayahEnd; a++) {
          discoveredVerses.push({ surah: match.surah, ayah: a });
        }
      }
    } else {
      // Streaming: feed chunks through RecitationTracker
      const tracker = new RecitationTracker(db, transcribe);
      const messages: WorkerOutbound[] = [];

      // Feed audio in chunks
      for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
        const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
        const chunk = audio.slice(offset, end);
        const msgs = await tracker.feed(chunk);
        messages.push(...msgs);
      }

      // Feed tail silence to let tracker complete discovery cycle
      const silenceChunk = new Float32Array(CHUNK_SAMPLES); // zeros = silence
      const silenceChunks = Math.ceil(
        (TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES,
      );
      for (let i = 0; i < silenceChunks; i++) {
        const msgs = await tracker.feed(silenceChunk);
        messages.push(...msgs);
      }

      // Collect discovered verses
      for (const msg of messages) {
        if (msg.type === "verse_match") {
          // Avoid duplicates
          const exists = discoveredVerses.some(
            (v) => v.surah === msg.surah && v.ayah === msg.ayah,
          );
          if (!exists) {
            discoveredVerses.push({ surah: msg.surah, ayah: msg.ayah });
          }
        }
      }
    }

    // Check: did we discover all expected verses?
    const expectedSet = new Set(
      sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`),
    );
    const discoveredSet = new Set(
      discoveredVerses.map((v) => `${v.surah}:${v.ayah}`),
    );

    // Per-sample metrics
    const matched = [...expectedSet].filter((v) => discoveredSet.has(v)).length;
    const sampleRecall = expectedSet.size > 0 ? matched / expectedSet.size : 1.0;
    const samplePrecision = discoveredSet.size > 0 ? matched / discoveredSet.size : 1.0;
    const sampleSeqAcc =
      expectedSet.size === discoveredSet.size &&
      [...expectedSet].every((v) => discoveredSet.has(v))
        ? 1.0
        : 0.0;

    totalRecall += sampleRecall;
    totalPrecision += samplePrecision;
    totalSeqAcc += sampleSeqAcc;

    // A sample is "correct" if all expected verses were discovered
    const allFound = [...expectedSet].every((v) => discoveredSet.has(v));

    if (allFound) {
      correct++;
      console.log(
        `  PASS  ${sample.id} (${durationSec}s) — found ${[...discoveredSet].join(", ")}`,
      );
    } else {
      const detail = `  FAIL  ${sample.id} (${durationSec}s) — expected [${[...expectedSet].join(", ")}] got [${[...discoveredSet].join(", ")}]`;
      console.log(detail);
      failures.push(detail);
    }
  }

  // 6. Summary
  console.log("\n" + "=".repeat(60));
  const pct = total > 0 ? ((correct / total) * 100).toFixed(1) : "0.0";
  const avgRecall = total > 0 ? (totalRecall / total) : 0;
  const avgPrecision = total > 0 ? (totalPrecision / total) : 0;
  const avgSeqAcc = total > 0 ? (totalSeqAcc / total) : 0;
  console.log(`Result: ${correct}/${total} (${pct}%)`);
  console.log(`Recall: ${(avgRecall * 100).toFixed(1)}%  Precision: ${(avgPrecision * 100).toFixed(1)}%  SeqAcc: ${(avgSeqAcc * 100).toFixed(1)}%`);

  if (failures.length > 0) {
    console.log(`\nFailures (${failures.length}):`);
    for (const f of failures) {
      console.log(f);
    }
  }

  console.log();
}

main()
  .then(() => {
    // Force clean exit — ONNX runtime can SIGABRT during cleanup
    process.exit(0);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
