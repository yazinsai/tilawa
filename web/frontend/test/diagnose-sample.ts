/**
 * Diagnostic script: runs a single sample through streaming mode with detailed
 * per-cycle logging to understand why certain samples produce empty results.
 *
 * Usage:
 *   tsx test/diagnose-sample.ts retasy_003
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, partialRatio } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import { ratio as levRatio } from "../src/lib/levenshtein.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  FIRST_MATCH_THRESHOLD,
  VERSE_MATCH_THRESHOLD,
  RAW_TRANSCRIPT_THRESHOLD,
} from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK = resolve(ROOT, "../../benchmark/test_corpus");
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const sampleId = process.argv[2];
if (!sampleId) {
  console.error("Usage: tsx test/diagnose-sample.ts <sample_id>");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Audio loading
// ---------------------------------------------------------------------------
function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
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
  // Load manifest
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  const sample = manifest.samples.find((s) => s.id === sampleId);
  if (!sample) {
    console.error(`Sample "${sampleId}" not found in manifest`);
    process.exit(1);
  }

  console.log(`=== DIAGNOSING: ${sample.id} ===`);
  console.log(`Expected: ${sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(", ")}`);
  console.log();

  // Load model
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  console.log("Loading ONNX model...");
  await createSession(modelPath);

  // Load vocab + decoder
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  const decoder = new CTCDecoder(vocabJson);

  // Load Quran data
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);

  // Expected verse info
  const expectedVerse = db.getVerse(sample.expected_verses[0].surah, sample.expected_verses[0].ayah);
  console.log(`Expected verse phonemes: "${expectedVerse?.phonemes_joined ?? "NOT FOUND"}"`);
  console.log();

  // Load audio
  const audioPath = resolve(BENCHMARK, sample.file);
  const audio = loadAudio(audioPath);
  const durationSec = (audio.length / SAMPLE_RATE).toFixed(1);
  console.log(`Audio: ${durationSec}s (${audio.length} samples)`);
  console.log();

  // --- Instrumented transcribe ---
  let transcribeCallCount = 0;

  async function transcribe(audioSlice: Float32Array): Promise<TranscribeResult> {
    transcribeCallCount++;
    const { features, timeFrames } = await computeMelSpectrogram(audioSlice);
    const numMels = 80;
    const { logprobs, timeSteps, vocabSize } = await runInference(features, numMels, timeFrames);
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

  // --- Instrumented DB wrapper ---
  // We wrap matchVerse to intercept results
  const origMatchVerse = db.matchVerse.bind(db);
  let lastMatchResult: Record<string, any> | null = null;

  db.matchVerse = function (
    text: string,
    threshold?: number,
    maxSpan?: number,
    hint?: [number, number] | null,
    returnTopK?: number,
  ): Record<string, any> | null {
    const result = origMatchVerse(text, threshold, maxSpan, hint, returnTopK);
    lastMatchResult = result;
    return result;
  };

  // --- Run streaming with cycle-by-cycle logging ---
  const tracker = new RecitationTracker(db, transcribe);
  const allMessages: WorkerOutbound[] = [];
  let cycleNum = 0;
  let hasEverMatched = false;
  let lastEmittedText = "";

  async function feedAndLog(chunk: Float32Array, label: string): Promise<void> {
    const prevCallCount = transcribeCallCount;
    const msgs = await tracker.feed(chunk);
    allMessages.push(...msgs);

    // Only log when transcribe was called (i.e. a cycle ran)
    if (transcribeCallCount > prevCallCount) {
      cycleNum++;
      const effectiveThreshold = hasEverMatched ? VERSE_MATCH_THRESHOLD : FIRST_MATCH_THRESHOLD;

      console.log(`--- Cycle ${cycleNum} (${label}) ---`);

      if (lastMatchResult) {
        const m = lastMatchResult;
        const topRunners = m.runners_up ?? [];
        console.log(`  Transcript: "${m.phonemes_joined ? '' : ''}"`);

        // We need the actual transcript text — get it from raw_transcript messages
        // Actually, the matchVerse doesn't store the input text. Let's get it another way.
        // The transcript is in raw_transcript messages or we stored it.
      }

      // Check messages emitted this cycle
      for (const msg of msgs) {
        if (msg.type === "raw_transcript") {
          console.log(`  Transcript: "${msg.text}"`);

          // Compute ratio to expected verse
          if (expectedVerse) {
            const ratioToExpected = levRatio(msg.text, expectedVerse.phonemes_joined);
            const ratioToExpectedNoBsm = expectedVerse.phonemes_joined_no_bsm
              ? levRatio(msg.text, expectedVerse.phonemes_joined_no_bsm)
              : 0;
            console.log(`  Ratio to expected (${sample.expected_verses[0].surah}:${sample.expected_verses[0].ayah}): ${ratioToExpected.toFixed(3)}${ratioToExpectedNoBsm > 0 ? ` (no-bsm: ${ratioToExpectedNoBsm.toFixed(3)})` : ""}`);
          }

          // Residual check
          if (lastEmittedText) {
            const residual = partialRatio(msg.text, lastEmittedText);
            console.log(`  Residual ratio: ${residual.toFixed(3)} (threshold: 0.7)`);
          }

          // Log match details
          if (lastMatchResult) {
            const m = lastMatchResult;
            console.log(`  Best match: ${m.surah}:${m.ayah} raw=${m.raw_score?.toFixed(3)} bonus=${m.bonus?.toFixed(3)} total=${m.score?.toFixed(3)}`);
            console.log(`  Effective threshold: ${effectiveThreshold}`);
            console.log(`  ${m.score >= effectiveThreshold ? "ABOVE" : "BELOW"} threshold`);

            const runners = m.runners_up ?? [];
            if (runners.length > 0) {
              console.log(`  Top runners:`);
              for (let i = 0; i < Math.min(runners.length, 3); i++) {
                const r = runners[i];
                console.log(`    ${i + 1}. ${r.surah}:${r.ayah} raw=${r.raw_score} bonus=${r.bonus} total=${r.score} "${r.phonemes_joined}..."`);
              }
            }
          } else {
            console.log(`  matchVerse returned null (below RAW_TRANSCRIPT_THRESHOLD=${RAW_TRANSCRIPT_THRESHOLD})`);
          }
        } else if (msg.type === "verse_match") {
          hasEverMatched = true;
          console.log(`  >>> VERSE MATCH: ${msg.surah}:${msg.ayah} confidence=${msg.confidence}`);
          lastEmittedText = db.getVerse(msg.surah, msg.ayah)?.phonemes_joined ?? "";
        } else if (msg.type === "word_progress") {
          console.log(`  TRACKING progress: ${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words}`);
        } else if (msg.type === "word_correction") {
          // skip for brevity
        }
      }

      // If no messages were emitted but transcribe ran, it means something was suppressed
      if (msgs.length === 0 && transcribeCallCount > prevCallCount) {
        console.log(`  [No messages emitted — suppressed by silence/residual/short-text check]`);
      }

      lastMatchResult = null;
    }
  }

  // Feed audio chunks
  console.log("=== FEEDING AUDIO ===\n");
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    const chunk = audio.slice(offset, end);
    const timeSec = (offset / SAMPLE_RATE).toFixed(1);
    await feedAndLog(chunk, `audio @${timeSec}s`);
  }

  // Feed tail silence
  console.log("\n=== FEEDING TAIL SILENCE ===\n");
  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    await feedAndLog(silenceChunk, `silence ${i + 1}/${silenceChunks}`);
  }

  // Summary
  console.log("\n=== SUMMARY ===");
  const discoveredVerses = allMessages
    .filter((m) => m.type === "verse_match")
    .map((m) => `${(m as any).surah}:${(m as any).ayah}`);
  const expectedStr = sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(", ");
  console.log(`Expected: [${expectedStr}]`);
  console.log(`Discovered: [${discoveredVerses.join(", ")}]`);
  console.log(`Total transcribe calls: ${transcribeCallCount}`);
  console.log(`Total cycles: ${cycleNum}`);

  // Final: do a full-audio transcription and match to show what full-file would get
  console.log("\n=== FULL-FILE BASELINE ===");
  const fullResult = await transcribe(audio);
  console.log(`Full transcript: "${fullResult.text}"`);
  if (expectedVerse) {
    const fullRatio = levRatio(fullResult.text, expectedVerse.phonemes_joined);
    console.log(`Full ratio to expected: ${fullRatio.toFixed(3)}`);
  }
  const fullMatch = db.matchVerse(fullResult.text, 0.25, 4, null, 5);
  if (fullMatch) {
    console.log(`Full match: ${fullMatch.surah}:${fullMatch.ayah} score=${fullMatch.score?.toFixed(3)}`);
    const runners = fullMatch.runners_up ?? [];
    for (let i = 0; i < Math.min(runners.length, 3); i++) {
      const r = runners[i];
      console.log(`  ${i + 1}. ${r.surah}:${r.ayah} raw=${r.raw_score} bonus=${r.bonus} total=${r.score}`);
    }
  } else {
    console.log("Full match: null (below threshold)");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
