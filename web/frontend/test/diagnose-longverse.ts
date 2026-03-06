/**
 * Diagnostic script for long verse matching failures in streaming mode.
 *
 * Usage:
 *   tsx test/diagnose-longverse.ts ref_002255
 *   tsx test/diagnose-longverse.ts ref_048029
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, partialRatio } from "../src/lib/quran-db.ts";
import { ratio } from "../src/lib/levenshtein.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import { createSession, runInference } from "./session-node.ts";
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  MAX_WINDOW_SAMPLES,
  TRACKING_TRIGGER_SAMPLES,
  TRACKING_MAX_WINDOW_SAMPLES,
  SILENCE_RMS_THRESHOLD,
  VERSE_MATCH_THRESHOLD,
  FIRST_MATCH_THRESHOLD,
  RAW_TRANSCRIPT_THRESHOLD,
  STALE_CYCLE_LIMIT,
} from "../src/lib/types.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK = resolve(ROOT, "../../benchmark/test_corpus");
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
const sampleId = process.argv[2];
if (!sampleId) {
  console.error("Usage: tsx test/diagnose-longverse.ts <sample_id>");
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
// bestSpanRatio: scores text against best contiguous N-word chunk of verse
// ---------------------------------------------------------------------------
function bestSpanRatio(text: string, verseText: string): number {
  // Word-level span (original spec) — works when text has word boundaries
  const textWords = text.split(" ");
  const verseWords = verseText.split(" ");
  if (textWords.length >= 2) {
    let best = 0;
    const spanLen = Math.min(textWords.length + 3, verseWords.length);
    for (let i = 0; i <= verseWords.length - spanLen; i++) {
      const span = verseWords.slice(i, i + spanLen).join(" ");
      best = Math.max(best, ratio(text, span));
    }
    return best;
  }

  // Character-level fallback: slide a window of text.length + padding over verseText
  if (text.length < 3) return 0;
  let best = 0;
  // Remove spaces from verseText to match the no-space transcription
  const versePlain = verseText.replace(/ /g, "");
  const window = Math.min(text.length + 10, versePlain.length);
  for (let i = 0; i <= versePlain.length - window; i++) {
    const span = versePlain.slice(i, i + window);
    best = Math.max(best, ratio(text, span));
  }
  return best;
}

// ---------------------------------------------------------------------------
// Transcribe
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
// Manifest
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  // Load model
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  console.log("Loading ONNX model...");
  await createSession(modelPath);

  // Load vocab
  const vocabJson = JSON.parse(
    readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"),
  );
  decoder = new CTCDecoder(vocabJson);

  // Load Quran data
  const quranData = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"),
  );
  const db = new QuranDB(quranData, decoder);

  // Load manifest and find sample
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  const sample = manifest.samples.find((s) => s.id === sampleId);
  if (!sample) {
    console.error(`Sample "${sampleId}" not found in manifest`);
    process.exit(1);
  }

  const expected = sample.expected_verses[0];
  const expectedVerse = db.getVerse(expected.surah, expected.ayah);
  if (!expectedVerse) {
    console.error(`Expected verse ${expected.surah}:${expected.ayah} not found in DB`);
    process.exit(1);
  }

  console.log(`\n=== DIAGNOSING ${sampleId} ===`);
  console.log(`Expected: ${expected.surah}:${expected.ayah}`);
  console.log(`Expected phonemes (first 120): ${expectedVerse.phonemes_joined.slice(0, 120)}`);
  console.log(`Expected phoneme word count: ${expectedVerse.phoneme_words.length}`);
  console.log(`Expected phonemes_joined length: ${expectedVerse.phonemes_joined.length} chars`);
  console.log();

  // Load audio
  const audioPath = resolve(BENCHMARK, sample.file);
  const audio = loadAudio(audioPath);
  const durationSec = (audio.length / SAMPLE_RATE).toFixed(1);
  console.log(`Audio duration: ${durationSec}s (${audio.length} samples)`);
  console.log(`MAX_WINDOW_SAMPLES: ${MAX_WINDOW_SAMPLES} (${(MAX_WINDOW_SAMPLES / SAMPLE_RATE).toFixed(1)}s)`);
  console.log(`TRIGGER_SAMPLES: ${TRIGGER_SAMPLES} (${(TRIGGER_SAMPLES / SAMPLE_RATE).toFixed(1)}s)`);
  console.log();

  // Simulate streaming — manually replicate tracker logic for diagnostics
  let fullAudio = new Float32Array(0);
  let newAudioCount = 0;
  let cycleNum = 0;
  let isTracking = false;
  let trackingVerse: string | null = null;

  // Track best scores seen across all cycles
  let bestRatio = 0;
  let bestPartial = 0;
  let bestSpan = 0;
  let bestRatioCycle = 0;
  let bestPartialCycle = 0;
  let bestSpanCycle = 0;
  let correctInTop3Count = 0;
  let totalCycles = 0;

  const header =
    "Cycle | Window(s) |  Mode     | Top1 Ref   Top1 Score | Top2 Ref   Top2 Score | Top3 Ref   Top3 Score | Exp ratio | Exp partial | Exp span  | Emitted?";
  console.log(header);
  console.log("-".repeat(header.length));

  function concatF32(a: Float32Array, b: Float32Array): Float32Array {
    const r = new Float32Array(a.length + b.length);
    r.set(a);
    r.set(b, a.length);
    return r;
  }

  // Feed audio chunks
  const totalChunks =
    Math.ceil(audio.length / CHUNK_SAMPLES) +
    Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);

  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    let chunk: Float32Array;
    const audioOffset = chunkIdx * CHUNK_SAMPLES;
    if (audioOffset < audio.length) {
      const end = Math.min(audioOffset + CHUNK_SAMPLES, audio.length);
      chunk = audio.slice(audioOffset, end);
    } else {
      chunk = new Float32Array(CHUNK_SAMPLES); // silence
    }

    fullAudio = concatF32(fullAudio, chunk);
    newAudioCount += chunk.length;

    // Trim to max window
    const maxSamples = isTracking
      ? TRACKING_MAX_WINDOW_SAMPLES
      : MAX_WINDOW_SAMPLES;
    if (fullAudio.length > maxSamples) {
      fullAudio = fullAudio.slice(-maxSamples);
    }

    // Check trigger threshold
    const triggerThreshold = isTracking
      ? TRACKING_TRIGGER_SAMPLES
      : TRIGGER_SAMPLES;
    if (newAudioCount < triggerThreshold) continue;
    newAudioCount = 0;

    // Check silence
    const tail = fullAudio.slice(-TRIGGER_SAMPLES);
    let sumSq = 0;
    for (let i = 0; i < tail.length; i++) sumSq += tail[i] * tail[i];
    const rms = Math.sqrt(sumSq / tail.length);
    if (rms < SILENCE_RMS_THRESHOLD) continue;

    // Transcribe
    const { text } = await transcribe(fullAudio.slice());
    if (!text || text.trim().length < 5) continue;

    cycleNum++;
    totalCycles++;

    const windowSec = (fullAudio.length / SAMPLE_RATE).toFixed(1);
    const audioTimeSec = (
      Math.min((chunkIdx + 1) * CHUNK_SAMPLES, audio.length) / SAMPLE_RATE
    ).toFixed(1);

    // Get top 3 from matchVerse
    const match = db.matchVerse(text, RAW_TRANSCRIPT_THRESHOLD, 4, null, 5);

    const top3: { ref: string; score: number }[] = [];
    if (match) {
      top3.push({
        ref: `${match.surah}:${match.ayah}${match.ayah_end ? `-${match.ayah_end}` : ""}`,
        score: match.score,
      });
      const runners = match.runners_up ?? [];
      for (let i = 0; i < Math.min(runners.length, 2); i++) {
        top3.push({
          ref: `${runners[i].surah}:${runners[i].ayah}`,
          score: runners[i].score,
        });
      }
    }

    // Compute scores for the expected verse specifically
    const expRatio = ratio(text, expectedVerse.phonemes_joined);
    const expPartial = partialRatio(text, expectedVerse.phonemes_joined);
    const expSpan = bestSpanRatio(text, expectedVerse.phonemes_joined);

    // Track best scores
    if (expRatio > bestRatio) {
      bestRatio = expRatio;
      bestRatioCycle = cycleNum;
    }
    if (expPartial > bestPartial) {
      bestPartial = expPartial;
      bestPartialCycle = cycleNum;
    }
    if (expSpan > bestSpan) {
      bestSpan = expSpan;
      bestSpanCycle = cycleNum;
    }

    // Check if correct verse is in top 3
    const correctInTop3 = top3.some((t) => {
      const parts = t.ref.split(/[-:]/);
      return (
        parseInt(parts[0]) === expected.surah &&
        parseInt(parts[1]) === expected.ayah
      );
    });
    if (correctInTop3) correctInTop3Count++;

    // Would this cycle emit a match?
    const effectiveThreshold = FIRST_MATCH_THRESHOLD; // conservative, first match
    const emitted = match && match.score >= effectiveThreshold ? "YES" : "no";

    // Determine mode label
    const mode = isTracking ? `TRACK(${trackingVerse})` : "DISCOVERY";

    // Format top3
    const t1 = top3[0]
      ? `${top3[0].ref.padEnd(10)} ${top3[0].score.toFixed(3)}`
      : "---        ---  ";
    const t2 = top3[1]
      ? `${top3[1].ref.padEnd(10)} ${top3[1].score.toFixed(3)}`
      : "---        ---  ";
    const t3 = top3[2]
      ? `${top3[2].ref.padEnd(10)} ${top3[2].score.toFixed(3)}`
      : "---        ---  ";

    console.log(
      `${String(cycleNum).padStart(5)} | ${windowSec.padStart(7)}s | ${mode.padEnd(9)} | ${t1} | ${t2} | ${t3} | ${expRatio.toFixed(3).padStart(9)} | ${expPartial.toFixed(3).padStart(11)} | ${expSpan.toFixed(3).padStart(9)} | ${emitted}`,
    );

    // Print transcribed text (first 100 chars) on a sub-line
    console.log(
      `        text@${audioTimeSec}s: "${text.slice(0, 100)}${text.length > 100 ? "..." : ""}"`,
    );
  }

  // ---------------------------------------------------------------------------
  // Final analysis
  // ---------------------------------------------------------------------------
  console.log("\n" + "=".repeat(80));
  console.log("ANALYSIS SUMMARY");
  console.log("=".repeat(80));
  console.log(
    `1. Correct verse (${expected.surah}:${expected.ayah}) appeared in top 3: ${correctInTop3Count}/${totalCycles} cycles`,
  );
  console.log(
    `2. Best ratio score for correct verse: ${bestRatio.toFixed(4)} (cycle ${bestRatioCycle})`,
  );
  console.log(
    `   Best partialRatio for correct verse: ${bestPartial.toFixed(4)} (cycle ${bestPartialCycle})`,
  );
  console.log(
    `   Best bestSpanRatio for correct verse: ${bestSpan.toFixed(4)} (cycle ${bestSpanCycle})`,
  );
  console.log(
    `3. The core issue: MAX_WINDOW is ${(MAX_WINDOW_SAMPLES / SAMPLE_RATE).toFixed(1)}s but the verse is ${durationSec}s long.`,
  );
  console.log(
    `   ratio() compares the FULL verse text (${expectedVerse.phonemes_joined.length} chars) against ` +
      `a window that can only capture ~${(MAX_WINDOW_SAMPLES / SAMPLE_RATE).toFixed(1)}s of audio.`,
  );
  console.log(
    `   This means at best ~${((MAX_WINDOW_SAMPLES / SAMPLE_RATE / parseFloat(durationSec)) * 100).toFixed(0)}% of the verse phonemes appear in the transcript.`,
  );
  console.log(`   ratio() penalizes the length mismatch severely.`);
  console.log(
    `4. partialRatio slides a window over the longer string, so it should score higher.`,
  );
  console.log(
    `   bestSpanRatio slides a word-window of transcript-length over the verse, also addressing length mismatch.`,
  );
  console.log(
    `   If partialRatio or bestSpanRatio >> ratio, then a partial/span scorer would fix long-verse matching.`,
  );
  console.log();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
