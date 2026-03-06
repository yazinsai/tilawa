/**
 * Diagnose multi-verse streaming: timeline of all tracker state transitions.
 *
 * Usage:
 *   tsx test/diagnose-multiverse.ts multi_067_001_004
 *   tsx test/diagnose-multiverse.ts multi_113_001_005
 *   tsx test/diagnose-multiverse.ts              # runs both
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, partialRatio } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  TRACKING_TRIGGER_SAMPLES,
  TRACKING_SILENCE_SAMPLES,
  STALE_CYCLE_LIMIT,
  VERSE_MATCH_THRESHOLD,
  FIRST_MATCH_THRESHOLD,
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
const sampleFilter = process.argv[2];
const DEFAULT_SAMPLES = ["multi_067_001_004", "multi_113_001_005"];

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------
function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---------------------------------------------------------------------------
// Transcribe
// ---------------------------------------------------------------------------
let decoder: CTCDecoder;

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);
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
// Timeline logger
// ---------------------------------------------------------------------------
let cycleNum = 0;
let audioTimeSec = 0;

function ts(): string {
  return `[${audioTimeSec.toFixed(1)}s c${cycleNum}]`;
}

function log(tag: string, detail: string): void {
  console.log(`  ${ts()} ${tag.padEnd(14)} ${detail}`);
}

// ---------------------------------------------------------------------------
// Tracker wrapper that intercepts internal state via message inspection
// ---------------------------------------------------------------------------
interface Sample {
  id: string;
  file: string;
  expected_verses: { surah: number; ayah: number }[];
}

async function diagnoseSample(db: QuranDB, sample: Sample): Promise<void> {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`SAMPLE: ${sample.id}`);
  console.log(`Expected: ${sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`).join(", ")}`);
  console.log("=".repeat(70));

  const audioPath = resolve(BENCHMARK, sample.file);
  const audio = loadAudio(audioPath);
  console.log(`Audio: ${(audio.length / SAMPLE_RATE).toFixed(1)}s`);
  console.log(
    `Thresholds: FIRST=${FIRST_MATCH_THRESHOLD} VERSE=${VERSE_MATCH_THRESHOLD} RAW=${RAW_TRANSCRIPT_THRESHOLD}`,
  );
  console.log(
    `Triggers: discovery=${(TRIGGER_SAMPLES / SAMPLE_RATE).toFixed(1)}s tracking=${(TRACKING_TRIGGER_SAMPLES / SAMPLE_RATE).toFixed(1)}s`,
  );
  console.log(
    `Limits: stale_cycles=${STALE_CYCLE_LIMIT} silence=${(TRACKING_SILENCE_SAMPLES / SAMPLE_RATE).toFixed(1)}s`,
  );
  console.log();

  // We wrap the transcribe function to log what the tracker sees
  let transcribeCallNum = 0;
  let lastTranscript = "";

  // State tracking via message inspection
  let currentMode: "discovery" | "tracking" = "discovery";
  let trackingVerse: string | null = null;
  let trackingWordCount = 0;
  let lastWordIdx = -1;
  let hasEverMatched = false;
  let lastEmittedRef: string | null = null;

  const wrappedTranscribe = async (a: Float32Array): Promise<TranscribeResult> => {
    transcribeCallNum++;
    const result = await transcribe(a);
    lastTranscript = result.text;

    const truncText = result.text.length > 80 ? result.text.slice(0, 80) + "..." : result.text;
    const mode = currentMode === "tracking" ? `TRK[${trackingVerse}]` : "DISC";
    log("TRANSCRIBE", `#${transcribeCallNum} (${mode}) "${truncText}"`);

    // In discovery mode, show what matchVerse would return
    if (currentMode === "discovery") {
      const lastRef = lastEmittedRef
        ? (lastEmittedRef.split(":").map(Number) as [number, number])
        : null;

      // Check residual
      if (lastEmittedRef) {
        const emittedVerse = db.getVerse(lastRef![0], lastRef![1]);
        if (emittedVerse) {
          const residual = partialRatio(result.text, emittedVerse.phonemes_joined);
          log("  residual", `vs ${lastEmittedRef}: ${residual.toFixed(3)} (skip if >0.7)`);
        }
      }

      const match = db.matchVerse(result.text, RAW_TRANSCRIPT_THRESHOLD, 4, lastRef, 5);
      const threshold = hasEverMatched ? VERSE_MATCH_THRESHOLD : FIRST_MATCH_THRESHOLD;
      if (match) {
        const ayahEnd = match.ayah_end ? `-${match.ayah_end}` : "";
        log(
          "  match",
          `${match.surah}:${match.ayah}${ayahEnd} score=${match.score.toFixed(3)} raw=${match.raw_score.toFixed(3)} bonus=${match.bonus.toFixed(3)} threshold=${threshold}`,
        );
        if (match.runners_up) {
          for (const ru of match.runners_up.slice(0, 3)) {
            log("  runner", `${ru.surah}:${ru.ayah} score=${ru.score} raw=${ru.raw_score}`);
          }
        }
        if (match.score < threshold) {
          log("  verdict", `BELOW threshold (${match.score.toFixed(3)} < ${threshold})`);
        }
      } else {
        log("  match", `none above ${RAW_TRANSCRIPT_THRESHOLD}`);
      }
    }

    return result;
  };

  const tracker = new RecitationTracker(db, wrappedTranscribe);

  cycleNum = 0;
  audioTimeSec = 0;

  // Feed audio chunks
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    const chunk = audio.slice(offset, end);
    audioTimeSec = end / SAMPLE_RATE;
    cycleNum++;

    const msgs = await tracker.feed(chunk);

    for (const msg of msgs) {
      switch (msg.type) {
        case "verse_match": {
          const prevMode = currentMode;
          log("VERSE_MATCH", `${msg.surah}:${msg.ayah} conf=${msg.confidence}`);
          currentMode = "tracking";
          const verse = db.getVerse(msg.surah, msg.ayah);
          trackingVerse = `${msg.surah}:${msg.ayah}`;
          trackingWordCount = verse?.phoneme_words.length ?? 0;
          lastWordIdx = -1;
          hasEverMatched = true;
          lastEmittedRef = trackingVerse;
          log("ENTER_TRACK", `${trackingVerse} (${trackingWordCount} words)`);
          break;
        }
        case "word_progress": {
          lastWordIdx = msg.word_index - 1; // word_index is 1-based position
          const pct = ((msg.word_index / msg.total_words) * 100).toFixed(0);
          log(
            "PROGRESS",
            `${msg.surah}:${msg.ayah} word ${msg.word_index}/${msg.total_words} (${pct}%)`,
          );
          break;
        }
        case "word_correction":
          // skip for brevity
          break;
        case "raw_transcript": {
          const truncText =
            msg.text.length > 60 ? msg.text.slice(0, 60) + "..." : msg.text;
          log("RAW_TRANS", `"${truncText}" conf=${msg.confidence}`);
          break;
        }
      }
    }

    // Detect tracking exit: if we were tracking and got no messages, and tracker
    // goes back to discovery on next cycle, we detect it by checking if a
    // verse_match without prior tracking exit occurred. Instead, we detect
    // stale/silence exits by the absence of progress + next cycle being discovery.
    // We do this heuristically: if messages is empty and we were tracking,
    // check if the tracker has exited by seeing if next feed goes to discovery.
    // Actually, we can detect exit more reliably: if we were in tracking mode
    // and the returned messages are empty (no progress, no match), and the
    // tracker internally reset, the next feed cycle will be in discovery mode.
    // We'll detect the transition when we see a raw_transcript or verse_match
    // from discovery after being in tracking.

    // Detect exit: if tracking and we got no verse_match but we had been tracking,
    // check if tracker is no longer tracking by looking at returned message pattern.
    // The most reliable way: if we get 0 messages while tracking, it could be
    // (a) waiting for trigger, (b) stale exit happened (no messages returned on exit),
    // or (c) silence exit. We can detect stale exit by the fact that on exit,
    // _exitTracking returns no messages, and then next cycle will be discovery.
    // Since _exitTracking doesn't emit messages, we detect it retroactively.

    if (
      currentMode === "tracking" &&
      msgs.length === 0 &&
      // No progress for a while - might have just exited
      false
    ) {
      // We can't reliably detect exit without access to internal state.
      // We'll infer it when the mode changes via message types.
    }

    // Detect mode change: if we get a raw_transcript while in tracking mode,
    // the tracker must have exited tracking and gone back to discovery.
    for (const msg of msgs) {
      if (msg.type === "raw_transcript" && currentMode === "tracking") {
        const progress =
          trackingWordCount > 0
            ? (((lastWordIdx + 1) / trackingWordCount) * 100).toFixed(0)
            : "0";
        log(
          "EXIT_TRACK",
          `${trackingVerse} (inferred from raw_transcript, progress=${progress}%, hasEverMatched=${hasEverMatched})`,
        );
        currentMode = "discovery";
        trackingVerse = null;
      }
    }

    // Also: if a new verse_match arrives while we were tracking a different verse,
    // that means auto-advance happened (verse complete -> enter next).
    // We handle that in the verse_match case above already.
  }

  // Feed tail silence
  console.log(`\n  --- Feeding ${TAIL_SILENCE_SECONDS}s tail silence ---`);
  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    audioTimeSec += CHUNK_SECONDS;
    cycleNum++;
    const msgs = await tracker.feed(silenceChunk);
    for (const msg of msgs) {
      if (msg.type === "verse_match") {
        log("VERSE_MATCH", `${msg.surah}:${msg.ayah} conf=${msg.confidence} (during silence)`);
      } else if (msg.type === "raw_transcript") {
        if (currentMode === "tracking") {
          const progress =
            trackingWordCount > 0
              ? (((lastWordIdx + 1) / trackingWordCount) * 100).toFixed(0)
              : "0";
          log(
            "EXIT_TRACK",
            `${trackingVerse} (silence phase, progress=${progress}%)`,
          );
          currentMode = "discovery";
        }
        log("RAW_TRANS", `"${msg.text.slice(0, 60)}" conf=${msg.confidence}`);
      }
    }
  }

  // If still tracking at the end
  if (currentMode === "tracking") {
    const progress =
      trackingWordCount > 0
        ? (((lastWordIdx + 1) / trackingWordCount) * 100).toFixed(0)
        : "0";
    log("STILL_TRACK", `${trackingVerse} progress=${progress}% at end`);
  }

  console.log();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log("Loading model...");
  await createSession(resolve(ROOT, "public/fastconformer_phoneme_q8.onnx"));

  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  console.log(`Loaded ${db.totalVerses} verses`);

  const manifest = JSON.parse(readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"));
  const ids = sampleFilter ? [sampleFilter] : DEFAULT_SAMPLES;

  for (const id of ids) {
    const sample = manifest.samples.find((s: any) => s.id === id);
    if (!sample) {
      console.error(`Sample "${id}" not found in manifest`);
      continue;
    }
    await diagnoseSample(db, sample);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
