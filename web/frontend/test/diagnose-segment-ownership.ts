/**
 * Diagnose streaming tracker "segment ownership" failures.
 *
 * This is intentionally diagnostic-only: it runs the same 300ms browser-style
 * stream as stability-report.ts, but records commit/tracking/stale events with
 * an approximate audio sample cursor so we can see whether extra verse commits
 * are coming from fresh audio or retained-tail replay.
 *
 * Usage:
 *   npx tsx test/diagnose-segment-ownership.ts --corpus=test_corpus_v3
 *   npx tsx test/diagnose-segment-ownership.ts --sample=ea_husary_002177 --json=test/segown-diag.json
 */

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { beamSearchDecode } from "../src/worker/beam-decode.ts";
import { buildTrie, type CompactTrie } from "../src/lib/phoneme-trie.ts";
import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker, type BeamVerseMatch, type TrackerDiagnosticEvent, type TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

const DEFAULT_SAMPLES = [
  "ea_husary_002177",
  "ea_alafasy_002282",
  "ea_husary_005003",
  "tlog_m000_100_001",
  "ea_husary_003008",
  "tlog_m020_010_105",
  "ea_husary_033006",
  "ea_alafasy_multi_073_001_005",
];

const args = process.argv.slice(2);
const corpusArg = args.find((a) => a.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus_v3";
const sampleArg = args.find((a) => a.startsWith("--sample="));
const sampleFilter = sampleArg
  ? new Set(sampleArg.split("=")[1].split(",").map((s) => s.trim()).filter(Boolean))
  : new Set(DEFAULT_SAMPLES);
const jsonArg = args.find((a) => a.startsWith("--json="));
const jsonOutPath = jsonArg ? jsonArg.split("=")[1] : null;
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

interface Sample {
  id: string;
  file: string;
  category: string;
  source: string;
  expected_verses: { surah: number; ayah: number }[];
}

interface TimedDiagnostic {
  sampleCursor: number;
  seconds: number;
  event: TrackerDiagnosticEvent;
}

interface TimedOutbound {
  sampleCursor: number;
  seconds: number;
  message: WorkerOutbound;
}

interface SampleDiagnostic {
  id: string;
  category: string;
  source: string;
  expected: string[];
  emitted: string[];
  exact: boolean;
  diagnostics: TimedDiagnostic[];
  messages: TimedOutbound[];
}

let decoder: CTCDecoder;
let trie: CompactTrie | null = null;

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    80,
    timeFrames,
  );
  const greedy = decoder.decode(logprobs, timeSteps, vocabSize);

  let beamMatches: BeamVerseMatch[] | undefined;
  if (trie) {
    const beamResults = beamSearchDecode(
      logprobs,
      timeSteps,
      vocabSize,
      decoder.getBlankId(),
      trie,
      8,
    );
    const seen = new Set<string>();
    beamMatches = [];
    for (const result of beamResults) {
      for (const ref of result.matchedVerses) {
        const key = `${ref.verseIndex}:${ref.spanLength}`;
        if (seen.has(key)) continue;
        seen.add(key);
        beamMatches.push({
          verseIndex: ref.verseIndex,
          spanLength: ref.spanLength,
          score: result.score,
        });
      }
    }
  }

  return {
    ...greedy,
    acoustic: { logprobs, timeSteps, vocabSize, blankId: decoder.getBlankId() },
    beamMatches,
  };
}

function refsEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((ref, idx) => b[idx] === ref);
}

async function runSample(sample: Sample, db: QuranDB, audio: Float32Array): Promise<SampleDiagnostic> {
  let cursor = 0;
  const diagnostics: TimedDiagnostic[] = [];
  const messages: TimedOutbound[] = [];
  const tracker = new RecitationTracker(db, transcribe, {
    onDiagnostic: (event) => {
      diagnostics.push({
        sampleCursor: cursor,
        seconds: Math.round((cursor / SAMPLE_RATE) * 1000) / 1000,
        event,
      });
    },
  });

  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const chunk = audio.slice(offset, Math.min(offset + CHUNK_SAMPLES, audio.length));
    cursor = Math.min(offset + chunk.length, audio.length);
    const out = await tracker.feed(chunk);
    for (const message of out) {
      messages.push({
        sampleCursor: cursor,
        seconds: Math.round((cursor / SAMPLE_RATE) * 1000) / 1000,
        message,
      });
    }
  }

  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    cursor = audio.length + (i + 1) * CHUNK_SAMPLES;
    const out = await tracker.feed(silenceChunk);
    for (const message of out) {
      messages.push({
        sampleCursor: cursor,
        seconds: Math.round((cursor / SAMPLE_RATE) * 1000) / 1000,
        message,
      });
    }
  }

  const emitted: string[] = [];
  for (const { message } of messages) {
    if (message.type !== "verse_match") continue;
    const ref = `${message.surah}:${message.ayah}`;
    if (!emitted.includes(ref)) emitted.push(ref);
  }
  const expected = sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`);

  return {
    id: sample.id,
    category: sample.category,
    source: sample.source,
    expected,
    emitted,
    exact: refsEqual(expected, emitted),
    diagnostics,
    messages: messages.filter(({ message }) => message.type === "verse_match" || message.type === "raw_transcript"),
  };
}

async function main() {
  console.log(`=== SEGMENT OWNERSHIP DIAGNOSTIC (${corpusName}) ===\n`);
  await createSession(resolve(ROOT, "public/fastconformer_phoneme_q8.onnx"));

  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  trie = buildTrie(quranData, vocabJson, 3).trie;

  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  const samples = manifest.samples.filter((sample) => sampleFilter.has(sample.id));
  if (samples.length === 0) {
    throw new Error(`No samples matched: ${[...sampleFilter].join(", ")}`);
  }

  const results: SampleDiagnostic[] = [];
  for (const sample of samples) {
    const audio = loadAudio(resolve(BENCHMARK, sample.file));
    const result = await runSample(sample, db, audio);
    results.push(result);

    const commits = result.diagnostics.filter((d) => d.event.type === "commit");
    const stale = result.diagnostics.filter((d) => d.event.type === "stale_exit");
    console.log(
      `${result.exact ? "EXACT" : "MISS "} ${sample.id} expected=[${result.expected.join(", ")}] ` +
        `emitted=[${result.emitted.join(", ")}] commits=${commits.length} stale=${stale.length}`,
    );
    for (const diag of commits) {
      if (diag.event.type !== "commit") continue;
      console.log(
        `  @${diag.seconds.toFixed(1)}s commit ${diag.event.ref} ` +
          `origin=${diag.event.origin ?? "?"} reason=${diag.event.reason} ` +
          `conf=${diag.event.confidence} clear=${diag.event.clear_margin ?? ""} repeat=${diag.event.repeated_leader ?? ""}`,
      );
    }
  }

  if (jsonOutPath) {
    const outPath = resolve(process.cwd(), jsonOutPath);
    writeFileSync(outPath, JSON.stringify({ corpus: corpusName, results }, null, 2));
    console.log(`\nJSON diagnostic saved to: ${outPath}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
