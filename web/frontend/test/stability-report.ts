/**
 * Structured per-sample stability report across N streaming validation runs.
 *
 * Runs the streaming validator N times, tracks per-sample pass/fail,
 * discovered verse lists, precision, recall, and classifies samples as
 * stable-pass, stable-fail, or flaky.
 *
 * Usage:
 *   npx tsx test/stability-report.ts                    # 5 repeats (default)
 *   npx tsx test/stability-report.ts --repeats=3        # custom repeats
 *   npx tsx test/stability-report.ts --corpus=test_v2   # different corpus
 *   npx tsx test/stability-report.ts --json=out.json    # save JSON report
 */

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { beamSearchDecode } from "../src/worker/beam-decode.ts";
import { buildTrie, type CompactTrie } from "../src/lib/phoneme-trie.ts";
import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker } from "../src/lib/tracker.ts";
import type { TranscribeResult, BeamVerseMatch } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
const repeatsArg = args.find((a) => a.startsWith("--repeats="));
const repeats = repeatsArg ? parseInt(repeatsArg.split("=")[1], 10) : 5;
const corpusArg = args.find((a) => a.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus";
const jsonArg = args.find((a) => a.startsWith("--json="));
const jsonOutPath = jsonArg ? jsonArg.split("=")[1] : null;
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

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
// Transcribe (same as validate-streaming.ts)
// ---------------------------------------------------------------------------
let decoder: CTCDecoder;
let trie: CompactTrie | null = null;

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const numMels = 80;
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    numMels,
    timeFrames,
  );

  const greedy = decoder.decode(logprobs, timeSteps, vocabSize);

  let beamMatches: BeamVerseMatch[] | undefined;
  if (trie) {
    const beamResults = beamSearchDecode(
      logprobs, timeSteps, vocabSize,
      decoder.getBlankId(), trie, 8,
    );
    const seen = new Set<string>();
    beamMatches = [];
    for (const result of beamResults) {
      for (const ref of result.matchedVerses) {
        const key = `${ref.verseIndex}:${ref.spanLength}`;
        if (!seen.has(key)) {
          seen.add(key);
          beamMatches.push({
            verseIndex: ref.verseIndex,
            spanLength: ref.spanLength,
            score: result.score,
          });
        }
      }
    }
  }

  return {
    ...greedy,
    acoustic: {
      logprobs,
      timeSteps,
      vocabSize,
      blankId: decoder.getBlankId(),
    },
    beamMatches,
  };
}

// ---------------------------------------------------------------------------
// Types
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

interface SampleRunResult {
  passed: boolean;
  discoveredVerses: string[];
  recall: number;
  precision: number;
  seqAcc: number;
}

interface SampleStability {
  id: string;
  category: string;
  expectedVerses: string[];
  runs: SampleRunResult[];
  passRate: number;
  classification: "stable-pass" | "stable-fail" | "flaky";
  medianPrecision: number;
  medianRecall: number;
}

interface StabilityReport {
  corpus: string;
  repeats: number;
  timestamp: string;
  samples: SampleStability[];
  aggregate: {
    totalSamples: number;
    stablePass: number;
    stableFail: number;
    flaky: number;
    medianPrecision: number;
    medianRecall: number;
    medianSeqAcc: number;
    perRunCorrect: number[];
    perRunPrecision: number[];
    perRunRecall: number[];
    perRunSeqAcc: number[];
  };
}

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

// ---------------------------------------------------------------------------
// Run one sample through streaming validation
// ---------------------------------------------------------------------------
async function runSample(
  sample: Sample,
  db: QuranDB,
  audio: Float32Array,
): Promise<SampleRunResult> {
  const tracker = new RecitationTracker(db, transcribe);
  const messages: WorkerOutbound[] = [];

  // Feed audio in chunks
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    const chunk = audio.slice(offset, end);
    const msgs = await tracker.feed(chunk);
    messages.push(...msgs);
  }

  // Feed tail silence
  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil(
    (TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES,
  );
  for (let i = 0; i < silenceChunks; i++) {
    const msgs = await tracker.feed(silenceChunk);
    messages.push(...msgs);
  }

  // Collect discovered verses (deduplicated, ordered)
  const discoveredVerses: string[] = [];
  for (const msg of messages) {
    if (msg.type === "verse_match") {
      const key = `${msg.surah}:${msg.ayah}`;
      if (!discoveredVerses.includes(key)) {
        discoveredVerses.push(key);
      }
    }
  }

  const expectedSet = new Set(
    sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`),
  );
  const discoveredSet = new Set(discoveredVerses);

  const matched = [...expectedSet].filter((v) => discoveredSet.has(v)).length;
  const recall = expectedSet.size > 0 ? matched / expectedSet.size : 1.0;
  const precision = discoveredSet.size > 0 ? matched / discoveredSet.size : 1.0;
  const seqAcc =
    expectedSet.size === discoveredSet.size &&
    [...expectedSet].every((v) => discoveredSet.has(v))
      ? 1.0
      : 0.0;

  const passed = [...expectedSet].every((v) => discoveredSet.has(v));

  return { passed, discoveredVerses, recall, precision, seqAcc };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log(`=== STABILITY REPORT (${repeats} repeats, corpus: ${corpusName}) ===\n`);

  // Setup
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  console.log("Loading ONNX model...");
  await createSession(modelPath);

  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);

  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  console.log(`Loaded ${db.totalVerses} verses`);

  const built = buildTrie(quranData, vocabJson, 3);
  trie = built.trie;

  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  const samples = manifest.samples;
  console.log(`Testing ${samples.length} samples × ${repeats} repeats\n`);

  // Pre-load all audio (avoid re-reading files each repeat)
  const audioCache = new Map<string, Float32Array>();
  for (const sample of samples) {
    const audioPath = resolve(BENCHMARK, sample.file);
    audioCache.set(sample.id, loadAudio(audioPath));
  }

  // Run N repeats
  const sampleResults = new Map<string, SampleRunResult[]>();
  for (const sample of samples) {
    sampleResults.set(sample.id, []);
  }

  for (let run = 0; run < repeats; run++) {
    console.log(`--- Run ${run + 1}/${repeats} ---`);
    let correct = 0;
    for (const sample of samples) {
      const audio = audioCache.get(sample.id)!;
      const result = await runSample(sample, db, audio);
      sampleResults.get(sample.id)!.push(result);
      if (result.passed) correct++;
      const status = result.passed ? "PASS" : "FAIL";
      process.stdout.write(`  ${status}  ${sample.id} [${result.discoveredVerses.join(", ")}]\n`);
    }
    console.log(`  Run ${run + 1}: ${correct}/${samples.length}\n`);
  }

  // Build stability report
  const sampleStabilities: SampleStability[] = samples.map((sample) => {
    const runs = sampleResults.get(sample.id)!;
    const passCount = runs.filter((r) => r.passed).length;
    const passRate = passCount / repeats;

    let classification: "stable-pass" | "stable-fail" | "flaky";
    if (passCount === repeats) classification = "stable-pass";
    else if (passCount === 0) classification = "stable-fail";
    else classification = "flaky";

    const precisions = runs.map((r) => r.precision);
    const recalls = runs.map((r) => r.recall);

    return {
      id: sample.id,
      category: sample.category,
      expectedVerses: sample.expected_verses.map((v) => `${v.surah}:${v.ayah}`),
      runs,
      passRate,
      classification,
      medianPrecision: median(precisions),
      medianRecall: median(recalls),
    };
  });

  // Aggregate per-run metrics
  const perRunCorrect: number[] = [];
  const perRunPrecision: number[] = [];
  const perRunRecall: number[] = [];
  const perRunSeqAcc: number[] = [];
  for (let run = 0; run < repeats; run++) {
    let correct = 0;
    let totalP = 0;
    let totalR = 0;
    let totalS = 0;
    for (const sample of samples) {
      const r = sampleResults.get(sample.id)![run];
      if (r.passed) correct++;
      totalP += r.precision;
      totalR += r.recall;
      totalS += r.seqAcc;
    }
    perRunCorrect.push(correct);
    perRunPrecision.push(totalP / samples.length);
    perRunRecall.push(totalR / samples.length);
    perRunSeqAcc.push(totalS / samples.length);
  }

  const report: StabilityReport = {
    corpus: corpusName,
    repeats,
    timestamp: new Date().toISOString(),
    samples: sampleStabilities,
    aggregate: {
      totalSamples: samples.length,
      stablePass: sampleStabilities.filter((s) => s.classification === "stable-pass").length,
      stableFail: sampleStabilities.filter((s) => s.classification === "stable-fail").length,
      flaky: sampleStabilities.filter((s) => s.classification === "flaky").length,
      medianPrecision: median(perRunPrecision),
      medianRecall: median(perRunRecall),
      medianSeqAcc: median(perRunSeqAcc),
      perRunCorrect,
      perRunPrecision,
      perRunRecall,
      perRunSeqAcc,
    },
  };

  // Print human summary
  console.log("=".repeat(60));
  console.log("STABILITY SUMMARY");
  console.log("=".repeat(60));
  console.log(`Corpus: ${corpusName} | Repeats: ${repeats}`);
  console.log(`Stable-pass: ${report.aggregate.stablePass}  Stable-fail: ${report.aggregate.stableFail}  Flaky: ${report.aggregate.flaky}`);
  console.log(`Per-run correct: [${perRunCorrect.join(", ")}]`);
  console.log(`Median precision: ${(report.aggregate.medianPrecision * 100).toFixed(1)}%`);
  console.log(`Median recall:    ${(report.aggregate.medianRecall * 100).toFixed(1)}%`);
  console.log(`Median SeqAcc:    ${(report.aggregate.medianSeqAcc * 100).toFixed(1)}%`);

  // Flaky samples detail
  const flaky = sampleStabilities.filter((s) => s.classification === "flaky");
  if (flaky.length > 0) {
    console.log(`\nFlaky samples (${flaky.length}):`);
    for (const s of flaky) {
      console.log(`  ${s.id} — pass rate: ${(s.passRate * 100).toFixed(0)}% | P: ${(s.medianPrecision * 100).toFixed(0)}% R: ${(s.medianRecall * 100).toFixed(0)}%`);
    }
  }

  // Stable-fail samples
  const stableFails = sampleStabilities.filter((s) => s.classification === "stable-fail");
  if (stableFails.length > 0) {
    console.log(`\nStable-fail samples (${stableFails.length}):`);
    for (const s of stableFails) {
      const lastRun = s.runs[s.runs.length - 1];
      console.log(`  ${s.id} — expected [${s.expectedVerses.join(", ")}] got [${lastRun.discoveredVerses.join(", ")}]`);
    }
  }

  // Save JSON
  if (jsonOutPath) {
    const outPath = resolve(process.cwd(), jsonOutPath);
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`\nJSON report saved to: ${outPath}`);
  }

  console.log();
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
