/**
 * Structured per-sample stability report across N streaming validation runs.
 *
 * Runs the streaming validator N times, tracks raw durable commits and the
 * final streaming-evidence sequence separately, and reports exact set vs
 * ordered sequence accuracy explicitly.
 *
 * Usage:
 *   npx tsx test/stability-report.ts                    # 5 repeats (default)
 *   npx tsx test/stability-report.ts --repeats=3        # custom repeats
 *   npx tsx test/stability-report.ts --corpus=test_v2   # different corpus
 *   npx tsx test/stability-report.ts --json=out.json    # save JSON report
 *   npx tsx test/stability-report.ts --focus=exact      # print exact-set failures
 *   npx tsx test/stability-report.ts --limit=3          # smoke-test first N samples
 *   npx tsx test/stability-report.ts --hypothesis=nextAyah=0.4,backward=-1.2
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
import type { VerseCandidate, WorkerOutbound } from "../src/lib/types.ts";
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
const focusArg = args.find((a) => a.startsWith("--focus="));
const focus = focusArg?.split("=")[1] === "exact" ? "exact" : "recall";
const limitArg = args.find((a) => a.startsWith("--limit="));
const sampleLimit = limitArg ? parseInt(limitArg.split("=")[1], 10) : null;
const hypothesisArg = args.find((a) => a.startsWith("--hypothesis="));
const hypothesisParams = parseHypothesisParams(
  hypothesisArg ? hypothesisArg.slice("--hypothesis=".length) : "",
);
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

for (const [name, value] of Object.entries(hypothesisParams)) {
  process.env[`STREAMING_HYPOTHESIS_${toEnvName(name)}`] = String(value);
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
  exactPassed: boolean;
  orderedPassed: boolean;
  rawCommitVerses: string[];
  finalSequenceVerses: string[];
  candidateHistory: CandidateHistoryEntry[];
  rawCommitMetrics: SequenceMetrics;
  finalSequenceMetrics: SequenceMetrics;
  productMetrics: ProductMetrics;
}

interface SampleStability {
  id: string;
  category: string;
  expectedVerses: string[];
  runs: SampleRunResult[];
  passRate: number;
  classification: "stable-pass" | "stable-fail" | "flaky";
  exactPassRate: number;
  exactClassification: "exact-stable-pass" | "exact-stable-fail" | "exact-flaky";
  orderedPassRate: number;
  orderedClassification: "ordered-stable-pass" | "ordered-stable-fail" | "ordered-flaky";
  rawCommitMedianPrecision: number;
  rawCommitMedianRecall: number;
  rawCommitMedianExactSetAcc: number;
  rawCommitMedianOrderedSeqAcc: number;
  finalMedianPrecision: number;
  finalMedianRecall: number;
  finalMedianExactSetAcc: number;
  finalMedianOrderedSeqAcc: number;
  medianFalseVisibleJumps: number;
  medianTimeToFirstCorrectCandidate: number | null;
}

interface StabilityReport {
  corpus: string;
  repeats: number;
  timestamp: string;
  metrics: {
    recall: string;
    precision: string;
    exactSetAcc: string;
    orderedSeqAcc: string;
    falseVisibleJumps: string;
    timeToFirstCorrectCandidate: string;
  };
  config: {
    chunkSeconds: number;
    tailSilenceSeconds: number;
    hypothesisParams: Record<string, number>;
  };
  samples: SampleStability[];
  aggregate: {
    totalSamples: number;
    stablePass: number;
    stableFail: number;
    flaky: number;
    exactStablePass: number;
    exactStableFail: number;
    exactFlaky: number;
    orderedStablePass: number;
    orderedStableFail: number;
    orderedFlaky: number;
    committedAccuracy: AggregateSequenceMetrics;
    rawCommits: AggregateSequenceMetrics;
    finalSequence: AggregateSequenceMetrics;
    product: AggregateProductMetrics;
  };
}

interface SequenceMetrics {
  recall: number;
  precision: number;
  exactSetAcc: number;
  orderedSeqAcc: number;
  recallPassed: boolean;
  exactSetPassed: boolean;
  orderedSeqPassed: boolean;
}

interface AggregateSequenceMetrics {
  medianPrecision: number;
  medianRecall: number;
  medianExactSetAcc: number;
  medianOrderedSeqAcc: number;
  perRunCorrect: number[];
  perRunExactSetAcc: number[];
  perRunOrderedSeqAcc: number[];
  perRunPrecision: number[];
  perRunRecall: number[];
}

interface AggregateProductMetrics {
  medianFalseVisibleJumps: number;
  medianTimeToFirstCorrectCandidate: number | null;
  perRunFalseVisibleJumps: number[];
  perRunTimeToFirstCorrectCandidate: Array<number | null>;
}

interface CandidateHistoryEntry {
  chunkIndex: number;
  timeSec: number;
  stable: boolean;
  finalFlush: boolean;
  candidates: Array<{
    ref: string;
    verses: string[];
    confidence: number;
    rank: number;
  }>;
}

interface VisibleEvent {
  type: "candidate" | "commit";
  key: string;
  verses: string[];
  timeSec: number;
}

interface ProductMetrics {
  falseVisibleJumps: number;
  timeToFirstCorrectCandidate: number | null;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function parseHypothesisParams(input: string): Record<string, number> {
  if (!input) return {};
  const params: Record<string, number> = {};
  for (const part of input.split(",")) {
    const eq = part.indexOf("=");
    const name = eq >= 0 ? part.slice(0, eq) : "";
    const rawValue = eq >= 0 ? part.slice(eq + 1) : "";
    const value = Number(rawValue);
    if (!name || !Number.isFinite(value)) {
      throw new Error(`Invalid --hypothesis entry: ${part}`);
    }
    params[name.trim()] = value;
  }
  return params;
}

function toEnvName(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/-/g, "_").toUpperCase();
}

function verseKey(surah: number, ayah: number): string {
  return `${surah}:${ayah}`;
}

function dedupeOrdered(keys: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const key of keys) {
    if (!seen.has(key)) {
      seen.add(key);
      result.push(key);
    }
  }
  return result;
}

function expandCandidateKeys(candidate: VerseCandidate): string[] {
  const end = candidate.ayah_end && candidate.ayah_end > candidate.ayah
    ? candidate.ayah_end
    : candidate.ayah;
  const keys: string[] = [];
  for (let ayah = candidate.ayah; ayah <= end; ayah++) {
    keys.push(verseKey(candidate.surah, ayah));
  }
  return keys;
}

function refForCandidate(candidate: VerseCandidate): string {
  return candidate.ayah_end && candidate.ayah_end > candidate.ayah
    ? `${candidate.surah}:${candidate.ayah}-${candidate.ayah_end}`
    : verseKey(candidate.surah, candidate.ayah);
}

function computeSequenceMetrics(expectedVerses: string[], emittedVerses: string[]): SequenceMetrics {
  const expectedSet = new Set(expectedVerses);
  const emittedSet = new Set(emittedVerses);
  const matched = [...expectedSet].filter((v) => emittedSet.has(v)).length;
  const recall = expectedSet.size > 0 ? matched / expectedSet.size : 1.0;
  const precision = emittedSet.size > 0 ? matched / emittedSet.size : 1.0;
  const exactSetAcc =
    expectedSet.size === emittedSet.size &&
    [...expectedSet].every((v) => emittedSet.has(v))
      ? 1.0
      : 0.0;
  const orderedSeqAcc =
    expectedVerses.length === emittedVerses.length &&
    expectedVerses.every((v, idx) => emittedVerses[idx] === v)
      ? 1.0
      : 0.0;
  return {
    recall,
    precision,
    exactSetAcc,
    orderedSeqAcc,
    recallPassed: matched === expectedSet.size,
    exactSetPassed: exactSetAcc === 1.0,
    orderedSeqPassed: orderedSeqAcc === 1.0,
  };
}

function parseVerseKey(key: string): { surah: number; ayah: number } {
  const [surah, ayah] = key.split(":").map(Number);
  return { surah, ayah };
}

function isNearVisibleJump(prev: string, next: string): boolean {
  const a = parseVerseKey(prev);
  const b = parseVerseKey(next);
  return a.surah === b.surah && Math.abs(a.ayah - b.ayah) <= 3;
}

function computeProductMetrics(
  expectedVerses: string[],
  candidateHistory: CandidateHistoryEntry[],
  visibleEvents: VisibleEvent[],
): ProductMetrics {
  const expectedSet = new Set(expectedVerses);
  let falseVisibleJumps = 0;
  let previousVisible: VisibleEvent | null = null;

  // Deterministic product metric assumptions:
  // - only the top candidate is treated as visible for jump counting
  // - same-surah movement within +/-3 ayahs is "nearby", not a false jump
  // - spans are correct if any verse in the top span is expected
  for (const event of visibleEvents) {
    const containsExpected = event.verses.some((v) => expectedSet.has(v));
    if (
      previousVisible &&
      event.key !== previousVisible.key &&
      !containsExpected &&
      !isNearVisibleJump(previousVisible.key, event.key)
    ) {
      falseVisibleJumps++;
    }
    previousVisible = event;
  }

  const firstCorrect = candidateHistory.find((entry) =>
    entry.candidates[0]?.verses.some((v) => expectedSet.has(v)),
  );

  return {
    falseVisibleJumps,
    timeToFirstCorrectCandidate: firstCorrect?.timeSec ?? null,
  };
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
  const rawCommitVerses: string[] = [];
  let finalSequenceVerses: string[] = [];
  const candidateHistory: CandidateHistoryEntry[] = [];
  const visibleEvents: VisibleEvent[] = [];

  const recordMessages = (
    msgs: WorkerOutbound[],
    chunkIndex: number,
    timeSec: number,
  ) => {
    for (const msg of msgs) {
      if (msg.type === "verse_match") {
        const key = verseKey(msg.surah, msg.ayah);
        rawCommitVerses.push(key);
        visibleEvents.push({
          type: "commit",
          key,
          verses: [key],
          timeSec,
        });
      } else if (msg.type === "verse_candidate") {
        const candidates = msg.candidates.map((candidate) => ({
          ref: refForCandidate(candidate),
          verses: expandCandidateKeys(candidate),
          confidence: candidate.confidence,
          rank: candidate.rank,
        }));
        candidateHistory.push({
          chunkIndex,
          timeSec,
          stable: msg.stable,
          finalFlush: msg.final_flush,
          candidates,
        });
        const top = candidates[0];
        if (top?.verses[0]) {
          visibleEvents.push({
            type: "candidate",
            key: top.verses[0],
            verses: top.verses,
            timeSec,
          });
        }
      } else if (msg.type === "final_sequence") {
        finalSequenceVerses = msg.verses.map((v) => verseKey(v.surah, v.ayah));
      }
    }
  };

  // Feed audio in chunks
  let chunkIndex = 0;
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    const chunk = audio.slice(offset, end);
    const msgs = await tracker.feed(chunk);
    chunkIndex++;
    recordMessages(msgs, chunkIndex, chunkIndex * CHUNK_SECONDS);
  }

  // Feed tail silence
  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil(
    (TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES,
  );
  for (let i = 0; i < silenceChunks; i++) {
    const msgs = await tracker.feed(silenceChunk);
    chunkIndex++;
    recordMessages(msgs, chunkIndex, chunkIndex * CHUNK_SECONDS);
  }

  const expectedVerses = sample.expected_verses.map((v) => verseKey(v.surah, v.ayah));
  const dedupedRawCommits = dedupeOrdered(rawCommitVerses);
  const dedupedFinalSequence = dedupeOrdered(finalSequenceVerses);
  const rawCommitMetrics = computeSequenceMetrics(expectedVerses, dedupedRawCommits);
  const finalSequenceMetrics = computeSequenceMetrics(expectedVerses, dedupedFinalSequence);
  const productMetrics = computeProductMetrics(expectedVerses, candidateHistory, visibleEvents);

  return {
    passed: finalSequenceMetrics.recallPassed,
    exactPassed: finalSequenceMetrics.exactSetPassed,
    orderedPassed: finalSequenceMetrics.orderedSeqPassed,
    rawCommitVerses: dedupedRawCommits,
    finalSequenceVerses: dedupedFinalSequence,
    candidateHistory,
    rawCommitMetrics,
    finalSequenceMetrics,
    productMetrics,
  };
}

function buildAggregateSequenceMetrics(
  samples: Sample[],
  sampleResults: Map<string, SampleRunResult[]>,
  pick: (run: SampleRunResult) => SequenceMetrics,
): AggregateSequenceMetrics {
  const perRunCorrect: number[] = [];
  const perRunExactSetAcc: number[] = [];
  const perRunOrderedSeqAcc: number[] = [];
  const perRunPrecision: number[] = [];
  const perRunRecall: number[] = [];

  for (let run = 0; run < repeats; run++) {
    let correct = 0;
    let totalExactSet = 0;
    let totalOrdered = 0;
    let totalP = 0;
    let totalR = 0;
    for (const sample of samples) {
      const metrics = pick(sampleResults.get(sample.id)![run]);
      if (metrics.recallPassed) correct++;
      totalExactSet += metrics.exactSetAcc;
      totalOrdered += metrics.orderedSeqAcc;
      totalP += metrics.precision;
      totalR += metrics.recall;
    }
    perRunCorrect.push(correct);
    perRunExactSetAcc.push(totalExactSet / samples.length);
    perRunOrderedSeqAcc.push(totalOrdered / samples.length);
    perRunPrecision.push(totalP / samples.length);
    perRunRecall.push(totalR / samples.length);
  }

  return {
    medianPrecision: median(perRunPrecision),
    medianRecall: median(perRunRecall),
    medianExactSetAcc: median(perRunExactSetAcc),
    medianOrderedSeqAcc: median(perRunOrderedSeqAcc),
    perRunCorrect,
    perRunExactSetAcc,
    perRunOrderedSeqAcc,
    perRunPrecision,
    perRunRecall,
  };
}

function buildAggregateProductMetrics(
  samples: Sample[],
  sampleResults: Map<string, SampleRunResult[]>,
): AggregateProductMetrics {
  const perRunFalseVisibleJumps: number[] = [];
  const perRunTimeToFirstCorrectCandidate: Array<number | null> = [];

  for (let run = 0; run < repeats; run++) {
    let totalJumps = 0;
    const candidateTimes: number[] = [];
    for (const sample of samples) {
      const metrics = sampleResults.get(sample.id)![run].productMetrics;
      totalJumps += metrics.falseVisibleJumps;
      if (metrics.timeToFirstCorrectCandidate !== null) {
        candidateTimes.push(metrics.timeToFirstCorrectCandidate);
      }
    }
    perRunFalseVisibleJumps.push(totalJumps / samples.length);
    perRunTimeToFirstCorrectCandidate.push(
      candidateTimes.length > 0 ? median(candidateTimes) : null,
    );
  }

  const nonNullTimes = perRunTimeToFirstCorrectCandidate.filter(
    (value): value is number => value !== null,
  );

  return {
    medianFalseVisibleJumps: median(perRunFalseVisibleJumps),
    medianTimeToFirstCorrectCandidate:
      nonNullTimes.length > 0 ? median(nonNullTimes) : null,
    perRunFalseVisibleJumps,
    perRunTimeToFirstCorrectCandidate,
  };
}

function printAggregate(label: string, metrics: AggregateSequenceMetrics): void {
  console.log(`\n${label}`);
  console.log(`  Per-run recall-pass:   [${metrics.perRunCorrect.join(", ")}]`);
  console.log(
    `  Per-run ExactSetAcc:   [${metrics.perRunExactSetAcc.map((v) => (v * 100).toFixed(1)).join("%, ")}%]`,
  );
  console.log(
    `  Per-run OrderedSeqAcc: [${metrics.perRunOrderedSeqAcc.map((v) => (v * 100).toFixed(1)).join("%, ")}%]`,
  );
  console.log(`  Median precision:      ${(metrics.medianPrecision * 100).toFixed(1)}%`);
  console.log(`  Median recall:         ${(metrics.medianRecall * 100).toFixed(1)}%`);
  console.log(`  Median ExactSetAcc:    ${(metrics.medianExactSetAcc * 100).toFixed(1)}%`);
  console.log(`  Median OrderedSeqAcc:  ${(metrics.medianOrderedSeqAcc * 100).toFixed(1)}%`);
}

function formatNullableSeconds(value: number | null): string {
  return value === null ? "n/a" : `${value.toFixed(1)}s`;
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
  const samples = sampleLimit && sampleLimit > 0
    ? manifest.samples.slice(0, sampleLimit)
    : manifest.samples;
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
    let finalCorrect = 0;
    let finalExactCorrect = 0;
    let rawExactCorrect = 0;
    for (const sample of samples) {
      const audio = audioCache.get(sample.id)!;
      const result = await runSample(sample, db, audio);
      sampleResults.get(sample.id)!.push(result);
      if (result.passed) finalCorrect++;
      if (result.exactPassed) finalExactCorrect++;
      if (result.rawCommitMetrics.exactSetPassed) rawExactCorrect++;
      const status = result.exactPassed ? "EXACT" : result.passed ? "PASS " : "FAIL ";
      process.stdout.write(
        `  ${status}  ${sample.id} raw=[${result.rawCommitVerses.join(", ")}] final=[${result.finalSequenceVerses.join(", ")}]\n`,
      );
    }
    console.log(
      `  Run ${run + 1}: final recall-pass ${finalCorrect}/${samples.length}, final exact-set ${finalExactCorrect}/${samples.length}, raw exact-set ${rawExactCorrect}/${samples.length}\n`,
    );
  }

  // Build stability report
  const sampleStabilities: SampleStability[] = samples.map((sample) => {
    const runs = sampleResults.get(sample.id)!;
    const passCount = runs.filter((r) => r.passed).length;
    const passRate = passCount / repeats;
    const exactPassCount = runs.filter((r) => r.exactPassed).length;
    const exactPassRate = exactPassCount / repeats;
    const orderedPassCount = runs.filter((r) => r.orderedPassed).length;
    const orderedPassRate = orderedPassCount / repeats;

    let classification: "stable-pass" | "stable-fail" | "flaky";
    if (passCount === repeats) classification = "stable-pass";
    else if (passCount === 0) classification = "stable-fail";
    else classification = "flaky";

    let exactClassification: "exact-stable-pass" | "exact-stable-fail" | "exact-flaky";
    if (exactPassCount === repeats) exactClassification = "exact-stable-pass";
    else if (exactPassCount === 0) exactClassification = "exact-stable-fail";
    else exactClassification = "exact-flaky";

    let orderedClassification: "ordered-stable-pass" | "ordered-stable-fail" | "ordered-flaky";
    if (orderedPassCount === repeats) orderedClassification = "ordered-stable-pass";
    else if (orderedPassCount === 0) orderedClassification = "ordered-stable-fail";
    else orderedClassification = "ordered-flaky";

    const rawPrecisions = runs.map((r) => r.rawCommitMetrics.precision);
    const rawRecalls = runs.map((r) => r.rawCommitMetrics.recall);
    const rawExactSetAccs = runs.map((r) => r.rawCommitMetrics.exactSetAcc);
    const rawOrderedSeqAccs = runs.map((r) => r.rawCommitMetrics.orderedSeqAcc);
    const finalPrecisions = runs.map((r) => r.finalSequenceMetrics.precision);
    const finalRecalls = runs.map((r) => r.finalSequenceMetrics.recall);
    const finalExactSetAccs = runs.map((r) => r.finalSequenceMetrics.exactSetAcc);
    const finalOrderedSeqAccs = runs.map((r) => r.finalSequenceMetrics.orderedSeqAcc);
    const falseVisibleJumps = runs.map((r) => r.productMetrics.falseVisibleJumps);
    const candidateTimes = runs
      .map((r) => r.productMetrics.timeToFirstCorrectCandidate)
      .filter((value): value is number => value !== null);

    return {
      id: sample.id,
      category: sample.category,
      expectedVerses: sample.expected_verses.map((v) => verseKey(v.surah, v.ayah)),
      runs,
      passRate,
      classification,
      exactPassRate,
      exactClassification,
      orderedPassRate,
      orderedClassification,
      rawCommitMedianPrecision: median(rawPrecisions),
      rawCommitMedianRecall: median(rawRecalls),
      rawCommitMedianExactSetAcc: median(rawExactSetAccs),
      rawCommitMedianOrderedSeqAcc: median(rawOrderedSeqAccs),
      finalMedianPrecision: median(finalPrecisions),
      finalMedianRecall: median(finalRecalls),
      finalMedianExactSetAcc: median(finalExactSetAccs),
      finalMedianOrderedSeqAcc: median(finalOrderedSeqAccs),
      medianFalseVisibleJumps: median(falseVisibleJumps),
      medianTimeToFirstCorrectCandidate:
        candidateTimes.length > 0 ? median(candidateTimes) : null,
    };
  });

  // Aggregate per-run metrics
  const rawAggregate = buildAggregateSequenceMetrics(
    samples,
    sampleResults,
    (r) => r.rawCommitMetrics,
  );
  const finalAggregate = buildAggregateSequenceMetrics(
    samples,
    sampleResults,
    (r) => r.finalSequenceMetrics,
  );
  const productAggregate = buildAggregateProductMetrics(samples, sampleResults);

  const report: StabilityReport = {
    corpus: corpusName,
    repeats,
    timestamp: new Date().toISOString(),
    metrics: {
      recall: "Fraction of expected verses found.",
      precision: "Fraction of emitted verses that were expected.",
      exactSetAcc: "Deduped emitted set exactly matches expected set; order ignored.",
      orderedSeqAcc: "Deduped emitted sequence exactly equals expected ordered sequence.",
      falseVisibleJumps: "Candidate or commit changed to a non-nearby non-expected verse.",
      timeToFirstCorrectCandidate: "Seconds until a top verse_candidate contains an expected verse, using 300ms chunks.",
    },
    config: {
      chunkSeconds: CHUNK_SECONDS,
      tailSilenceSeconds: TAIL_SILENCE_SECONDS,
      hypothesisParams,
    },
    samples: sampleStabilities,
    aggregate: {
      totalSamples: samples.length,
      stablePass: sampleStabilities.filter((s) => s.classification === "stable-pass").length,
      stableFail: sampleStabilities.filter((s) => s.classification === "stable-fail").length,
      flaky: sampleStabilities.filter((s) => s.classification === "flaky").length,
      exactStablePass: sampleStabilities.filter((s) => s.exactClassification === "exact-stable-pass").length,
      exactStableFail: sampleStabilities.filter((s) => s.exactClassification === "exact-stable-fail").length,
      exactFlaky: sampleStabilities.filter((s) => s.exactClassification === "exact-flaky").length,
      orderedStablePass: sampleStabilities.filter((s) => s.orderedClassification === "ordered-stable-pass").length,
      orderedStableFail: sampleStabilities.filter((s) => s.orderedClassification === "ordered-stable-fail").length,
      orderedFlaky: sampleStabilities.filter((s) => s.orderedClassification === "ordered-flaky").length,
      // Alias kept because product-facing reports call durable verse_match metrics committed accuracy.
      committedAccuracy: rawAggregate,
      rawCommits: rawAggregate,
      finalSequence: finalAggregate,
      product: productAggregate,
    },
  };

  // Print human summary
  console.log("=".repeat(60));
  console.log("STABILITY SUMMARY");
  console.log("=".repeat(60));
  console.log(`Corpus: ${corpusName} | Repeats: ${repeats}`);
  printAggregate("Raw commits (verse_match)", report.aggregate.rawCommits);
  printAggregate("Final sequence", report.aggregate.finalSequence);
  console.log(
    `Final stability: recall-pass ${report.aggregate.stablePass}/${samples.length}, exact-set ${report.aggregate.exactStablePass}/${samples.length}, ordered ${report.aggregate.orderedStablePass}/${samples.length}`,
  );
  console.log(
    `Product: median false visible jumps ${report.aggregate.product.medianFalseVisibleJumps.toFixed(1)}, median time-to-first-correct-candidate ${formatNullableSeconds(report.aggregate.product.medianTimeToFirstCorrectCandidate)}`,
  );

  // Flaky samples detail
  const flaky = sampleStabilities.filter((s) =>
    focus === "exact"
      ? s.exactClassification === "exact-flaky"
      : s.classification === "flaky",
  );
  if (flaky.length > 0) {
    console.log(`\n${focus === "exact" ? "Exact-flaky" : "Flaky"} samples (${flaky.length}):`);
    for (const s of flaky) {
      const passRate = focus === "exact" ? s.exactPassRate : s.passRate;
      console.log(`  ${s.id} — pass rate: ${(passRate * 100).toFixed(0)}% | final P: ${(s.finalMedianPrecision * 100).toFixed(0)}% R: ${(s.finalMedianRecall * 100).toFixed(0)}%`);
    }
  }

  // Stable-fail samples
  const stableFails = sampleStabilities.filter((s) =>
    focus === "exact"
      ? s.exactClassification === "exact-stable-fail"
      : s.classification === "stable-fail",
  );
  if (stableFails.length > 0) {
    console.log(`\n${focus === "exact" ? "Exact stable-fail" : "Stable-fail"} samples (${stableFails.length}):`);
    for (const s of stableFails) {
      const lastRun = s.runs[s.runs.length - 1];
      console.log(`  ${s.id} — expected [${s.expectedVerses.join(", ")}] raw [${lastRun.rawCommitVerses.join(", ")}] final [${lastRun.finalSequenceVerses.join(", ")}]`);
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
