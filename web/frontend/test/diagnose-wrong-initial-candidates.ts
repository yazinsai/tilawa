/**
 * Diagnostic: audit wrong-initial exact-fail runs from the cached v3 stability
 * report and ask whether CTC candidate scoring could have prevented them.
 *
 * Usage:
 *   npx tsx test/diagnose-wrong-initial-candidates.ts
 */

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { QuranDB, type QuranCandidate } from "../src/lib/quran-db.ts";
import { RecitationTracker, type TrackerDiagnosticEvent, type TranscribeResult } from "../src/lib/tracker.ts";
import { scoreCtcSequence, type AcousticEvidence } from "../src/lib/ctc-rescore.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import {
  ACOUSTIC_OVERRIDE_TEXT_THRESHOLD,
  DISCOVERY_EXPANDED_CANDIDATES,
  DISCOVERY_FUSION_ACOUSTIC_WEIGHT,
  DISCOVERY_FUSION_LENGTH_WEIGHT,
  DISCOVERY_FUSION_LOW_ACOUSTIC_WEIGHT,
  DISCOVERY_FUSION_LOW_LENGTH_WEIGHT,
  DISCOVERY_FUSION_LOW_TEXT_WEIGHT,
  DISCOVERY_FUSION_TEXT_WEIGHT,
  DISCOVERY_LOW_CONFIDENCE_CHARS,
  DISCOVERY_LOW_CONFIDENCE_WORDS,
  DISCOVERY_MAX_SPAN,
  DISCOVERY_TOP_SINGLE_CANDIDATES,
  DISCOVERY_TOP_SURAHS,
  RAW_TRANSCRIPT_THRESHOLD,
  SAMPLE_RATE,
} from "../src/lib/types.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const BENCHMARK = resolve(ROOT, "../../benchmark/test_corpus_v3");
const STABILITY_JSON = resolve(ROOT, "test/stab-gate-on-v3.json");
const OUT_JSON = resolve(ROOT, "test/wrong-initial-candidate-audit-v3.json");
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;
const MAX_REPLAY_RUNS = 24;

type RunClass =
  | "exact"
  | "missing_only"
  | "extra_after_expected"
  | "extra_before_expected"
  | "wrong_initial"
  | "wrong_surah_jump"
  | "no_emit"
  | "partial_multi";

interface StabilityRun {
  discoveredVerses: string[];
  seqAcc: number;
}

interface StabilitySample {
  id: string;
  category: string;
  expectedVerses: string[];
  runs: StabilityRun[];
}

interface StabilityReport {
  corpus: string;
  repeats: number;
  samples: StabilitySample[];
}

interface ManifestSample {
  id: string;
  file: string;
  category: string;
  source: string;
  expected_verses: { surah: number; ayah: number }[];
}

interface RankedCandidate {
  ref: string;
  candidate: QuranCandidate;
  textScore: number;
  rawScore: number;
  ctcScore: number;
  ctcFeasible: boolean;
  lengthFit: number;
  acousticFit: number;
  fusionScore: number;
  textRank: number;
  ctcRank: number | null;
  fusionRank: number;
}

let decoder: CTCDecoder;
const fullAudioCache = new WeakMap<Float32Array, TranscribeResult>();

function isSubsequence<T>(needle: readonly T[], haystack: readonly T[]): boolean {
  let pos = 0;
  for (const item of haystack) {
    if (needle[pos] === item) pos++;
    if (pos === needle.length) return true;
  }
  return needle.length === 0;
}

function isPrefix<T>(prefix: readonly T[], values: readonly T[]): boolean {
  if (prefix.length > values.length) return false;
  for (let i = 0; i < prefix.length; i++) {
    if (prefix[i] !== values[i]) return false;
  }
  return true;
}

function surahOf(ref: string): string {
  return ref.split(":")[0] ?? "";
}

function classifyRun(expected: readonly string[], discovered: readonly string[]): RunClass {
  if (discovered.length === 0) return "no_emit";
  if (
    expected.length === discovered.length &&
    expected.every((ref, idx) => discovered[idx] === ref)
  ) {
    return "exact";
  }

  const expectedSet = new Set(expected);
  const discoveredSet = new Set(discovered);
  const missing = expected.filter((ref) => !discoveredSet.has(ref));
  const extras = discovered.filter((ref) => !expectedSet.has(ref));

  if (expected.length > 1 && missing.length > 0) return "partial_multi";
  if (extras.length === 0 && missing.length > 0) return "missing_only";
  if (extras.length > 0 && discovered[0] && !expectedSet.has(discovered[0])) {
    return "wrong_initial";
  }
  if (extras.length > 0) {
    const expectedSurahs = new Set(expected.map(surahOf));
    const hasWrongSurahAfterExpected =
      isSubsequence(expected, discovered) &&
      discovered.some((ref, idx) => {
        const expectedPrefixSeen = isSubsequence(expected, discovered.slice(0, idx + 1));
        return expectedPrefixSeen && !expectedSurahs.has(surahOf(ref));
      });
    if (hasWrongSurahAfterExpected) return "wrong_surah_jump";
    if (isPrefix(expected, discovered) || isSubsequence(expected, discovered)) {
      return "extra_after_expected";
    }
    return "extra_before_expected";
  }
  return "missing_only";
}

function refKey(surah: number, ayah: number, ayahEnd?: number | null): string {
  return ayahEnd && ayahEnd !== ayah ? `${surah}:${ayah}-${ayahEnd}` : `${surah}:${ayah}`;
}

function expandRef(ref: string): string[] {
  const span = ref.match(/^(\d+):(\d+)-(\d+)$/);
  if (!span) return [ref];
  const surah = Number(span[1]);
  const start = Number(span[2]);
  const end = Number(span[3]);
  const refs: string[] = [];
  for (let ayah = start; ayah <= end; ayah++) refs.push(`${surah}:${ayah}`);
  return refs;
}

function refsEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((ref, idx) => b[idx] === ref);
}

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const cached = fullAudioCache.get(audio);
  if (cached) return cached;

  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);
  const greedy = decoder.decode(logprobs, timeSteps, vocabSize);

  return {
    ...greedy,
    acoustic: { logprobs, timeSteps, vocabSize, blankId: decoder.getBlankId() },
  };
}

function rankCandidates(
  candidates: QuranCandidate[],
  evidence: AcousticEvidence,
  observedTokenLength: number,
  observedText: string,
): RankedCandidate[] {
  const observedLength = Math.max(observedTokenLength, 1);
  const observedWords = observedText.trim().split(/\s+/).filter(Boolean).length;
  const observedChars = observedText.replace(/\s+/g, "").length;
  const textWeak =
    observedWords <= DISCOVERY_LOW_CONFIDENCE_WORDS ||
    observedChars <= DISCOVERY_LOW_CONFIDENCE_CHARS;
  const textWeight = textWeak
    ? DISCOVERY_FUSION_LOW_TEXT_WEIGHT
    : DISCOVERY_FUSION_TEXT_WEIGHT;
  const acousticWeight = textWeak
    ? DISCOVERY_FUSION_LOW_ACOUSTIC_WEIGHT
    : DISCOVERY_FUSION_ACOUSTIC_WEIGHT;
  const lengthWeight = textWeak
    ? DISCOVERY_FUSION_LOW_LENGTH_WEIGHT
    : DISCOVERY_FUSION_LENGTH_WEIGHT;

  const withScores = candidates.map((candidate) => ({
    candidate,
    ref: refKey(candidate.surah, candidate.ayah, candidate.ayah_end),
    ctcScore: scoreCtcSequence(evidence, candidate.phoneme_token_ids),
  }));
  const feasibleScores = withScores
    .filter((entry) => Number.isFinite(entry.ctcScore) && entry.ctcScore < 1e9)
    .map((entry) => entry.ctcScore);
  const minAcoustic = feasibleScores.length ? Math.min(...feasibleScores) : 0;
  const maxAcoustic = feasibleScores.length ? Math.max(...feasibleScores) : 1;
  const acousticRange = Math.max(maxAcoustic - minAcoustic, 1e-6);

  const textSorted = withScores
    .slice()
    .sort((a, b) => b.candidate.stage_a_score - a.candidate.stage_a_score);
  const ctcSorted = withScores
    .filter((entry) => Number.isFinite(entry.ctcScore) && entry.ctcScore < 1e9)
    .sort((a, b) => a.ctcScore - b.ctcScore);
  const textRanks = new Map(textSorted.map((entry, idx) => [entry.ref, idx + 1]));
  const ctcRanks = new Map(ctcSorted.map((entry, idx) => [entry.ref, idx + 1]));

  const ranked = withScores.map((entry) => {
    const candidateLength = Math.max(entry.candidate.phoneme_token_ids.length, 1);
    const lengthFit =
      Math.min(candidateLength, observedLength) /
      Math.max(candidateLength, observedLength);
    const ctcFeasible = Number.isFinite(entry.ctcScore) && entry.ctcScore < 1e9;
    const acousticFit = ctcFeasible
      ? 1 - (entry.ctcScore - minAcoustic) / acousticRange
      : 0;
    const fusionScore = Math.min(
      1,
      entry.candidate.stage_a_score * textWeight +
        acousticFit * acousticWeight +
        lengthFit * lengthWeight,
    );
    return {
      ref: entry.ref,
      candidate: entry.candidate,
      textScore: entry.candidate.stage_a_score,
      rawScore: entry.candidate.raw_score,
      ctcScore: entry.ctcScore,
      ctcFeasible,
      lengthFit,
      acousticFit,
      fusionScore,
      textRank: textRanks.get(entry.ref) ?? candidates.length,
      ctcRank: ctcRanks.get(entry.ref) ?? null,
      fusionRank: 0,
    };
  });

  ranked.sort((a, b) => {
    if (b.fusionScore !== a.fusionScore) return b.fusionScore - a.fusionScore;
    if (b.textScore !== a.textScore) return b.textScore - a.textScore;
    return a.ctcScore - b.ctcScore;
  });
  ranked.forEach((entry, idx) => {
    entry.fusionRank = idx + 1;
  });
  return ranked;
}

function expectedCandidates(expected: string[], ranked: RankedCandidate[]): RankedCandidate[] {
  return ranked.filter((entry) => {
    const expanded = expandRef(entry.ref);
    return refsEqual(expanded, expected) || expected.some((ref) => expanded.includes(ref));
  });
}

function compactCandidate(entry: RankedCandidate) {
  return {
    ref: entry.ref,
    kind: entry.candidate.kind,
    textScore: Number(entry.textScore.toFixed(4)),
    rawScore: Number(entry.rawScore.toFixed(4)),
    ctcScore: Number(entry.ctcScore.toFixed(4)),
    ctcFeasible: entry.ctcFeasible,
    fusionScore: Number(entry.fusionScore.toFixed(4)),
    lengthFit: Number(entry.lengthFit.toFixed(4)),
    textRank: entry.textRank,
    ctcRank: entry.ctcRank,
    fusionRank: entry.fusionRank,
  };
}

async function replaySample(
  sample: ManifestSample,
  db: QuranDB,
): Promise<{
  discovered: string[];
  diagnostics: (TrackerDiagnosticEvent & { timeSec: number })[];
  firstCommit: (TrackerDiagnosticEvent & { type: "commit"; timeSec: number }) | null;
  precedingDiscovery: (TrackerDiagnosticEvent & { type: "discovery_cycle"; timeSec: number }) | null;
  commitEvidence: AcousticEvidence | null;
  commitTokenLength: number;
}> {
  const diagnostics: (TrackerDiagnosticEvent & { timeSec: number })[] = [];
  let currentTimeSec = 0;
  let lastResult: TranscribeResult | null = null;
  let firstCommit: (TrackerDiagnosticEvent & { type: "commit"; timeSec: number }) | null = null;
  let precedingDiscovery: (TrackerDiagnosticEvent & { type: "discovery_cycle"; timeSec: number }) | null = null;
  let lastDiscovery: (TrackerDiagnosticEvent & { type: "discovery_cycle"; timeSec: number }) | null = null;
  let commitEvidence: AcousticEvidence | null = null;
  let commitTokenLength = 0;

  async function transcribeForReplay(audio: Float32Array): Promise<TranscribeResult> {
    lastResult = await transcribe(audio);
    return lastResult;
  }

  const tracker = new RecitationTracker(db, transcribeForReplay, {
    onDiagnostic: (event) => {
      const timed = { ...event, timeSec: Number(currentTimeSec.toFixed(3)) } as TrackerDiagnosticEvent & { timeSec: number };
      diagnostics.push(timed);
      if (timed.type === "discovery_cycle") {
        lastDiscovery = timed;
      }
      if (timed.type === "commit" && !firstCommit) {
        firstCommit = timed;
        precedingDiscovery = lastDiscovery;
        commitEvidence = lastResult?.acoustic ?? null;
        commitTokenLength = lastResult?.tokenIds?.length ?? 0;
      }
    },
  });

  const messages: WorkerOutbound[] = [];
  const audio = loadAudio(resolve(BENCHMARK, sample.file));
  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
    currentTimeSec = end / SAMPLE_RATE;
    messages.push(...(await tracker.feed(audio.slice(offset, end))));
  }

  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    currentTimeSec = audio.length / SAMPLE_RATE + (i + 1) * CHUNK_SECONDS;
    messages.push(...(await tracker.feed(silenceChunk)));
  }

  const discovered: string[] = [];
  for (const message of messages) {
    if (message.type !== "verse_match") continue;
    const key = `${message.surah}:${message.ayah}`;
    if (!discovered.includes(key)) discovered.push(key);
  }

  return {
    discovered,
    diagnostics,
    firstCommit,
    precedingDiscovery,
    commitEvidence,
    commitTokenLength,
  };
}

async function main() {
  const stability: StabilityReport = JSON.parse(readFileSync(STABILITY_JSON, "utf-8"));
  const manifest: { samples: ManifestSample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  const manifestById = new Map(manifest.samples.map((sample) => [sample.id, sample]));

  const wrongInitialRuns: Array<{
    sample: StabilitySample;
    run: StabilityRun;
    runIndex: number;
  }> = [];
  for (const sample of stability.samples) {
    sample.runs.forEach((run, runIndex) => {
      if (classifyRun(sample.expectedVerses, run.discoveredVerses) === "wrong_initial") {
        wrongInitialRuns.push({ sample, run, runIndex });
      }
    });
  }

  const replayRuns = wrongInitialRuns.slice(0, MAX_REPLAY_RUNS);
  console.log(
    `Auditing ${replayRuns.length}/${wrongInitialRuns.length} cached wrong_initial runs...`,
  );
  console.log("Loading ONNX model...");
  await createSession(resolve(ROOT, "public/fastconformer_phoneme_q8.onnx"));
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  const entries = [];
  const summary = {
    cachedWrongInitialRuns: wrongInitialRuns.length,
    replayLimit: MAX_REPLAY_RUNS,
    replayedRuns: replayRuns.length,
    replayedWrongInitialFirstCommit: 0,
    expectedInTop20: 0,
    expectedInTop64: 0,
    expectedInExpanded: 0,
    ctcBestEqualsExpected: 0,
    ctcBestDiffersFromCommitted: 0,
    deferralWouldPreventWrongCommit: 0,
  };

  for (let idx = 0; idx < replayRuns.length; idx++) {
    const { sample, run, runIndex } = replayRuns[idx];
    const manifestSample = manifestById.get(sample.id);
    if (!manifestSample) continue;
    if (idx % 10 === 0) {
      console.log(`  ${idx + 1}/${wrongInitialRuns.length}: ${sample.id}`);
    }

    const replay = await replaySample(manifestSample, db);
    const firstCommitRef = replay.firstCommit?.ref ?? null;
    const replayClass = classifyRun(sample.expectedVerses, replay.discovered);
    if (firstCommitRef && !sample.expectedVerses.includes(expandRef(firstCommitRef)[0])) {
      summary.replayedWrongInitialFirstCommit++;
    }

    let candidateAudit = null;
    if (replay.precedingDiscovery && replay.commitEvidence) {
      const text = replay.precedingDiscovery.text;
      const match = db.matchVerse(
        text,
        RAW_TRANSCRIPT_THRESHOLD,
        DISCOVERY_MAX_SPAN,
        null,
        5,
      );
      const textConfidenceLow = !match || match.score < ACOUSTIC_OVERRIDE_TEXT_THRESHOLD;
      const production = db.retrieveCandidates(text, {
        maxSpan: DISCOVERY_MAX_SPAN,
        hint: null,
        singleLimit: textConfidenceLow
          ? DISCOVERY_EXPANDED_CANDIDATES
          : DISCOVERY_TOP_SINGLE_CANDIDATES,
        topSurahs: textConfidenceLow ? 10 : DISCOVERY_TOP_SURAHS,
        spanLimit: DISCOVERY_TOP_SINGLE_CANDIDATES,
      }).combined;
      const expanded = db.retrieveCandidates(text, {
        maxSpan: DISCOVERY_MAX_SPAN,
        hint: null,
        singleLimit: DISCOVERY_EXPANDED_CANDIDATES,
        topSurahs: 10,
        spanLimit: DISCOVERY_TOP_SINGLE_CANDIDATES,
      }).combined;
      const rankedProduction = rankCandidates(
        production,
        replay.commitEvidence,
        replay.commitTokenLength,
        text,
      );
      const rankedExpanded = rankCandidates(
        expanded,
        replay.commitEvidence,
        replay.commitTokenLength,
        text,
      );
      const textSorted = rankedExpanded
        .slice()
        .sort((a, b) => b.textScore - a.textScore);
      const ctcSorted = rankedExpanded
        .filter((entry) => entry.ctcFeasible)
        .slice()
        .sort((a, b) => a.ctcScore - b.ctcScore);
      const textBest = textSorted[0] ?? null;
      const ctcBest = ctcSorted[0] ?? null;
      const fusionBest = rankedExpanded[0] ?? null;
      const expectedTop20 = expectedCandidates(sample.expectedVerses, rankedProduction.slice(0, 20));
      const expectedTop64 = expectedCandidates(sample.expectedVerses, rankedProduction.slice(0, 64));
      const expectedExpanded = expectedCandidates(sample.expectedVerses, rankedExpanded);
      const ctcBestIsExpected =
        !!ctcBest && refsEqual(expandRef(ctcBest.ref), sample.expectedVerses);
      const ctcBestDiffersFromCommitted = !!ctcBest && ctcBest.ref !== firstCommitRef;
      const firstCommitWrong =
        !!firstCommitRef &&
        !expandRef(firstCommitRef).some((ref) => sample.expectedVerses.includes(ref));
      const wouldPrevent =
        firstCommitWrong &&
        !!textBest &&
        !!ctcBest &&
        textBest.ref !== ctcBest.ref &&
        ctcBest.ref !== firstCommitRef;

      if (expectedTop20.length) summary.expectedInTop20++;
      if (expectedTop64.length) summary.expectedInTop64++;
      if (expectedExpanded.length) summary.expectedInExpanded++;
      if (ctcBestIsExpected) summary.ctcBestEqualsExpected++;
      if (ctcBestDiffersFromCommitted) summary.ctcBestDiffersFromCommitted++;
      if (wouldPrevent) summary.deferralWouldPreventWrongCommit++;

      candidateAudit = {
        text,
        productionCandidateCount: production.length,
        expandedCandidateCount: expanded.length,
        topDiagnosticCandidates: replay.precedingDiscovery.candidates,
        presence: {
          expectedInProductionTop20: expectedTop20.length > 0,
          expectedInProductionTop64: expectedTop64.length > 0,
          expectedInExpanded: expectedExpanded.length > 0,
        },
        expectedCandidates: expectedExpanded.slice(0, 12).map(compactCandidate),
        textBest: textBest ? compactCandidate(textBest) : null,
        ctcBest: ctcBest ? compactCandidate(ctcBest) : null,
        fusionBest: fusionBest ? compactCandidate(fusionBest) : null,
        ctcBestAgreesWithTextBest: !!textBest && !!ctcBest && textBest.ref === ctcBest.ref,
        ctcBestAgreesWithCommitted: !!ctcBest && ctcBest.ref === firstCommitRef,
        ctcBestIsExpected,
        wouldDeferringOnTextCtcDisagreementPreventWrongCommit: wouldPrevent,
        topExpandedByFusion: rankedExpanded.slice(0, 10).map(compactCandidate),
        topExpandedByCtc: ctcSorted.slice(0, 10).map(compactCandidate),
      };
    }

    entries.push({
      sampleId: sample.id,
      category: sample.category,
      runIndex,
      expectedRefs: sample.expectedVerses,
      cachedDiscoveredRefs: run.discoveredVerses,
      cachedClass: "wrong_initial",
      replayDiscoveredRefs: replay.discovered,
      replayClass,
      firstCommit: replay.firstCommit
        ? {
            ref: replay.firstCommit.ref,
            reason: replay.firstCommit.reason,
            origin: replay.firstCommit.origin ?? null,
            confidence: replay.firstCommit.confidence,
            effectiveScore: replay.firstCommit.effective_score ?? null,
            acousticMargin: replay.firstCommit.acoustic_margin ?? null,
            lengthFit: replay.firstCommit.length_fit ?? null,
            selectedRank: replay.firstCommit.selected_rank ?? null,
            selectedFusion: replay.firstCommit.selected_fusion ?? null,
            timeSec: replay.firstCommit.timeSec,
          }
        : null,
      candidateAudit,
    });
  }

  const output = {
    input: "test/stab-gate-on-v3.json",
    generatedAt: new Date().toISOString(),
    corpus: stability.corpus,
    summary,
    entries,
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  console.log(`JSON saved to ${OUT_JSON}`);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
