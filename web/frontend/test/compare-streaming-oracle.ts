/**
 * Compare browser-style streaming output against a full-file ONNX oracle.
 *
 * This is diagnostic-only. It helps separate:
 *   - model/matcher floor: full-file prediction is already wrong
 *   - streaming tracker loss: full-file is exact but streaming misses/extras
 *   - corpus/label ambiguity: neither path matches expected cleanly
 *
 * Usage:
 *   npx tsx test/compare-streaming-oracle.ts --corpus=test_corpus_v3 --sample=ea_alafasy_002143
 *   npx tsx test/compare-streaming-oracle.ts --corpus=test_corpus_v3 --limit=10 --json=test/oracle.json
 */

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { QuranDB } from "../src/lib/quran-db.ts";
import { RecitationTracker, type BeamVerseMatch, type TranscribeResult } from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { beamSearchDecode } from "../src/worker/beam-decode.ts";
import { buildTrie, type CompactTrie } from "../src/lib/phoneme-trie.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

const args = process.argv.slice(2);
const corpusArg = args.find((arg) => arg.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus";
const sampleArg = args.find((arg) => arg.startsWith("--sample="));
const sampleFilter = sampleArg ? sampleArg.split("=")[1] : null;
const limitArg = args.find((arg) => arg.startsWith("--limit="));
const limit = limitArg ? Number(limitArg.split("=")[1]) : null;
const jsonArg = args.find((arg) => arg.startsWith("--json="));
const jsonOut = jsonArg ? jsonArg.split("=")[1] : null;
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

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

interface Comparison {
  id: string;
  category: string;
  source: string;
  expected: string[];
  streaming: string[];
  fullFile: string[];
  fullFileScore: number;
  fullFileTranscriptChars: number;
  streamingExact: boolean;
  fullFileExact: boolean;
  shape: string;
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

function refsEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((ref, idx) => b[idx] === ref);
}

function classify(expected: string[], streaming: string[], fullFile: string[]): string {
  const streamingExact = refsEqual(expected, streaming);
  const fullFileExact = refsEqual(expected, fullFile);
  if (streamingExact && fullFileExact) return "both_exact";
  if (!streamingExact && fullFileExact) return "streaming_tracker_loss";
  if (streamingExact && !fullFileExact) return "streaming_beats_full_file";
  if (fullFile.length === 0) return "full_file_no_match";

  const expectedSet = new Set(expected);
  const fullHasAny = fullFile.some((ref) => expectedSet.has(ref));
  const streamingHasAny = streaming.some((ref) => expectedSet.has(ref));
  if (!fullHasAny && !streamingHasAny) return "model_or_label_floor";
  if (fullHasAny && !streamingHasAny) return "streaming_lost_expected";
  if (!fullHasAny && streamingHasAny) return "full_file_lost_expected";
  return "both_partial_or_extra";
}

async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(features, 80, timeFrames);
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
    acoustic: { logprobs, timeSteps, vocabSize, blankId: decoder.getBlankId() },
    beamMatches,
  };
}

async function runStreaming(db: QuranDB, audio: Float32Array): Promise<string[]> {
  const tracker = new RecitationTracker(db, transcribe);
  const messages: WorkerOutbound[] = [];

  for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
    messages.push(...(await tracker.feed(audio.slice(offset, offset + CHUNK_SAMPLES))));
  }

  const silenceChunk = new Float32Array(CHUNK_SAMPLES);
  const silenceChunks = Math.ceil((TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES);
  for (let i = 0; i < silenceChunks; i++) {
    messages.push(...(await tracker.feed(silenceChunk)));
  }

  const refs: string[] = [];
  for (const message of messages) {
    if (message.type !== "verse_match") continue;
    const ref = `${message.surah}:${message.ayah}`;
    if (!refs.includes(ref)) refs.push(ref);
  }
  return refs;
}

async function runFullFile(db: QuranDB, audio: Float32Array): Promise<{
  refs: string[];
  score: number;
  transcriptChars: number;
}> {
  const result = await transcribe(audio);
  const match = db.matchVerse(result.text, 0.25, 4, null, 0);
  if (!match || match.score < 0.45) {
    return { refs: [], score: match?.score ?? 0, transcriptChars: result.text.length };
  }

  const refs: string[] = [];
  const end = match.ayah_end ?? match.ayah;
  for (let ayah = match.ayah; ayah <= end; ayah++) {
    refs.push(`${match.surah}:${ayah}`);
  }
  return { refs, score: match.score, transcriptChars: result.text.length };
}

async function main() {
  await createSession(resolve(ROOT, "public/fastconformer_phoneme_q8.onnx"));
  const vocabJson = JSON.parse(readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"));
  decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"));
  const db = new QuranDB(quranData, decoder);
  trie = buildTrie(quranData, vocabJson, 3).trie;

  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  let samples = manifest.samples;
  if (sampleFilter) samples = samples.filter((sample) => sample.id === sampleFilter);
  if (limit !== null) samples = samples.slice(0, limit);
  if (sampleFilter && samples.length === 0) {
    throw new Error(`Sample "${sampleFilter}" not found in ${corpusName}`);
  }

  const comparisons: Comparison[] = [];
  for (const sample of samples) {
    const audio = loadAudio(resolve(BENCHMARK, sample.file));
    const expected = sample.expected_verses.map((verse) => `${verse.surah}:${verse.ayah}`);
    const streaming = await runStreaming(db, audio);
    const fullFile = await runFullFile(db, audio);
    const comparison: Comparison = {
      id: sample.id,
      category: sample.category,
      source: sample.source,
      expected,
      streaming,
      fullFile: fullFile.refs,
      fullFileScore: Math.round(fullFile.score * 1000) / 1000,
      fullFileTranscriptChars: fullFile.transcriptChars,
      streamingExact: refsEqual(expected, streaming),
      fullFileExact: refsEqual(expected, fullFile.refs),
      shape: classify(expected, streaming, fullFile.refs),
    };
    comparisons.push(comparison);
    console.log(
      `${comparison.shape.padEnd(24)} ${sample.id} expected=[${expected.join(", ")}] ` +
        `streaming=[${streaming.join(", ")}] full=[${fullFile.refs.join(", ")}]`,
    );
  }

  const shapeCounts = new Map<string, number>();
  for (const comparison of comparisons) {
    shapeCounts.set(comparison.shape, (shapeCounts.get(comparison.shape) ?? 0) + 1);
  }
  console.log("\nSummary:");
  for (const [shape, count] of [...shapeCounts.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${shape}: ${count}`);
  }

  if (jsonOut) {
    writeFileSync(
      resolve(process.cwd(), jsonOut),
      JSON.stringify({ corpus: corpusName, comparisons }, null, 2),
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
