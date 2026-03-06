/**
 * Structured streaming diagnostics runner.
 *
 * Keeps validate-streaming.ts as the frozen scorer and emits a separate
 * per-sample JSON artifact with tracker diagnostics for discovery/tracking.
 *
 * Usage:
 *   tsx test/diagnose-streaming.ts
 *   tsx test/diagnose-streaming.ts --corpus=test_corpus_v2
 *   tsx test/diagnose-streaming.ts --sample retasy_003
 *   tsx test/diagnose-streaming.ts --out test/debug/custom.json
 */

import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { QuranDB } from "../src/lib/quran-db.ts";
import {
  RecitationTracker,
  type TrackerDiagnosticEvent,
  type TranscribeResult,
} from "../src/lib/tracker.ts";
import type { WorkerOutbound } from "../src/lib/types.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;
const CHUNK_SECONDS = 0.3;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SECONDS);
const TAIL_SILENCE_SECONDS = 4.0;

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

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const args = process.argv.slice(2);
const corpusArg = args.find((arg) => arg.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus";
const sampleArg = args.find((arg) => arg.startsWith("--sample="));
const sampleFilter = sampleArg ? sampleArg.split("=")[1] : null;
const outArg = args.find((arg) => arg.startsWith("--out="));
const outPath =
  outArg?.split("=")[1] ??
  resolve(
    ROOT,
    `test/debug/streaming/${new Date().toISOString().replace(/[:.]/g, "-")}-${corpusName}.json`,
  );
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

async function main() {
  await createSession(resolve(ROOT, "public/fastconformer_phoneme_q8.onnx"));
  const vocabJson = JSON.parse(
    readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"),
  );
  const decoder = new CTCDecoder(vocabJson);
  const quranData = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"),
  );
  const db = new QuranDB(quranData, decoder);
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );

  const samples = sampleFilter
    ? manifest.samples.filter((sample) => sample.id === sampleFilter)
    : manifest.samples;
  if (sampleFilter && !samples.length) {
    throw new Error(`Sample "${sampleFilter}" not found in ${corpusName}`);
  }

  async function transcribe(audio: Float32Array): Promise<TranscribeResult> {
    const { features, timeFrames } = await computeMelSpectrogram(audio);
    const { logprobs, timeSteps, vocabSize } = await runInference(
      features,
      80,
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

  const run = {
    corpus: corpusName,
    generated_at: new Date().toISOString(),
    model: "fastconformer_phoneme_q8.onnx",
    samples: [] as Array<Record<string, unknown>>,
  };

  for (const sample of samples) {
    const audioPath = resolve(BENCHMARK, sample.file);
    const audio = loadAudio(audioPath);
    const discoveredVerses: { surah: number; ayah: number }[] = [];
    const diagnostics: Array<TrackerDiagnosticEvent & { time_sec: number }> = [];
    const workerMessages: WorkerOutbound[] = [];
    let currentTimeSec = 0;

    const tracker = new RecitationTracker(db, transcribe, {
      onDiagnostic: (event) => diagnostics.push({ ...event, time_sec: currentTimeSec }),
    });

    for (let offset = 0; offset < audio.length; offset += CHUNK_SAMPLES) {
      currentTimeSec = offset / SAMPLE_RATE;
      const end = Math.min(offset + CHUNK_SAMPLES, audio.length);
      const messages = await tracker.feed(audio.slice(offset, end));
      workerMessages.push(...messages);
    }

    const silenceChunk = new Float32Array(CHUNK_SAMPLES);
    const silenceChunks = Math.ceil(
      (TAIL_SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SAMPLES,
    );
    for (let i = 0; i < silenceChunks; i++) {
      currentTimeSec = audio.length / SAMPLE_RATE + i * CHUNK_SECONDS;
      const messages = await tracker.feed(silenceChunk);
      workerMessages.push(...messages);
    }

    for (const message of workerMessages) {
      if (message.type !== "verse_match") continue;
      if (
        !discoveredVerses.some(
          (verse) => verse.surah === message.surah && verse.ayah === message.ayah,
        )
      ) {
        discoveredVerses.push({ surah: message.surah, ayah: message.ayah });
      }
    }

    const expectedSet = new Set(
      sample.expected_verses.map((verse) => `${verse.surah}:${verse.ayah}`),
    );
    const discoveredSet = new Set(
      discoveredVerses.map((verse) => `${verse.surah}:${verse.ayah}`),
    );
    const passed = [...expectedSet].every((verse) => discoveredSet.has(verse));

    run.samples.push({
      id: sample.id,
      file: sample.file,
      category: sample.category,
      source: sample.source,
      expected: [...expectedSet],
      discovered: [...discoveredSet],
      passed,
      duration_sec: Math.round((audio.length / SAMPLE_RATE) * 1000) / 1000,
      diagnostics,
      worker_messages: workerMessages,
    });
  }

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(run, null, 2));
  console.log(`Wrote ${run.samples.length} sample diagnostics to ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
