/**
 * Context-sweep diagnostic: measure phoneme WER as a function of input
 * audio length, to test whether the streaming/offline accuracy gap is
 * driven by the model needing more context than streaming inference
 * provides.
 *
 * For each sample, we run inference on prefixes of the audio at several
 * cut points (1s, 2s, 3s, 5s, 10s, full) and compute phoneme-level edit
 * distance against:
 *   (a) the expected verse phoneme reference (truncated proportionally)
 *   (b) the full-audio decode (truncated proportionally)
 *
 * Hypothesis: if WER vs reference drops sharply only when we feed full
 * audio, the model needs late context to disambiguate early frames →
 * cache-aware streaming retrain is justified. If WER is roughly flat
 * across context sizes (above ~2s), the streaming gap lives elsewhere.
 *
 * Usage:
 *   npx tsx test/diagnose-context-sweep.ts
 *   npx tsx test/diagnose-context-sweep.ts --max-samples 10
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { computeMelSpectrogram } from "../src/worker/mel.ts";
import { CTCDecoder } from "../src/worker/ctc-decode.ts";
import { createSession, runInference } from "./session-node.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const SAMPLE_RATE = 16000;

const CONTEXT_SECONDS = [1.0, 2.0, 3.0, 5.0, 10.0]; // plus "full"

const args = process.argv.slice(2);
const maxArg = args.find((a) => a.startsWith("--max-samples="));
const maxSamples = maxArg ? parseInt(maxArg.split("=")[1]) : 0;
const corpusArg = args.find((a) => a.startsWith("--corpus="));
const corpusName = corpusArg ? corpusArg.split("=")[1] : "test_corpus";
const BENCHMARK = resolve(ROOT, `../../benchmark/${corpusName}`);

interface Sample {
  id: string;
  file: string;
  surah: number;
  ayah: number;
  ayah_end: number | null;
  expected_verses: { surah: number; ayah: number }[];
}

interface VerseEntry {
  surah: number;
  ayah: number;
  phonemes: string;
}

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function tokenize(s: string): string[] {
  return s.trim().split(/\s+/).filter((t) => t.length > 0);
}

// Levenshtein on token sequences. Returns {edits, refLen}.
function tokenEdits(ref: string[], hyp: string[]): { edits: number; refLen: number } {
  const n = ref.length;
  const m = hyp.length;
  if (n === 0) return { edits: m, refLen: 0 };
  let prev = new Int32Array(m + 1);
  let cur = new Int32Array(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;
  for (let i = 1; i <= n; i++) {
    cur[0] = i;
    for (let j = 1; j <= m; j++) {
      const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1);
    }
    const tmp = prev;
    prev = cur;
    cur = tmp;
  }
  return { edits: prev[m], refLen: n };
}

async function transcribePhonemes(
  decoder: CTCDecoder,
  audio: Float32Array,
): Promise<string[]> {
  const { features, timeFrames } = await computeMelSpectrogram(audio);
  const { logprobs, timeSteps, vocabSize } = await runInference(
    features,
    80,
    timeFrames,
  );
  const result = decoder.decode(logprobs, timeSteps, vocabSize);
  return tokenize(result.rawPhonemes);
}

async function main() {
  console.log("=== CONTEXT-SWEEP DIAGNOSTIC ===\n");

  // Load model
  const modelPath = resolve(ROOT, "public/fastconformer_phoneme_q8.onnx");
  console.log("Loading ONNX model...");
  await createSession(modelPath);

  // Load vocab
  const vocabJson = JSON.parse(
    readFileSync(resolve(ROOT, "public/phoneme_vocab.json"), "utf-8"),
  );
  const decoder = new CTCDecoder(vocabJson);

  // Load Quran phoneme data and index by (surah, ayah)
  const quranPhonemes: VerseEntry[] = JSON.parse(
    readFileSync(resolve(ROOT, "public/quran_phonemes.json"), "utf-8"),
  );
  const phonemeMap = new Map<string, string>();
  for (const v of quranPhonemes) {
    phonemeMap.set(`${v.surah}:${v.ayah}`, v.phonemes);
  }

  // Load manifest
  const manifest: { samples: Sample[] } = JSON.parse(
    readFileSync(resolve(BENCHMARK, "manifest.json"), "utf-8"),
  );
  let samples = manifest.samples;
  if (maxSamples > 0) samples = samples.slice(0, maxSamples);

  console.log(`Evaluating ${samples.length} samples\n`);

  type Bucket = { totalEdits: number; totalRefLen: number; samples: number };
  const refBuckets = new Map<string, Bucket>();
  const fullBuckets = new Map<string, Bucket>();
  for (const sec of CONTEXT_SECONDS) {
    refBuckets.set(`${sec}s`, { totalEdits: 0, totalRefLen: 0, samples: 0 });
    fullBuckets.set(`${sec}s`, { totalEdits: 0, totalRefLen: 0, samples: 0 });
  }
  refBuckets.set("full", { totalEdits: 0, totalRefLen: 0, samples: 0 });

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const audioPath = resolve(BENCHMARK, sample.file);
    const audio = loadAudio(audioPath);
    const durationSec = audio.length / SAMPLE_RATE;

    // Build expected phoneme sequence from expected verses
    const expectedParts: string[] = [];
    for (const v of sample.expected_verses) {
      const ph = phonemeMap.get(`${v.surah}:${v.ayah}`);
      if (ph) expectedParts.push(ph);
    }
    if (expectedParts.length === 0) {
      console.log(`  SKIP ${sample.id} — no expected phonemes`);
      continue;
    }
    // Insert | between verses to mimic word-boundary continuation across verses
    const expectedTokens = tokenize(expectedParts.join(" | "));

    // Full-audio decode (reference for the "vs full" comparison)
    const fullTokens = await transcribePhonemes(decoder, audio);

    // (a) WER of prefix-decode vs FULL expected reference. As context grows,
    //     deletions go down monotonically; tells us how much of the verse is
    //     recovered with t seconds of audio.
    // (b) Stability: longest common prefix of prefix-decode and full-audio
    //     decode, divided by |prefix-decode|. If close to 1, the prefix
    //     decode is a stable prefix of the full decode → no late-context
    //     refinement. If << 1, full-audio context changes early predictions
    //     → cache-aware streaming would help.
    const fullVsRef = tokenEdits(expectedTokens, fullTokens);
    const fullWer = fullVsRef.edits / Math.max(fullVsRef.refLen, 1);
    refBuckets.get("full")!.totalEdits += fullVsRef.edits;
    refBuckets.get("full")!.totalRefLen += fullVsRef.refLen;
    refBuckets.get("full")!.samples += 1;

    const perSampleLine: string[] = [
      `${sample.id.padEnd(28)} dur=${durationSec.toFixed(1)}s ref=${expectedTokens.length}t fullDec=${fullTokens.length}t fullWer=${fullWer.toFixed(2)}`,
    ];

    for (const sec of CONTEXT_SECONDS) {
      if (sec >= durationSec) continue;
      const cutSamples = Math.floor(sec * SAMPLE_RATE);
      const prefixAudio = audio.slice(0, cutSamples);
      const prefixTokens = await transcribePhonemes(decoder, prefixAudio);

      // (a) vs full expected reference (deletions count)
      const refE = tokenEdits(expectedTokens, prefixTokens);
      const rb = refBuckets.get(`${sec}s`)!;
      rb.totalEdits += refE.edits;
      rb.totalRefLen += refE.refLen;
      rb.samples += 1;

      // (b) stability: how much of the prefix decode is a prefix of the full decode
      let lcp = 0;
      const limit = Math.min(prefixTokens.length, fullTokens.length);
      while (lcp < limit && prefixTokens[lcp] === fullTokens[lcp]) lcp++;
      const stability = prefixTokens.length > 0 ? lcp / prefixTokens.length : 1;
      const fb = fullBuckets.get(`${sec}s`)!;
      fb.totalEdits += prefixTokens.length - lcp; // mismatched-or-extra tokens
      fb.totalRefLen += prefixTokens.length;
      fb.samples += 1;

      perSampleLine.push(
        `  ${sec}s: dec=${prefixTokens.length}t ref_wer=${(refE.edits / refE.refLen).toFixed(2)} stab=${stability.toFixed(2)}`,
      );
    }

    console.log(`[${i + 1}/${samples.length}] ${perSampleLine.join("")}`);
  }

  // Summary
  console.log("\n" + "=".repeat(72));
  console.log("PHONEME WER vs EXPECTED REFERENCE (lower is better)");
  console.log("=".repeat(72));
  console.log(`  ${"context".padEnd(8)} ${"WER".padStart(6)}  n`);
  for (const sec of CONTEXT_SECONDS) {
    const b = refBuckets.get(`${sec}s`)!;
    if (b.totalRefLen === 0) {
      console.log(`  ${(sec + "s").padEnd(8)} ${"  N/A".padStart(6)}  ${b.samples}`);
      continue;
    }
    const wer = b.totalEdits / b.totalRefLen;
    console.log(
      `  ${(sec + "s").padEnd(8)} ${wer.toFixed(3).padStart(6)}  ${b.samples}`,
    );
  }
  const fb = refBuckets.get("full")!;
  if (fb.totalRefLen > 0) {
    console.log(
      `  ${"full".padEnd(8)} ${(fb.totalEdits / fb.totalRefLen).toFixed(3).padStart(6)}  ${fb.samples}`,
    );
  }

  console.log("\n" + "=".repeat(72));
  console.log(
    "PREFIX-DECODE STABILITY (1 - LCP/decodeLen): fraction of emitted tokens",
  );
  console.log(
    "that DIFFER from what full-audio context produces. Lower = more stable.",
  );
  console.log("=".repeat(72));
  console.log(`  ${"context".padEnd(8)} ${"instab".padStart(6)}  n`);
  for (const sec of CONTEXT_SECONDS) {
    const b = fullBuckets.get(`${sec}s`)!;
    if (b.totalRefLen === 0) {
      console.log(`  ${(sec + "s").padEnd(8)} ${"  N/A".padStart(6)}  ${b.samples}`);
      continue;
    }
    const instab = b.totalEdits / b.totalRefLen;
    console.log(
      `  ${(sec + "s").padEnd(8)} ${instab.toFixed(3).padStart(6)}  ${b.samples}`,
    );
  }
  console.log();
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
