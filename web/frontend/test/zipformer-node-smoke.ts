/**
 * Node smoke for the Zipformer browser host. Feeds 001002.mp3 through the
 * same ZipformerSession the worker uses (onnxruntime-node, cpu EP).
 *
 *   npx tsx test/zipformer-node-smoke.ts
 */
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { WorkerOutbound } from "../src/lib/types.ts";
import { displayQuranFromRaw, ZipformerSession } from "../src/worker/zipformer-session.ts";
import type { ZipformerIo } from "@tilawa/core";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "..");
const WORKTREE = resolve(FRONTEND, "../..");
const SAMPLE_RATE = 16000;

const MODEL =
  process.env.ZIPFORMER_MODEL ??
  resolve(FRONTEND, "public/models/zipformer_a0w_ep1_a05.int8.onnx");
const IO_PATH =
  process.env.ZIPFORMER_IO ??
  resolve(FRONTEND, "public/models/zipformer_a0w_ep1_a05.io.json");
const CORPUS =
  process.env.ZIPFORMER_CORPUS ??
  resolve(FRONTEND, "public/zipformer_quran.json");
const QURAN = resolve(FRONTEND, "public/quran.json");
const AUDIO =
  process.argv[2] ??
  resolve(WORKTREE, "lab/benchmark/test_corpus/001002.mp3");
const ORT_DIR = process.env.ZIPFORMER_ORT_DIR ?? resolve(FRONTEND, "node_modules");

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function requireExisting(path: string, hint: string): void {
  if (!existsSync(path)) {
    throw new Error(`missing ${path}\n${hint}`);
  }
}

requireExisting(MODEL, "Run: bash web/frontend/scripts/fetch-zipformer-assets.sh");
requireExisting(IO_PATH, "Run: bash web/frontend/scripts/fetch-zipformer-assets.sh");
requireExisting(CORPUS, "Run: bash web/frontend/scripts/fetch-zipformer-assets.sh");
requireExisting(QURAN, "public/quran.json should be in the repo");
requireExisting(AUDIO, "expected lab/benchmark/test_corpus/001002.mp3");

const require = createRequire(`${ORT_DIR}/`);
const ort = require("onnxruntime-node");

const io = JSON.parse(readFileSync(IO_PATH, "utf8")) as ZipformerIo;
const corpusJson = JSON.parse(readFileSync(CORPUS, "utf8"));
const quranDb = displayQuranFromRaw(JSON.parse(readFileSync(QURAN, "utf8")));
const pcm = loadAudio(AUDIO);

const host = await ZipformerSession.create({
  ort,
  model: new Uint8Array(readFileSync(MODEL)),
  io,
  corpus: corpusJson,
  quran: quranDb,
  executionProviders: ["cpu"],
});

const CHUNK = 7680;
const events: WorkerOutbound[] = [];
for (let i = 0; i < pcm.length; i += CHUNK) {
  events.push(...await host.feed(pcm.subarray(i, Math.min(pcm.length, i + CHUNK))));
}
events.push(...await host.stop());

const matches = events.filter((e) => e.type === "verse_match");
const finals = events.filter((e) => e.type === "final_sequence");
const lastFinal = finals.at(-1);
const refs = (lastFinal && lastFinal.type === "final_sequence")
  ? lastFinal.verses.map((v) => `${v.surah}:${v.ayah}`)
  : matches.map((e) => (e.type === "verse_match" ? `${e.surah}:${e.ayah}` : ""));

console.log(JSON.stringify({
  audio: AUDIO,
  samples: pcm.length,
  eventTypes: events.reduce<Record<string, number>>((acc, e) => {
    acc[e.type] = (acc[e.type] ?? 0) + 1;
    return acc;
  }, {}),
  verse_match: matches.map((e) => (
    e.type === "verse_match" ? { surah: e.surah, ayah: e.ayah, confidence: e.confidence } : e
  )),
  final_sequence: lastFinal && lastFinal.type === "final_sequence" ? lastFinal.verses : [],
  transcript: events.filter((e) => e.type === "raw_transcript").at(-1),
  refs,
}, null, 2));

const ok = refs.includes("1:2");
if (!ok) {
  console.error(`expected 1:2, got ${refs.join(",") || "(none)"}`);
  process.exit(1);
}
console.error(`ok: 1:2 (${events.length} events)`);
