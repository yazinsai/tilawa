// Per-chunk latency of a correction-mode ZipformerSession with the structural rules off / on.
//
//   tsx structural-latency.ts --pcm-dir /tmp/lat --backend node|wasm [--expected]
// Each <id>.f32 in --pcm-dir is 16 kHz float32 PCM, with <id>.json {expected?}. Prints aggregates only.

import { createRequire } from "node:module";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ZipformerSession, type StructuralOptions, type ZipformerIo } from "@tilawa/core";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..", "..");
const arg = (k: string, d = ""): string => {
  const i = process.argv.indexOf(`--${k}`);
  return i >= 0 ? process.argv[i + 1]! : d;
};
const dir = arg("pcm-dir", "/tmp/lat");
const backend = arg("backend", "node");
const withExpected = process.argv.includes("--expected");
const CHUNK = 7680;
const require = createRequire(path.join(ROOT, "web", "frontend", "node_modules", "/"));
const ort = backend === "wasm" ? require("onnxruntime-web") : require("onnxruntime-node");
if (backend === "wasm") {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
} else {
  const create = ort.InferenceSession.create.bind(ort.InferenceSession);
  ort.InferenceSession.create = (m: unknown, o: Record<string, unknown> = {}) =>
    create(m, { ...o, intraOpNumThreads: 1, interOpNumThreads: 1 });
}
const model = new Uint8Array(readFileSync(process.env.ZIPFORMER_MODEL ?? "/tmp/models/a0w.int8.onnx"));
const io = JSON.parse(readFileSync(process.env.ZIPFORMER_IO ?? "/tmp/models/a0w.io.json", "utf8")) as ZipformerIo;
const corpus = JSON.parse(readFileSync(path.join(ROOT, "web", "frontend", "public", "zipformer_quran.json"), "utf8"));
const clips = readdirSync(dir).filter((f) => f.endsWith(".f32")).sort().map((f) => {
  const buf = readFileSync(path.join(dir, f));
  const meta = JSON.parse(readFileSync(path.join(dir, f.replace(/\.f32$/, ".json")), "utf8"));
  return { pcm: new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), expected: meta.expected ?? null };
});

const q = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))]! : 0;
};
const r1 = (x: number) => Math.round(x * 10) / 10;

const configs: Array<[string, StructuralOptions | undefined]> = [
  ["off", undefined],
  ["stop", { ayahOrder: true, similarVerse: true }],
  ["pause", { ayahOrder: true, similarVerse: true, timing: "pause" }],
];
async function main(): Promise<void> {
for (const [name, structural] of configs) {
  const host = await ZipformerSession.create({ ort, model, io, corpus, executionProviders: [backend === "wasm" ? "wasm" : "cpu"],
    ...(structural ? { structural } : {}) });
  const evalMs: number[] = [];
  const own = host as unknown as Record<string, unknown>;
  const proto = Object.getPrototypeOf(host) as Record<string, (...a: unknown[]) => unknown>;
  own.structuralEvaluate = (...a: unknown[]) => {
    const t = performance.now();
    const out = proto.structuralEvaluate!.apply(host, a);
    evalMs.push(performance.now() - t);
    return out;
  };
  const chunkMs: number[] = [];
  const stopMs: number[] = [];
  let audioS = 0;
  const drain = (msgs: Array<{ type: string; state?: { phase: string } }>): void => {
    for (const m of msgs) if (m.type === "correction" && m.state?.phase === "error") drain(host.correct("dismiss") as never);
  };
  for (const clip of clips) {
    host.reset();
    host.setMode("correction");
    host.setExpected(withExpected ? clip.expected : null);
    audioS += clip.pcm.length / 16000;
    for (let i = 0; i < clip.pcm.length; i += CHUNK) {
      const t = performance.now();
      const msgs = await host.feed(clip.pcm.subarray(i, Math.min(clip.pcm.length, i + CHUNK)));
      chunkMs.push(performance.now() - t);
      drain(msgs as never);
    }
    const t = performance.now();
    const msgs = await host.stop();
    stopMs.push(performance.now() - t);
    drain(msgs as never);
  }
  console.log(JSON.stringify({
    backend, config: name, expected: withExpected, clips: clips.length, audio_s: Math.round(audioS), chunks: chunkMs.length,
    chunk_ms: { p50: r1(q(chunkMs, 0.5)), p95: r1(q(chunkMs, 0.95)), p99: r1(q(chunkMs, 0.99)), max: r1(Math.max(...chunkMs)) },
    stop_ms: { p50: r1(q(stopMs, 0.5)), max: r1(Math.max(...stopMs)) },
    rule_eval_ms: evalMs.length ? { n: evalMs.length, p50: r1(q(evalMs, 0.5)), p95: r1(q(evalMs, 0.95)), max: r1(Math.max(...evalMs)) } : null,
  }));
}
}

void main();
