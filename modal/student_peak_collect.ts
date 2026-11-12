/** Cache deployment-exact frame outputs for occurrence-calibration training. No checker labels or word slots. */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { parseArgs } from "node:util";
import { loadNodeModel, decodeWav } from "../packages/sdk/src/node.js";
import { createListener } from "../packages/sdk/src/listener.js";
import type { PhonemeEvidence } from "../packages/core/src/evidence.js";

const { values } = parseArgs({ options: {
  model: { type: "string" }, items: { type: "string" }, out: { type: "string" },
} });
if (!values.model || !values.items || !values.out) throw new Error("--model, --items and --out required");
const items = JSON.parse(readFileSync(values.items, "utf8")) as {
  id: string; speaker: string; wav: string; transcripts: string[];
  source: { kind: string; split: string; labelsSha256: string };
}[];
for (const item of items) {
  if (!item.speaker || !item.transcripts.length || item.source?.kind !== "none" || item.source?.split !== "train") {
    throw new Error("Only prepared none-labelled training-speaker recordings can enter calibration");
  }
}
const model = await loadNodeModel(values.model);
delete model.runConfidence;
const backendRun = model.backend.run.bind(model.backend);
let rows: number[][] = [];
model.backend.run = async feeds => {
  const out = await backendRun(feeds);
  const flat = out.log_probs.data as Float32Array;
  const vocab = model.manifest.tokens.length + 1;
  for (let i = 0; i < flat.length; i += vocab) rows.push(Array.from(flat.subarray(i, i + vocab)));
  return out;
};
mkdirSync(values.out, { recursive: true });
for (const item of items) {
  rows = [];
  const wav = decodeWav(readFileSync(item.wav));
  const listener = createListener(model, { posteriors: true });
  const chunks: PhonemeEvidence[] = [];
  const step = Math.round(wav.sampleRate * 0.02);
  for (let i = 0; i < wav.samples.length; i += step) {
    const e = await listener.push(wav.samples.subarray(i, i + step), wav.sampleRate);
    if (e) chunks.push(e);
  }
  const end = await listener.end();
  if (end) chunks.push(end);
  const n = chunks.reduce((a, e) => a + (e.posteriors?.nFrames ?? 0), 0);
  const destination = resolve(values.out, `${item.id}.json`);
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(destination, JSON.stringify({ ...item, modelSha256: model.manifest.sha256,
    inventory: model.manifest.inventory, tokens: model.manifest.tokens,
    phonemes: chunks.flatMap(e => e.phonemes), logProbs: rows.slice(0, n) }));
  console.log(`${item.id}: ${n} frames`);
}
await model.backend.release();
