/** CPU training input extraction. No labels other than authorized none-labelled training takes are decoded. */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createHash } from "node:crypto";
import { loadNodeModel, decodeWav, createListener, SpectralBands } from "../packages/sdk/src/node.js";

const { values } = parseArgs({ options: { data: { type: "string" }, model: { type: "string" }, out: { type: "string" } } });
if (!values.data || !values.model || !values.out) throw new Error("--data, --model and --out required");
const root = resolve(values.data), out = resolve(values.out);
const rows = readFileSync(resolve(root, "labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s));
// The help-train directory excludes release speakers by contract. Also exclude every speaker explicitly
// marked eval/test/release in supplied labels; never split a speaker between training and release.
const excluded = new Set(rows.filter(r => ["eval", "test", "release"].includes(r.split)).map(r => r.speaker));
const clean = rows.filter(r => r.kind === "none" && r.speaker && !excluded.has(r.speaker) &&
  (r.events ?? []).length === 0 && (r.non_errors ?? []).length === 0);
if (new Set(clean.map(r => r.speaker)).size < 3) throw new Error("Need at least three authorized training speakers' none-labelled takes");
const ref = JSON.parse(readFileSync(new URL("../packages/core/assets/ref-v2.json", import.meta.url), "utf8"));
const ayahs = new Map<string, string>(ref.ayahs.map((a: { s: number; a: number; p: string }) => [`${a.s}:${a.a}`, a.p]));
const model = await loadNodeModel(resolve(values.model));
// This utility must seed alignments with the unchanged primary listener, never its auxiliary judges.
const edges = Array.from({ length: 25 }, (_, i) => Math.round(1 + i * 244 / 24));
mkdirSync(out, { recursive: true });
const plan = clean.sort((a, b) => a.id.localeCompare(b.id));
for (let i = 0; i < plan.length; i++) {
  const row = plan[i], path = resolve(root, row.wav);
  if (!path.startsWith(root + "/")) throw new Error("Training audio must be inside the authorized directory");
  const audio = decodeWav(readFileSync(path));
  if (audio.sampleRate !== 16000) throw new Error("Prepare expects supplied 16 kHz training WAVs");
  const labels = row.ayahs.map((a: string) => ayahs.get(a));
  if (labels.some((p: string | undefined) => !p)) throw new Error("Missing canonical pronunciation");
  const listener = createListener(model, { posteriors: true, topK: model.manifest.tokens.length + 1 });
  const chunks = [];
  for (let a = 0; a < audio.samples.length; a += 320) {
    const e = await listener.push(audio.samples.subarray(a, a + 320), audio.sampleRate);
    if (e) chunks.push(e);
  }
  const tail = await listener.end(); if (tail) chunks.push(tail);
  const lp: number[][] = [];
  for (const c of chunks) if (c.posteriors) {
    const p = c.posteriors;
    for (let f = 0; f < p.nFrames; f++) {
      const r = Array(model.manifest.tokens.length + 1).fill(Math.log(1e-6));
      for (let k = 0; k < p.k; k++) r[p.ids[f * p.k + k]] = Math.log(Math.max(1e-6, p.probs[f * p.k + k]));
      lp.push(r);
    }
  }
  const key = createHash("sha256").update(row.id).digest("hex");
  writeFileSync(resolve(out, key + ".json"), JSON.stringify({ speaker: row.speaker, transcript: labels.join(""),
    x: new SpectralBands(edges).push(audio.samples).map(f => Array.from(f)), lp }));
  console.log(`${i + 1}/${plan.length}`);
}
writeFileSync(resolve(out, "input.json"), JSON.stringify({ inventory: model.manifest.inventory,
  tokens: model.manifest.tokens, edges, recordings: plan.length, excludedSpeakers: excluded.size }));
await model.backend.release();
