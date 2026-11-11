/** Produce primary listener trajectories for authorized none-labelled training takes, without tracker labels. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import type { PhonemeEvidence } from "@tilawa/core";
import { createListener } from "../listener.js";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";

interface Label { id: string; speaker: string; split: string; kind: string; wav: string; events: unknown[]; non_errors: unknown[]; exclude?: unknown }
const { values } = parseArgs({ options: { "help-dir": { type: "string" }, model: { type: "string" }, out: { type: "string" } } });
if (!values["help-dir"] || !values.model || !values.out) throw new Error("--help-dir, --model and --out are required");
const dir = resolve(values["help-dir"]), out = resolve(values.out);
const labels: Label[] = readFileSync(resolve(dir, "labels.jsonl"), "utf8").trim().split("\n").map(l => JSON.parse(l));
const training = (l: Label) => ["train", "training"].includes(l.split);
const held = new Set(labels.filter(l => !training(l)).map(l => l.speaker));
const selected = labels.filter(l => training(l) && !held.has(l.speaker) && l.kind === "none" && !l.events?.length && !l.non_errors?.length && !l.exclude);
const shard = Number(process.env.LAB_SHARD ?? 0), shards = Number(process.env.LAB_SHARDS ?? 1);
mkdirSync(out, { recursive: true });
const model = await loadNodeModel(resolve(values.model));
for (const [i, label] of selected.entries()) {
  if (i % shards !== shard) continue;
  const file = resolve(out, `${label.id}.evidence.json`);
  if (existsSync(file)) continue;
  const wav = decodeWav(readFileSync(resolve(dir, label.wav)));
  const listener = createListener(model, { posteriors: true });
  const chunks: PhonemeEvidence[] = [];
  const size = Math.round(wav.sampleRate * 0.32);
  for (let at = 0; at < wav.samples.length; at += size) {
    const e = await listener.push(wav.samples.subarray(at, at + size), wav.sampleRate);
    if (e) chunks.push(e);
  }
  const tail = await listener.end();
  if (tail) chunks.push(tail);
  const rows = chunks.flatMap(e => e.posteriors ? [e.posteriors] : []);
  writeFileSync(file, JSON.stringify({ inventory: model.manifest.inventory, frameMs: model.manifest.frameMs,
    phonemes: chunks.flatMap(e => e.phonemes), posteriors: { k: rows[0].k,
      ids: rows.flatMap(p => Array.from(p.ids)), probs: rows.flatMap(p => Array.from(p.probs)) } }));
  console.log(JSON.stringify({ id: label.id, speaker: label.speaker, frames: listener.decoded }));
}
await model.backend.release();
