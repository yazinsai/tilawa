/** Run the unchanged shipped engine on the admitted real-phone training recordings. */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createEngine } from "../session.js";
import { loadNodeModel } from "../node.js";
import { decodeWav } from "../wav.js";
import { measureLatency } from "../latency.js";
import { loadReferenceIndex } from "@tilawa/core";
import type { CheckerEvent } from "@tilawa/core";
import { eligibleLearner, minedAccusations } from "./accusations.js";
import type { LearnerLabel } from "./accusations.js";

const { values: v } = parseArgs({ options: { model: { type: "string" }, data: { type: "string" }, out: { type: "string" } } });
if (!v.model || !v.data || !v.out) throw new Error("--model --data --out required");
const labels = readFileSync(resolve(v.data, "labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s) as LearnerLabel);
if (labels.some(l => l.split !== "train")) throw new Error("Non-training rows in mining input");
const asset = (f: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${f}`, import.meta.url), "utf8"));
const reference = loadReferenceIndex(asset("ref-v2.json"));
const model = await loadNodeModel(resolve(v.model));
const engine = createEngine({ reference, phrases: asset("phrases-v1.json"), model, recordEvidence: true });
mkdirSync(v.out, { recursive: true });
const rows = [];
for (const l of labels.filter(eligibleLearner)) {
  const path = resolve(v.data, l.wav);
  if (!path.startsWith(resolve(v.data, "wav") + "/")) throw new Error("Audio outside training directory");
  const wav = decodeWav(readFileSync(path));
  const session = engine.startFreeRecitation();
  const run = await measureLatency(session, wav.samples, wav.sampleRate, 20);
  // end() can replace context-free gaps in recorded evidence. Mine the engine's final decoded evidence.
  const replay = engine.startFreeRecitation();
  const events = session.evidence().flatMap(e => replay.pushEvidence(e));
  events.push(...replay.endEvidence());
  const checks = events.filter((e): e is CheckerEvent => e.type === "checker");
  const mined = minedAccusations(l, checks);
  rows.push({ label: l, modelSha256: model.manifest.sha256, mined, checks,
    phonemes: session.evidence().flatMap(e => e.phonemes), audioS: wav.samples.length / wav.sampleRate });
  if (rows.length % 20 === 0) console.log(JSON.stringify({ clips: rows.length, accusations: rows.reduce((n, r) => n + r.mined.length, 0) }));
}
writeFileSync(resolve(v.out, "mined.json"), JSON.stringify(rows));
console.log(JSON.stringify({ clips: rows.length, speakers: new Set(rows.map(r => r.label.speaker)).size,
  accusations: rows.reduce((n, r) => n + r.mined.length, 0), substitutions: rows.reduce((n, r) => n + r.label.events.filter(e => e.errorClass === "wrong_word").length, 0) }));
await model.backend.release();
