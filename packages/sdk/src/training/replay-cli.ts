/** Full-file deployment replay for inspecting a candidate listener; writes no labels or scores. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { loadReferenceIndex } from "@tilawa/core";
import { loadNodeModel } from "../node.js";
import { createEngine } from "../session.js";
import { decodeWav } from "../wav.js";
import { measureLatency } from "../latency.js";

const { values: v } = parseArgs({ options: { model: { type: "string" }, wav: { type: "string" }, out: { type: "string" } } });
if (!v.model || !v.wav || !v.out) throw new Error("--model --wav --out required");
const asset = (f: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${f}`, import.meta.url), "utf8"));
const reference = loadReferenceIndex(asset("ref-v2.json"));
const model = await loadNodeModel(resolve(v.model));
const engine = createEngine({ reference, phrases: asset("phrases-v1.json"), model, recordEvidence: true });
const wav = decodeWav(readFileSync(v.wav));
const session = engine.startFreeRecitation();
const raw = await measureLatency(session, wav.samples, wav.sampleRate, 20);
const evidence = session.evidence();
const replay = engine.startFreeRecitation();
const events = evidence.flatMap(e => replay.pushEvidence(e));
events.push(...replay.endEvidence());
const posteriors = evidence.flatMap(e => e.posteriors ? [e.posteriors] : []);
mkdirSync(v.out, { recursive: true });
writeFileSync(resolve(v.out, "replay.json"), JSON.stringify({ modelSha256: model.manifest.sha256,
  frameMs: model.manifest.frameMs, audioS: wav.samples.length / wav.sampleRate,
  nFrames: posteriors.reduce((n, p) => Math.max(n, p.startFrame + p.nFrames), 0),
  phonemes: evidence.flatMap(e => e.phonemes), rawEvents: raw.events, events,
  posteriors: posteriors.length ? { k: posteriors[0].k, ids: posteriors.flatMap(p => Array.from(p.ids)),
    probs: posteriors.flatMap(p => Array.from(p.probs)) } : undefined }));
console.log(JSON.stringify({ model: model.manifest.id, audioS: wav.samples.length / wav.sampleRate, computeS: raw.computeMs / 1000 }));
await model.backend.release();
