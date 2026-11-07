/** Whole-session counting-visit upper bound. No labels or primary windows enter the second engine. */
import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { CountingReadVisit, PhonemeEvidence } from "@tilawa/core";
import { insertionCells } from "../seam-insertion.js";
import { pauseMaskSpans } from "../pause-mask.js";
import { createEngine } from "../session.js";
import { loadNodeModel, readWav } from "../node.js";
import type { LockPoint } from "../gap-decode.js";

const { values } = parseArgs({ options: {
  model: { type: "string" }, wav: { type: "string" }, evidence: { type: "string" },
  "teacher-model": { type: "string" }, out: { type: "string" }, save: { type: "string" },
} });
if (!values.out || (!values.evidence && !(values.model && values.wav))) throw new Error("--out and --evidence or --model/--wav required");
const asset = (name: string) => JSON.parse(readFileSync(new URL(`../../../core/assets/${name}`, import.meta.url), "utf8"));
const index = loadReferenceIndex(asset("ref-v2.json")), phrases = asset("phrases-v1.json");
function review(chunks: PhonemeEvidence[]) {
  const visits = new Map<string, CountingReadVisit>();
  const locks: LockPoint[] = [];
  const events = [];
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases, onCountingVisit(v) { visits.set(JSON.stringify(v), v); } });
  const wordIndex = new Map(index.words.map((w, i) => [`${w.surah}:${w.ayah}:${w.word}`, i]));
  const note = (p: ReturnType<typeof tracker.push>) => {
    for (const e of p) if (e.type === "position" && e.state === "locked" && e.expected) {
      const wi = wordIndex.get(`${e.expected.surah}:${e.expected.ayah}:${e.expected.word}`);
      if (wi !== undefined && locks.at(-1)?.wi !== wi) locks.push({ wi, timeMs: e.timeMs });
    }
  };
  for (const e of insertionCells(chunks)) {
    const p = tracker.push(trackerInput(e)); note(p); events.push(...checker.push(e, p));
  }
  const p = tracker.end(); note(p); events.push(...checker.end(p));
  return { visits: [...visits.values()], locks, events, mistakes: checker.mistakes() };
}
function readEvidence(path: string): PhonemeEvidence[] {
  const d = JSON.parse(readFileSync(path, "utf8"));
  if (Array.isArray(d)) return d;
  return [{ inventory: d.inventory, frameMs: d.frameMs, phonemes: d.phonemes,
    ...(d.posteriors && { posteriors: { ...d.posteriors, startFrame: 0, nFrames: d.nFrames } }) }];
}
let report: unknown;
if (values.evidence) report = { provisionalUpperBound: true, ...review(readEvidence(values.evidence)) };
else {
  const manifest = JSON.parse(readFileSync(join(values.model!, "manifest.json"), "utf8"));
  const weights = readFileSync(join(values.model!, manifest.file));
  const sha256 = createHash("sha256").update(weights).digest("hex");
  if (weights.length !== manifest.bytes || sha256 !== manifest.sha256) throw new Error("Pause-mask export integrity check failed");
  console.log(`Verified pause-mask export: ${weights.length} bytes, SHA-256 ${sha256}`);
  const model = await loadNodeModel(values.model!);
  if (!model.manifest.pauseMask) throw new Error("The original native pause-mask export is required");
  const primary = values["teacher-model"] ? await loadNodeModel(values["teacher-model"]) : undefined;
  if (primary && !primary.insertionTeacher) throw new Error("Inherited insertion teacher is required");
  model.insertionTeacher = primary?.insertionTeacher;
  const wav = await readWav(values.wav!);
  const run = async (pauseMask?: ReturnType<typeof pauseMaskSpans>) => {
    const session = createEngine({ reference: index, phrases, model, recordEvidence: true,
      listener: { pauseMask } }).startFreeRecitation();
    for (let a = 0; a < wav.samples.length; a += wav.sampleRate)
      await session.pushAudio(wav.samples.subarray(a, a + wav.sampleRate), wav.sampleRate);
    await session.end();
    return session.evidence();
  };
  console.log("Running complete unmasked preparation");
  const warm = await run();
  const prepared = review(warm);
  const spans = pauseMaskSpans(warm, prepared.locks, index);
  console.log(`Running complete masked pass with ${spans.length} independently selected spans`);
  const masked = spans.length ? await run(spans) : warm;
  const final = review(masked);
  if (values.save) writeFileSync(values.save, JSON.stringify(masked));
  report = { provisionalUpperBound: true, manifest: model.manifest, teacherInstalled: !!model.insertionTeacher,
    spans, ...final };
  await model.backend.release();
  await primary?.backend.release();
}
writeFileSync(values.out, JSON.stringify(report));
console.log(`Whole-session counting-visit report: ${values.out}`);
