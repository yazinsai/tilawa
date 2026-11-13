/** Read-only comparison after training is frozen. Labels never enter scorer inference. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { createTracker, trackerInput, loadReferenceIndex, createChecker, PosteriorFrames } from "@tilawa/core";
import type { EngineEvent, PhonemeEvidence } from "@tilawa/core";
import { createAcousticTransitionScorer, posteriorTransitionRows } from "../../core/src/acoustic-transition.js";
import { chunks } from "../../core/src/bench/lab.js";
import { Fbank } from "./fbank.js";
import { Resampler } from "./audio.js";
import { decodeWav } from "./wav.js";
import { carryAcousticTransitions, summarizeTransition } from "./acoustic-transition-features.js";

const { values } = parseArgs({ options: { evidence: { type: "string" }, audio: { type: "string" },
  fit: { type: "string" }, at: { type: "string" }, apply: { type: "boolean", default: false } } });
if (!values.evidence || !values.audio || !values.fit || !values.at || !process.env.LAB_OUT) throw new Error("evidence, audio, fit directory, at and LAB_OUT required");
const fit = JSON.parse(readFileSync(join(values.fit, "acoustic-transition-fit.json"), "utf8"));
if (fit.trainingOnly !== true || !fit.selected || !fit.trainingSha256) throw new Error("Frozen training-only fit required");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const models = Object.fromEntries(["original", "posterior"].map(arm => [arm,
  JSON.parse(readFileSync(join(values.fit!, `acoustic-transition-${arm}-model.json`), "utf8"))]));
const scorers = Object.fromEntries(Object.entries(models).map(([arm, model]) => [arm, createAcousticTransitionScorer(model, index.inventoryId)]));
for (const arm of ["original", "posterior"]) {
  for (const sample of JSON.parse(readFileSync(join(values.fit, `acoustic-transition-${arm}-parity.json`), "utf8"))) {
    for (let i = 0; i < sample.choices.length; i++) {
      const cost = scorers[arm](sample.rows, sample.choices[i]);
      if (cost === null || Math.abs(cost - sample.costs[i]) > 1e-4 * Math.max(1, Math.abs(sample.costs[i]))) throw new Error("Python/JS parity failed");
    }
  }
}
const raw = readFileSync(values.evidence, "utf8"), evidence = JSON.parse(raw);
const wav = decodeWav(readFileSync(values.audio));
let samples = wav.samples;
if (wav.sampleRate !== 16000) {
  const r = new Resampler(wav.sampleRate, 16000), parts = [r.push(samples), r.flush()];
  samples = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  samples.set(parts[0]); samples.set(parts[1], parts[0].length);
}
const stride = evidence.frameMs / 10;
if (!Number.isInteger(stride)) throw new Error("Unsupported original feature geometry");
const originalFrames = new Fbank().push(samples);
const originalFeatures = { version: 1 as const, startFrame: 0, nFrames: evidence.nFrames,
  width: 2 * (index.tokens.size + 1), values: [] as number[] };
for (let t = 0; t < evidence.nFrames; t++) {
  const window = originalFrames.slice(t * stride, (t + 1) * stride);
  if (window.length !== stride) throw new Error("Evidence timeline exceeds original audio");
  originalFeatures.values.push(...summarizeTransition(window, originalFeatures.width));
}
const cells = carryAcousticTransitions(chunks(evidence), [originalFeatures]);
const control = new PosteriorFrames(index.tokens.size);
control.append({ ...evidence.posteriors, startFrame: 0, nFrames: evidence.nFrames });
const comparisons: unknown[] = [], events: EngineEvent[] = [];
let time = 0;
const at = Number(values.at);
const tracker = createTracker(index, { mode: "free", phrases, acousticTransition: { model: models.original,
  audit: !values.apply, inspect(candidates, original, selected, interval) {
    if (!interval || Math.abs(time - at) > 5) return;
    const rows = posteriorTransitionRows(control, interval.fromFrame, interval.toFrame, models.posterior.frames);
    comparisons.push({ time, interval, original, selected, candidates: candidates.map(c => ({ ...c,
      word: index.words[index.wordAt(Math.max(0, c.offset - 1))],
      controlCost: rows && scorers.posterior(rows, c.pronunciation) })) });
  } } });
const checker = createChecker(index, { phrases });
for (const e of cells as PhonemeEvidence[]) {
  time = (e.decodedTo ?? 0) * e.frameMs / 1000;
  const p = tracker.push(trackerInput(e));
  events.push(...p, ...checker.push(e, p));
}
const final = tracker.end(); events.push(...final, ...checker.end(final));
writeFileSync(join(process.env.LAB_OUT, `acoustic-transition-${values.apply ? "apply" : "audit"}.json`), JSON.stringify({
  evidenceSha256: createHash("sha256").update(raw).digest("hex"), fitSha256: fit.trainingSha256,
  comparisons, events: events.filter(e => Math.abs(e.timeMs / 1000 - at) <= 10 || e.type === "checker" && e.state === "confirmed"),
}, null, 2));
console.log(JSON.stringify({ comparisons: comparisons.length, apply: values.apply }));
