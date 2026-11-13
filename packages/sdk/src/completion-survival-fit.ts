/** Training-only estimator. Reads only eligible help-train takes; never opens practice or scores warnings. */
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { isAbsolute, join, relative } from "node:path";
import { createTracker, loadReferenceIndex, trackerInput } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceAsset } from "@tilawa/core";
import { completionLogLikelihood, fitCompletionModel } from "../../core/dist/completion-survival.js";
import type { CompletionExample } from "../../core/dist/completion-survival.js";
import type { LocalSnapshot } from "../../core/dist/local.js";
import { completionExamples, completionTrainingLabels } from "./completion-training.js";
import type { CompletionLabel } from "./completion-training.js";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const help = realpathSync(join(data, "help-train"));
const bytes = readFileSync(join(help, "labels.jsonl"));
const labels = completionTrainingLabels(bytes.toString("utf8").trim().split("\n").map(s => JSON.parse(s)) as CompletionLabel[]);
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 3) throw new Error("Completion fitting needs at least three eligible training speakers");
const foldOf = new Map(speakers.map((s, i) => [s, i % 3]));
const refBytes = readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url));
const index = loadReferenceIndex(JSON.parse(refBytes.toString("utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../core/assets/phrases-v1.json", import.meta.url), "utf8"));
const digest = (x: Uint8Array | string) => createHash("sha256").update(x).digest("hex");
const extension = new URL(import.meta.url).pathname.endsWith(".ts") ? ".ts" : ".js";
const decoderHash = digest(["listener", "audio", "fbank"].map(f => readFileSync(new URL(f + extension, import.meta.url))).map(b => digest(b)).join(""));
const cache = process.env.COMPLETION_CACHE ?? join(out, "decodes"); mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const records: { speaker: string; examples: CompletionExample[] }[] = [], hashes: string[] = [];
try {
  for (const l of labels) {
    const path = realpathSync(join(help, l.wav)), inside = relative(help, path);
    if (inside === ".." || inside.startsWith("../") || isAbsolute(inside)) throw new Error("Training audio leaves help-train");
    const audio = readFileSync(path), hash = digest(audio); hashes.push(hash);
    const file = join(cache, `${model.manifest.sha256}-${decoderHash}-${hash}.json`);
    let evidence: PhonemeEvidence[];
    if (existsSync(file)) evidence = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(audio), listener = createListener(model); evidence = [];
      const stride = Math.max(1, Math.round(wav.sampleRate * 0.04));
      for (let at = 0; at < wav.samples.length; at += stride) {
        const e = await listener.push(wav.samples.subarray(at, at + stride), wav.sampleRate); if (e) evidence.push(e);
      }
      const e = await listener.end(); if (e) evidence.push(e);
      writeFileSync(file, JSON.stringify(evidence));
    }
    const ayahs = new Set(l.ayahs), transcript = new Set(index.words.flatMap((w, i) => ayahs.has(`${w.surah}:${w.ayah}`) ? [i] : []));
    const trace: LocalSnapshot[] = [];
    const tracker = createTracker(index, { phrases, acquireMs: 40, observeLocal: s => trace.push(s) });
    for (const e of evidence) tracker.push(trackerInput(e)); tracker.end();
    const examples = completionExamples(trace, transcript); records.push({ speaker: l.speaker, examples });
    console.log(JSON.stringify({ recordings: records.length, of: labels.length, prefixes: examples.length,
      complete: examples.filter(e => e.outcome === "complete").length, abandon: examples.filter(e => e.outcome === "abandon").length,
      censored: examples.filter(e => e.outcome === "censored").length }));
  }
  const selection: { ridge: number; full: number[]; endpoint: number[]; score: number }[] = [];
  for (const ridge of [0.001, 0.01, 0.1]) {
    const full: number[] = [], endpoint: number[] = [];
    for (let fold = 0; fold < 3; fold++) {
      const train = records.filter(r => foldOf.get(r.speaker) !== fold).flatMap(r => r.examples);
      const held = records.filter(r => foldOf.get(r.speaker) === fold).flatMap(r => r.examples);
      full.push(completionLogLikelihood(fitCompletionModel(train, false, ridge), held));
      endpoint.push(completionLogLikelihood(fitCompletionModel(train, true, ridge), held));
    }
    selection.push({ ridge, full, endpoint, score: [...full, ...endpoint].reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  const chosen = [...selection].sort((a, b) => b.score - a.score || b.ridge - a.ridge)[0];
  const examples = records.flatMap(r => r.examples);
  const pair = { full: fitCompletionModel(examples, false, chosen.ridge), endpoint: fitCompletionModel(examples, true, chosen.ridge) };
  writeFileSync(join(out, "completion-model.json"), JSON.stringify(pair));
  const report = { trainingOnly: true, recordings: labels.length, speakers: speakers.length, prefixes: examples.length,
    outcomes: Object.fromEntries(["complete", "abandon", "censored"].map(s => [s, examples.filter(e => e.outcome === s).length])),
    selection, chosen, modelSha256: model.manifest.sha256, decoderHash, referenceHash: digest(refBytes), labelsHash: digest(bytes),
    eligibleLabelsHash: digest(JSON.stringify(labels)), audioHashes: hashes,
    isolation: "none-labelled scored training takes without edits/repeats; all non-training or mixed-split speakers excluded",
    supervision: "causal actual-tracker prefixes; later transcript-consistent in-order progress with the same run completes; unresolved endings censored",
    control: "eight features, identical recordings, event/censor likelihood, training compute, ridge and 3 nats/edit calibration; endpoint history ablated" };
  writeFileSync(join(out, "completion-fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await model.backend.release(); }
