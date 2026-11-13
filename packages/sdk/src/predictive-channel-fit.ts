/** Training-only: none-labelled training speakers, no practice input or destination supervision. */
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, relative, isAbsolute } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitPredictiveChannel, predictiveCalibration, predictiveLogP } from "../../core/dist/predictive-channel.js";
import type { PredictiveRecording } from "../../core/dist/predictive-channel.js";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { predictiveSpeakerFolds, predictiveTrainingLabels } from "./predictive-channel-training.js";
import type { PredictiveTrainingLabel } from "./predictive-channel-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const help = realpathSync(join(data, "help-train"));
const labelBytes = readFileSync(join(help, "labels.jsonl"));
const labels = predictiveTrainingLabels(labelBytes.toString("utf8").trim().split("\n").map(s => JSON.parse(s)) as PredictiveTrainingLabel[]);
const groups = predictiveSpeakerFolds(labels), folds = Math.min(3, groups.size);
const referenceBytes = readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url));
const reference = loadReferenceIndex(JSON.parse(referenceBytes.toString("utf8")) as ReferenceAsset);
const digest = (data: Uint8Array | string) => createHash("sha256").update(data).digest("hex");
const extension = new URL(import.meta.url).pathname.endsWith(".ts") ? ".ts" : ".js";
const decoderSha256 = digest(["listener", "audio", "fbank"].map(file =>
  readFileSync(new URL(file + extension, import.meta.url), "utf8")).join("\n"));
const cache = process.env.PREDICTIVE_CACHE ?? join(out, "decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: PredictiveRecording }[] = [], audioHashes: string[] = [];
  for (const l of labels) {
    const path = realpathSync(join(help, l.wav)), inside = relative(help, path);
    if (inside === ".." || inside.startsWith("../") || isAbsolute(inside)) throw new Error("Training audio leaves help-train");
    const bytes = readFileSync(path), hash = digest(bytes); audioHashes.push(hash);
    const file = join(cache, `${model.manifest.sha256}-${decoderSha256}-${hash}.json`);
    let observations: DecodedPhoneme[];
    if (existsSync(file)) observations = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(bytes), listener = createListener(model);
      observations = [];
      const stride = Math.max(1, Math.round(wav.sampleRate * 0.04));
      for (let at = 0; at < wav.samples.length; at += stride) {
        const e = await listener.push(wav.samples.subarray(at, at + stride), wav.sampleRate);
        if (e) observations.push(...e.phonemes);
      }
      const tail = await listener.end(); if (tail) observations.push(...tail.phonemes);
      writeFileSync(file, JSON.stringify(observations));
    }
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !observations.length || words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Empty or noncontiguous training take");
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), heard: observations.map(p => p.p).join("") } });
    console.log(`Decoded ${clips.length}/${labels.length} eligible none-labelled training takes`);
  }
  const inventory = [...reference.tokens];
  const selection: { context: number; pooling: number; forward: number[]; reverse: number[]; sum: number; calibration: number[] }[] = [];
  for (const context of [1, 2]) for (const pooling of [8, 32]) {
    const forward: number[] = [], reverse: number[] = [], calibration: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => groups.get(c.speaker) !== fold).map(c => c.record);
      const held = clips.filter(c => groups.get(c.speaker) === fold);
      const f = fitPredictiveChannel(train, inventory, context, pooling, false);
      const r = fitPredictiveChannel(train, inventory, context, pooling, true);
      forward.push(held.reduce((sum, c) => sum + predictiveLogP(f, c.record), 0));
      reverse.push(held.reduce((sum, c) => sum + predictiveLogP(r, c.record, true), 0));
      calibration.push(predictiveCalibration([f, r]));
    }
    selection.push({ context, pooling, forward, reverse, calibration, sum: [...forward, ...reverse].reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  const selected = [...selection].sort((a, b) => b.sum - a.sum || a.context - b.context || b.pooling - a.pooling)[0];
  const records = clips.map(c => c.record);
  const pair = { forward: fitPredictiveChannel(records, inventory, selected.context, selected.pooling, false),
    reverse: fitPredictiveChannel(records, inventory, selected.context, selected.pooling, true),
    natsPerEdit: selected.calibration.reduce((a, b) => a + b, 0) / folds };
  writeFileSync(join(out, "predictive-model.json"), JSON.stringify(pair));
  const report = { trainingOnly: true, recordings: clips.length, speakers: groups.size, folds, selection, selected,
    listener: model.manifest.id, modelSha256: model.manifest.sha256, decoderSha256, referenceSha256: digest(referenceBytes),
    labelsSha256: digest(labelBytes), eligibleLabelsSha256: digest(JSON.stringify(labels)), audioHashes,
    isolation: "none-labelled scored training takes without events/repeats; all non-training and mixed-split speakers excluded",
    control: "identical recordings, 12 EM iterations, full edit marginalization, same context length and pooling; direction only differs",
    calibration: "mean training-fold pooled mismatch log-odds per existing weighted edit, frozen before practice" };
  writeFileSync(join(out, "predictive-fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ recordings: clips.length, speakers: groups.size, selected, natsPerEdit: pair.natsPerEdit }));
} finally { await model.backend.release(); }
