/** Training-only estimator. It opens help-train and the deployed model, never practice sessions. */
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, relative, isAbsolute } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { copyRecordingLogP, fitCopyChannel } from "../../core/dist/copy-channel.js";
import type { CopyRecording } from "../../core/dist/copy-channel.js";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { copySpeakerFolds, copyTrainingLabels } from "./copy-channel-training.js";
import type { CopyTrainingLabel } from "./copy-channel-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const help = realpathSync(join(data, "help-train"));
const labelBytes = readFileSync(join(help, "labels.jsonl"));
const labels = copyTrainingLabels(labelBytes.toString("utf8").trim().split("\n").map(s => JSON.parse(s)) as CopyTrainingLabel[]);
const groups = copySpeakerFolds(labels), folds = Math.min(3, groups.size);
const referenceBytes = readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url));
const reference = loadReferenceIndex(JSON.parse(referenceBytes.toString("utf8")) as ReferenceAsset);
const digest = (data: Uint8Array | string) => createHash("sha256").update(data).digest("hex");
const extension = new URL(import.meta.url).pathname.endsWith(".ts") ? ".ts" : ".js";
const decoderSha256 = digest(["listener", "audio", "fbank"].map(file =>
  readFileSync(new URL(file + extension, import.meta.url), "utf8")).join("\n"));
const cache = process.env.COPY_CHANNEL_CACHE ?? join(out, "copy-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: CopyRecording }[] = [], audioHashes: string[] = [];
  for (const l of labels) {
    const path = realpathSync(join(help, l.wav)), inside = relative(help, path);
    if (inside === ".." || inside.startsWith("../") || isAbsolute(inside)) throw new Error("Training audio leaves help-train");
    const bytes = readFileSync(path), hash = digest(bytes); audioHashes.push(hash);
    const file = join(cache, `${model.manifest.sha256}-${decoderSha256}-${hash}-copy-v1.json`);
    let observations: DecodedPhoneme[];
    if (existsSync(file)) observations = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(bytes), listener = createListener(model);
      observations = [];
      // Use the deployed streaming decoder's normal chunk boundaries.
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
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), observations } });
    console.log(`Decoded ${clips.length}/${labels.length} eligible none-labelled training takes`);
  }
  const inventory = [...reference.tokens];
  const selection: { linked: boolean; pooling: number; foldLogP: number[]; logP: number }[] = [];
  for (const linked of [false, true]) for (const pooling of [8, 32]) {
    const foldLogP: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => groups.get(c.speaker) !== fold).map(c => c.record);
      const held = clips.filter(c => groups.get(c.speaker) === fold);
      const fit = fitCopyChannel(train, inventory, pooling, linked);
      foldLogP.push(held.reduce((sum, c) => sum + copyRecordingLogP(fit, c.record), 0));
    }
    selection.push({ linked, pooling, foldLogP, logP: foldLogP.reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  const best = (linked: boolean) => selection.filter(x => x.linked === linked).sort((a, b) => b.logP - a.logP || b.pooling - a.pooling)[0];
  const linked = best(true), control = best(false);
  writeFileSync(join(out, "copy-linked-model.json"), JSON.stringify(fitCopyChannel(clips.map(c => c.record), inventory, linked.pooling, true)));
  writeFileSync(join(out, "copy-control-model.json"), JSON.stringify(fitCopyChannel(clips.map(c => c.record), inventory, control.pooling, false)));
  const report = { trainingOnly: true, recordings: clips.length, speakers: groups.size, folds, selection,
    preferLinked: linked.logP > control.logP, validationGainNats: linked.logP - control.logP,
    listener: model.manifest.id, modelSha256: model.manifest.sha256, decoderSha256, referenceSha256: digest(referenceBytes),
    labelsSha256: digest(labelBytes), eligibleLabelsSha256: digest(JSON.stringify(labels)), audioHashes,
    isolation: "none-labelled scored training takes without events/repeats; mixed-split and all non-training speakers excluded",
    paths: "complete-string normalized forward/backward; latent edits and source assignments; 24 EM iterations",
    delay: "at most 16 already-consumed observations, normalized delay and manner-class source weights; distinct timestamp occurrences only",
    control: "same edit graph, emission model, priors and class sharing; source sampled from fitted pooled distribution instead of retained observed history" };
  writeFileSync(join(out, "copy-fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await model.backend.release(); }
