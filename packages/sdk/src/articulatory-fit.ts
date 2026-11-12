/** Training-only estimator. This entry point never opens practice recordings or scores warnings. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { ReferenceAsset } from "@tilawa/core";
import { articulatoryLogLikelihood, fitArticulatoryModel, posteriorObservation } from "../../core/dist/articulatory-channel.js";
import type { ArticulatoryRecording, ArticulatoryModel } from "../../core/dist/articulatory-channel.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phones = [...reference.tokens];
interface Label { id: string; speaker: string; kind: string; split: string; scored: boolean; ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown }
const all = readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(s => JSON.parse(s)) as Label[];
const nonTrain = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
const labels = all.filter(l => l.split === "train" && !nonTrain.has(l.speaker) && l.kind === "none" && l.scored && !l.excluded && !l.events.length && !l.non_errors.length);
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 2) throw new Error("Need two eligible training speakers");
const folds = Math.min(5, speakers.length);
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const cache = join(out, "articulatory-training-cache");
mkdirSync(cache, { recursive: true });
try {
  const clips: { speaker: string; recording: ArticulatoryRecording }[] = [];
  for (const lab of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${lab.id}.json`);
    let recording: ArticulatoryRecording;
    if (existsSync(file)) recording = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wanted = new Set(lab.ayahs);
      const words = reference.words.filter(w => wanted.has(`${w.surah}:${w.ayah}`));
      if (!words.length) throw new Error("Empty training reference");
      const wav = decodeWav(readFileSync(join(data, "help-train", lab.wav)));
      const listener = createListener(listenerModel, { posteriors: true });
      const observed: number[][] = [];
      for (const e of [await listener.push(wav.samples, wav.sampleRate), await listener.end()]) {
        if (!e?.posteriors) continue;
        for (const ph of e.phonemes) {
          const q = posteriorObservation(e.posteriors, phones, ph);
          if (!q) throw new Error("Training run outside its posterior interval");
          observed.push(q);
        }
      }
      recording = { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), observed };
      if (!observed.length) throw new Error("Empty training audio");
      writeFileSync(file, JSON.stringify(recording));
    }
    clips.push({ speaker: lab.speaker, recording });
    console.log(`none-labelled training recording ${clips.length}/${labels.length}: ${recording.observed.length} observations`);
  }
  const selections: Record<string, unknown> = {};
  for (const kind of ["joint", "identity"] as const) {
    const selection: { mass: number; logP: number; folds: number[] }[] = [];
    for (const mass of [4, 16, 64]) {
      const likelihoods: number[] = [];
      for (let fold = 0; fold < folds; fold++) {
        const model = fitArticulatoryModel(clips.filter(c => foldOf.get(c.speaker) !== fold).map(c => c.recording), phones, kind, mass);
        likelihoods.push(clips.filter(c => foldOf.get(c.speaker) === fold).reduce((sum, c) => sum + articulatoryLogLikelihood(model, c.recording), 0));
      }
      selection.push({ mass, logP: likelihoods.reduce((a, b) => a + b, 0), folds: likelihoods });
      console.log(JSON.stringify({ kind, ...selection.at(-1) }));
    }
    selection.sort((a, b) => b.logP - a.logP || b.mass - a.mass);
    const fitted: ArticulatoryModel = { ...fitArticulatoryModel(clips.map(c => c.recording), phones, kind, selection[0].mass), inventory: reference.inventoryId };
    writeFileSync(join(out, `articulatory-${kind}.json`), JSON.stringify(fitted));
    selections[kind] = selection;
  }
  const report = {
    trainingOnly: true, recordings: clips.length, speakers: speakers.length, folds,
    model: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    partitionsSha256: createHash("sha256").update(JSON.stringify([...foldOf])).digest("hex"),
    excluded: "all non-train speakers, mistakes, repeats and excluded labels",
    observations: "average unconditioned posterior rows across each emitted run, then condition on nonblank",
    placement: "normalized pair-channel forward/backward; 20 EM iterations maximum",
    units: "likelihood versus pooled observations, existing 3 nats per edit",
    selections,
    limit: "joint and identity raw log likelihoods have different alphabets; compare only within each kind for selection",
  };
  writeFileSync(join(out, "articulatory-fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await listenerModel.backend.release(); }
