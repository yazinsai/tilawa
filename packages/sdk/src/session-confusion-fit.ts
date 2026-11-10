/** Authorized help-train speakers only; no warning scoring or practice data is used in model selection. */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { confusionPredictiveLogP, confusionRecording, fitConfusionModel } from "../../core/src/session-confusion.js";
import type { ConfusionRecording } from "../../core/src/session-confusion.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA!;
const out = process.env.LAB_OUT!;
const cache = process.env.CONFUSION_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phones = [...reference.tokens];
interface Label { id: string; speaker: string; kind: string; split: string; scored: boolean; ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[] }
const labels = (readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[])
  .filter(x => x.kind === "none" && x.split === "train" && x.scored && !x.events.length && !x.non_errors.length);
const speakers = [...new Set(labels.map(x => x.speaker))].sort();
if (speakers.length < 2) throw new Error("Need training-speaker partitions for confusion selection");
const folds = Math.min(5, speakers.length);
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const clips: { speaker: string; recording: ConfusionRecording }[] = [];
for (const l of labels) {
  const file = join(cache, `${model.manifest.id}-${l.id}.json`);
  let phonemes: DecodedPhoneme[];
  if (existsSync(file)) phonemes = JSON.parse(readFileSync(file, "utf8"));
  else {
    const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
    const listener = createListener(model);
    phonemes = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
    writeFileSync(file, JSON.stringify(phonemes));
  }
  const ayahs = new Set(l.ayahs);
  const words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
  if (!words.length || !phonemes.length) throw new Error(`Empty training recording: ${l.id}`);
  const written = reference.phonemes.slice(words[0].start, words.at(-1)!.end);
  clips.push({ speaker: l.speaker, recording: confusionRecording(written, phonemes.map(p => p.p).join(""), phones) });
  console.log(`none-labelled recording ${clips.length}/${labels.length}: ${phonemes.length} heard phones`);
}
const selection: { mappings: number; mass: number; logP: number }[] = [];
const smallestTrainingPartition = Math.min(...Array.from({ length: folds }, (_, f) => clips.filter(c => foldOf.get(c.speaker) !== f).length));
for (const mappings of [1, 2, 3].filter(k => k <= smallestTrainingPartition)) for (const mass of [16, 64, 256]) {
  let logP = 0;
  for (let fold = 0; fold < folds; fold++) {
    const train = clips.filter(c => foldOf.get(c.speaker) !== fold).map(c => c.recording);
    const held = clips.filter(c => foldOf.get(c.speaker) === fold);
    const fitted = fitConfusionModel(train, phones, mappings, mass);
    for (const c of held) logP += confusionPredictiveLogP(fitted, c.recording);
  }
  selection.push({ mappings, mass, logP });
  console.log(JSON.stringify(selection.at(-1)));
}
selection.sort((a, b) => b.logP - a.logP || a.mappings - b.mappings || b.mass - a.mass);
const fitted = fitConfusionModel(clips.map(c => c.recording), phones, selection[0].mappings, selection[0].mass);
const provenance = { listener: model.manifest.id, modelSha256: model.manifest.sha256, clips: clips.length, speakers: speakers.length, folds,
  labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), smallestTrainingPartition, selection,
  alignment: "fixed ordinary-edit forward/backward sufficient statistics; one shared mapping per recording",
  cleanTrainingOnly: true };
writeFileSync(join(out, "session-confusion.json"), JSON.stringify(fitted));
writeFileSync(join(out, "fit-report.json"), JSON.stringify(provenance, null, 2));
console.log(JSON.stringify(provenance));
await model.backend.release();
