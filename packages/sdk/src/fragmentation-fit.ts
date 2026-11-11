/** Training-only CLI. Opens help-train and the deployed listener, never practice evidence. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitFragmentation, fragmentationLogP } from "../../core/dist/fragmentation.js";
import type { FragmentationRecording } from "../../core/dist/fragmentation.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { fragmentationSpeakerFolds, fragmentationTrainingLabels } from "./fragmentation-training.js";
import type { FragmentationLabel } from "./fragmentation-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labels = fragmentationTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as FragmentationLabel[]);
const foldOf = fragmentationSpeakerFolds(labels), folds = Math.min(5, foldOf.size);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.FRAGMENTATION_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: FragmentationRecording }[] = [];
  for (const l of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${l.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
      const listener = createListener(listenerModel);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length) throw new Error(`Empty eligible training recording ${l.id}`);
    if (words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Non-contiguous training reference");
    const from = words[0].start, to = words.at(-1)!.end;
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(from, to),
      heard: phones.map(p => p.p).join(""), starts: words.map(w => w.start - from) } });
    console.log(`Decoded eligible none recording ${clips.length}/${labels.length}`);
  }
  const inventory = [...reference.tokens];
  const selection: { enabled: boolean; mass: number; logP: number; foldLogP: number[] }[] = [];
  for (const candidate of [{ enabled: false, mass: 32 }, ...[8, 32, 128].map(mass => ({ enabled: true, mass }))]) {
    const foldLogP: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => foldOf.get(c.speaker) !== fold).map(c => c.record);
      const held = clips.filter(c => foldOf.get(c.speaker) === fold);
      const model = fitFragmentation(train, inventory, candidate.mass, candidate.enabled);
      foldLogP.push(held.reduce((sum, c) => sum + fragmentationLogP(model, c.record), 0));
    }
    selection.push({ ...candidate, foldLogP, logP: foldLogP.reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.logP - a.logP || Number(a.enabled) - Number(b.enabled) || b.mass - a.mass);
  const selected = selection[0];
  const model = fitFragmentation(clips.map(c => c.record), inventory, selected.mass, selected.enabled);
  const report = { trainingOnly: true, listener: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: foldOf.size, folds, selection,
    isolation: "scored none-labelled train recordings without events or repeats; speakers with any non-train label excluded",
    alignment: "complete-string normalized edit/fragment transducer; latent boundaries; at most 30 hierarchical EM iterations",
    fragmentSupport: "all heterogeneous length-2 and length-3 inventory strings; no validation-vocabulary selection",
    duration: "only first identity symbol in a run is fragmentable; remaining symbols keep existing duration charges",
    units: "likelihood ratio against ordinary exact recognition; existing three nats per edit" };
  writeFileSync(join(out, "fragmentation-model.json"), JSON.stringify(model));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await listenerModel.backend.release();
}
