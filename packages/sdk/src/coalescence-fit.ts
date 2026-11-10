/** Training-only CLI. Opens help-train and the deployed model; never practice evidence or warning labels. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { coalescenceLogP, fitCoalescence } from "../../core/dist/coalescence.js";
import type { CoalescenceRecording } from "../../core/dist/coalescence.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { coalescenceTrainingLabels, coalescenceSpeakerFolds } from "./coalescence-training.js";
import type { CoalescenceLabel } from "./coalescence-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labels = coalescenceTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as CoalescenceLabel[]);
const foldOf = coalescenceSpeakerFolds(labels);
const folds = Math.min(5, foldOf.size);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.COALESCENCE_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: CoalescenceRecording }[] = [];
  for (const l of labels) {
    const file = join(cache, `${model.manifest.sha256}-${l.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
      const listener = createListener(model);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(l.ayahs);
    const words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length) throw new Error(`Empty eligible training clip ${l.id}`);
    const from = words[0].start, to = words.at(-1)!.end;
    if (words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Non-contiguous training reference");
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(from, to),
      heard: phones.map(p => p.p).join(""), starts: words.map(w => w.start - from) } });
    console.log(`Decoded eligible none clip ${clips.length}/${labels.length}`);
  }
  const inventory = [...reference.tokens];
  const selection: { mass: number; enabled: boolean; logP: number; foldLogP: number[] }[] = [];
  // The no-coalescence model is an explicit training-only alternative. No practice-driven model choice.
  for (const candidate of [{ mass: 32, enabled: false }, ...[8, 32, 128].map(mass => ({ mass, enabled: true }))]) {
    const foldLogP: number[] = [];
    for (let f = 0; f < folds; f++) {
      const train = clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.record);
      const held = clips.filter(c => foldOf.get(c.speaker) === f);
      const fitted = fitCoalescence(train, inventory, candidate.mass, candidate.enabled);
      foldLogP.push(held.reduce((sum, c) => sum + coalescenceLogP(fitted, c.record), 0));
    }
    selection.push({ ...candidate, foldLogP, logP: foldLogP.reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.logP - a.logP || Number(a.enabled) - Number(b.enabled) || b.mass - a.mass);
  const selected = selection[0];
  const fitted = fitCoalescence(clips.map(c => c.record), inventory, selected.mass, selected.enabled);
  const report = { trainingOnly: true, listener: model.manifest.id, modelSha256: model.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: foldOf.size, folds, selection,
    isolation: "none, scored, no events or repeats; all speakers with any non-train label excluded",
    alignment: "normalized edit transducer; latent ordinary/substitution/deletion/insertion/coalescence paths; at most 30 EM iterations",
    exclusions: "word boundaries, identical phones, and either articulation touching a repeated reference run",
    predictionUnits: "likelihood ratios against exact ordinary recognition; existing 3 nats/edit" };
  writeFileSync(join(out, "coalescence-model.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await model.backend.release();
}
