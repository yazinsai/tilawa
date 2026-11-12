/** Training-only estimator: reads help-train labels/audio and the deployed listener, never practice evidence. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { fitSpokenPhoneModel, loadReferenceIndex, phoneLogP } from "@tilawa/core";
import type { DecodedPhoneme, PhoneRecording, ReferenceAsset } from "@tilawa/core";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";
import { spokenPhoneRecording, spokenPhoneTrainingLabels } from "./spoken-phone-training.js";
import type { SpokenPhoneLabel } from "./spoken-phone-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT required");
const labels = spokenPhoneTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as SpokenPhoneLabel[]);
const speakers = [...new Set(labels.map(l => l.speaker))].sort(), folds = Math.min(3, speakers.length);
if (folds < 2) throw new Error("Need multiple eligible training speakers");
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.SPOKEN_PHONE_CACHE ?? join(out, "decodes");
mkdirSync(cache, { recursive: true });
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; kind: string; record: PhoneRecording }[] = [];
  for (const label of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${label.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", label.wav))), listener = createListener(listenerModel);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    clips.push({ speaker: label.speaker, kind: label.kind, record: spokenPhoneRecording(reference, label, phones.map(p => p.p).join("")) });
    console.log(`Decoded eligible training take ${clips.length}/${labels.length}`);
  }
  const inventory = [...reference.tokens], iterations = 20;
  const selection: { mass: number; twoStage: number[]; control: number[]; combined: number }[] = [];
  for (const mass of [8, 64]) {
    const twoStage: number[] = [], control: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => foldOf.get(c.speaker) !== fold), held = clips.filter(c => foldOf.get(c.speaker) === fold);
      const fitted = fitSpokenPhoneModel(train.map(c => c.record), inventory, mass, iterations);
      // Validation has no script mask at inference: pronunciation variation is possible at every phone.
      twoStage.push(held.reduce((s, c) => s + phoneLogP(fitted.recognition, { ...c.record, variable: Array(c.record.written.length).fill(true) }, fitted.pronunciation), 0));
      control.push(held.reduce((s, c) => s + phoneLogP(fitted.control, c.record), 0));
      console.log(JSON.stringify({ mass, fold, twoStage: twoStage.at(-1), control: control.at(-1) }));
    }
    selection.push({ mass, twoStage, control, combined: [...twoStage, ...control].reduce((a, b) => a + b, 0) });
  }
  selection.sort((a, b) => b.combined - a.combined || b.mass - a.mass);
  const fitted = fitSpokenPhoneModel(clips.map(c => c.record), inventory, selection[0].mass, iterations);
  const report = { trainingOnly: true, listener: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: speakers.length, folds, kinds: Object.fromEntries(["none", "vowel", "tajweed"].map(k => [k, clips.filter(c => c.kind === k).length])),
    selection, iterations, selectedMass: selection[0].mass,
    isolation: "only train speakers; mixed-split speakers, lexical scripts, repeats and extra mistakes excluded",
    recognition: "none anchors; forward/backward geometric-insertion transducer; frozen before pronunciation fit",
    pronunciation: "latent one-phone realization with phonetic sharing; variable only at labelled words during training",
    control: "same full recordings, inventory, smoothing family, selected mass, iterations and speaker folds",
    local: "composed phone likelihood ratio, same observation baseline; unchanged movement and duplicate-symbol duration charges" };
  writeFileSync(join(out, "spoken-phone-model.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await listenerModel.backend.release(); }
