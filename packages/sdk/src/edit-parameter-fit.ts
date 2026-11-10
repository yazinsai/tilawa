/** Training-only estimator. Selects prior strength by speaker partitions before opening any practice case. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitEditParameters, parameterRecordingLogP } from "../../core/dist/edit-parameter.js";
import type { ParameterRecording } from "../../core/dist/edit-parameter.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { editSpeakerFolds, editTrainingLabels } from "./edit-parameter-training.js";
import type { EditTrainingLabel } from "./edit-parameter-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labels = editTrainingLabels(readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim()
  .split("\n").map(x => JSON.parse(x)) as EditTrainingLabel[]);
const foldOf = editSpeakerFolds(labels), folds = Math.min(5, foldOf.size);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.EDIT_PARAMETER_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; record: ParameterRecording }[] = [];
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
    const ayahs = new Set(l.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length) throw new Error(`Empty eligible training clip ${l.id}`);
    if (words.some((w, i) => i > 0 && words[i - 1].end !== w.start)) throw new Error("Non-contiguous training reference");
    clips.push({ speaker: l.speaker, record: { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), heard: phones.map(p => p.p).join("") } });
    console.log(`Decoded eligible none clip ${clips.length}/${labels.length}`);
  }
  const selection: { strength: number; logP: number; foldLogP: number[] }[] = [];
  for (const strength of [1, 4, 16, 64, 256]) {
    const foldLogP: number[] = [];
    for (let f = 0; f < folds; f++) {
      const fitted = fitEditParameters(clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.record), [...reference.tokens], strength);
      foldLogP.push(clips.filter(c => foldOf.get(c.speaker) === f).reduce((n, c) => n + parameterRecordingLogP(fitted, c.record), 0));
    }
    selection.push({ strength, foldLogP, logP: foldLogP.reduce((a, b) => a + b, 0) });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.logP - a.logP || b.strength - a.strength);
  const selected = selection[0];
  const fitted = fitEditParameters(clips.map(c => c.record), [...reference.tokens], selected.strength);
  const report = { trainingOnly: true, listener: model.manifest.id, modelSha256: model.manifest.sha256,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"),
    clips: clips.length, speakers: foldOf.size, folds, selection, selectedStrength: selected.strength,
    isolation: "none, scored, no events or repeats; any speaker with a non-train label excluded",
    posterior: "per-reference-phone Dirichlet match/substitute/delete rows and a shared insertion Beta; exact ordered-path posterior predictive",
    limitations: "conditional on existing directed weighted training alignments; substitution and insertion identity distributions frozen; no latent alignment integration",
    fixedCharges: "duration indels cost 0.2 edits; movement transitions retain their existing costs; 3 nats/edit" };
  writeFileSync(join(out, "edit-parameter-model.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await model.backend.release(); }
