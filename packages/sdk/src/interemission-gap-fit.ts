/** Training-only calibration; does not evaluate warnings or alter any labels or evidence. */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { fitGapCounts, gapTrainingObservations, gapLogRatio, GAP_BINS, gapBin } from "../../core/src/interemission-gap.js";
import type { GapObservation } from "../../core/src/interemission-gap.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA!;
const out = process.env.LAB_OUT!;
const cache = process.env.GAP_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));
interface Label { id: string; speaker: string; kind: string; split: string; scored: boolean; ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[] }
const labels = (readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[])
  .filter(x => x.kind === "none" && x.split === "train" && x.scored && !x.events.length && !x.non_errors.length);
const speakers = [...new Set(labels.map(x => x.speaker))].sort();
if (speakers.length < 2) throw new Error("Need at least two training speakers for grouped selection");
const folds = Math.min(5, speakers.length);
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const clips: { speaker: string; observations: GapObservation[] }[] = [];
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
  if (!words.length || !phonemes.length) throw new Error(`Empty reference or decode: ${l.id}`);
  const start = words[0].start, end = words.at(-1)!.end;
  const text = reference.phonemes.slice(start, end);
  const boundaries = new Set(words.slice(1).map(w => w.start - start));
  const observations = gapTrainingObservations(text, boundaries, phonemes);
  const mass = observations.reduce((n, x) => n + x.weight, 0);
  if (Math.abs(mass - (phonemes.length - 1)) > 1e-5) throw new Error(`Gap count mismatch: ${l.id}`);
  clips.push({ speaker: l.speaker, observations });
  console.log(`clean clip ${clips.length}/${labels.length}: ${phonemes.length} phones, ${mass.toFixed(2)} gaps`);
}
const selections = [8, 32, 128].map(mass => {
  let logP = 0, n = 0;
  for (let f = 0; f < folds; f++) {
    const train = clips.filter(c => foldOf.get(c.speaker) !== f).flatMap(c => c.observations);
    const held = clips.filter(c => foldOf.get(c.speaker) === f).flatMap(c => c.observations);
    const fitted = fitGapCounts(train, mass);
    const total = fitted.pooled.reduce((a, x) => a + x, 0);
    for (const o of held) {
      logP += o.weight * (Math.log((fitted.pooled[gapBin(o.frames)] + 0.5) / (total + GAP_BINS * 0.5)) + gapLogRatio(fitted, o, o.frames));
      n += o.weight;
    }
  }
  return { mass, meanLogP: logP / n, gaps: n };
});
selections.sort((a, b) => b.meanLogP - a.meanLogP || b.mass - a.mass);
const counts = fitGapCounts(clips.flatMap(c => c.observations), selections[0].mass);
// Remove fractional counts that round to zero: the discarded total is bounded and recorded, not a search filter.
let roundedAway = 0;
for (const row of [counts.pooled, ...Object.values(counts.contexts)]) for (let i = 0; i < row.length; i++) {
  const v = Math.round(row[i] * 1e9) / 1e9;
  roundedAway += Math.abs(v - row[i]); row[i] = v;
}
for (const [k, row] of Object.entries(counts.contexts)) if (!row.some(x => x > 0)) delete counts.contexts[k];
const provenance = { listener: model.manifest.id, modelSha256: model.manifest.sha256, frameMs: model.manifest.frameMs, clips: clips.length, speakers: speakers.length,
  labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), folds, selection: selections, roundedAway,
  gaps: counts.pooled.reduce((a, x) => a + x, 0), contexts: Object.keys(counts.contexts).length };
writeFileSync(join(out, "gap-channel.json"), JSON.stringify(counts));
writeFileSync(join(out, "fit-report.json"), JSON.stringify(provenance, null, 2));
console.log(JSON.stringify(provenance));
await model.backend.release();
