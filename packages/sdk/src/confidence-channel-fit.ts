/** Training-only confidence-channel estimator. It never opens practice evidence or computes warning scores. */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { confidenceLogLikelihood, fitConfidenceChannel } from "../../core/dist/confidence-channel.js";
import type { ConfidenceRecording } from "../../core/dist/confidence-channel.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";

const data = process.env.LAB_DATA;
const out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const cache = process.env.CONFIDENCE_CACHE ?? join(out, "clean-decodes");
mkdirSync(cache, { recursive: true });
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
interface Label { id: string; speaker: string; kind: string; split: string; scored: boolean; ayahs: string[]; wav: string; events: unknown[]; non_errors: unknown[]; excluded?: unknown }
const allLabels = readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[];
// Reject mixed split speakers too, even if a mistakenly annotated row claims to be training.
const excludedSpeakers = new Set(allLabels.filter(l => l.split !== "train").map(l => l.speaker));
const labels = allLabels.filter(l => l.split === "train" && !excludedSpeakers.has(l.speaker) && l.kind === "none" && l.scored && !l.excluded && !l.events.length && !l.non_errors.length);
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 2) throw new Error("Need at least two eligible training speakers");
const folds = Math.min(5, speakers.length);
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
try {
  const clips: { speaker: string; recording: ConfidenceRecording }[] = [];
  for (const l of labels) {
    const file = join(cache, `${listenerModel.manifest.sha256}-${l.id}.json`);
    let phonemes: DecodedPhoneme[];
    if (existsSync(file)) phonemes = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", l.wav)));
      const listener = createListener(listenerModel);
      phonemes = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phonemes));
    }
    const ayahs = new Set(l.ayahs);
    const words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phonemes.length) throw new Error(`Empty eligible training recording: ${l.id}`);
    clips.push({ speaker: l.speaker, recording: { written: reference.phonemes.slice(words[0].start, words.at(-1)!.end), heard: phonemes.map(p => ({ p: p.p, conf: p.conf })) } });
    console.log(`none-labelled training recording ${clips.length}/${labels.length}: ${phonemes.length} phones`);
  }
  const phones = [...reference.tokens];
  const selection: { bins: number; mass: number; logDensity: number; foldLogDensity: number[] }[] = [];
  for (const bins of [1, 2, 4, 8]) for (const mass of [16, 64, 256]) {
    const foldLogDensity: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = clips.filter(c => foldOf.get(c.speaker) !== fold).map(c => c.recording);
      const held = clips.filter(c => foldOf.get(c.speaker) === fold);
      const fitted = fitConfidenceChannel(train, phones, bins, mass);
      foldLogDensity.push(held.reduce((sum, c) => sum + confidenceLogLikelihood(fitted, c.recording), 0));
    }
    selection.push({ bins, mass, logDensity: foldLogDensity.reduce((a, b) => a + b, 0), foldLogDensity });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.logDensity - a.logDensity || a.bins - b.bins || b.mass - a.mass);
  const fitted = fitConfidenceChannel(clips.map(c => c.recording), phones, selection[0].bins, selection[0].mass);
  const provenance = { listener: listenerModel.manifest.id, modelSha256: listenerModel.manifest.sha256, clips: clips.length, speakers: speakers.length, folds,
    labelsSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), selection,
    trainingOnly: true, split: "speaker-grouped train partitions; all non-train speakers excluded",
    alignment: "30 EM iterations maximum; normalized pair-edit channel with latent diagonals and indels",
    contribution: "-log P(confidence | heard identity, reference/heard phonetic classes, identity relation) / P(confidence | heard identity), existing 3 nats/edit" };
  writeFileSync(join(out, "confidence-channel.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(provenance, null, 2));
  console.log(JSON.stringify(provenance));
} finally {
  await listenerModel.backend.release();
}
