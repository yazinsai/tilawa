/** Train-only CLI. No practice evidence, session identifiers or warning scores enter this estimator. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset, WordRef } from "@tilawa/core";
import { episodeTrainingGraph, episodeTrainingLogP, fitEpisodeMotion } from "../../core/dist/episode-motion.js";
import type { EpisodeTrainingGraph, EpisodeTrainingWord } from "../../core/dist/episode-motion.js";
import { createListener } from "./listener.js";
import { decodeWav, loadNodeModel } from "./node.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const cache = process.env.EPISODE_CACHE ?? join(out, "training-decodes");
mkdirSync(cache, { recursive: true });
interface Label {
  id: string; speaker: string; split: string; scored: boolean; excluded?: unknown; kind: string; ayahs: string[]; wav: string;
  events: { errorClass: string; location: WordRef; words?: number }[];
  non_errors: { kind: string; location: WordRef }[];
}
const all = readFileSync(join(data, "help-train/labels.jsonl"), "utf8").trim().split("\n").map(x => JSON.parse(x)) as Label[];
const excludedSpeakers = new Set(all.filter(l => l.split !== "train").map(l => l.speaker));
const labels = all.filter(l => l.split === "train" && !excludedSpeakers.has(l.speaker) && l.scored && !l.excluded && l.ayahs.length === 1
  && ["none", "skip_word", "substitution", "repeat"].includes(l.kind)
  && l.events.every(e => ["omitted_word", "wrong_word"].includes(e.errorClass) && (e.words ?? 1) === 1)
  && l.non_errors.every(e => e.kind === "repeat"));
const speakers = [...new Set(labels.map(l => l.speaker))].sort();
if (speakers.length < 2) throw new Error("Need at least two eligible training speakers");
const folds = Math.min(5, speakers.length);
const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phones = [...reference.tokens];
const listenerModel = await loadNodeModel(join(data, "models/spliced-clean-int8"));
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
try {
  const clips: { speaker: string; graph: EpisodeTrainingGraph }[] = [];
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
    const omitted = new Set(l.events.filter(e => e.errorClass === "omitted_word").map(e => key(e.location)));
    const repeated = new Set(l.non_errors.map(e => key(e.location)));
    const refWords = reference.words.filter(w => `${w.surah}:${w.ayah}` === l.ayahs[0]);
    if (!refWords.length || !phonemes.length) throw new Error("Empty eligible training recording");
    const words: EpisodeTrainingWord[] = [];
    for (const w of refWords) {
      const p = reference.phonemes.slice(w.start, w.end);
      if (!p.length) continue;
      words.push({ phones: p, omit: omitted.has(key(w)) });
      if (repeated.has(key(w))) words.push({ phones: p, repeat: true });
    }
    if (omitted.size !== words.filter(w => w.omit).length || repeated.size !== words.filter(w => w.repeat).length) throw new Error("Scripted event outside training ayah");
    clips.push({ speaker: l.speaker, graph: episodeTrainingGraph({ words, heard: phonemes.map(p => p.p).join("") }, phones) });
    console.log(`Eligible training decode ${clips.length}/${labels.length}`);
  }
  const selection: { dependent: boolean; mass: number; logP: number; folds: number[] }[] = [];
  for (const dependent of [false, true]) for (const mass of [1, 16]) {
    const scores: number[] = [];
    for (let f = 0; f < folds; f++) {
      const train = clips.filter(c => foldOf.get(c.speaker) !== f).map(c => c.graph);
      const fitted = fitEpisodeMotion(train, mass, dependent);
      scores.push(clips.filter(c => foldOf.get(c.speaker) === f).reduce((sum, c) => sum + episodeTrainingLogP(fitted, c.graph), 0));
    }
    selection.push({ dependent, mass, logP: scores.reduce((a, b) => a + b, 0), folds: scores });
    console.log(JSON.stringify(selection.at(-1)));
  }
  selection.sort((a, b) => b.logP - a.logP || Number(a.dependent) - Number(b.dependent) || b.mass - a.mass);
  const best = selection[0];
  const fitted = fitEpisodeMotion(clips.map(c => c.graph), best.mass, best.dependent);
  const report = { listener: listenerModel.manifest.id, listenerSha256: listenerModel.manifest.sha256,
    labelSha256: createHash("sha256").update(JSON.stringify(labels)).digest("hex"), recordings: clips.length, speakers: speakers.length, folds,
    kinds: Object.fromEntries([...new Set(labels.map(l => l.kind))].map(k => [k, labels.filter(l => l.kind === k).length])), selection,
    trainingOnly: true, excludedNonTrainSpeakers: excludedSpeakers.size,
    boundaryModel: "normalized edit DAG with mandatory scripted omissions and unfolded repeats; joint boundary/state forward-backward",
    passageRelocation: "unchanged; not estimated from single-ayah clips" };
  writeFileSync(join(out, "episode-motion.json"), JSON.stringify(fitted));
  writeFileSync(join(out, "fit-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally {
  await listenerModel.backend.release();
}
