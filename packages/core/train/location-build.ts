/** Authorized training observations only. Run through the supplied Modal supervisor. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";
import { loadNodeModel } from "../../sdk/src/node.js";
import { createListener } from "../../sdk/src/listener.js";
import { decodeWav } from "../../sdk/src/wav.js";
import { loadReferenceIndex } from "../src/reference.js";
import { TRAJECTORY_FEATURES } from "../src/trajectory.js";
import { trainingLabels, transcriptPath, wordBoundaries } from "./location-data.js";
import type { TrainingLabel } from "./location-data.js";

const data = process.env.LAB_DATA, cache = process.env.LOCATION_CACHE;
if (!data || !cache || !process.env.LAB_OUT) throw new Error("LAB_DATA, LOCATION_CACHE and LAB_OUT required");
if (!resolve(cache).startsWith("/scratch/tracker-adversarial-location-margin/")) throw new Error("Experiment cache required");
mkdirSync(cache, { recursive: true });
const root = resolve(data, "help-train"), bytes = readFileSync(`${root}/labels.jsonl`);
const labels: TrainingLabel[] = bytes.toString().trim().split("\n").map(s => JSON.parse(s));
const eligible = trainingLabels(labels).filter(r => r.kind === "none");
if (new Set(eligible.map(r => r.speaker)).size < 3) throw new Error("Need three exclusively training speakers' clean takes");
const assetBytes = readFileSync(new URL("../assets/ref-v2.json", import.meta.url));
const index = loadReferenceIndex(JSON.parse(assetBytes.toString()));
const sha = (s: string | Uint8Array) => createHash("sha256").update(s).digest("hex");
const model = await loadNodeModel(`${data}/models/spliced-clean-int8`);
const recordings: { speaker: string; phones: string; path: number[]; boundaries: number[]; hash: string }[] = [];
try {
  for (const r of eligible) {
    const file = resolve(root, r.wav);
    if (!file.startsWith(root + sep)) throw new Error("Audio escapes training root");
    const audio = readFileSync(file), hash = sha(audio), stored = `${cache}/${model.manifest.sha256}-${hash}.json`;
    let phones: string;
    if (existsSync(stored)) {
      const saved = JSON.parse(readFileSync(stored, "utf8"));
      if (saved.audioSha256 !== hash || saved.listenerSha256 !== model.manifest.sha256) throw new Error("Stale decode cache");
      phones = saved.phones;
    } else {
      const wav = decodeWav(audio), listener = createListener(model), parts: string[] = [];
      const size = Math.round(.32 * wav.sampleRate);
      for (let i = 0; i < wav.samples.length; i += size) {
        const e = await listener.push(wav.samples.slice(i, i + size), wav.sampleRate);
        if (e) parts.push(e.phonemes.map(p => p.p).join(""));
      }
      const e = await listener.end(); if (e) parts.push(e.phonemes.map(p => p.p).join(""));
      phones = parts.join("");
      writeFileSync(stored, JSON.stringify({ phones, audioSha256: hash, listenerSha256: model.manifest.sha256 }));
    }
    const path = transcriptPath(index, r);
    if (phones && path.length >= 3) recordings.push({ speaker: r.speaker, phones, path,
      boundaries: wordBoundaries(index, path, phones), hash });
    console.log("DECODE", recordings.length, "/", eligible.length);
  }
} finally { await model.backend.release(); }

// Every speaker and all that speaker's augmentations stay together in the fitter's partitions.
const examples: unknown[] = [];
for (const r of recordings) {
  for (let end = 2; end < r.path.length; end++) {
    const begin = Math.max(0, end - 3), a = r.boundaries[begin], b = r.boundaries[end + 1];
    if (b <= a || b - a > 64) continue;
    const target = index.words[r.path[end]], first = index.words[r.path[begin]].start;
    const record = (phones: string, kind: string) => {
      if (phones.length >= 3) examples.push({ speaker: r.speaker, phones, start: first,
        target: [target.start + 1, target.end + 1], kind, source: r.hash, sources: [r.hash] });
    };
    record(r.phones.slice(a, b), "clean");
    // Omit one whole word; deleting a longer internal span provides positive passage movement examples.
    const remove = end - 1, left = r.boundaries[remove], right = r.boundaries[remove + 1];
    if (right > left) record(r.phones.slice(a, left) + r.phones.slice(right, b), "omission");
    if (end - begin >= 3 && r.boundaries[end] > r.boundaries[begin + 1]) {
      record(r.phones.slice(a, r.boundaries[begin + 1]) + r.phones.slice(r.boundaries[end], b), "jump");
    }
  }
  // A true passage movement crosses an unread ayah. Both clips come from the same training speaker;
  // the destination remains inside the unchanged local search's two-ayah reach.
  const sourceWord = index.words[r.path.at(-1)!];
  const dest = recordings.find(s => s.speaker === r.speaker
    && index.words[s.path[0]].surah === sourceWord.surah
    && index.words[s.path[0]].ayah === sourceWord.ayah + 2);
  if (dest && dest.path.length >= 2) {
    const begin = Math.max(0, r.path.length - 2), target = index.words[dest.path[1]];
    const phones = r.phones.slice(r.boundaries[begin]) + dest.phones.slice(0, dest.boundaries[2]);
    if (phones.length >= 3 && phones.length <= 64) examples.push({ speaker: r.speaker, phones,
      start: index.words[r.path[begin]].start, target: [target.start + 1, target.end + 1],
      kind: "passage-jump", source: r.hash, sources: [r.hash, dest.hash] });
  }
}
const payload = { version: 1, features: TRAJECTORY_FEATURES,
  reference: { phonemes: index.phonemes, words: index.words }, examples,
  recordings: recordings.map(r => ({ ...r, split: "train" })),
  provenance: { labelSha256: sha(bytes), referenceSha256: sha(assetBytes), listenerSha256: model.manifest.sha256,
    listener: model.manifest.id, eligible: eligible.length, excluded: labels.length - eligible.length } };
writeFileSync(`${cache}/training.json.gz`, gzipSync(JSON.stringify(payload)));
writeFileSync(`${process.env.LAB_OUT}/training-summary.json`, JSON.stringify({ ...payload.provenance,
  recordings: recordings.length, examples: examples.length, speakers: new Set(recordings.map(r => r.speaker)).size }, null, 2));
console.log("TRAINING_READY", recordings.length, examples.length);
