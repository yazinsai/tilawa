/** Training-only export. No practice input, tracker visits used as labels, or held-out speaker audio. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";
import { loadNodeModel } from "../../sdk/src/node.js";
import { createListener } from "../../sdk/src/listener.js";
import { decodeWav } from "../../sdk/src/wav.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { TRAJECTORY_FEATURES } from "../src/trajectory.js";
import type { TrajectoryFrame } from "../src/trajectory.js";
import { trainingLabels, transcriptPath, wordBoundaries } from "./trajectory-data.js";
import type { TrainingLabel } from "./trajectory-data.js";

const dataRoot = process.env.LAB_DATA, cache = process.env.TRAJECTORY_CACHE;
if (!dataRoot || !cache?.startsWith("/scratch/tracker-conditional-trajectory-training/")) throw new Error("LAB_DATA and experiment TRAJECTORY_CACHE required");
mkdirSync(cache, { recursive: true });
const root = resolve(dataRoot, "help-train"), bytes = readFileSync(`${root}/labels.jsonl`);
const labels: TrainingLabel[] = bytes.toString().trim().split("\n").map(s => JSON.parse(s));
const eligible = trainingLabels(labels);
if (new Set(eligible.map(r => r.speaker)).size < 3) throw new Error("Need three training speakers");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const model = await loadNodeModel(`${dataRoot}/models/spliced-clean-int8`);
const sha = (s: string | Uint8Array) => createHash("sha256").update(s).digest("hex");
const recordings: { speaker: string; phones: string; path: number[]; clean: boolean; kind: string; hash: string }[] = [];
try {
  for (const r of eligible) {
    const file = resolve(root, r.wav);
    if (!file.startsWith(root + sep)) throw new Error("Audio escapes training root");
    const audio = readFileSync(file), hash = sha(audio), stored = `${cache}/${model.manifest.sha256}-${hash}.json`;
    let phones: string;
    if (existsSync(stored)) phones = JSON.parse(readFileSync(stored, "utf8")).phones;
    else {
      const wav = decodeWav(audio), listener = createListener(model), chunks: string[] = [];
      const size = Math.round(.32 * wav.sampleRate);
      for (let i = 0; i < wav.samples.length; i += size) {
        const e = await listener.push(wav.samples.slice(i, i + size), wav.sampleRate);
        if (e) chunks.push(e.phonemes.map(p => p.p).join(""));
      }
      const e = await listener.end(); if (e) chunks.push(e.phonemes.map(p => p.p).join(""));
      phones = chunks.join("");
      writeFileSync(stored, JSON.stringify({ phones, audioSha256: hash, listenerSha256: model.manifest.sha256 }));
    }
    const path = transcriptPath(index, r);
    const kind = r.kind === "none" ? "none" : r.events?.some(e => e.errorClass === "skipped_passage")
      ? "scripted-jump" : r.non_errors?.length ? "scripted-repeat" : "scripted-omit";
    if (phones && path.length > 2) recordings.push({ speaker: r.speaker, phones, path, clean: r.kind === "none", kind, hash });
    console.log("DECODE", recordings.length, "/", eligible.length);
  }
} finally { await model.backend.release(); }

const rows: unknown[] = [];
const record = (speaker: string, phones: string, path: number[], kind: string, source: string[]) => {
  const first = index.words[path[0]].start;
  const frames: TrajectoryFrame[] = [];
  const local = createLocalSearch(index, { offset: first, origin: first, edits: 0 }, { observe: f => frames.push(f) });
  const initial = local.save() as { lo: number; cost: Float64Array };
  for (const p of phones) local.step(p);
  // Intersect each layer with this ordered sequence of reference-phone occurrences in the fitter.
  const coordinates = [first, ...path.flatMap(i => Array.from({ length: index.words[i].end - index.words[i].start }, (_, j) => index.words[i].start + j + 1))];
  rows.push({ speaker, kind, source, frames, coordinates, initial: { lo: initial.lo, n: initial.cost.length, at: first } });
};
for (const r of recordings) {
  record(r.speaker, r.phones, r.path, r.kind, [r.hash]);
  if (!r.clean) continue;
  const boundaries = wordBoundaries(index, r.path, r.phones);
  const changed = 1 + parseInt(r.hash.slice(0, 8), 16) % (r.path.length - 2);
  const a = boundaries[changed], b = boundaries[changed + 1];
  if (b > a) {
    record(r.speaker, r.phones.slice(0, a) + r.phones.slice(b), r.path.filter((_, i) => i !== changed), "omit-word", [r.hash]);
    record(r.speaker, r.phones.slice(0, b) + r.phones.slice(a, b) + r.phones.slice(b),
      [...r.path.slice(0, changed + 1), r.path[changed], ...r.path.slice(changed + 1)], "repeat", [r.hash]);
  }
  // Multiword forward jumps use exactly the existing word-start restart arcs.
  if (r.path.length >= 6) {
    const end = Math.min(r.path.length - 1, changed + 3);
    record(r.speaker, r.phones.slice(0, a) + r.phones.slice(boundaries[end]),
      [...r.path.slice(0, changed), ...r.path.slice(end)], "jump", [r.hash]);
  }
  const next = recordings.find(s => s.clean && s.speaker === r.speaker && s.path[0] === r.path.at(-1)! + 1);
  if (next) record(r.speaker, r.phones + next.phones, [...r.path, ...next.path], "joined-none", [r.hash, next.hash]);
  const sourceWord = index.words[r.path.at(-1)!];
  const distant = recordings.find(s => {
    const dest = index.words[s.path[0]];
    return s.clean && s.speaker === r.speaker && dest.surah === sourceWord.surah && dest.ayah === sourceWord.ayah + 2;
  });
  if (distant) record(r.speaker, r.phones + distant.phones, [...r.path, ...distant.path], "jump", [r.hash, distant.hash]);
}
writeFileSync(`${cache}/training.json.gz`, gzipSync(JSON.stringify({ version: 1, features: TRAJECTORY_FEATURES, rows,
  provenance: recordings.map(({ phones, ...r }) => ({ ...r, split: "train", decodeSha256: sha(phones) })),
  labelSha256: sha(bytes), listenerSha256: model.manifest.sha256, listener: model.manifest.id })));
console.log("TRAINING_READY", JSON.stringify({ recordings: recordings.length, streams: rows.length }));
