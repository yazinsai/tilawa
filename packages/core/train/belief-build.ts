/** Authorized training only. Run in /app; output and decode caches live in the experiment's scratch directory. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { gzipSync } from "node:zlib";
import { createListener } from "../../sdk/src/listener.js";
import { loadNodeModel } from "../../sdk/src/node.js";
import { decodeWav } from "../../sdk/src/wav.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { smoothLocations, boundaryDistribution, sampleBoundary } from "./belief-smoother.js";
import { trainingNoneLabels } from "./belief-eligibility.js";
import type { TrainingLabel } from "./belief-eligibility.js";

const dataRoot = process.env.LAB_DATA;
const cache = process.env.BELIEF_CACHE;
if (!dataRoot || !cache || !cache.startsWith("/scratch/tracker-smoothed-belief-distill/")) {
  throw new Error("LAB_DATA and this experiment's BELIEF_CACHE are required");
}
mkdirSync(cache, { recursive: true });
const root = resolve(dataRoot, "help-train");
const bytes = readFileSync(`${root}/labels.jsonl`);
const labels: TrainingLabel[] = bytes.toString().trim().split("\n").map(s => JSON.parse(s));
const eligible = trainingNoneLabels(labels);
if (new Set(eligible.map(r => r.speaker)).size < 3) throw new Error("Need at least three eligible training speakers");
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const model = await loadNodeModel(`${dataRoot}/models/spliced-clean-int8`);
const sha = (b: string | Uint8Array) => createHash("sha256").update(b).digest("hex");
const recordings: { speaker: string; phones: string; path: number[]; audioSha256: string }[] = [];
try {
  for (const r of eligible) {
    const wavPath = resolve(root, r.wav);
    if (!wavPath.startsWith(root + sep)) throw new Error("Training audio escapes help-train");
    const audio = readFileSync(wavPath), audioSha256 = sha(audio);
    const file = `${cache}/${model.manifest.sha256}-${audioSha256}.json`;
    let phones: string;
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8")).phones;
    else {
      const wav = decodeWav(audio), listener = createListener(model);
      const output: string[] = [], size = Math.round(.32 * wav.sampleRate);
      for (let i = 0; i < wav.samples.length; i += size) {
        const e = await listener.push(wav.samples.slice(i, i + size), wav.sampleRate);
        if (e) output.push(e.phonemes.map(p => p.p).join(""));
      }
      const e = await listener.end(); if (e) output.push(e.phonemes.map(p => p.p).join(""));
      phones = output.join("");
      writeFileSync(file, JSON.stringify({ phones, audioSha256, listenerSha256: model.manifest.sha256 }));
    }
    const path = index.words.flatMap((w, i) => r.ayahs.includes(`${w.surah}:${w.ayah}`) && w.end > w.start ? [i] : []);
    if (phones && path.length > 2) recordings.push({ speaker: r.speaker, phones, path, audioSha256 });
    console.log("DECODE", recordings.length, "/", eligible.length);
  }
} finally { await model.backend.release(); }

const rows: unknown[] = [];
const exclusions: Record<string, number> = {};
const record = (speaker: string, phones: string, path: number[], kind: string) => {
  const reference = path.map(i => index.phonemes.slice(index.words[i].start, index.words[i].end)).join("");
  const endpoints = [index.words[path[0]].start,
    ...path.flatMap(i => Array.from({ length: index.words[i].end - index.words[i].start }, (_, j) => index.words[i].start + j + 1))];
  const q = smoothLocations(reference, phones).slice(1);
  const frames: unknown[] = [];
  let previousLo = 0, previousN = 0, time = 0, retained = 0;
  const start = index.words[path[0]].start;
  const local = createLocalSearch(index, { offset: start, origin: start, edits: 0 }, { observe(frame) {
    const target = frame.features.map(() => 0);
    q[time].forEach((p, j) => {
      const cell = endpoints[j] - frame.lo;
      if (cell >= 0 && cell < target.length && frame.features[cell].length) target[cell] += p;
    });
    const mass = target.reduce((a, b) => a + b, 0);
    retained += mass;
    // Keep lost posterior mass visible for the audit; it is never mapped to a wrong location.
    frames.push({ x: frame.features.map(x => x.length ? x : new Array(12).fill(0)),
      valid: frame.features.map(x => !!x.length),
      costs: Array.from(frame.costs, c => Number.isFinite(c) ? c : 1e6),
      predecessor: Array.from(frame.predecessor, at => at >= previousLo && at < previousLo + previousN ? at - previousLo : -1),
      target: target.map(p => mass ? p / mass : 0) });
    previousLo = frame.lo; previousN = target.length; time++;
  } });
  for (const p of phones) local.step(p);
  if (retained / phones.length < .95) { exclusions[kind] = (exclusions[kind] ?? 0) + 1; return; }
  rows.push({ speaker, kind, frames, retainedMass: retained / phones.length });
};

for (const r of recordings) {
  record(r.speaker, r.phones, r.path, "none");
  const text = r.path.map(i => index.phonemes.slice(index.words[i].start, index.words[i].end)).join("");
  const posterior = smoothLocations(text, r.phones);
  // Hash selection and boundary quantiles are fixed before any practice observation.
  const changed = 1 + parseInt(r.audioSha256.slice(0, 8), 16) % (r.path.length - 2);
  const lo = r.path.slice(0, changed).reduce((n, i) => n + index.words[i].end - index.words[i].start, 0);
  const hi = lo + index.words[r.path[changed]].end - index.words[r.path[changed]].start;
  for (const quantile of [.2, .5, .8]) {
    const a = sampleBoundary(boundaryDistribution(posterior, lo), quantile);
    const b = sampleBoundary(boundaryDistribution(posterior, hi), quantile);
    if (b <= a) continue;
    record(r.speaker, r.phones.slice(0, a) + r.phones.slice(b), r.path.filter((_, i) => i !== changed), "omit-word");
    record(r.speaker, r.phones.slice(0, b) + r.phones.slice(a, b) + r.phones.slice(b),
      [...r.path.slice(0, changed + 1), r.path[changed], ...r.path.slice(changed + 1)], "repeat-word");
  }
  // Adjacent authorized recordings by the same speaker form a training-only multi-ayah stream.
  const next = recordings.find(s => s.speaker === r.speaker && s.path[0] === r.path.at(-1)! + 1);
  if (next) record(r.speaker, r.phones + next.phones, [...r.path, ...next.path], "joined-ayahs");
  console.log("STREAMS", rows.length);
}
writeFileSync(`${cache}/training.json.gz`, gzipSync(JSON.stringify({ schema: 1, rows, exclusions,
  provenance: recordings.map(({ phones, ...r }) => ({ ...r, split: "train", kind: "none", decodeSha256: sha(phones) })),
  labelSha256: sha(bytes), listener: model.manifest.id, listenerSha256: model.manifest.sha256 })));
console.log("TRAINING_READY", JSON.stringify({ recordings: recordings.length, streams: rows.length, exclusions }));
