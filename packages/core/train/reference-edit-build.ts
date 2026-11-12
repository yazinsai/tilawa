// Training only. Decode once; all reference variants consume the identical retained token string.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { loadNodeModel } from "../../sdk/src/node.js";
import { createListener } from "../../sdk/src/listener.js";
import { decodeWav } from "../../sdk/src/wav.js";
import { loadReferenceIndex } from "../src/reference.js";
import { createLocalSearch } from "../src/local.js";
import { editedReference, jumpFeature } from "../src/reference-edit.js";
import { deletionCost, insertionCost, subCost, UNIT } from "../src/fuzzy.js";
import type { ReferenceIndex } from "../src/reference.js";

const root = `${process.env.LAB_DATA}/help-train`;
const cache = "/scratch/tracker-reference-edit-equivariance/decodes";
mkdirSync(cache, { recursive: true });
const labels = readFileSync(`${root}/labels.jsonl`, "utf8").trim().split("\n").map(x => JSON.parse(x));
if (labels.some(r => r.split !== "train")) throw new Error("Non-training speaker in export");
const clean = labels.filter(r => r.kind === "none" && !r.events?.length && !r.non_errors?.length
  && !r.unsupported?.length && !r.exclude && !r.excluded && r.scored !== false).sort((a, b) => a.id.localeCompare(b.id));
const index = loadReferenceIndex(JSON.parse(readFileSync("packages/core/assets/ref-v2.json", "utf8")));
const listenerModel = await loadNodeModel(`${process.env.LAB_DATA}/models/spliced-clean-int8`);

function graph(ref: ReferenceIndex, first: number, last: number, phones: string, mapping: number[]) {
  const t = createLocalSearch(ref, { offset: ref.words[first].start, origin: ref.words[first].start, edits: 0 });
  const init = t.save() as any;
  const lo = init.lo, n = init.cost.length, starts = new Set(ref.words.map(w => w.start));
  const source: number[] = [], jump: number[][] = [], ins: number[][] = [], sub: number[][] = [];
  const fixed: number[][] = [];
  const entry = (at: number) => { const wi = ref.wordAt(at), w = ref.words[wi];
    return wi > 0 && w.start === at && w.surah !== ref.words[wi - 1].surah && w.surah !== 9 ? 3 : 0; };
  for (const p of phones) {
    const before = t.save() as any, best = before.best;
    if (best < lo || best >= lo + n) return null;
    source.push(best - lo);
    jump.push(Array.from({ length: n }, (_, i) => i && starts.has(lo + i - 1) && lo + i - 1 !== best
      ? jumpFeature(ref, best, lo + i - 1) : -1));
    ins.push(Array.from({ length: n }, (_, i) => insertionCost(ref.phonemes, lo + i, p) / UNIT));
    sub.push(Array.from({ length: n }, (_, i) => i ? subCost(ref.phonemes[lo + i - 1], p) / UNIT : 0));
    fixed.push(Array.from({ length: n }, (_, i) => i ? Math.abs(lo + i - 1 - best) * 0.0001
      + (ref.words[ref.wordAt(lo + i - 1)].surah !== ref.words[ref.wordAt(Math.max(0, best - 1))].surah && !entry(lo + i - 1) ? 3 : 0) : 0));
    t.step(p);
  }
  const owners = Array.from({ length: n }, (_, i) => mapping[ref.wordAt(Math.max(0, lo + i - 1))]);
  return { source, jump, ins, sub, fixed, initial: ref.words[first].start - lo,
    del: Array.from({ length: n }, (_, i) => i ? deletionCost(ref.phonemes, lo + i - 1) / UNIT : 0),
    entry: Array.from({ length: n }, (_, i) => i ? entry(lo + i - 1) : 0), owners,
    allowed: Array.from({ length: n }, (_, i) => lo + i >= ref.words[first].start && lo + i <= ref.words[last].end) };
}

const pairs: any[] = [], provenance: any[] = [], excluded: any[] = [];
for (const r of clean) {
  const data = readFileSync(`${root}/${r.wav}`), hash = createHash("sha256").update(data).digest("hex");
  const path = `${cache}/${listenerModel.manifest.sha256}-${hash}.json`;
  let phones: string;
  if (existsSync(path)) phones = JSON.parse(readFileSync(path, "utf8")).phones;
  else {
    const wav = decodeWav(data), listener = createListener(listenerModel), evidence: any[] = [];
    const step = Math.round(wav.sampleRate * 0.32);
    for (let i = 0; i < wav.samples.length; i += step) {
      const e = await listener.push(wav.samples.slice(i, i + step), wav.sampleRate); if (e) evidence.push(e);
    }
    const end = await listener.end(); if (end) evidence.push(end);
    phones = evidence.flatMap(e => e.phonemes).map(p => p.p).join("");
    writeFileSync(path, JSON.stringify({ phones }));
  }
  const targets = index.words.map((w, i) => r.ayahs.includes(`${w.surah}:${w.ayah}`) ? i : -1).filter(i => i >= 0);
  if (targets.length < 3 || !phones.length) { excluded.push({ id: r.id, reason: "no internal editable word" }); continue; }
  // One internal word chosen deterministically per recording; selection uses no practice observation.
  const editable = targets.slice(1, -1).filter(i => index.words[i].end > index.words[i].start);
  if (!editable.length) { excluded.push({ id: r.id, reason: "no nonempty internal word" }); continue; }
  const changed = editable[parseInt(hash.slice(0, 8), 16) % editable.length];
  const written = index.phonemes.slice(index.words[changed].start, index.words[changed].end);
  const donor = targets.map(i => index.wordForms?.[i]?.[0]).find(p => p && p !== written);
  if (!donor) throw new Error("Missing training replacement pronunciation");
  const original = graph(index, targets[0], targets.at(-1)!, phones, index.words.map((_, i) => i));
  if (!original) { excluded.push({ id: r.id, reason: "baseline left initial local search window" }); continue; }
  const originalOwners = [...original.owners];
  for (const kind of ["delete", "insert", "substitute"] as const) {
    const copy = editedReference(index, changed, kind, donor);
    const first = copy.correspondence.indexOf(targets[0]), last = copy.correspondence.indexOf(targets.at(-1)!);
    const edited = graph(copy.index, first, last, phones, copy.correspondence);
    if (!edited) { excluded.push({ id: r.id, kind, reason: "baseline left initial local search window" }); continue; }
    const common = targets.filter(i => i !== changed);
    for (const g of [original, edited]) {
      // Local locations outside the recorded transcript remain separate competitors in the graph.
      g.owners = g.owners.map(i => common.indexOf(i));
    }
    pairs.push({ speaker: r.speaker, kind, original: { ...original }, edited, common: common.length });
    // Restore original ordinals for the next paired edit.
    original.owners = [...originalOwners];
  }
  provenance.push({ speaker: r.speaker, audioSha256: hash, decodedSha256: createHash("sha256").update(phones).digest("hex") });
  console.log("built", provenance.length, "/", clean.length, "pairs", pairs.length);
}
await listenerModel.backend.release();
const output = JSON.stringify({ pairs, provenance, excluded,
  listener: listenerModel.manifest.id, listenerSha256: listenerModel.manifest.sha256 });
writeFileSync(`${process.env.LAB_OUT}/pairs.json`, output);
writeFileSync("/scratch/tracker-reference-edit-equivariance/pairs.json", output);
console.log("TRAINING_EXPORT", JSON.stringify({ recordings: provenance.length, pairs: pairs.length, excluded }));
