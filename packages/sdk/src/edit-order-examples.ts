/** Training-only examples: transcript-consistent word windows against Quran pronunciations of equal length. */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { loadReferenceIndex } from "@tilawa/core";
import type { DecodedPhoneme, ReferenceAsset } from "@tilawa/core";
import { editSequence, orderedAlignments, orderFeatures, shuffledOrderFeatures } from "../../core/dist/edit-order.js";
import type { ScoredEditScript } from "../../core/dist/edit-order.js";
import { loadNodeModel, decodeWav } from "./node.js";
import { createListener } from "./listener.js";
import { editOrderSpeakerFolds, editOrderTrainingLabels } from "./edit-order-training.js";
import type { EditOrderLabel } from "./edit-order-training.js";

const data = process.env.LAB_DATA, out = process.env.LAB_OUT;
if (!data || !out) throw new Error("LAB_DATA and LAB_OUT are required");
const labelsText = readFileSync(join(data, "help-train/labels.jsonl"), "utf8");
const labels = editOrderTrainingLabels(labelsText.trim().split("\n").map(l => JSON.parse(l)) as EditOrderLabel[]);
const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const cache = process.env.EDIT_ORDER_CACHE ?? join(out, "edit-order-decodes");
mkdirSync(cache, { recursive: true });
const model = await loadNodeModel(join(data, "models/spliced-clean-int8"));

type Pronunciation = { written: string; boundaries: Set<number> };
const windows = new Map<number, Map<string, Pronunciation>>();
// Pronunciation content is the only candidate key. No locations or reader identity reach the fitted model.
for (let i = 0; i < reference.words.length; i++) for (let count = 1; count <= 3 && i + count <= reference.words.length; count++) {
  const ws = reference.words.slice(i, i + count);
  if (ws.some(w => w.surah !== ws[0].surah || w.ayah !== ws[0].ayah)) break;
  const start = ws[0].start, end = ws.at(-1)!.end;
  if (end - start < 4 || end - start > 40) continue;
  const written = reference.phonemes.slice(start, end), boundaries = new Set(ws.map(w => w.start - start));
  const key = `${written}/${[...boundaries]}`;
  if (!windows.has(written.length)) windows.set(written.length, new Map());
  windows.get(written.length)!.set(key, { written, boundaries });
}

const pairs: unknown[] = [];
const groups = (paths: ScoredEditScript[]) => {
  const result = new Map<string, ScoredEditScript[]>();
  for (const path of paths) {
    const key = path.script.counts.join(",");
    if (!result.has(key)) result.set(key, []);
    result.get(key)!.push(path);
  }
  return result;
};
const encode = (paths: ScoredEditScript[]) => paths.map(p => ({ features: orderFeatures(p.script), cost: p.cost, sequence: editSequence(p.script) }));
const grams = (s: string) => new Set(Array.from({ length: Math.max(0, s.length - 1) }, (_, i) => s.slice(i, i + 2)));

try {
  for (const label of labels) {
    const file = join(cache, `${model.manifest.sha256}-${label.id}.json`);
    let phones: DecodedPhoneme[];
    if (existsSync(file)) phones = JSON.parse(readFileSync(file, "utf8"));
    else {
      const wav = decodeWav(readFileSync(join(data, "help-train", label.wav)));
      const listener = createListener(model);
      phones = [...(await listener.push(wav.samples, wav.sampleRate))?.phonemes ?? [], ...(await listener.end())?.phonemes ?? []];
      writeFileSync(file, JSON.stringify(phones));
    }
    const ayahs = new Set(label.ayahs), words = reference.words.filter(w => ayahs.has(`${w.surah}:${w.ayah}`));
    if (!words.length || !phones.length || words.some((w, i) => i > 0 && words[i - 1].end !== w.start))
      throw new Error("Eligible recording has empty or non-contiguous transcript/evidence");
    const base = words[0].start, written = reference.phonemes.slice(base, words.at(-1)!.end), heard = phones.map(p => p.p).join("");
    const boundaries = new Set(words.map(w => w.start - base));
    // Alternative complete transcript alignments give latent window boundaries; no tracker visits label them.
    const transcript = orderedAlignments(written, heard, boundaries);
    const crops = new Map<string, { from: number; to: number; obsFrom: number; obsTo: number }>();
    for (const path of transcript) {
      let r = 0, o = 0;
      const at: number[] = [0];
      for (const op of editSequence(path.script)) {
        if (op === "boundary") continue;
        if (op !== "del") o++;
        if (op !== "ins") { r++; at[r] = o; }
      }
      for (let i = 0; i < words.length; i++) for (let n = 1; n <= 3 && i + n <= words.length; n++) {
        const ws = words.slice(i, i + n);
        if (ws.some(w => w.ayah !== ws[0].ayah || w.surah !== ws[0].surah)) break;
        const from = ws[0].start - base, to = ws.at(-1)!.end - base;
        if (to - from < 4 || to - from > 40 || at[from] === at[to]) continue;
        const crop = { from, to, obsFrom: at[from], obsTo: at[to] };
        crops.set(JSON.stringify(crop), crop);
      }
    }
    for (const crop of crops.values()) {
      const text = written.slice(crop.from, crop.to), obs = heard.slice(crop.obsFrom, crop.obsTo);
      const bs = new Set([...boundaries].filter(b => b >= crop.from && b < crop.to).map(b => b - crop.from));
      const positive = groups(orderedAlignments(text, obs, bs));
      const g = grams(obs);
      const competitors = [...(windows.get(text.length)?.values() ?? [])].filter(p => p.written !== text && p.boundaries.size === bs.size)
        .map(p => ({ p, overlap: [...grams(p.written)].filter(x => g.has(x)).length }))
        .sort((a, b) => b.overlap - a.overlap || a.p.written.localeCompare(b.p.written)).slice(0, 24);
      for (const { p } of competitors) {
        const negative = groups(orderedAlignments(p.written, obs, p.boundaries));
        for (const [counts, positives] of positive) {
          const negatives = negative.get(counts);
          if (!negatives) continue;
          const c = positives[0].script.counts;
          pairs.push({ speaker: label.speaker, referenceLength: text.length, counts: c,
            positive: encode(positives), negative: encode(negatives), control: shuffledOrderFeatures(c) });
        }
      }
    }
    console.log(JSON.stringify({ decoded: label.id, pairs: pairs.length }));
  }
  const speakers = new Set((pairs as { speaker: string }[]).map(p => p.speaker));
  const folds = editOrderSpeakerFolds([...speakers].map(speaker => ({ speaker })));
  const metadata = { trainingOnly: true, listener: model.manifest.id, modelSha256: model.manifest.sha256,
    labelsSha256: createHash("sha256").update(labelsText).digest("hex"), clips: labels.length, speakers: speakers.size,
    pairs: pairs.length, folds: Object.fromEntries(folds), scripts: 16, competitors: 24,
    isolation: "none, train, scored, no events/repeats; all non-train speakers excluded",
    matching: "exact match/sub/ins/del/boundary counts and reference phone length; normalized acoustic weights within each matched script group",
    recurrence: "preceding operation, 25 transition parameters, centered at exact uniform-permutation expected transitions" };
  writeFileSync(join(out, "edit-order-examples.json"), JSON.stringify({ metadata, pairs }));
  console.log(JSON.stringify(metadata));
} finally { await model.backend.release(); }
