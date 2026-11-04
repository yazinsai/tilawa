import { waqfMarked } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceIndex, SeamEvidence } from "@tilawa/core";
import type { LockPoint } from "./gap-decode.js";
import type { LoadedModel } from "./model.js";

export function seamSpans(index: ReferenceIndex, steps: LockPoint[]) {
  const spans: { word: number; fromMs: number; toMs: number }[] = [];
  for (let i = 0; i < steps.length; i++) {
    const left = steps[i], prev = index.words[left.wi], next = index.words[left.wi + 1];
    if (!prev || !next || !(prev.surah !== next.surah || prev.ayah !== next.ayah || waqfMarked(left.wi))) continue;
    let j = i + 1;
    while (j < steps.length && steps[j].wi === left.wi + 1) j++;
    const right = steps[j];
    if (right?.wi === left.wi + 2 && right.timeMs > left.timeMs) spans.push({ word: left.wi + 1, fromMs: left.timeMs, toMs: right.timeMs + 300 });
  }
  return spans;
}

/** Enrich the first token with bounded second-ear evidence; every main token and posterior row is retained. */
export async function attachSeamEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[], steps: LockPoint[], index: ReferenceIndex) {
  const first = chunks.findIndex(c => c.phonemes.length);
  if (!model.seamEar || first < 0) return chunks;
  const frameMs = chunks[first].frameMs, k = chunks[first].posteriors?.k ?? 5;
  const seams: SeamEvidence[] = [];
  for (const span of seamSpans(index, steps)) {
    const i0 = Math.max(0, Math.floor((span.fromMs - 4000) * model.manifest.sampleRate / 1000));
    const i1 = Math.min(audio.length, Math.ceil(span.toMs * model.manifest.sampleRate / 1000));
    if (i1 <= i0) continue;
    const crop = await model.seamEar.decode(audio.subarray(i0, i1), k);
    if (crop.inventory !== index.inventoryId || crop.frameMs !== frameMs || !crop.posteriors) throw new Error("Teacher evidence differs from the main listener clock or inventory");
    const shift = Math.round(i0 / model.manifest.sampleRate * 1000 / frameMs);
    const fromFrame = Math.round(span.fromMs / frameMs), toFrame = Math.floor(Math.min(span.toMs, audio.length / model.manifest.sampleRate * 1000) / frameMs);
    const post = crop.posteriors;
    if (toFrame > shift + post.nFrames) throw new Error("Teacher crop does not cover the seam");
    const word = index.words[span.word];
    seams.push({ word: { surah: word.surah, ayah: word.ayah, word: word.word }, fromFrame, toFrame,
      phonemes: crop.phonemes.map(p => ({ ...p, start: p.start + shift, end: Math.min(toFrame - 1, p.end + shift) })).filter(p => p.start >= fromFrame && p.start < toFrame),
      posteriors: { ...post, startFrame: post.startFrame + shift, ids: Array.from(post.ids), probs: Array.from(post.probs) } });
  }
  if (!seams.length) return chunks;
  return chunks.map((c, i) => i !== first ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p : { ...p, seams }) });
}
