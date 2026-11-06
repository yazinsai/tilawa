import { minGapFrames, waqfMarked } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceIndex, PauseMaskEvidence } from "@tilawa/core";
import type { LockPoint } from "./gap-decode.js";
import type { LoadedModel } from "./model.js";

/** Half-open native CTC span. Cut at pause onset; keep the pause as warm-up. */
export interface PauseSpan { fromFrame: number; toFrame: number }

/** Detector pauses at ayah/marked inner boundaries, ending at the primary step onto word 2. */
export function pauseMaskSpans(chunks: PhonemeEvidence[], locks: LockPoint[], index: ReferenceIndex): PauseSpan[] {
  if (!chunks.length || chunks.some(c => !c.posteriors)) return [];
  const frameMs = chunks[0].frameMs, rows = chunks.flatMap(c => {
    const p = c.posteriors!;
    return Array.from({ length: p.nFrames }, (_, t) => ({ frame: p.startFrame+t,
      blank: Array.from({ length: p.k }, (_, j) => j).some(j => p.ids[t*p.k+j] === 0 && p.probs[t*p.k+j] >= .5) }));
  });
  const pauses: PauseSpan[] = [];
  for (let a = 0; a < rows.length;) {
    let b = a+1;
    while (b < rows.length && rows[b].frame === rows[b-1].frame+1 && rows[b].blank === rows[a].blank) b++;
    if (rows[a].blank && b-a >= minGapFrames(frameMs)) pauses.push({ fromFrame: rows[a].frame, toFrame: rows[b-1].frame+1 });
    a = b;
  }
  const spans: PauseSpan[] = [];
  const nextWord = (wi: number) => { do { wi++; } while (wi < index.words.length && index.words[wi].end === index.words[wi].start); return wi; };
  for (let i = 1; i+1 < locks.length; i++) {
    const left = locks[i-1], first = locks[i], second = locks[i+1];
    if (first.wi !== nextWord(left.wi) || second.wi !== nextWord(first.wi)) continue;
    const prev = index.words[left.wi], word = index.words[first.wi];
    if (!prev || !word || prev.surah !== word.surah) continue;
    const ayah = prev.ayah !== word.ayah && word.word === 1;
    const inner = prev.ayah === word.ayah && waqfMarked(left.wi);
    if (!ayah && !inner) continue;
    const from = left.timeMs/frameMs, entry = first.timeMs/frameMs, end = Math.ceil(second.timeMs/frameMs);
    for (const pause of pauses.filter(p => p.fromFrame >= from && p.fromFrame <= entry && p.toFrame < end))
      spans.push({ fromFrame: pause.fromFrame, toFrame: end });
  }
  return spans;
}

/** A model input on the native embedding clock; positions outside spans are unmasked. */
export function pauseFloor(spans: PauseSpan[], processed: number, chunk: number): BigInt64Array {
  const floors = new BigInt64Array(chunk).fill(-1n);
  for (let q = 0; q < chunk; q++) {
    const frame = (processed+q)/2;
    const active = spans.filter(s => frame >= s.fromFrame && frame < s.toFrame).at(-1);
    if (active) floors[q] = BigInt(active.fromFrame*2);
  }
  return floors;
}

/** Preserve primary values and carry checker-only rows through token-preserving JSON. */
export function carryPauseEvidence(primary: PhonemeEvidence[], masked: PhonemeEvidence[], spans: PauseSpan[]): PhonemeEvidence[] {
  if (!spans.length || !primary.length || !masked.length) return primary;
  const at = primary.findIndex(c => c.phonemes.length);
  if (at < 0) return primary;
  if (masked.some(c => c.inventory !== primary[0].inventory || c.frameMs !== primary[0].frameMs || !c.posteriors))
    throw new Error("Pause-trained evidence differs from the primary inventory or clock");
  const k = primary[0].posteriors?.k;
  if (!k || masked.some(c => c.posteriors!.k !== k)) throw new Error("Pause-trained posterior width differs from primary evidence");
  const ranges: PauseSpan[] = [];
  for (const s of [...spans].sort((a,b) => a.fromFrame-b.fromFrame)) {
    const last = ranges.at(-1);
    if (last && s.fromFrame <= last.toFrame) last.toFrame = Math.max(last.toFrame,s.toFrame);
    else ranges.push({ ...s });
  }
  const phones = masked.flatMap(c => c.phonemes), rows = new Map<number,{ ids: number[]; probs: number[] }>();
  for (const c of masked) {
    const r = c.posteriors!;
    for (let t = 0; t < r.nFrames; t++) rows.set(r.startFrame+t,{
      ids: Array.from({ length: k },(_,j) => r.ids[t*k+j]),probs: Array.from({ length: k },(_,j) => r.probs[t*k+j]),
    });
  }
  const metadata: PauseMaskEvidence[] = [];
  for (const s of ranges) {
    const data = Array.from({ length: s.toFrame-s.fromFrame },(_,i) => rows.get(s.fromFrame+i));
    if (data.some(r => !r)) continue;
    metadata.push({ ...s,inventory: primary[0].inventory,frameMs: primary[0].frameMs,
      phonemes: phones.filter(p => p.start < s.toFrame && p.end >= s.fromFrame).map(p => ({
        p: p.p,start: Math.max(p.start,s.fromFrame),end: Math.min(p.end,s.toFrame-1),conf: p.conf,
      })),posteriors: { startFrame: s.fromFrame,nFrames: data.length,k,
        ids: data.flatMap(r => r!.ids),probs: data.flatMap(r => r!.probs) } });
  }
  if (!metadata.length) return primary;
  return primary.map((c,i) => i !== at ? c : { ...c,phonemes: c.phonemes.map((p,j) => j ? p : { ...p,pauseMaskEvidence: metadata }) });
}

/** Full streaming pass with the original grid, frontend and carried positions. */
export async function decodePauseMasked(model: LoadedModel, samples: Float32Array, spans: PauseSpan[], k: number): Promise<PhonemeEvidence[]> {
  if (!model.manifest.pauseMask || model.manifest.pauseMask.clockMs !== 20 || model.manifest.frameMs !== 40)
    throw new Error("Pause-trained streaming requires the declared mask and native 40 ms CTC clock");
  const { createListener } = await import("./listener.js");
  const listener = createListener(model, { posteriors: true, topK: k, pauseMask: spans });
  const chunks: PhonemeEvidence[] = [];
  for (let a = 0; a < samples.length; a += model.manifest.sampleRate) {
    const e = await listener.push(samples.subarray(a,a+model.manifest.sampleRate),model.manifest.sampleRate);
    if (e) chunks.push(e);
  }
  const tail = await listener.end();
  if (tail) chunks.push(tail);
  return chunks;
}
