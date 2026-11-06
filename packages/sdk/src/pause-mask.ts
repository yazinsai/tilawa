import { minGapFrames, waqfMarked } from "@tilawa/core";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
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

/** Full streaming pass with the original grid, frontend and carried positions. */
export async function decodePauseMasked(model: LoadedModel, samples: Float32Array, spans: PauseSpan[], k: number): Promise<PhonemeEvidence[]> {
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
