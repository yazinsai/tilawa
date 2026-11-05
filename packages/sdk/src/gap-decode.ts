import {
  GAP_PAD_S,
  gapCarriesAddition,
  interWordGaps,
  minGapFrames,
  placeGap,
  recoverGapPhonemes,
  lexicalWords,
  pauseSpellingSpans,
  preserveColdSpellings,
  seamSpellingText,
} from "@tilawa/core";
import type { GapPhoneme, PhonemeEvidence, ReferenceIndex, WordGap } from "@tilawa/core";
import { createListener } from "./listener.js";
import type { LoadedModel } from "./model.js";

export interface LockPoint {
  wi: number;
  timeMs: number;
}

interface FlatEvidence {
  inventory: string;
  frameMs: number;
  phonemes: GapPhoneme[];
  k: number;
  ids: number[];
  probs: number[];
  nFrames: number;
}

function flatten(chunks: PhonemeEvidence[]): FlatEvidence | null {
  if (!chunks.length || chunks.some((c) => !c.posteriors)) return null;
  const frameMs = chunks[0].frameMs;
  const k = chunks[0].posteriors!.k;
  if (chunks.some((c) => c.frameMs !== frameMs || c.posteriors!.k !== k)) return null;
  const phonemes = chunks.flatMap((c) => c.phonemes);
  const ids: number[] = [];
  const probs: number[] = [];
  let nFrames = 0;
  for (const c of chunks) {
    const p = c.posteriors!;
    const rows = Math.min(p.nFrames, Math.floor(p.ids.length / k));
    for (let i = 0; i < rows * k; i++) {
      ids.push(Number(p.ids[i]));
      probs.push(Number(p.probs[i]));
    }
    nFrames += rows;
  }
  return { inventory: chunks[0].inventory, frameMs, phonemes, k, ids, probs, nFrames };
}

export async function decodeIsolated(model: LoadedModel, samples: Float32Array, sampleRate: number, k: number) {
  const listener = createListener(model, { posteriors: true, topK: k });
  const out: PhonemeEvidence[] = [];
  const step = Math.max(1, Math.round(sampleRate * 0.02));
  for (let i = 0; i < samples.length; i += step) {
    const e = await listener.push(samples.subarray(i, Math.min(samples.length, i + step)), sampleRate);
    if (e) out.push(e);
  }
  const tail = await listener.end();
  if (tail) out.push(tail);
  const posts = out.flatMap((c) => (c.posteriors ? [c.posteriors] : []));
  return {
    phonemes: out.flatMap((c) => c.phonemes),
    k: posts[0]?.k ?? k,
    ids: posts.flatMap((p) => Array.from(p.ids)),
    probs: posts.flatMap((p) => Array.from(p.probs)),
  };
}

/** Side evidence for the acoustic superset of exact, pause-bounded spellings.
 * The checker still selects and certifies its own seam candidate unchanged. */
export function coldSpellingDecoder(model: LoadedModel, index: ReferenceIndex, cache = new Map<string, string>()) {
  const spans = pauseSpellingSpans([...index.tokens]), lexical = lexicalWords(index);
  const done = cache;
  return {
    cache,
    async push(e: PhonemeEvidence, audio: () => Float32Array, sampleRate: number): Promise<void> {
      let samples: Float32Array | undefined;
      for (const span of spans.push(e)) {
        if (!lexical.spelledWord(seamSpellingText(span.raw))) continue;
        const key = `${span.fromFrame}:${span.toFrame}`;
        let phonemes = done.get(key);
        if (phonemes === undefined) {
          const cold = await decodeColdSpan(model, samples ??= audio(), sampleRate, span.fromFrame, span.toFrame, e.frameMs, e.posteriors!.k);
          phonemes = cold.map(p => p.p).join("");
          done.set(key, phonemes);
        }
        const proofs = span.carrier.seamCold ??= [];
        if (!proofs.some(p => p.fromFrame === span.fromFrame && p.toFrame === span.toFrame)) {
          proofs.push({ fromFrame: span.fromFrame, toFrame: span.toFrame, phonemes });
        }
      }
    },
  };
}

/** Re-encode a selected span from a fresh listener with the existing gap padding.
 * Padding gives the model edge context, but only token centres inside the span
 * contribute to its identity confirmation. No neighbour phones are removed. */
export async function decodeColdSpan(
  model: LoadedModel, audio: Float32Array, sampleRate: number,
  fromFrame: number, toFrame: number, frameMs: number, k: number,
): Promise<GapPhoneme[]> {
  if (fromFrame < 0 || toFrame <= fromFrame || frameMs <= 0 || sampleRate <= 0 ||
      toFrame * frameMs / 1000 > audio.length / sampleRate) return [];
  const frameS = frameMs / 1000;
  const t0 = Math.max(0, fromFrame * frameS - GAP_PAD_S);
  const t1 = Math.min(audio.length / sampleRate, toFrame * frameS + GAP_PAD_S);
  const i0 = Math.floor(t0 * sampleRate), i1 = Math.min(audio.length, Math.ceil(t1 * sampleRate));
  const crop = await decodeIsolated(model, audio.subarray(i0, i1), sampleRate, k);
  const shift = Math.round(t0 / frameS);
  return crop.phonemes.map(p => ({ ...p, start: p.start + shift, end: p.end + shift }))
    .filter(p => (p.start + p.end) / 2 >= fromFrame && (p.start + p.end) / 2 < toFrame);
}

function gapsFor(index: ReferenceIndex, locks: LockPoint[], obs: GapPhoneme[], frameMs: number): WordGap[] {
  const runs: LockPoint[][] = [];
  let cur: LockPoint[] = [];
  for (const lock of locks) {
    if (cur.length && lock.wi <= cur.at(-1)!.wi) {
      runs.push(cur);
      cur = [];
    }
    cur.push(lock);
  }
  if (cur.length) runs.push(cur);
  const minFrames = minGapFrames(frameMs);
  const seen = new Set<string>();
  const gaps: WordGap[] = [];
  const wordText = (wi: number) => {
    const w = index.words[wi];
    return w ? index.phonemes.slice(w.start, w.end) : "";
  };
  for (const run of runs) {
    // Overlapping windows keep each pause inside a bounded alignment.
    for (let i = 0; i < run.length; i += 39) {
      const window = run.slice(i, i + 40);
      if (window.length < 2) continue;
      const first = index.words[window[0].wi];
      const last = index.words[window.at(-1)!.wi];
      if (!first || !last || last.end <= first.start) continue;
      const t0 = window[0].timeMs - 3000;
      const t1 = window.at(-1)!.timeMs + 1000;
      const slice = obs.filter((p) => p.end * frameMs >= t0 && p.start * frameMs <= t1);
      for (const gap of interWordGaps({
        phonemes: index.phonemes,
        from: first.start,
        to: last.end,
        wordAt: (offset) => index.wordAt(offset),
        wordText,
        obs: slice,
        minFrames,
      })) {
        const key = `${gap.fromFrame}:${gap.toFrame}`;
        if (seen.has(key)) continue;
        seen.add(key);
        gaps.push(gap);
      }
    }
  }
  return gaps;
}

/**
 * Re-encode each pause between two confidently matched words with a fresh model, and where that decode
 * carries an addition the neighbours do not explain, replace the gap's phones and posterior rows with it.
 * Returns the evidence to record. The in-context decode is kept wherever the cropped window adds nothing.
 */
export async function spliceContextFreeGaps(
  model: LoadedModel,
  audio: Float32Array,
  sampleRate: number,
  chunks: PhonemeEvidence[],
  locks: LockPoint[],
  index: ReferenceIndex,
): Promise<PhonemeEvidence[]> {
  const flat = flatten(chunks);
  if (!flat || !locks.length || !audio.length) return chunks;
  const gaps = gapsFor(index, locks, flat.phonemes, flat.frameMs);
  if (!gaps.length) return chunks;
  const frameS = flat.frameMs / 1000;
  let phonemes = flat.phonemes;
  const ids = flat.ids.slice();
  const probs = flat.probs.slice();
  let changed = false;
  for (const gap of gaps) {
    const t0 = Math.max(0, gap.fromFrame * frameS - GAP_PAD_S);
    const t1 = Math.min(audio.length / sampleRate, gap.toFrame * frameS + GAP_PAD_S);
    const i0 = Math.floor(t0 * sampleRate);
    const i1 = Math.min(audio.length, Math.ceil(t1 * sampleRate));
    // A multi-second hole is a break or a skip, not the pause between two words.
    if (t1 - t0 > 8) continue;
    if (i1 - i0 < sampleRate * 0.2) continue;
    const crop = await decodeIsolated(model, audio.subarray(i0, i1), sampleRate, flat.k);
    if (crop.k !== flat.k) continue;
    const shift = Math.round(t0 / frameS);
    const shifted = crop.phonemes.map((p) => ({ ...p, start: p.start + shift, end: p.end + shift }));
    const kept = recoverGapPhonemes(shifted, gap.fromFrame, gap.toFrame, gap.left, gap.right);
    if (!gapCarriesAddition(kept)) continue;
    phonemes = placeGap(phonemes, gap, kept);
    for (let f = gap.fromFrame; f < gap.toFrame && f < flat.nFrames; f++) {
      const src = f - shift;
      if (src < 0 || (src + 1) * crop.k > crop.ids.length) continue;
      for (let c = 0; c < flat.k; c++) {
        ids[f * flat.k + c] = crop.ids[src * crop.k + c];
        probs[f * flat.k + c] = crop.probs[src * crop.k + c];
      }
    }
    changed = true;
  }
  if (!changed) return chunks;
  return [{
    inventory: flat.inventory,
    frameMs: flat.frameMs,
    phonemes: preserveColdSpellings(flat.phonemes, phonemes),
    posteriors: { k: flat.k, startFrame: 0, nFrames: flat.nFrames, ids, probs },
  }];
}
