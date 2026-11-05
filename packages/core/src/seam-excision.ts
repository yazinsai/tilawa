import { hypothesis } from "./ayah-ledger.js";
import type { PosteriorFrames } from "./ctc.js";
import type { WordSeparatorEvidence } from "./evidence.js";

/** Missing head coverage cannot certify a word. Endpoint separators are allowed;
 * only a separator strictly inside the first-to-last token interval vetoes it. */
export class SeamWordBoundaries {
  private evidence: WordSeparatorEvidence[] = [];

  append(e: WordSeparatorEvidence) {
    if (e.toFrame > e.fromFrame) this.evidence.push(e);
  }

  oneWord(first: number, last: number): boolean {
    if (last < first) return false;
    let covered = first;
    for (const e of [...this.evidence].sort((a, b) => a.fromFrame - b.fromFrame)) {
      if (e.frames.some(f => f > first && f < last)) return false;
      if (e.fromFrame <= covered) covered = Math.max(covered, e.toFrame);
    }
    return covered > last;
  }

  dropBefore(frame: number) {
    this.evidence = this.evidence.filter(e => e.toFrame > frame);
  }
}

/** A held nasal after phones tentatively excised as surplus is a stopped nun, not
 * the previous word's tail. A bare nasal at the span's edge still strips normally.
 * The counterfactual neighbour fit must establish that these phones are surplus. */
export function surplusNasal(raw: string): string {
  return raw.replace(/([^ۦۥں۾ڇؙۜ])ں+/gu, "$1ن");
}

/** Closed read versus each neighbour omitted, on the same remaining frames.
 * Each word may use its canonical or pausal form. Both competitors get the same
 * choices and contiguous word-repeat arcs; neither has a filler path. The third
 * word is right context, not another neighbour being certified. */
export function excisedNeighbourMargins(
  frames: PosteriorFrames,
  idOf: (phone: string) => number,
  forms: readonly (readonly string[])[],
  from: number,
  to: number,
  cut: readonly [number, number],
): number[] | null {
  if (forms.length !== 3 || forms.some(f => !f.length) || from < frames.startFrame || to > frames.endFrame ||
      cut[0] < from || cut[1] > to || cut[1] <= cut[0]) return null;
  const best = (drop: number) => {
    let nll = Infinity;
    const paths: string[][] = [[]];
    for (let i = 0; i < forms.length; i++) {
      if (i === drop) continue;
      const prev = paths.splice(0);
      for (const path of prev) for (const form of forms[i]) paths.push([...path, form]);
    }
    for (const path of paths) {
      // Singleton groups give exactly the ledger's word-repeat arcs, without
      // inventing a repeat of a partial ayah across this seam.
      const h = hypothesis(path.map(p => [p]), idOf);
      nll = Math.min(nll, frames.nll(h.ids, from, to, h.repeats, cut));
    }
    return nll;
  };
  const read = best(-1);
  if (!Number.isFinite(read)) return null;
  return [best(0) - read, best(1) - read];
}
