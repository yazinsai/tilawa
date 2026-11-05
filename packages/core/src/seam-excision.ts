import { hypothesis } from "./ayah-ledger.js";
import { PosteriorFrames } from "./ctc.js";
import type { DecodedPhoneme, PhonemeEvidence } from "./evidence.js";
import { minGapFrames } from "./gap-decode.js";

/** A held nasal after phones tentatively excised as surplus is a stopped nun, not
 * the previous word's tail. A bare nasal at the span's edge still strips normally.
 * The counterfactual neighbour fit must establish that these phones are surplus. */
export function surplusNasal(raw: string): string {
  return raw.replace(/([^ۦۥں۾ڇؙۜ])ں+/gu, "$1ن");
}

/** Same nasal retention and waqf-edge strip used by the parent candidate. */
export function seamSpellingText(raw: string): string {
  return surplusNasal(raw).replace(/^[ۦۥں۾ڇؙۜ]+|[ۦۥں۾ڇؙۜ]+$/gu, "");
}

/** Incremental acoustic superset of the parent's pause-bounded candidates.
 * It uses the same run-end pause and blank ownership tests, without changing
 * any tracker or checker window. A result's carrier is the next warm token. */
export function pauseSpellingSpans(tokens: readonly string[]) {
  const frames = new PosteriorFrames(tokens.length);
  let prev: DecodedPhoneme | null = null;
  let utterance: DecodedPhoneme[] | null = null;
  return {
    push(e: PhonemeEvidence): { fromFrame: number; toFrame: number; raw: string; carrier: DecodedPhoneme }[] {
      if (!e.posteriors) return [];
      frames.append(e.posteriors);
      const out: { fromFrame: number; toFrame: number; raw: string; carrier: DecodedPhoneme }[] = [];
      const argmax = (f: number) => {
        const row = frames.row(f);
        let best = 0;
        for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
        return best;
      };
      for (const ph of e.phonemes) {
        let pause = false;
        if (prev) {
          let a = prev.end + 1;
          const b = ph.start, id = tokens.indexOf(prev.p) + 1;
          if (a >= frames.startFrame && b <= frames.endFrame) {
            if (id) while (a < b && argmax(a) === id) a++;
            pause = b - a >= minGapFrames(e.frameMs);
            for (let f = a; pause && f < b; f++) if (frames.row(f)[0] < Math.log(0.5)) pause = false;
          }
        }
        if (pause) {
          if (utterance?.length) out.push({ fromFrame: utterance[0].start,
            toFrame: prev!.end + 1, raw: utterance.map(p => p.p).join(""), carrier: ph });
          utterance = [ph];
        } else if (utterance) utterance.push(ph);
        prev = ph;
      }
      if (prev) frames.dropBefore(prev.end + 1);
      return out;
    },
  };
}

/** Move side evidence off any warm token removed by the existing cold-gap splice. */
export function preserveColdSpellings(before: DecodedPhoneme[], after: DecodedPhoneme[]): DecodedPhoneme[] {
  const proofs = before.flatMap(p => p.seamCold ?? []);
  if (!proofs.length || !after.length) return after;
  const out = after.map(({ seamCold: _, ...p }) => p as DecodedPhoneme);
  for (const proof of proofs) {
    const carrier = out.find(p => p.start >= proof.toFrame) ?? out.at(-1)!;
    (carrier.seamCold ??= []).push(proof);
  }
  return out;
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
