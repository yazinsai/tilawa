import type { DecodedPhoneme, PhonemeEvidence } from "@tilawa/core";

/**
 * Cuts an evidence stream into cells of `ms` audio by phoneme start, whatever chunks it arrives in. A cell is ready once
 * a phoneme from a later cell arrives or decoding has passed its end.
 */
export function evidenceGrid(ms: number) {
  let held: DecodedPhoneme[] = [];
  let frameMs = 0;
  let inventory = "";
  let post: { k: number; startFrame: number; ids: number[]; probs: number[] } | null = null;
  const cellOf = (frame: number) => Math.floor((frame * frameMs) / ms);
  const cut = (n: number, until: number, decodedTo?: number): PhonemeEvidence => {
    const e: PhonemeEvidence = { inventory, frameMs, phonemes: held.slice(0, n), ...(decodedTo !== undefined && { decodedTo }) };
    held = held.slice(n);
    if (post) {
      const take = Math.max(0, Math.min(until, post.startFrame + post.ids.length / post.k) - post.startFrame);
      e.posteriors = { k: post.k, startFrame: post.startFrame, nFrames: take, ids: post.ids.splice(0, take * post.k), probs: post.probs.splice(0, take * post.k) };
      post.startFrame += take;
    }
    return e;
  };
  return {
    add(e: PhonemeEvidence) {
      frameMs = e.frameMs;
      inventory = e.inventory;
      held.push(...e.phonemes);
      if (e.posteriors) {
        post ??= { k: e.posteriors.k, startFrame: e.posteriors.startFrame, ids: [], probs: [] };
        post.ids.push(...Array.from(e.posteriors.ids));
        post.probs.push(...Array.from(e.posteriors.probs));
      }
    },
    /** Complete cells; `decoded`: the first frame whose phonemes may still be coming, when the source knows it. */
    ready(decoded?: number): PhonemeEvidence[] {
      const out: PhonemeEvidence[] = [];
      while (held.length) {
        const c = cellOf(held[0].start);
        const n = held.findIndex((p) => cellOf(p.start) > c);
        if (n < 0 && !(decoded !== undefined && cellOf(decoded) > c)) break;
        const end = Math.ceil(((c + 1) * ms) / frameMs);
        out.push(cut(n < 0 ? held.length : n, n < 0 ? end : held[n].start, end));
      }
      return out;
    },
    rest(): PhonemeEvidence[] {
      const out = this.ready();
      if (held.length || post?.ids.length) out.push(cut(held.length, Infinity));
      return out;
    },
  };
}
