import { EvidenceError } from "./evidence.js";
import type { TopKPosteriors } from "./evidence.js";

/** Floor for any token's probability, so log-scores stay finite. See docs/evidence-contract.md. */
const MIN_P = 1e-6;

/**
 * Top-k posterior rows accumulated across a session, expanded on demand into per-frame log-probabilities over the
 * blank (id 0) and the inventory tokens (ids 1..size).
 */
export class PosteriorFrames {
  private ids: number[] = [];
  private probs: number[] = [];
  private k = 0;
  private start = -1;
  private readonly cache = new Map<number, Float64Array>();

  constructor(private readonly size: number) {}

  get startFrame() {
    return this.start;
  }

  get endFrame() {
    return this.start + this.ids.length / Math.max(1, this.k);
  }

  append(p: TopKPosteriors) {
    if (p.ids.length !== p.nFrames * p.k || p.probs.length !== p.nFrames * p.k) {
      throw new EvidenceError(`Posterior rows hold ${p.ids.length} ids and ${p.probs.length} probs, expected nFrames x k = ${p.nFrames * p.k}`);
    }
    if (this.k && p.k !== this.k) throw new EvidenceError(`Posterior k changed from ${this.k} to ${p.k} within a session`);
    if (this.start !== -1 && p.startFrame !== this.endFrame) {
      throw new EvidenceError(`Posterior rows start at frame ${p.startFrame}, expected ${this.endFrame} to continue the session`);
    }
    for (let i = 0; i < p.ids.length; i++) {
      if (!Number.isInteger(p.ids[i]) || p.ids[i] < 0 || p.ids[i] > this.size) {
        throw new EvidenceError(`Posterior token id ${p.ids[i]} is outside the inventory's 0..${this.size}`);
      }
    }
    if (this.start === -1) this.start = p.startFrame;
    this.k = p.k;
    for (let i = 0; i < p.ids.length; i++) {
      this.ids.push(p.ids[i]);
      this.probs.push(p.probs[i]);
    }
  }

  /** Forget rows before `frame`. */
  dropBefore(frame: number) {
    const drop = Math.min(this.ids.length / Math.max(1, this.k), frame - this.start);
    if (drop <= 0) return;
    this.ids.splice(0, drop * this.k);
    this.probs.splice(0, drop * this.k);
    this.start += drop;
    for (const f of this.cache.keys()) if (f < this.start) this.cache.delete(f);
  }

  /** Whether token `id` is among the listed top-k at `frame`. */
  lists(frame: number, id: number): boolean {
    const r = (frame - this.start) * this.k;
    for (let c = 0; c < this.k; c++) if (this.ids[r + c] === id) return true;
    return false;
  }

  /** Log-probabilities of blank and every token at `frame`; omitted mass is spread over the unlisted tokens. */
  row(frame: number): Float64Array {
    let row = this.cache.get(frame);
    if (row) return row;
    const r = (frame - this.start) * this.k;
    const p = new Float64Array(this.size + 1);
    let listed = 0;
    let unlisted = this.size + 1;
    for (let c = 0; c < this.k; c++) {
      const id = this.ids[r + c];
      if (p[id] === 0) unlisted--;
      p[id] += this.probs[r + c];
      listed += this.probs[r + c];
    }
    const rest = unlisted ? Math.max(0, 1 - listed) / unlisted : 0;
    let total = 0;
    for (let i = 0; i <= this.size; i++) {
      if (p[i] === 0) p[i] = rest;
      total += p[i];
    }
    row = new Float64Array(this.size + 1);
    for (let i = 0; i <= this.size; i++) row[i] = Math.log(Math.max(MIN_P, p[i] / total));
    this.cache.set(frame, row);
    return row;
  }

  /** CTC negative log-likelihood, in nats, of `labels` (token ids, no blanks) over frames `[from, to)`. */
  nll(labels: ArrayLike<number>, from: number, to: number): number {
    const L = labels.length;
    if (to <= from) return L ? Infinity : 0;
    const S = 2 * L + 1;
    const label = (s: number) => (s % 2 === 0 ? 0 : labels[(s - 1) >> 1]);
    let prev = new Float64Array(S).fill(-Infinity);
    let next = new Float64Array(S);
    const r0 = this.row(from);
    prev[0] = r0[0];
    if (L) prev[1] = r0[label(1)];
    for (let f = from + 1; f < to; f++) {
      const row = this.row(f);
      for (let s = 0; s < S; s++) {
        let a = prev[s];
        if (s > 0) a = logAdd(a, prev[s - 1]);
        if (s > 1 && s % 2 === 1 && label(s) !== label(s - 2)) a = logAdd(a, prev[s - 2]);
        next[s] = a + row[label(s)];
      }
      [prev, next] = [next, prev];
    }
    const end = L ? logAdd(prev[S - 1], prev[S - 2]) : prev[0];
    return -end;
  }

  /**
   * Extra nats, above a phone-loop filler, of the best path that utters `labels` somewhere in frames
   * `[from, to)`. The filler pays nothing: each frame keeps its best token. The keyword is a CTC
   * sequence entered at any frame and left at any later frame; every other frame stays on the filler.
   * Infinity when the sequence cannot be placed.
   */
  keywordExtra(labels: ArrayLike<number>, from: number, to: number): number {
    const L = labels.length;
    if (!L || to - from < L) return L ? Infinity : 0;
    const S = 2 * L + 1;
    const tokenAt = (s: number) => (s % 2 === 0 ? 0 : labels[(s - 1) >> 1]);
    let prev = new Float64Array(S).fill(Infinity);
    let next = new Float64Array(S);
    let best = Infinity;
    for (let f = from; f < to; f++) {
      const row = this.row(f);
      let top = row[0];
      for (let i = 1; i < row.length; i++) if (row[i] > top) top = row[i];
      const extra = (k: number) => Math.max(0, top - row[k]);
      next.fill(Infinity);
      // Filler so far, then enter the sequence on this frame.
      next[0] = extra(0);
      next[1] = extra(labels[0]);
      for (let s = 0; s < S; s++) {
        let src = prev[s];
        if (s > 0 && prev[s - 1] < src) src = prev[s - 1];
        if (s > 1 && s % 2 === 1 && tokenAt(s) !== tokenAt(s - 2) && prev[s - 2] < src) src = prev[s - 2];
        if (src === Infinity) continue;
        const cost = src + extra(tokenAt(s));
        if (cost < next[s]) next[s] = cost;
      }
      const done = next[S - 1] < next[S - 2] ? next[S - 1] : next[S - 2];
      if (done < best) best = done;
      [prev, next] = [next, prev];
    }
    return best;
  }
}

function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}
