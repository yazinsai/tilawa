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

  /**
   * CTC negative log-likelihood, in nats, of `labels` (token ids, no blanks) over frames `[from, to)`.
   * `repeats` are inclusive label spans that may be said again: from the end of the span, the forward
   * sum may re-enter its first label. The same arcs the alignment already uses to repeat a contiguous
   * word or ayah. No filler loop. Omitted, the sum is the plain chain.
   * `cut`, when present, removes those frames from both the emissions and the
   * recurrence; it does not replace them with blank or modify stored evidence.
   */
  nll(labels: ArrayLike<number>, from: number, to: number, repeats?: ReadonlyArray<readonly [number, number]>, cut?: readonly [number, number]): number {
    const L = labels.length;
    // Excise frames, rather than forcing them to blank. The state distribution carries directly
    // across the cut, exactly as it would on the concatenated remaining rows.
    if (cut && from >= cut[0] && from < cut[1]) from = cut[1];
    if (to <= from) return L ? Infinity : 0;
    const S = 2 * L + 1;
    const label = (s: number) => (s % 2 === 0 ? 0 : labels[(s - 1) >> 1]);
    const back = repeatBack(repeats, L);
    let prev = new Float64Array(S).fill(-Infinity);
    let next = new Float64Array(S);
    const r0 = this.row(from);
    prev[0] = r0[0];
    if (L) prev[1] = r0[label(1)];
    for (let f = from + 1; f < to; f++) {
      if (cut && f >= cut[0] && f < cut[1]) continue;
      const row = this.row(f);
      for (let s = 0; s < S; s++) {
        let a = prev[s];
        if (s > 0) a = logAdd(a, prev[s - 1]);
        if (s > 1 && s % 2 === 1 && label(s) !== label(s - 2)) a = logAdd(a, prev[s - 2]);
        if (back && s % 2 === 1) {
          const extra = back[(s - 1) >> 1];
          for (let k = 0; k < extra.length; k++) a = logAdd(a, prev[extra[k]]);
        }
        next[s] = a + row[label(s)];
      }
      [prev, next] = [next, prev];
    }
    const end = L ? logAdd(prev[S - 1], prev[S - 2]) : prev[0];
    return -end;
  }
}

/** States that may re-enter each label, for contiguous repeat arcs. Null when there are none. */
function repeatBack(repeats: ReadonlyArray<readonly [number, number]> | undefined, L: number): Int32Array[] | null {
  if (!repeats?.length || !L) return null;
  const S = 2 * L + 1;
  const lists: number[][] = Array.from({ length: L }, () => []);
  const seen: Set<number>[] = Array.from({ length: L }, () => new Set());
  let any = false;
  for (const [start, end] of repeats) {
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= L) continue;
    const add = (state: number) => {
      if (state < 0 || state >= S || state === 2 * start + 1 || seen[start].has(state)) return;
      seen[start].add(state);
      lists[start].push(state);
      any = true;
    };
    add(2 * end + 1);
    add(2 * end + 2);
  }
  return any ? lists.map((xs) => Int32Array.from(xs)) : null;
}

function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}
