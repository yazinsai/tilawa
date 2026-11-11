import { PosteriorFrames } from "@tilawa/core";
import type { Channel } from "@tilawa/core";

const add = (a: number, b: number): number => a === -Infinity ? b : b === -Infinity ? a :
  Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));

/** A proper distribution over emitted strings, including the empty string.
 * The existing insertion rates are expected counts per input boundary. A geometric
 * insertion loop preserves that mean; leaving the loop divides the emission/deletion
 * mass rather than giving the extra paths free weight. Every input phone is consumed.
 */
export class ComposedChannel {
  private readonly exit: number;
  private readonly ins: Float64Array;
  private readonly del: Float64Array;
  private readonly sub: Float64Array[];
  private readonly size: number;
  private readonly index: Map<string, number>;

  constructor(channel: Channel, readonly tokens: readonly string[]) {
    this.size = tokens.length;
    this.index = new Map(tokens.map((p, i) => [p, i]));
    if (this.index.size !== this.size || tokens.some(p => !channel.index.has(p)) || channel.phones.length !== this.size)
      throw new Error("Composed channel and CTC inventories differ");
    const order = tokens.map(p => channel.index.get(p)!);
    const rate = order.reduce((s, i) => s + Math.exp(channel.logIns[i]), 0);
    this.exit = -Math.log1p(rate);
    this.ins = Float64Array.from(order, i => channel.logIns[i] + this.exit);
    this.del = Float64Array.from(order, i => channel.logDel[i] + this.exit);
    this.sub = order.map(w => Float64Array.from(order, h => channel.logSub[w][h] + this.exit));
  }

  private labels(written: string): number[] | null {
    const ids = [...written].map(p => this.index.get(p) ?? -1);
    return ids.some(i => i < 0) ? null : ids;
  }

  private finish(totals: Float64Array, labels: number[]): number {
    let tail = this.exit, score = -Infinity;
    for (let j = labels.length; j >= 0; j--) {
      score = add(score, totals[j] + tail);
      if (j) tail += this.del[labels[j - 1]];
    }
    return score;
  }

  /** Sum over all edit scripts of this fixed emitted string, using the SAME channel
   * as the composed scorer. No Viterbi approximation or independent re-normalization.
   */
  stringLogP(written: string, emitted: string): number | null {
    const labels = this.labels(written), heard = this.labels(emitted);
    if (!labels || !heard) return null;
    let prev = new Float64Array(labels.length + 1).fill(-Infinity);
    prev[0] = 0;
    for (const h of heard) {
      const next = new Float64Array(prev.length).fill(-Infinity);
      let closure = -Infinity;
      for (let j = 0; j < prev.length; j++) {
        const before = closure;
        closure = add(prev[j], j ? closure + this.del[labels[j - 1]] : -Infinity);
        next[j] = add(closure + this.ins[h], j ? before + this.sub[labels[j - 1]][h] : -Infinity);
      }
      prev = next;
    }
    return this.finish(prev, labels);
  }

  /** log sum_y P_channel(y | written) P_CTC(y | rows).
   * State = input position and previous RAW CTC class. Blanks and continued spikes
   * do not trigger another channel edit. A repeated emitted label requires a blank.
   * Deletion epsilon closure is applied only on a new emission (or acceptance), so
   * paths are not recounted at each blank frame. O(frames * input length * inventory).
   */
  acousticLogP(written: string, frames: PosteriorFrames, from: number, to: number): number | null {
    const labels = this.labels(written);
    if (!labels || !Number.isInteger(from) || !Number.isInteger(to) || from < frames.startFrame || to > frames.endFrame || to < from)
      return null;
    const v = this.size + 1, positions = labels.length + 1;
    let prev = new Float64Array(positions * v).fill(-Infinity);
    let next = new Float64Array(prev.length);
    const totals = new Float64Array(positions);
    const excluded = new Float64Array(prev.length);
    prev[0] = 0;
    for (let f = from; f < to; f++) {
      const row = frames.row(f);
      if (row.length !== v) return null;
      // Prefix/suffix sums exclude one class without subtracting nearly equal mass.
      for (let j = 0; j < positions; j++) {
        const base = j * v;
        let prefix = -Infinity;
        for (let h = 0; h < v; h++) {
          excluded[base + h] = prefix;
          prefix = add(prefix, prev[base + h]);
        }
        totals[j] = prefix;
        let suffix = -Infinity;
        for (let h = v - 1; h >= 0; h--) {
          excluded[base + h] = add(excluded[base + h], suffix);
          suffix = add(suffix, prev[base + h]);
        }
        next[base] = totals[j] + row[0];
      }
      for (let h = 1; h < v; h++) {
        let closure = -Infinity;
        for (let j = 0; j < positions; j++) {
          const at = j * v + h, before = closure;
          closure = add(excluded[at], j ? closure + this.del[labels[j - 1]] : -Infinity);
          const fresh = add(closure + this.ins[h - 1], j ? before + this.sub[labels[j - 1]][h - 1] : -Infinity);
          next[at] = add(prev[at], fresh) + row[h];
        }
      }
      [prev, next] = [next, prev];
    }
    for (let j = 0; j < positions; j++) {
      let total = -Infinity;
      for (let h = 0; h < v; h++) total = add(total, prev[j * v + h]);
      totals[j] = total;
    }
    return this.finish(totals, labels);
  }

  greedy(frames: PosteriorFrames, from: number, to: number): string {
    let prev = 0, emitted = "";
    for (let f = from; f < to; f++) {
      const row = frames.row(f);
      let best = 0;
      for (let h = 1; h < row.length; h++) if (row[h] > row[best]) best = h;
      if (best && best !== prev) emitted += this.tokens[best - 1];
      prev = best;
    }
    return emitted;
  }
}
