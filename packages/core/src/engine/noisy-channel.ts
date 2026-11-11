import { collapseHeld, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/**
 * P(heard token | written phone) of the shipped listener, counted by {@link accumulateChannel} on
 * gold-excluded decodes under the shipped augmentation. Additive smoothing is the estimator, not a
 * fitted gate. A deletion is the written phone producing nothing. An insertion is a heard token
 * with no written phone.
 */
export interface ChannelCounts {
  phones: string[];
  /** Substitution counts, row = written phone, column = heard token. */
  sub: number[][];
  /** Deletion counts per written phone. */
  del: number[];
  /** Insertion counts per heard token. */
  ins: number[];
  /** Written phones that were aligned (substitutions + deletions). */
  nWritten: number;
  /**
   * Highest log-odds by which a non-neighbour word beat the best competitor on a clean
   * gold-excluded seam, under this span score. Unused by omission placement.
   */
  cut: number;
}

/** Laplace mass on each unseen edit. Part of the estimator, not a placement margin. */
const ALPHA = 0.5;

export interface Channel {
  phones: string[];
  index: Map<string, number>;
  /** log P(heard | written), including the match. Rows sum with `del` over the simplex. */
  logSub: Float64Array[];
  logDel: Float64Array;
  /** log probability of inserting this heard token, scaled per written phone. */
  logIns: Float64Array;
  cut: number;
}

export function channelFromCounts(c: ChannelCounts): Channel {
  const n = c.phones.length;
  const index = new Map(c.phones.map((p, i) => [p, i]));
  const logSub: Float64Array[] = [];
  const logDel = new Float64Array(n);
  for (let w = 0; w < n; w++) {
    let total = c.del[w] ?? 0;
    const rowCounts = c.sub[w] ?? [];
    for (let h = 0; h < n; h++) total += rowCounts[h] ?? 0;
    const z = total + ALPHA * (n + 1);
    const row = new Float64Array(n);
    for (let h = 0; h < n; h++) row[h] = Math.log(((rowCounts[h] ?? 0) + ALPHA) / z);
    logDel[w] = Math.log(((c.del[w] ?? 0) + ALPHA) / z);
    logSub.push(row);
  }
  const logIns = new Float64Array(n);
  const zIns = Math.max(1, c.nWritten) + ALPHA * n;
  for (let h = 0; h < n; h++) logIns[h] = Math.log(((c.ins[h] ?? 0) + ALPHA) / zIns);
  return { phones: c.phones, index, logSub, logDel, logIns, cut: c.cut };
}

const NEG = -1e9;

/** Viterbi log P(heard | written) under the channel. */
export function spanLogP(heard: readonly string[], written: readonly string[], ch: Channel): number {
  const n = heard.length;
  const m = written.length;
  const hi = heard.map((p) => ch.index.get(p) ?? -1);
  const wi = written.map((p) => ch.index.get(p) ?? -1);
  const w = m + 1;
  const dp = new Float64Array((n + 1) * w).fill(NEG);
  const at = (i: number, j: number) => i * w + j;
  dp[0] = 0;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= m; j++) {
      const cur = dp[at(i, j)];
      if (cur <= NEG / 2) continue;
      if (i < n) {
        const h = hi[i];
        const add = h < 0 ? Math.log(ALPHA / (ALPHA * ch.phones.length)) : ch.logIns[h];
        const to = at(i + 1, j);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
      if (j < m) {
        const wr = wi[j];
        const add = wr < 0 ? Math.log(ALPHA / (ALPHA * (ch.phones.length + 1))) : ch.logDel[wr];
        const to = at(i, j + 1);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
      if (i < n && j < m) {
        const h = hi[i];
        const wr = wi[j];
        const add = h < 0 || wr < 0 ? Math.log(ALPHA / (ALPHA * (ch.phones.length + 1))) : ch.logSub[wr][h];
        const to = at(i + 1, j + 1);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
    }
  }
  return dp[at(n, m)];
}

/**
 * Count listener confusions along one minimum-cost alignment of a gold-excluded decode to its
 * canonical phones. Costs are the checker's existing edit costs, so the path is not fit here.
 */
export function accumulateChannel(counts: ChannelCounts, writtenRaw: string, heardRaw: string): void {
  const index = new Map(counts.phones.map((p, i) => [p, i]));
  const written = collapseHeld([...writtenRaw]);
  const heard = collapseHeld([...heardRaw]);
  const n = written.length;
  const m = heard.length;
  const w = m + 1;
  const dp = new Float64Array((n + 1) * w);
  const bt = new Uint8Array((n + 1) * w);
  const at = (i: number, j: number) => i * w + j;
  for (let j = 1; j <= m; j++) {
    dp[j] = dp[j - 1] + insertionCost("", 0, heard[j - 1]);
    bt[j] = 1;
  }
  for (let i = 1; i <= n; i++) {
    dp[at(i, 0)] = dp[at(i - 1, 0)] + deletionCost(written.join(""), i - 1);
    bt[at(i, 0)] = 2;
    for (let j = 1; j <= m; j++) {
      const sub = dp[at(i - 1, j - 1)] + subCost(written[i - 1], heard[j - 1]);
      const ins = dp[at(i, j - 1)] + insertionCost(written.join(""), i, heard[j - 1]);
      const del = dp[at(i - 1, j)] + deletionCost(written.join(""), i - 1);
      let best = sub;
      let how = 0;
      if (ins < best) { best = ins; how = 1; }
      if (del < best) { best = del; how = 2; }
      dp[at(i, j)] = best;
      bt[at(i, j)] = how;
    }
  }
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    const how = bt[at(i, j)];
    if (how === 0 && i > 0 && j > 0) {
      const wr = index.get(written[i - 1]);
      const h = index.get(heard[j - 1]);
      if (wr !== undefined && h !== undefined) counts.sub[wr][h] += 1;
      if (wr !== undefined) counts.nWritten += 1;
      i--;
      j--;
    } else if (how === 1 && j > 0) {
      const h = index.get(heard[j - 1]);
      if (h !== undefined) counts.ins[h] += 1;
      j--;
    } else if (i > 0) {
      const wr = index.get(written[i - 1]);
      if (wr !== undefined) { counts.del[wr] += 1; counts.nWritten += 1; }
      i--;
    } else break;
  }
}

export function emptyCounts(phones: readonly string[], cut = 0): ChannelCounts {
  const n = phones.length;
  return {
    phones: [...phones],
    sub: Array.from({ length: n }, () => Array(n).fill(0)),
    del: Array(n).fill(0),
    ins: Array(n).fill(0),
    nWritten: 0,
    cut,
  };
}
