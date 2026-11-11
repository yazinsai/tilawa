import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** A reference articulation on a local trajectory, including its original movement edge. */
export interface CoupledPhone {
  at: number;
  /** Extra price of entering this phone (a jump or surah entry), in weighted edit units. */
  entry: number;
  /** A jump enters by consuming an observation, rather than by deleting its first phone. */
  consumes: boolean;
  /** Observation consumed on the independent optimum; null means reference-only deletion. */
  observed: number | null;
  /** Observation frontier before a deletion, or before consuming `observed`. */
  frontier: number;
}

export interface CoupledPath {
  phones: readonly CoupledPhone[];
  /** Reference cursor at the common predecessor. */
  start: number;
  /** Independent cost after the common predecessor, in weighted edit units. */
  cost: number;
}

export interface CoupledCosts {
  left: number;
  right: number;
  correspondences: number;
  disputed: number;
}

const addLog = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

interface Binding {
  total: number;
  left: number;
  right: number;
  count: number;
}

/** Equal joint-cost bindings have equal normalized weight, independently of operand order. */
function offer(row: (Binding | undefined)[], at: number, from: Binding, left: number, right: number): void {
  if (!Number.isFinite(left + right)) return;
  const next = { total: from.total + (left + right), left: from.left - left / UNIT,
    right: from.right - right / UNIT, count: from.count };
  const old = row[at];
  if (!old || next.total < old.total) row[at] = next;
  else if (next.total === old.total) row[at] = { total: old.total, left: addLog(old.left, next.left),
    right: addLog(old.right, next.right), count: addLog(old.count, next.count) };
}

/**
 * Couple the common pronunciation of two trajectories over exactly the same observed interval. Maximum common
 * subsequences define the reference correspondences; each distinct correspondence gets equal probability. For
 * each one, a paired edit graph chooses common observation frontiers, while the intervening differing strings
 * retain independent edit paths. Shared phones are deleted together or consume the same observed token.
 *
 * Movement edges and contextual edit prices come from the original reference, including its duration runs. Only
 * common-phone binding is constrained. There is no fitted temperature: weighted edits convert through UNIT.
 */
export function coupledPhoneCosts(ref: string, observed: string, left: CoupledPath, right: CoupledPath): CoupledCosts {
  const a = left.phones, b = right.phones, n = observed.length;
  const lcs = Array.from({ length: a.length + 1 }, () => new Int32Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--)
    lcs[i][j] = ref[a[i].at] === ref[b[j].at] ? 1 + lcs[i + 1][j + 1] : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  if (!lcs[0][0]) return { left: left.cost, right: right.cost, correspondences: 0, disputed: 0 };

  const cursor = (path: CoupledPath, i: number) => i ? path.phones[i - 1].at + 1 : path.start;
  // Exact gap fits for every observed [from,to). A gap may insert at its right boundary, as in the local DP.
  const cache = new Map<string, Float64Array[]>();
  const gaps = (path: CoupledPath, begin: number, end: number): Float64Array[] => {
    const key = `${path === left ? "l" : "r"}:${begin}:${end}`;
    const have = cache.get(key);
    if (have) return have;
    const rows: Float64Array[] = [];
    for (let from = 0; from <= n; from++) {
      let prev = new Float64Array(end - begin + 1).fill(Infinity);
      prev[0] = 0;
      for (let i = begin; i < end; i++) {
        const p = path.phones[i];
        if (!p.consumes) prev[i - begin + 1] = prev[i - begin] + deletionCost(ref, p.at) + p.entry;
      }
      const costs = new Float64Array(n + 1).fill(Infinity);
      costs[from] = prev[end - begin];
      for (let t = from; t < n; t++) {
        const row = new Float64Array(prev.length).fill(Infinity);
        for (let i = begin; i <= end; i++) {
          const k = i - begin;
          row[k] = prev[k] + insertionCost(ref, cursor(path, i), observed[t]);
          if (i > begin) {
            const p = path.phones[i - 1];
            row[k] = Math.min(row[k], prev[k - 1] + subCost(ref[p.at], observed[t]) + p.entry);
            if (!p.consumes) row[k] = Math.min(row[k], row[k - 1] + deletionCost(ref, p.at) + p.entry);
          }
        }
        prev = row;
        costs[t + 1] = row[end - begin];
      }
      rows.push(costs);
    }
    cache.set(key, rows);
    return rows;
  };

  let mappings = 0, disputed = 0, sumLeft = -Infinity, sumRight = -Infinity;
  const score = (pairs: readonly (readonly [number, number])[]) => {
    mappings++;
    const identical = pairs.every(([i, j]) => a[i].observed === b[j].observed && a[i].frontier === b[j].frontier);
    if (identical) {
      sumLeft = addLog(sumLeft, -left.cost / UNIT);
      sumRight = addLog(sumRight, -right.cost / UNIT);
      return;
    }
    disputed++;
    let row: (Binding | undefined)[] = Array(n + 1);
    row[0] = { total: 0, left: 0, right: 0, count: 0 };
    let ai = 0, bi = 0;
    for (const [i, j] of pairs) {
      const ga = gaps(left, ai, i), gb = gaps(right, bi, j);
      const next: (Binding | undefined)[] = Array(n + 1);
      for (let from = 0; from <= n; from++) {
        const state = row[from];
        if (!state) continue;
        for (let to = from; to <= n; to++) {
          const ca = ga[from][to], cb = gb[from][to];
          if (!a[i].consumes && !b[j].consumes)
            offer(next, to, state, ca + deletionCost(ref, a[i].at) + a[i].entry,
              cb + deletionCost(ref, b[j].at) + b[j].entry);
          if (to < n) offer(next, to + 1, state, ca + subCost(ref[a[i].at], observed[to]) + a[i].entry,
            cb + subCost(ref[b[j].at], observed[to]) + b[j].entry);
        }
      }
      row = next;
      ai = i + 1; bi = j + 1;
    }
    const ga = gaps(left, ai, a.length), gb = gaps(right, bi, b.length);
    const final: (Binding | undefined)[] = [];
    for (let from = 0; from <= n; from++) if (row[from]) offer(final, 0, row[from]!, ga[from][n], gb[from][n]);
    const result = final[0];
    if (result) {
      sumLeft = addLog(sumLeft, result.left - result.count);
      sumRight = addLog(sumRight, result.right - result.count);
    }
  };
  // Enumerate index-pair sequences, not LCS deletion routes: otherwise the same correspondence gets counted twice.
  const visit = (i: number, j: number, remaining: number, pairs: [number, number][]) => {
    if (!remaining) { score(pairs); return; }
    for (let x = i; x < a.length; x++) for (let y = j; y < b.length; y++)
      if (ref[a[x].at] === ref[b[y].at] && lcs[x + 1][y + 1] === remaining - 1) {
        pairs.push([x, y]); visit(x + 1, y + 1, remaining - 1, pairs); pairs.pop();
      }
  };
  visit(0, 0, lcs[0][0], []);
  // All identical bindings are an exact no-op, without round-off from the normalized mixture.
  if (!disputed) return { left: left.cost, right: right.cost, correspondences: mappings, disputed };
  return { left: -UNIT * (sumLeft - Math.log(mappings)), right: -UNIT * (sumRight - Math.log(mappings)),
    correspondences: mappings, disputed };
}
