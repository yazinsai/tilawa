import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

/** One row partitions probability between an ordinary articulation and 2-to-1 recognition outcomes. */
export interface CoalescenceModel {
  phones: string[];
  /** Rows: [ordinary, ...P(two reference articulations -> heard phone)]. */
  pairs: Record<string, number[]>;
  classes: Record<string, number[]>;
  pooled: number[];
  mass: number;
}

export interface CoalescenceRecording {
  written: string;
  heard: string;
  /** Word starts, including zero. Merges never swallow a word boundary. */
  starts: number[];
}

const NATS_PER_EDIT = 3;
const INSERT = 0.1;
const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};
const normalize = (r: number[]) => {
  const sum = r.reduce((a, b) => a + b, 0);
  return r.map(x => x / sum);
};

/** Manner classes, fixed by the phone inventory, independent of reader and text location. */
export function articulationClass(p: string): string {
  const groups = ["َُِاۦۥ۪ٲ", "ءبتدطضقك", "ثحخذزسشصظغفهڇ", "ج", "منں۾", "رل", "وي"];
  return String(groups.findIndex(g => g.includes(p)));
}
const classKey = (a: string, b: string) => `${articulationClass(a)}/${articulationClass(b)}`;

/** No identical sounds, duration-expanded runs, or word boundaries qualify as two articulations. */
export function coalescenceEligible(ref: string, at: number, starts: ReadonlySet<number>): boolean {
  return at >= 0 && at + 1 < ref.length && !starts.has(at + 1) && ref[at] !== ref[at + 1]
    && ref[at - 1] !== ref[at] && ref[at + 1] !== ref[at + 2];
}

function rowFor(model: CoalescenceModel, a: string, b: string): number[] {
  return model.pairs[a + b] ?? model.classes[classKey(a, b)] ?? model.pooled;
}

/** The ordinary edit emission distribution used for latent training alignments. */
function ordinary(ref: string, at: number, phones: readonly string[]): number[] {
  return normalize([...phones.map(p => Math.exp(-NATS_PER_EDIT * subCost(ref[at], p) / UNIT)),
    Math.exp(-NATS_PER_EDIT * deletionCost(ref, at) / UNIT)]);
}

export interface CoalescenceCosts {
  /** Probability allocated to the ordinary branch. Added only when consuming a reference phone. */
  ordinary(at: number): number;
  /** Two distinct reference articulations and one observed token, as one recognition event. */
  merged(at: number, heard: string): number;
}

/**
 * Costs are likelihood ratios against exact ordinary recognition of the consumed articulations, in existing edit
 * units (3 nats/edit). Branches partition probability; an added route does not get free likelihood counting weight.
 * Insertions, movement prices, duration edits and reference ownership retain their existing definitions.
 */
export function coalescenceCosts(model: CoalescenceModel, ref: string, starts: ReadonlySet<number>): CoalescenceCosts {
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const rows = new Map<number, number[]>();
  const baselines = new Map<number, number>();
  const get = (at: number) => {
    if (!coalescenceEligible(ref, at, starts) || !ids.has(ref[at]) || !ids.has(ref[at + 1])) return undefined;
    let row = rows.get(at);
    if (!row) rows.set(at, row = rowFor(model, ref[at], ref[at + 1]));
    return row;
  };
  return {
    ordinary(at) {
      const row = get(at);
      return row ? -Math.log(row[0]) * UNIT / NATS_PER_EDIT : 0;
    },
    merged(at, heard) {
      const row = get(at), h = ids.get(heard);
      if (!row || h === undefined || !row[h + 1]) return Infinity;
      let baseline = baselines.get(at);
      if (baseline === undefined) {
        baseline = (1 - INSERT) * ordinary(ref, at, model.phones)[ids.get(ref[at])!]
          * ordinary(ref, at + 1, model.phones)[ids.get(ref[at + 1])!];
        baselines.set(at, baseline);
      }
      return -Math.log(row[h + 1] / baseline) * UNIT / NATS_PER_EDIT;
    },
  };
}

type Counts = Map<string, number[]>;
const addCount = (counts: Counts, key: string, k: number, weight: number, n: number) => {
  let row = counts.get(key);
  if (!row) counts.set(key, row = Array(n + 1).fill(0));
  row[k] += weight;
};

/** Forward/backward over complete recordings. All token/edit boundaries, including coalescence, are latent. */
function expectation(model: CoalescenceModel, record: CoalescenceRecording, counts?: Counts): number {
  const { written: ref, heard } = record;
  const n = ref.length, m = heard.length, width = m + 1, size = (n + 1) * width;
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const starts = new Set(record.starts);
  const eligible = Array.from({ length: n }, (_, i) => coalescenceEligible(ref, i, starts));
  const rows = Array.from({ length: n }, (_, i) => eligible[i] ? rowFor(model, ref[i], ref[i + 1]) : undefined);
  const emissions = Array.from({ length: n }, (_, i) => ordinary(ref, i, model.phones));
  const ins = Math.log(INSERT / model.phones.length);
  const sub = (i: number, j: number) => Math.log((1 - INSERT) * (rows[i]?.[0] ?? 1) * emissions[i][ids.get(heard[j])!]);
  const del = (i: number) => Math.log((1 - INSERT) * (rows[i]?.[0] ?? 1) * emissions[i][model.phones.length]);
  const coal = (i: number, j: number) => Math.log((1 - INSERT) * rows[i]![ids.get(heard[j])! + 1]);
  const at = (i: number, j: number) => i * width + j;
  const f = new Float64Array(size).fill(-Infinity), b = new Float64Array(size).fill(-Infinity);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const cur = f[at(i, j)];
    if (j < m) f[at(i, j + 1)] = logAdd(f[at(i, j + 1)], cur + ins);
    if (i < n) {
      f[at(i + 1, j)] = logAdd(f[at(i + 1, j)], cur + del(i));
      if (j < m) {
        f[at(i + 1, j + 1)] = logAdd(f[at(i + 1, j + 1)], cur + sub(i, j));
        if (eligible[i]) f[at(i + 2, j + 1)] = logAdd(f[at(i + 2, j + 1)], cur + coal(i, j));
      }
    }
  }
  // End of the reference: insertion or stop are the remaining normalized alternatives.
  const logP = f[at(n, m)] + Math.log(1 - INSERT);
  if (!counts) return logP;
  b[at(n, m)] = Math.log(1 - INSERT);
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (j < m) b[at(i, j)] = logAdd(b[at(i, j)], ins + b[at(i, j + 1)]);
    if (i < n) {
      b[at(i, j)] = logAdd(b[at(i, j)], del(i) + b[at(i + 1, j)]);
      if (j < m) {
        b[at(i, j)] = logAdd(b[at(i, j)], sub(i, j) + b[at(i + 1, j + 1)]);
        if (eligible[i]) b[at(i, j)] = logAdd(b[at(i, j)], coal(i, j) + b[at(i + 2, j + 1)]);
      }
      if (eligible[i]) {
        let mass = Math.exp(f[at(i, j)] + del(i) + b[at(i + 1, j)] - logP);
        if (j < m) mass += Math.exp(f[at(i, j)] + sub(i, j) + b[at(i + 1, j + 1)] - logP);
        addCount(counts, ref.slice(i, i + 2), 0, mass, model.phones.length);
        if (j < m) addCount(counts, ref.slice(i, i + 2), ids.get(heard[j])! + 1,
          Math.exp(f[at(i, j)] + coal(i, j) + b[at(i + 2, j + 1)] - logP), model.phones.length);
      }
    }
  }
  return logP;
}

export function coalescenceLogP(model: CoalescenceModel, record: CoalescenceRecording): number {
  return expectation(model, record);
}

/** Hierarchical EM: exact articulation pairs -> manner-class pairs -> pooled recognition events. */
export function fitCoalescence(records: readonly CoalescenceRecording[], phones: readonly string[], mass: number, enabled = true): CoalescenceModel {
  if (!records.length || !phones.length || !(mass > 0)) throw new Error("Invalid coalescence training data");
  const inventory = new Set(phones);
  for (const r of records) {
    if (!r.written || !r.heard || [...r.written + r.heard].some(p => !inventory.has(p))) throw new Error("Unknown or empty training phones");
  }
  const prior = [0.95, ...phones.map(() => 0.05 / phones.length)];
  let model: CoalescenceModel = { phones: [...phones], pairs: {}, classes: {}, pooled: enabled ? prior : [1, ...phones.map(() => 0)], mass };
  if (!enabled) return model;
  for (let iteration = 0; iteration < 30; iteration++) {
    const counts: Counts = new Map();
    for (const r of records) expectation(model, r, counts);
    const pooled = prior.map(x => 2 * x);
    const classes: Counts = new Map();
    for (const [key, row] of counts) for (let k = 0; k < row.length; k++) {
      pooled[k] += row[k];
      addCount(classes, classKey(key[0], key[1]), k, row[k], phones.length);
    }
    const next: CoalescenceModel = { phones: [...phones], mass, pooled: normalize(pooled), classes: {}, pairs: {} };
    for (const [key, row] of classes) next.classes[key] = normalize(row.map((x, k) => x + mass * next.pooled[k]));
    let change = 0;
    for (const [key, row] of counts) {
      const parent = next.classes[classKey(key[0], key[1])];
      const fitted = normalize(row.map((x, k) => x + mass * parent[k]));
      const old = rowFor(model, key[0], key[1]);
      change = Math.max(change, ...fitted.map((x, k) => Math.abs(x - old[k])));
      next.pairs[key] = fitted;
    }
    model = next;
    if (change < 1e-6) break;
  }
  return model;
}
