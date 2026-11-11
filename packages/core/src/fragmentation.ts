import { deletionCost, subCost, UNIT } from "./fuzzy.js";

const NATS_PER_EDIT = 3;
const INSERT = 0.1;
const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

/** Excess sequence probabilities plus a uniform residual over all heterogeneous sequences. */
export interface FragmentationRow {
  ordinary: number;
  sequences: Record<string, number>;
  residual: number;
}
export interface FragmentationModel {
  phones: string[];
  phonesRows: Record<string, FragmentationRow>;
  classes: Record<string, FragmentationRow>;
  pooled: FragmentationRow;
  mass: number;
}
export interface FragmentationRecording {
  written: string;
  heard: string;
  /** Reference word starts; an identical sound across a boundary has another owner. */
  starts: number[];
}

/** Fixed manner classes, independent of the training or practice locations. */
export function fragmentationClass(p: string): string {
  return String(["َُِاۦۥ۪ٲ", "ءبتدطضقك", "ثحخذزسشصظغفهڇ", "ج", "منں۾", "رل", "وي"].findIndex(g => g.includes(p)));
}
export function heterogeneous(sequence: string): boolean {
  return (sequence.length === 2 || sequence.length === 3) && [...sequence].some(p => p !== sequence[0]);
}
/** Duration expansions remain separate chargeable reference symbols after the articulation's first symbol. */
export function fragmentationEligible(ref: string, at: number, starts: ReadonlySet<number>): boolean {
  return at >= 0 && at < ref.length && (at === 0 || starts.has(at) || ref[at] !== ref[at - 1]);
}
export const fragmentationRow = (model: FragmentationModel, p: string): FragmentationRow =>
  model.phonesRows[p] ?? model.classes[fragmentationClass(p)] ?? model.pooled;

/** Half the residual mass goes to each length. This defines full support without held-partition vocabulary. */
export function fragmentationProbability(model: FragmentationModel, p: string, sequence: string): number {
  if (!heterogeneous(sequence) || [...sequence].some(x => !model.phones.includes(x))) return 0;
  const row = fragmentationRow(model, p), k = model.phones.length;
  return (row.sequences[sequence] ?? 0) + row.residual / 2 / (k ** sequence.length - k);
}
function emissions(ref: string, at: number, phones: readonly string[]): number[] {
  const raw = [...phones.map(p => Math.exp(-NATS_PER_EDIT * subCost(ref[at], p) / UNIT)),
    Math.exp(-NATS_PER_EDIT * deletionCost(ref, at) / UNIT)];
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map(x => x / sum);
}

/** Normalized event likelihood ratios, in the existing edit units. Ordinary edits keep their identity prices. */
export function fragmentationCosts(model: FragmentationModel, ref: string, starts: ReadonlySet<number>) {
  const inventory = new Set(model.phones);
  const eligible = (at: number) => fragmentationEligible(ref, at, starts) && inventory.has(ref[at]);
  const baseline = new Map<number, number>();
  return {
    ordinary(at: number): number {
      return eligible(at) ? -Math.log(fragmentationRow(model, ref[at]).ordinary) * UNIT / NATS_PER_EDIT : 0;
    },
    fragmented(at: number, sequence: string): number {
      if (!eligible(at)) return Infinity;
      const prob = fragmentationProbability(model, ref[at], sequence);
      if (!prob) return Infinity;
      let exact = baseline.get(at);
      if (exact === undefined) {
        exact = emissions(ref, at, model.phones)[model.phones.indexOf(ref[at])];
        baseline.set(at, exact);
      }
      return -Math.log(prob / exact) * UNIT / NATS_PER_EDIT;
    },
  };
}

interface Counts { ordinary: number; sequences: Record<string, number> }
const emptyCounts = (): Counts => ({ ordinary: 0, sequences: {} });
const plus = (to: Counts, from: Counts) => {
  to.ordinary += from.ordinary;
  for (const [s, v] of Object.entries(from.sequences)) to.sequences[s] = (to.sequences[s] ?? 0) + v;
};
function shrink(counts: Counts, parent: FragmentationRow, mass: number): FragmentationRow {
  const total = counts.ordinary + Object.values(counts.sequences).reduce((a, b) => a + b, 0) + mass;
  const sequences: Record<string, number> = {};
  for (const s of new Set([...Object.keys(parent.sequences), ...Object.keys(counts.sequences)])) {
    sequences[s] = ((counts.sequences[s] ?? 0) + mass * (parent.sequences[s] ?? 0)) / total;
  }
  return { ordinary: (counts.ordinary + mass * parent.ordinary) / total, sequences, residual: mass * parent.residual / total };
}

/** Complete-string forward/backward; phone boundaries and fragment ownership are latent. */
function expectation(model: FragmentationModel, record: FragmentationRecording, counts?: Map<string, Counts>): number {
  const { written: ref, heard } = record, n = ref.length, m = heard.length, width = m + 1;
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const starts = new Set(record.starts);
  const eligible = Array.from({ length: n }, (_, i) => fragmentationEligible(ref, i, starts));
  const q = Array.from({ length: n }, (_, i) => emissions(ref, i, model.phones));
  const row = Array.from({ length: n }, (_, i) => eligible[i] ? fragmentationRow(model, ref[i]).ordinary : 1);
  const at = (i: number, j: number) => i * width + j;
  const ins = Math.log(INSERT / model.phones.length);
  const del = (i: number) => Math.log((1 - INSERT) * row[i] * q[i][model.phones.length]);
  const sub = (i: number, j: number) => Math.log((1 - INSERT) * row[i] * q[i][ids.get(heard[j])!]);
  const fragments = Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) =>
    [2, 3].map(k => eligible[i] && j + k <= m
      ? Math.log((1 - INSERT) * fragmentationProbability(model, ref[i], heard.slice(j, j + k))) : -Infinity)));
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const cur = f[at(i, j)];
    if (j < m) f[at(i, j + 1)] = logAdd(f[at(i, j + 1)], cur + ins);
    if (i < n) {
      f[at(i + 1, j)] = logAdd(f[at(i + 1, j)], cur + del(i));
      if (j < m) {
        f[at(i + 1, j + 1)] = logAdd(f[at(i + 1, j + 1)], cur + sub(i, j));
        for (const k of [2, 3]) if (j + k <= m) f[at(i + 1, j + k)] = logAdd(f[at(i + 1, j + k)], cur + fragments[i][j][k - 2]);
      }
    }
  }
  const logP = f[at(n, m)] + Math.log(1 - INSERT);
  if (!counts) return logP;
  const b = new Float64Array(f.length).fill(-Infinity);
  b[at(n, m)] = Math.log(1 - INSERT);
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (j < m) b[at(i, j)] = logAdd(b[at(i, j)], ins + b[at(i, j + 1)]);
    if (i < n) {
      b[at(i, j)] = logAdd(b[at(i, j)], del(i) + b[at(i + 1, j)]);
      if (j < m) {
        b[at(i, j)] = logAdd(b[at(i, j)], sub(i, j) + b[at(i + 1, j + 1)]);
        for (const k of [2, 3]) if (j + k <= m) b[at(i, j)] = logAdd(b[at(i, j)], fragments[i][j][k - 2] + b[at(i + 1, j + k)]);
      }
      if (eligible[i]) {
        let c = counts.get(ref[i]);
        if (!c) counts.set(ref[i], c = emptyCounts());
        c.ordinary += Math.exp(f[at(i, j)] + del(i) + b[at(i + 1, j)] - logP);
        if (j < m) {
          c.ordinary += Math.exp(f[at(i, j)] + sub(i, j) + b[at(i + 1, j + 1)] - logP);
          for (const k of [2, 3]) if (j + k <= m && fragments[i][j][k - 2] !== -Infinity) {
            const s = heard.slice(j, j + k);
            c.sequences[s] = (c.sequences[s] ?? 0) + Math.exp(f[at(i, j)] + fragments[i][j][k - 2] + b[at(i + 1, j + k)] - logP);
          }
        }
      }
    }
  }
  return logP;
}
export const fragmentationLogP = (model: FragmentationModel, record: FragmentationRecording): number => expectation(model, record);

/** Hierarchical EM: articulation identity -> fixed manner class -> pooled ordered fragments. */
export function fitFragmentation(records: readonly FragmentationRecording[], phones: readonly string[], mass: number, enabled = true): FragmentationModel {
  if (!records.length || phones.length < 2 || new Set(phones).size !== phones.length || !(mass > 0)) throw new Error("Invalid fragmentation training data");
  const inventory = new Set(phones);
  for (const r of records) if (!r.written || !r.heard || [...r.written + r.heard].some(p => !inventory.has(p))) throw new Error("Unknown or empty training phones");
  const prior: FragmentationRow = { ordinary: 0.95, sequences: {}, residual: 0.05 };
  let model: FragmentationModel = { phones: [...phones], mass, phonesRows: {}, classes: {}, pooled: enabled ? prior : { ordinary: 1, sequences: {}, residual: 0 } };
  if (!enabled) return model;
  for (let iteration = 0; iteration < 30; iteration++) {
    const counts = new Map<string, Counts>();
    for (const r of records) expectation(model, r, counts);
    const pooled = emptyCounts(), classes = new Map<string, Counts>();
    for (const [p, c] of counts) {
      plus(pooled, c);
      const key = fragmentationClass(p);
      let parent = classes.get(key);
      if (!parent) classes.set(key, parent = emptyCounts());
      plus(parent, c);
    }
    const next: FragmentationModel = { phones: [...phones], mass, phonesRows: {}, classes: {}, pooled: shrink(pooled, prior, 2) };
    for (const [key, c] of classes) next.classes[key] = shrink(c, next.pooled, mass);
    let change = 0;
    for (const [p, c] of counts) {
      const row = shrink(c, next.classes[fragmentationClass(p)], mass), old = fragmentationRow(model, p);
      change = Math.max(change, Math.abs(row.ordinary - old.ordinary), ...Object.keys(row.sequences).map(s => Math.abs(row.sequences[s] - (old.sequences[s] ?? 0))));
      next.phonesRows[p] = row;
    }
    model = next;
    if (change < 1e-6) break;
  }
  return model;
}
