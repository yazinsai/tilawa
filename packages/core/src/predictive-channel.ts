/** Normalized observation transducers. Reverse inference reverses sequences, never a greedy edit script. */
import { subCost, UNIT } from "./fuzzy.js";

export interface PredictiveChannel {
  phones: string[];
  context: number;
  pooling: number;
  insert: number;
  delete: number;
  inserted: number[];
  emitted: number[][];
  conditional: Record<string, number[]>;
}
export interface PredictivePair {
  forward: PredictiveChannel;
  reverse: PredictiveChannel;
  /** Frozen using training partitions, nats equivalent to one existing weighted edit. */
  natsPerEdit: number;
}
export interface PredictiveRecording { written: string; heard: string }

export function snapshotPredictivePair(pair: PredictivePair): PredictivePair {
  const clone = (m: PredictiveChannel): PredictiveChannel => {
    if (!Number.isInteger(m.context) || ![1, 2].includes(m.context) || !(m.pooling > 0)
      || !m.phones.length || new Set(m.phones).size !== m.phones.length
      || ![m.insert, m.delete].every(p => Number.isFinite(p) && p > 0 && p < 1)) throw new Error("Invalid predictive model");
    const row = (xs: number[]) => {
      if (xs.length !== m.phones.length || !xs.every(p => Number.isFinite(p) && p > 0)
        || Math.abs(xs.reduce((a, b) => a + b, 0) - 1) > 1e-8) throw new Error("Unnormalized predictive row");
      return [...xs];
    };
    if (m.emitted.length !== m.phones.length) throw new Error("Invalid predictive emissions");
    return { ...m, phones: [...m.phones], inserted: row(m.inserted), emitted: m.emitted.map(row),
      conditional: Object.fromEntries(Object.entries(m.conditional).map(([k, xs]) => [k, row(xs)])) };
  };
  if (!(pair.natsPerEdit > 0) || !Number.isFinite(pair.natsPerEdit)
    || pair.forward.context !== pair.reverse.context || pair.forward.pooling !== pair.reverse.pooling
    || pair.forward.phones.join("") !== pair.reverse.phones.join("")) throw new Error("Unmatched predictive control");
  return { forward: clone(pair.forward), reverse: clone(pair.reverse), natsPerEdit: pair.natsPerEdit };
}

export function phoneClass(p: string | undefined): string {
  if (!p) return "end";
  if ("َُِ".includes(p)) return "vowel";
  if ("اۦۥ".includes(p)) return "long";
  if ("نمں۾".includes(p)) return "nasal";
  if ("بتدطضكقء".includes(p)) return "stop";
  if ("ثذسزصشخغحعهف".includes(p)) return "fricative";
  return "sonorant";
}
const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
};
const normalize = (xs: number[]) => { const sum = xs.reduce((a, b) => a + b, 0); return xs.map(x => x / sum); };
const contextKey = (r: string, x: string, j: number, i: number, n: number): string =>
  [r[j], ...Array.from({ length: n }, (_, k) => phoneClass(r[j - k - 1])),
    ...Array.from({ length: n }, (_, k) => phoneClass(x[i - k - 1]))].join("/");

/** A reference phone is deleted or emits one observation; insertions form a geometric run at each frontier. */
function lattice(model: PredictiveChannel, r: string, x: string) {
  const n = r.length, m = x.length, width = m + 1;
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  const b = new Float64Array(f.length).fill(-Infinity);
  const inventory = new Map(model.phones.map((p, i) => [p, i]));
  const ri = [...r].map(p => inventory.get(p)), xi = [...x].map(p => inventory.get(p));
  if (ri.some(i => i === undefined) || xi.some(i => i === undefined)) throw new Error("Unknown predictive-channel phone");
  const ins = Math.log(model.insert), del = Math.log((1 - model.insert) * model.delete);
  const match = Math.log((1 - model.insert) * (1 - model.delete));
  const emissions = new Float64Array(n * m);
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) {
    const row = model.conditional[contextKey(r, x, j, i, model.context)] ?? model.emitted[ri[j]!];
    emissions[j * m + i] = Math.log(row[xi[i]!]);
  }
  const insertion = xi.map(i => ins + Math.log(model.inserted[i!]));
  f[0] = 0;
  for (let j = 0; j <= n; j++) for (let i = 0; i <= m; i++) {
    const at = j * width + i, v = f[at];
    if (i < m) f[at + 1] = logAdd(f[at + 1], v + insertion[i]);
    if (j < n) {
      f[at + width] = logAdd(f[at + width], v + del);
      if (i < m) f[at + width + 1] = logAdd(f[at + width + 1], v + match + emissions[j * m + i]);
    }
  }
  b[n * width + m] = Math.log(1 - model.insert);
  for (let j = n; j >= 0; j--) for (let i = m; i >= 0; i--) {
    const at = j * width + i;
    if (i < m) b[at] = logAdd(b[at], insertion[i] + b[at + 1]);
    if (j < n) {
      b[at] = logAdd(b[at], del + b[at + width]);
      if (i < m) b[at] = logAdd(b[at], match + emissions[j * m + i] + b[at + width + 1]);
    }
  }
  return { f, b, width, logP: b[0], emissions, insertion, del, match, ri, xi };
}

export function predictiveLogP(model: PredictiveChannel, recording: PredictiveRecording, reverse = false): number {
  const orient = (s: string) => reverse ? [...s].reverse().join("") : s;
  return lattice(model, orient(recording.written), orient(recording.heard)).logP;
}

/** EM sums all insertion/deletion placements, including ambiguous duration repetitions. No transcript alignment labels. */
export function fitPredictiveChannel(records: readonly PredictiveRecording[], phones: string[], context: number,
  pooling: number, reverse: boolean, iterations = 12): PredictiveChannel {
  if (!records.length || !phones.length || ![1, 2].includes(context) || !(pooling > 0)) throw new Error("Invalid predictive training inputs");
  const orient = (s: string) => reverse ? [...s].reverse().join("") : s;
  const data = records.map(r => ({ written: orient(r.written), heard: orient(r.heard) }));
  const initial = phones.map(r => normalize(phones.map(x => Math.exp(-subCost(r, x) / UNIT * 3))));
  let model: PredictiveChannel = { phones: [...phones], context, pooling, insert: 0.1, delete: 0.1,
    inserted: phones.map(() => 1 / phones.length), emitted: initial, conditional: {} };
  for (let iteration = 0; iteration < iterations; iteration++) {
    const emitted = phones.map(() => phones.map(() => 0)), inserted = phones.map(() => 0);
    const conditional: Record<string, number[]> = {};
    let nIns = 0, nDel = 0, nMatch = 0;
    for (const { written: r, heard: x } of data) {
      const { f, b, width, logP, emissions, insertion, del, match, ri, xi } = lattice(model, r, x);
      for (let j = 0; j <= r.length; j++) for (let i = 0; i <= x.length; i++) {
        const at = j * width + i;
        if (i < x.length) {
          const w = Math.exp(f[at] + insertion[i] + b[at + 1] - logP);
          nIns += w; inserted[xi[i]!] += w;
        }
        if (j < r.length) {
          nDel += Math.exp(f[at] + del + b[at + width] - logP);
          if (i < x.length) {
            const w = Math.exp(f[at] + match + emissions[j * x.length + i] + b[at + width + 1] - logP);
            nMatch += w; emitted[ri[j]!][xi[i]!] += w;
            const key = contextKey(r, x, j, i, context);
            const row = conditional[key] ??= phones.map(() => 0);
            row[xi[i]!] += w;
          }
        }
      }
    }
    const base = emitted.map((row, j) => normalize(row.map((v, i) => v + pooling * initial[j][i])));
    model = { phones: [...phones], context, pooling,
      insert: (nIns + 1) / (nIns + nMatch + nDel + data.length + 2),
      delete: (nDel + 1) / (nDel + nMatch + 2),
      inserted: normalize(inserted.map(v => v + 1 / phones.length)), emitted: base,
      conditional: Object.fromEntries(Object.entries(conditional).map(([key, row]) => {
        const j = phones.indexOf(key.split("/")[0]);
        return [key, normalize(row.map((v, i) => v + pooling * base[j][i]))];
      })) };
  }
  return model;
}

/** Pooled emission odds convert evidence to existing edit units; computed without practice observations. */
export function predictiveCalibration(models: readonly PredictiveChannel[]): number {
  let sum = 0, weight = 0;
  for (const m of models) for (let j = 0; j < m.phones.length; j++) for (let i = 0; i < m.phones.length; i++) {
    if (i === j) continue;
    const odds = Math.log(m.emitted[j][j] / m.emitted[j][i]);
    if (odds <= 0) continue;
    const w = m.emitted[j][i];
    sum += w * odds / (subCost(m.phones[j], m.phones[i]) / UNIT); weight += w;
  }
  if (!(weight > 0) || !(sum > 0)) throw new Error("Uncalibrated predictive model");
  return sum / weight;
}

export function predictiveContribution(pair: PredictivePair, record: PredictiveRecording) {
  const forward = predictiveLogP(pair.forward, record);
  const reverse = predictiveLogP(pair.reverse, record, true);
  return { forward, reverse, edits: (reverse - forward) / pair.natsPerEdit };
}
