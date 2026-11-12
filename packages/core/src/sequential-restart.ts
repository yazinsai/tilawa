import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

export interface RestartChannel {
  phones: string[];
  /** Each row sums to one: observed identities, then reference deletion. */
  rows: number[][];
  insertion: number;
  inserted: number[];
}
export interface RestartCalibration {
  channel: RestartChannel;
  /** Frozen upper tail of destination log odds on speaker-separated clean training streams. */
  threshold: number;
}
export interface RestartGeometry {
  reference: string;
  lo: number;
  hi: number;
  starts: readonly number[];
  movement(from: number, to: number): number;
  entry(at: number): number;
}
interface RestartSnapshot {
  continuation: Float64Array;
  restarted: Map<number, Float64Array>;
  tokens: number;
  continuationLogLikelihood: number;
  increments: Map<number, number>;
}
export interface RestartEvidence {
  tokens: number;
  destination: number;
  logOdds: number;
  increment: number;
  restartLogLikelihood: number;
  continuationLogLikelihood: number;
  admitted: boolean;
  /** Candidates outside this retained reference interval use the original movement rule. */
  supported: boolean;
}
const add = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(Math.min(a, b) - m));
};
const sum = (xs: ArrayLike<number>): number => {
  let total = -Infinity;
  for (let i = 0; i < xs.length; i++) total = add(total, xs[i]);
  return total;
};
const normalized = (xs: number[]) => {
  const total = xs.reduce((a, b) => a + b, 0);
  return xs.map(x => x / total);
};

/** Every state defines a normalized distribution for the next observation, including latent deletions. */
export function restartKernel(model: RestartChannel, geometry: RestartGeometry) {
  const ids = new Map(model.phones.map((p, i) => [p, i])), k = ids.size;
  if (k !== model.phones.length || !k || !(model.insertion > 0 && model.insertion < 1)
    || model.rows.length !== k || model.rows.some(r => r.length !== k + 1) || model.inserted.length !== k)
    throw new Error("Invalid restart channel dimensions");
  for (const row of [...model.rows, model.inserted]) {
    if (row.some(p => !Number.isFinite(p) || p <= 0) || Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-8)
      throw new Error("Unnormalized restart channel");
  }
  const { reference: ref, lo, hi } = geometry;
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi > ref.length || hi <= lo)
    throw new Error("Invalid restart interval");
  const size = hi - lo + 1;
  const insert: number[][] = [], match: number[][] = [], deleted: number[] = [];
  for (let at = lo; at < hi; at++) {
    const identity = ids.get(ref[at]);
    if (identity === undefined) throw new Error("Reference phone absent from restart channel");
    const row = model.rows[identity], entry = Math.exp(-3 * geometry.entry(at) / UNIT);
    const ins = model.phones.map((p, i) => model.insertion * model.inserted[i]
      * Math.exp(3 * (UNIT - insertionCost(ref, at, p)) / UNIT));
    const emit = row.slice(0, k).map(p => (1 - model.insertion) * p * entry);
    const del = (1 - model.insertion) * row[k] * entry * Math.exp(3 * (UNIT - deletionCost(ref, at)) / UNIT);
    const mass = [...ins, ...emit, del].reduce((a, b) => a + b, 0);
    insert.push(ins.map(p => Math.log(p / mass)));
    match.push(emit.map(p => Math.log(p / mass)));
    deleted.push(Math.log(del / mass));
  }
  // End-of-window overflow has a proper observation distribution, rather than silently losing probability mass.
  insert.push(model.inserted.map(Math.log));
  return (previous: Float64Array, phone: string): Float64Array => {
    if (previous.length !== size) throw new Error("Restart row geometry changed");
    const id = ids.get(phone);
    if (id === undefined) throw new Error("Observation absent from restart channel");
    const flow = previous.slice(), next = new Float64Array(size).fill(-Infinity);
    for (let i = 0; i < size; i++) {
      next[i] = add(next[i], flow[i] + insert[i][id]);
      if (i + 1 < size) {
        next[i + 1] = add(next[i + 1], flow[i] + match[i][id]);
        flow[i + 1] = add(flow[i + 1], flow[i] + deleted[i]);
      }
    }
    return next;
  };
}

/**
 * One persistent comparison per confirmed run. q(t)=1/(t(t+1)) is a proper prior over positive restart times.
 * A destination's row sums all restart times and alignments; no overlapping window becomes another trial.
 */
export function createRestartEvidence(calibration: RestartCalibration, geometry: RestartGeometry, offset: number,
  /** Total movement-family odds retain the existing two-edit jump price. */
  jumpCost: number) {
  if (!Number.isFinite(calibration.threshold) || calibration.threshold < 0 || !Number.isFinite(jumpCost) || jumpCost < 0)
    throw new Error("Invalid restart calibration");
  // The structural destination set is frozen independently of future acoustic rankings.
  geometry = { ...geometry, starts: [...new Set(geometry.starts)] };
  const step = restartKernel(calibration.channel, geometry), size = geometry.hi - geometry.lo + 1;
  if (offset < geometry.lo || offset > geometry.hi) throw new Error("Restart seed outside interval");
  let continuation = new Float64Array(size).fill(-Infinity); continuation[offset - geometry.lo] = 0;
  let restarted = new Map<number, Float64Array>();
  let tokens = 0, continuationLogLikelihood = 0;
  let increments = new Map<number, number>();
  const priorOdds = -3 * jumpCost / UNIT;
  return {
    step(phone: string, source: number) {
      const t = tokens + 1;
      const next = step(continuation, phone), predictive = sum(next);
      for (let i = 0; i < size; i++) next[i] -= predictive;
      // Enumerate from the pre-observation state, independently of the new token's acoustic ranking.
      const destinations = geometry.starts.filter(at => at !== source && at >= geometry.lo && at < geometry.hi
        && Number.isFinite(geometry.movement(source, at)));
      const weights = destinations.map(at => -3 * geometry.movement(source, at) / UNIT);
      const normalizer = sum(weights);
      const nextRestarted = new Map<number, Float64Array>();
      const nextIncrements = new Map<number, number>();
      for (const destination of new Set([...restarted.keys(), ...destinations])) {
        const previous = restarted.get(destination) ?? new Float64Array(size).fill(-Infinity);
        const injected = previous.slice(), j = destinations.indexOf(destination);
        if (j >= 0) injected[destination - geometry.lo] = add(injected[destination - geometry.lo],
          -Math.log(t) - Math.log(t + 1) + weights[j] - normalizer);
        const row = step(injected, phone);
        for (let i = 0; i < size; i++) row[i] -= predictive;
        nextRestarted.set(destination, row);
        // Difference of prefix log likelihoods, not a second charge for the retained prefix.
        const old = sum(previous), current = sum(row);
        nextIncrements.set(destination, old === -Infinity ? current : current - old);
      }
      continuation = next; restarted = nextRestarted; increments = nextIncrements;
      tokens = t; continuationLogLikelihood += predictive;
    },
    comparison(destination: number): RestartEvidence {
      const relative = restarted.has(destination) ? sum(restarted.get(destination)!) : -Infinity;
      const logOdds = relative + priorOdds;
      return { tokens, destination, logOdds, increment: increments.get(destination) ?? -Infinity,
        restartLogLikelihood: relative + continuationLogLikelihood,
        continuationLogLikelihood, admitted: logOdds > calibration.threshold, supported: geometry.starts.includes(destination) };
    },
    comparisons(): RestartEvidence[] { return [...restarted.keys()].map(d => this.comparison(d)); },
    save(): RestartSnapshot { return { continuation, restarted, tokens, continuationLogLikelihood, increments }; },
    restore(state: unknown) {
      const s = state as RestartSnapshot;
      ({ continuation, restarted, tokens, continuationLogLikelihood, increments } = s);
    },
  };
}

export interface RestartRecording { written: string; heard: string }
/** Prefix likelihood under the same normalized observation process used by the runtime. */
export function restartRecordingLogP(channel: RestartChannel, record: RestartRecording): number {
  const step = restartKernel(channel, { reference: record.written, lo: 0, hi: record.written.length,
    starts: [], movement: () => 0, entry: () => 0 });
  let row = new Float64Array(record.written.length + 1).fill(-Infinity); row[0] = 0;
  for (const p of record.heard) row = step(row, p);
  return sum(row);
}

/** Smoothed directed edit counts on eligible clean training recordings; inference sums latent alignments. */
export function fitRestartChannel(records: readonly RestartRecording[], phones: string[], strength: number): RestartChannel {
  if (!(strength > 0) || !records.length) throw new Error("Training records and smoothing required");
  const ids = new Map(phones.map((p, i) => [p, i])), k = phones.length;
  const prior = phones.map(p => normalized([...phones.map(q => Math.exp(-3 * subCost(p, q) / UNIT)), Math.exp(-3)]));
  const counts = prior.map(row => row.map(p => p * strength));
  const inserted = Array(k).fill(strength / k);
  let ins = strength / 2, advance = strength / 2;
  for (const { written, heard } of records) {
    if (![...written, ...heard].every(p => ids.has(p))) throw new Error("Training inventory mismatch");
    const width = heard.length + 1, d = new Float64Array((written.length + 1) * width);
    for (let i = 0; i <= written.length; i++) for (let j = 0; j <= heard.length; j++) {
      const at = i * width + j;
      if (!i) d[at] = j * UNIT;
      else if (!j) d[at] = d[at - width] + deletionCost(written, i - 1);
      else d[at] = Math.min(d[at - width - 1] + subCost(written[i - 1], heard[j - 1]),
        d[at - 1] + insertionCost(written, i, heard[j - 1]), d[at - width] + deletionCost(written, i - 1));
    }
    let i = written.length, j = heard.length;
    while (i || j) {
      const at = i * width + j;
      if (i && j && d[at] === d[at - width - 1] + subCost(written[i - 1], heard[j - 1])) {
        counts[ids.get(written[--i])!][ids.get(heard[--j])!]++; advance++;
      } else if (i && d[at] === d[at - width] + deletionCost(written, i - 1)) {
        counts[ids.get(written[--i])!][k]++; advance++;
      } else { inserted[ids.get(heard[--j])!]++; ins++; }
    }
  }
  return { phones: [...phones], rows: counts.map(normalized), inserted: normalized(inserted), insertion: ins / (ins + advance) };
}
