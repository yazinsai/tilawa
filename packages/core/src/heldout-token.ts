import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

export interface TokenEditModel {
  phones: string[];
  /** Per written phone: normalized heard identities followed by deletion. */
  rows: number[][];
  insertion: number;
  inserted: number[];
  block: number;
}

export interface TokenRecording { written: string; heard: string }
export interface TokenLayer { lo: number; size: number; heard: string }
export interface TokenInterval {
  lo: number;
  cost: ArrayLike<number>;
  origin: ArrayLike<number>;
  layers: TokenLayer[];
  nearOrigin: number;
  proposedOrigin: number;
}
export interface TokenGeometry {
  reference: string;
  starts: ReadonlySet<number>;
  /** Existing movement price in weighted edit units, independent of observations. */
  movement(from: number, to: number): number;
  entry(at: number): number;
}

export const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(Math.min(a, b) - m));
};
const normalized = (xs: number[]) => {
  const sum = xs.reduce((a, b) => a + b, 0);
  return xs.map(x => x / sum);
};

/** Mask identities, never positions: contiguous blocks alternate between complementary partitions. */
export function tokenPartition(n: number, block: number, retained: number): boolean[] {
  if (!Number.isInteger(block) || block < 1 || (retained !== 0 && retained !== 1)) throw new Error("Invalid token partition");
  return Array.from({ length: n }, (_, i) => Math.floor(i / block) % 2 === retained);
}

function channel(model: TokenEditModel, geometry: TokenGeometry) {
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  if (model.rows.length !== ids.size || model.rows.some(r => r.length !== ids.size + 1) || model.inserted.length !== ids.size
    || !(model.insertion > 0 && model.insertion < 1)) throw new Error("Invalid token channel");
  for (const row of [...model.rows, model.inserted]) {
    if (row.some(p => !Number.isFinite(p) || p <= 0) || Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-8) throw new Error("Unnormalized token channel");
  }
  const { reference: ref } = geometry;
  const row = (at: number) => model.rows[ids.get(ref[at])!];
  const deleted = (at: number) => row(at)[ids.size] * Math.exp(3 * (UNIT - deletionCost(ref, at)) / UNIT);
  const insertWeight = (at: number, p: string) => model.insertion * model.inserted[ids.get(p)!]
    * Math.exp(3 * (UNIT - insertionCost(ref, at, p)) / UNIT);
  // Duration discounts retain their existing relative edit charges; normalization accounts for every identity.
  const insertMass = (at: number) => model.phones.reduce((sum, p) => sum + insertWeight(at, p), 0);
  const advanceMass = (at: number) => at < ref.length ? (1 - model.insertion) * (1 - row(at)[ids.size] + deleted(at)) : 1 - model.insertion;
  return {
    normalizer(at: number, jumps: readonly number[]) {
      return Math.log(insertMass(at) + advanceMass(at) + jumps.reduce((sum, to) => sum
        + Math.exp(-3 * geometry.movement(at, to) / UNIT) * (1 - model.insertion) * (1 - row(to)[ids.size]), 0));
    },
    insert(at: number, heard: string | null) { return Math.log(heard === null ? insertMass(at) : insertWeight(at, heard)); },
    match(at: number, heard: string | null) {
      return Math.log((1 - model.insertion) * (heard === null ? 1 - row(at)[ids.size] : row(at)[ids.get(heard)!])) - 3 * geometry.entry(at) / UNIT;
    },
    delete(at: number) { return Math.log((1 - model.insertion) * deleted(at)) - 3 * geometry.entry(at) / UNIT; },
  };
}

interface SelectedLocation { offset: number; origin: number }
interface Selection { sources: number[]; near: SelectedLocation | null; restart: SelectedLocation | null }

/** Retained-only Viterbi inference. The jump source and every edit boundary are inferred without withheld identities. */
function select(model: TokenEditModel, geometry: TokenGeometry, interval: TokenInterval, mask: boolean[]): Selection {
  const ch = channel(model, geometry);
  let lo = interval.lo;
  let nearScores = Float64Array.from(interval.cost, (c, i) => interval.origin[i] === interval.nearOrigin ? -3 * c / UNIT : -Infinity);
  let scores = Float64Array.from(interval.cost, (c, i) => interval.origin[i] !== interval.nearOrigin ? -3 * c / UNIT : -Infinity);
  let origins = Int32Array.from(interval.origin);
  const sources: number[] = [];
  for (let t = 0; t < interval.layers.length; t++) {
    const layer = interval.layers[t];
    const prev = new Float64Array(layer.size).fill(-Infinity), pn = new Float64Array(layer.size).fill(-Infinity), po = new Int32Array(layer.size);
    for (let i = 0; i < layer.size; i++) {
      const old = layer.lo + i - lo;
      if (old >= 0 && old < scores.length) { prev[i] = scores[old]; pn[i] = nearScores[old]; po[i] = origins[old]; }
    }
    let best = 0;
    for (let i = 1; i < prev.length; i++) if (Math.max(prev[i], pn[i]) > Math.max(prev[best], pn[best])) best = i;
    const source = layer.lo + best;
    sources.push(source);
    const jumps = [...geometry.starts].filter(at => at >= layer.lo && at < layer.lo + layer.size - 1 && at !== source);
    const norms = Array.from({ length: layer.size }, (_, i) => ch.normalizer(layer.lo + i, i === best ? jumps : []));
    const next = new Float64Array(layer.size).fill(-Infinity), nn = new Float64Array(layer.size).fill(-Infinity), no = new Int32Array(layer.size);
    const heard = mask[t] ? layer.heard : null;
    for (let i = 0; i < next.length; i++) {
      const at = layer.lo + i;
      next[i] = prev[i] + ch.insert(at, heard) - norms[i]; no[i] = po[i];
      nn[i] = pn[i] + ch.insert(at, heard) - norms[i];
      if (i === 0) continue;
      const consume = prev[i - 1] + ch.match(at - 1, heard) - norms[i - 1];
      if (consume >= next[i]) { next[i] = consume; no[i] = po[i - 1]; }
      nn[i] = Math.max(nn[i], pn[i - 1] + ch.match(at - 1, heard) - norms[i - 1]);
      if (geometry.starts.has(at - 1) && at - 1 !== source) {
        const jump = Math.max(prev[best], pn[best]) - 3 * geometry.movement(source, at - 1) / UNIT + ch.match(at - 1, heard) - norms[best];
        if (at - 1 === interval.nearOrigin) nn[i] = Math.max(nn[i], jump);
        else if (jump > next[i]) { next[i] = jump; no[i] = at - 1; }
      }
      const del = next[i - 1] + ch.delete(at - 1) - norms[i - 1];
      if (del > next[i]) { next[i] = del; no[i] = no[i - 1]; }
      nn[i] = Math.max(nn[i], nn[i - 1] + ch.delete(at - 1) - norms[i - 1]);
    }
    scores = next; nearScores = nn; origins = no; lo = layer.lo;
  }
  let near = -1, restart = -1;
  for (let i = 0; i < scores.length; i++) {
    if (Number.isFinite(nearScores[i]) && (near < 0 || nearScores[i] > nearScores[near])) near = i;
    if (Number.isFinite(scores[i]) && (restart < 0 || scores[i] > scores[restart])) restart = i;
  }
  const location = (i: number) => i < 0 ? null : { offset: lo + i, origin: origins[i] };
  return { sources, near: near < 0 ? null : { offset: lo + near, origin: interval.nearOrigin }, restart: location(restart) };
}

/** Sum all alignments ending at the retained-selected location, with its retained-selected movement graph frozen. */
function partition(model: TokenEditModel, geometry: TokenGeometry, interval: TokenInterval,
  mask: boolean[], sources: number[], location: SelectedLocation): number {
  const ch = channel(model, geometry);
  let lo = interval.lo;
  let yes = Float64Array.from(interval.cost, (c, i) => interval.origin[i] === location.origin ? -3 * c / UNIT : -Infinity);
  let other = Float64Array.from(interval.cost, (c, i) => interval.origin[i] !== location.origin ? -3 * c / UNIT : -Infinity);
  for (let t = 0; t < interval.layers.length; t++) {
    const layer = interval.layers[t], source = sources[t];
    const y = new Float64Array(layer.size).fill(-Infinity), o = new Float64Array(layer.size).fill(-Infinity);
    for (let i = 0; i < layer.size; i++) {
      const old = layer.lo + i - lo;
      if (old >= 0 && old < yes.length) { y[i] = yes[old]; o[i] = other[old]; }
    }
    const jumps = [...geometry.starts].filter(at => at >= layer.lo && at < layer.lo + layer.size - 1 && at !== source);
    const norms = Array.from({ length: layer.size }, (_, i) => ch.normalizer(layer.lo + i, layer.lo + i === source ? jumps : []));
    const ny = new Float64Array(layer.size).fill(-Infinity), no = new Float64Array(layer.size).fill(-Infinity);
    const heard = mask[t] ? layer.heard : null;
    for (let i = 0; i < layer.size; i++) {
      const at = layer.lo + i;
      const ins = ch.insert(at, heard) - norms[i];
      ny[i] = y[i] + ins; no[i] = o[i] + ins;
      if (i === 0) continue;
      const match = ch.match(at - 1, heard) - norms[i - 1];
      ny[i] = logAdd(ny[i], y[i - 1] + match); no[i] = logAdd(no[i], o[i - 1] + match);
      if (geometry.starts.has(at - 1) && at - 1 !== source) {
        const j = source - layer.lo;
        const jump = logAdd(y[j], o[j]) - 3 * geometry.movement(source, at - 1) / UNIT + ch.match(at - 1, heard) - norms[j];
        if (at - 1 === location.origin) ny[i] = logAdd(ny[i], jump); else no[i] = logAdd(no[i], jump);
      }
      const del = ch.delete(at - 1) - norms[i - 1];
      ny[i] = logAdd(ny[i], ny[i - 1] + del); no[i] = logAdd(no[i], no[i - 1] + del);
    }
    yes = ny; other = no; lo = layer.lo;
  }
  return yes[location.offset - lo];
}

export interface TokenValidation {
  directions: { retained: number; near: SelectedLocation | null; restart: SelectedLocation | null;
    nearPredictive: number | null; restartPredictive: number | null; gain: number | null }[];
  retain: boolean;
}

export function validateTokenRestart(model: TokenEditModel, geometry: TokenGeometry, interval: TokenInterval): TokenValidation {
  const full = interval.layers.map(() => true);
  const directions = [0, 1].map(retained => {
    const mask = tokenPartition(full.length, model.block, retained);
    if (!mask.includes(true) || !mask.includes(false)) return { retained, near: null, restart: null, nearPredictive: null, restartPredictive: null, gain: null };
    const selected = select(model, geometry, interval, mask);
    const predictive = (location: SelectedLocation | null) => location ? partition(model, geometry, interval, full, selected.sources, location)
      - partition(model, geometry, interval, mask, selected.sources, location) : null;
    const nearPredictive = predictive(selected.near), restartPredictive = predictive(selected.restart);
    return { retained, near: selected.near, restart: selected.restart, nearPredictive, restartPredictive,
      gain: nearPredictive === null || restartPredictive === null ? null : restartPredictive - nearPredictive };
  });
  return { directions, retain: directions.every(d => d.restart?.origin === interval.proposedOrigin && d.gain !== null && Number.isFinite(d.gain) && d.gain > 0) };
}

/** Complete-reference normalized edit transducer for training; sums latent token/edit boundaries. */
export function tokenRecordingLogP(model: TokenEditModel, record: TokenRecording, retained?: boolean[]): number {
  const geometry: TokenGeometry = { reference: record.written, starts: new Set(), movement: () => Infinity, entry: () => 0 };
  const ch = channel(model, geometry), n = record.written.length;
  let row = new Float64Array(n + 1).fill(-Infinity); row[0] = 0;
  const norms = Array.from({ length: n + 1 }, (_, i) => ch.normalizer(i, []));
  for (let i = 1; i <= n; i++) row[i] = row[i - 1] + ch.delete(i - 1) - norms[i - 1];
  for (let t = 0; t < record.heard.length; t++) {
    const heard = retained && !retained[t] ? null : record.heard[t];
    const next = new Float64Array(n + 1).fill(-Infinity);
    for (let i = 0; i <= n; i++) {
      next[i] = row[i] + ch.insert(i, heard) - norms[i];
      if (i > 0) next[i] = logAdd(next[i], logAdd(row[i - 1] + ch.match(i - 1, heard) - norms[i - 1], next[i - 1] + ch.delete(i - 1) - norms[i - 1]));
    }
    row = next;
  }
  return row[n] + Math.log(1 - model.insertion);
}

/** Smoothed directed training alignment counts. Likelihood evaluation marginalizes alignments. */
export function fitTokenEditModel(records: readonly TokenRecording[], phones: string[], strength: number, block: number): TokenEditModel {
  const ids = new Map(phones.map((p, i) => [p, i])), k = phones.length;
  const rows = phones.map(p => [...phones.map(q => Math.exp(-3 * subCost(p, q) / UNIT)), Math.exp(-3)]);
  const prior = rows.map(normalized);
  const counts = prior.map(row => row.map(p => p * strength));
  const inserted = Array(k).fill(strength / k);
  let ins = strength / 2, advance = strength / 2;
  for (const { written, heard } of records) {
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
      if (i && j && d[at] === d[at - width - 1] + subCost(written[i - 1], heard[j - 1])) { counts[ids.get(written[--i])!][ids.get(heard[--j])!]++; advance++; }
      else if (i && d[at] === d[at - width] + deletionCost(written, i - 1)) { counts[ids.get(written[--i])!][k]++; advance++; }
      else { inserted[ids.get(heard[--j])!]++; ins++; }
    }
  }
  return { phones: [...phones], rows: counts.map(normalized), inserted: normalized(inserted), insertion: ins / (ins + advance), block };
}
