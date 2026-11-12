import { subCost, UNIT } from "./fuzzy.js";
import type { DecodedPhoneme } from "./evidence.js";

/** A recording's paired views share its reference, speaker and unchanged audio. */
export interface PhaseRecording {
  written: string;
  views: readonly (readonly { p: string; phase: number }[])[];
}

export interface PhaseChannel {
  /** When frozen for deployment, reject observations from a different inference graph or advance. */
  geometry?: { model: string; chunkFrames: number };
  phones: readonly string[];
  bins: number;
  mass: number;
  /** P(emitted identity | reference identity); the final row describes insertions. */
  pooled: readonly (readonly number[])[];
  /** Same normalized channel, conditioned on native inference phase. */
  conditioned: readonly (readonly (readonly number[])[])[];
}

const NATS_PER_EDIT = 3;
const norm = (row: readonly number[]) => {
  const sum = row.reduce((a, b) => a + b, 0);
  return row.map(x => x / sum);
};
const add = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
};
const phaseBin = (phase: number, bins: number) =>
  Number.isFinite(phase) && phase >= 0 && phase < 1 ? Math.floor(phase * bins) : null;

/** No caller clock or push length participates in this computation. */
export function inferencePhase(token: DecodedPhoneme): number | undefined {
  const g = token.inference;
  if (!g || !Number.isInteger(g.frame) || !Number.isInteger(g.chunkStart) ||
      !Number.isInteger(g.chunkFrames) || g.chunkFrames <= 0 ||
      g.frame < g.chunkStart || g.frame >= g.chunkStart + g.chunkFrames) return undefined;
  return (g.frame - g.chunkStart) / g.chunkFrames;
}

export function channelPhase(channel: PhaseChannel | undefined, token: DecodedPhoneme): number | undefined {
  if (channel?.geometry && (token.inference?.model !== channel.geometry.model ||
      token.inference?.chunkFrames !== channel.geometry.chunkFrames)) return undefined;
  return inferencePhase(token);
}

/** Phase versus pooled correction in the tracker's existing edit units. */
export function phaseObservation(model: PhaseChannel, heard: string, phase?: number) {
  const b = phaseBin(phase ?? NaN, model.bins), h = model.phones.indexOf(heard);
  const correction = (r: number) => b === null || h < 0 || model.bins === 1 ? 0 :
    -UNIT / NATS_PER_EDIT * Math.log(model.conditioned[r][b][h] / model.pooled[r][h]);
  const costs = new Map(model.phones.map((p, r) => [p, correction(r)]));
  return { substitution: (reference: string) => costs.get(reference) ?? 0, insertion: correction(model.phones.length) };
}

/** Phonetic manner backoff; no word, reader or passage features. */
function phoneClass(p: string): number {
  const classes = ["َُِ۪", "اۦۥٲ", "ءبتدطضقك", "ثحخذزسشصظغفهڇ", "ج", "منں۾", "رل", "وي"];
  const c = classes.findIndex(row => row.includes(p));
  return c < 0 ? classes.length : c;
}

/**
 * Transcript-constrained latent edits, used only to fit the observation channel. Interior choices are normalized
 * match/deletion/insertion (.8/.1/.1); the exhausted reference stops with .9 or inserts with .1. A paired view
 * receives 1 / views.length weight, so changing the number of origins cannot overweight a speaker's recording.
 */
function align(model: PhaseChannel, written: string, view: PhaseRecording["views"][number],
  collect?: (r: number, b: number, h: number, weight: number) => void): number {
  const k = model.phones.length, ids = new Map(model.phones.map((p, i) => [p, i]));
  const r = [...written].map(p => ids.get(p)), h = view.map(p => ids.get(p.p));
  const bins = view.map(p => phaseBin(p.phase, model.bins));
  if (r.some(x => x === undefined) || h.some(x => x === undefined) || bins.some(x => x === null)) throw new Error("Invalid phase training observation");
  const n = r.length, m = h.length, width = m + 1, at = (i: number, j: number) => i * width + j;
  const f = new Float64Array((n + 1) * width).fill(-Infinity), back = new Float64Array(f.length).fill(-Infinity);
  const lp = model.conditioned.map(rows => rows.map(row => row.map(Math.log)));
  const diagonal = (i: number, j: number) => Math.log(.8) + lp[r[i]!][bins[j]!][h[j]!];
  const insertion = (j: number) => Math.log(.1) + lp[k][bins[j]!][h[j]!];
  const deletion = Math.log(.1);
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const v = f[at(i, j)];
    if (i < n && j < m) f[at(i + 1, j + 1)] = add(f[at(i + 1, j + 1)], v + diagonal(i, j));
    if (i < n) f[at(i + 1, j)] = add(f[at(i + 1, j)], v + deletion);
    if (j < m) f[at(i, j + 1)] = add(f[at(i, j + 1)], v + insertion(j));
  }
  const logP = f[at(n, m)] + Math.log(.9);
  if (!collect) return logP;
  back[at(n, m)] = Math.log(.9);
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (i < n && j < m) back[at(i, j)] = add(back[at(i, j)], diagonal(i, j) + back[at(i + 1, j + 1)]);
    if (i < n) back[at(i, j)] = add(back[at(i, j)], deletion + back[at(i + 1, j)]);
    if (j < m) back[at(i, j)] = add(back[at(i, j)], insertion(j) + back[at(i, j + 1)]);
  }
  for (let i = 0; i <= n; i++) for (let j = 0; j < m; j++) {
    if (i < n) collect(r[i]!, bins[j]!, h[j]!, Math.exp(f[at(i, j)] + diagonal(i, j) + back[at(i + 1, j + 1)] - logP));
    collect(k, bins[j]!, h[j]!, Math.exp(f[at(i, j)] + insertion(j) + back[at(i, j + 1)] - logP));
  }
  return logP;
}

export function phaseLogLikelihood(model: PhaseChannel, record: PhaseRecording): number {
  return record.views.reduce((sum, view) => sum + align(model, record.written, view), 0) / record.views.length;
}

/** Fit with hierarchical reference-class phase backoff to reference-specific pooled emissions. */
export function fitPhaseChannel(records: readonly PhaseRecording[], phones: readonly string[], bins: number, mass: number): PhaseChannel {
  if (!records.length || !phones.length || !Number.isInteger(bins) || bins < 1 || !(mass > 0) ||
      records.some(r => !r.written || !r.views.length)) throw new Error("Invalid phase fit");
  const k = phones.length;
  const prior = phones.map(r => norm(phones.map(h => Math.exp(-NATS_PER_EDIT * subCost(r, h) / UNIT))));
  prior.push(norm(phones.map(() => 1)));
  let model: PhaseChannel = { phones: [...phones], bins, mass, pooled: prior, conditioned: prior.map(row => Array.from({ length: bins }, () => [...row])) };
  const classes = phones.map(phoneClass); classes.push(10);
  for (let iteration = 0; iteration < 12; iteration++) {
    const counts = Array.from({ length: k + 1 }, () => Array.from({ length: bins }, () => Array(k).fill(0) as number[]));
    for (const record of records) for (const view of record.views) align(model, record.written, view,
      (r, b, h, w) => counts[r][b][h] += w / record.views.length);
    const pooled = counts.map((rows, r) => norm(prior[r].map((p, h) => mass * p + rows.reduce((sum, row) => sum + row[h], 0))));
    const classCounts = Array.from({ length: 11 }, () => Array.from({ length: bins }, () => Array(k).fill(.5) as number[]));
    counts.forEach((rows, r) => rows.forEach((row, b) => row.forEach((v, h) => classCounts[classes[r]][b][h] += v)));
    const classPool = classCounts.map(rows => norm(phones.map((_, h) => rows.reduce((s, row) => s + row[h], 0))));
    const classPhase = classCounts.map(rows => rows.map(norm));
    const conditioned = counts.map((rows, r) => rows.map((row, b) => {
      if (bins === 1) return [...pooled[r]];
      const backoff = norm(pooled[r].map((p, h) => p * classPhase[classes[r]][b][h] / classPool[classes[r]][h]));
      return norm(row.map((v, h) => v + mass * backoff[h]));
    }));
    model = { ...model, pooled, conditioned };
  }
  return model;
}
