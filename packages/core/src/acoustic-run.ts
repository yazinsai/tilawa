import { PosteriorFrames } from "./ctc.js";

/** Small cross-attention scorer. All tensors and calibration are fitted on training speakers only. */
export interface AcousticRunModel {
  version: 1;
  inventory: string;
  tokens: string;
  frames: number;
  acoustic: number[][][];
  acousticBias: number[];
  text: number[][][];
  textBias: number[];
  joint: number[][];
  jointBias: number[];
  output: number[];
  positionWeight: number;
  temperature: number;
}

/** Average complete distributions into temporal bins; blank and sub-blank classes are retained. */
export function acousticRows(frames: PosteriorFrames, from: number, to: number, bins = 64): number[][] | null {
  if (from < frames.startFrame || to > frames.endFrame || to <= from) return null;
  const n = Math.min(bins, to - from);
  return Array.from({ length: n }, (_, i) => {
    const a = from + Math.floor(i * (to - from) / n);
    const b = from + Math.floor((i + 1) * (to - from) / n);
    const row = new Array(frames.row(a).length).fill(0);
    for (let f = a; f < b; f++) {
      const lp = frames.row(f);
      for (let k = 0; k < row.length; k++) row[k] += Math.exp(lp[k]) / (b - a);
    }
    return row;
  });
}

/** Pure JS CPU inference, with the same convolutions and attention as the training module. */
export function createAcousticRunScorer(model: AcousticRunModel, inventory: string) {
  const dim = model.acousticBias.length;
  const vocab = model.tokens.length + 1;
  if (model.version !== 1 || model.inventory !== inventory || !dim || model.temperature <= 0
    || model.acoustic.length !== dim || model.text.length !== dim || model.textBias.length !== dim
    || model.joint.length !== dim || model.output.length !== dim || model.jointBias.length !== dim
    || model.acoustic.some(w => w.length !== 3 || w.some(r => r.length !== vocab))
    || model.text.some(w => w.length !== 3 || w.some(r => r.length !== vocab))
    || model.joint.some(r => r.length !== 3 * dim)
    || !Number.isFinite(model.positionWeight)
    || ![...model.acoustic.flat(2), ...model.text.flat(2), ...model.joint.flat(),
      ...model.acousticBias, ...model.textBias, ...model.jointBias, ...model.output, model.temperature].every(Number.isFinite)) {
    throw new Error("Invalid or incompatible acoustic run model");
  }
  const ids = new Map([...model.tokens].map((p, i) => [p, i + 1]));
  const cache = new WeakMap<readonly (readonly number[])[], number[][]>();
  return (rows: readonly (readonly number[])[], pronunciation: string): number | null => {
    if (!rows.length || !pronunciation || rows.some(r => r.length !== vocab)) return null;
    const phones = [...pronunciation].map(p => ids.get(p));
    if (phones.some(p => p === undefined)) return null;
    const phoneIds = phones as number[];
    const acoustic = cache.get(rows) ?? rows.map((_, t) => Array.from({ length: dim }, (_, d) => {
      let x = model.acousticBias[d];
      for (let k = 0; k < 3; k++) {
        const r = rows[t + k - 1];
        if (r) for (let v = 0; v < vocab; v++) x += model.acoustic[d][k][v] * r[v];
      }
      return Math.tanh(x);
    }));
    cache.set(rows, acoustic);
    let energy = 0;
    for (let j = 0; j < phones.length; j++) {
      const text = Array.from({ length: dim }, (_, d) => {
        let x = model.textBias[d];
        for (let k = 0; k < 3; k++) x += model.text[d][k][phoneIds[j + k - 1] ?? 0];
        return Math.tanh(x);
      });
      const logits = acoustic.map((r, t) => {
        const delta = (t + 0.5) / rows.length - (j + 0.5) / phones.length;
        return r.reduce((s, x, d) => s + x * text[d], 0) / Math.sqrt(dim)
          - Math.max(0, model.positionWeight) * delta * delta;
      });
      const peak = Math.max(...logits);
      const weights = logits.map(x => Math.exp(x - peak));
      const total = weights.reduce((a, b) => a + b, 0);
      const attended = Array.from({ length: dim }, (_, d) => acoustic.reduce((s, r, t) => s + r[d] * weights[t] / total, 0));
      const joint = [...text, ...attended, ...text.map((x, d) => x * attended[d])];
      energy += model.output.reduce((s, w, d) => s + w * Math.tanh(model.jointBias[d]
        + model.joint[d].reduce((a, v, k) => a + v * joint[k], 0)), 0) / phones.length;
    }
    // Existing tracker convention: 3 nats have the weight of one edit (10 units).
    return -energy / model.temperature * 10 / 3;
  };
}

export interface AcousticPath {
  previous: AcousticPath | null;
  reference: string;
  acousticCost: number;
  /** Consumes one decoded observation; reference-only deletion and movement do not. */
  consumes: boolean;
  /** Bounds retained provenance; older acoustic charges stay in the original DP score. */
  retained?: number;
}

export function appendAcousticPath(previous: AcousticPath | null, reference: string, acousticCost: number, consumes: boolean): AcousticPath {
  let parent = previous;
  const retained = (path: AcousticPath | null): number => path?.retained ?? 0;
  if (consumes && (previous?.retained ?? 0) >= 32) {
    const suffix: AcousticPath[] = [];
    let observations = 0;
    for (let p = previous; p; p = p.previous) {
      suffix.push(p);
      if (p.consumes && ++observations === 16) break;
    }
    parent = null;
    for (const p of suffix.reverse()) parent = { ...p, previous: parent, retained: retained(parent) + Number(p.consumes) };
  }
  return { previous: parent, reference, acousticCost, consumes, retained: retained(parent) + Number(consumes) };
}

/** Common suffix of at most the local search's existing recent-cost horizon. */
export function acousticPathSuffix(path: AcousticPath | null, limit = 16) {
  const pieces: string[] = [];
  let acousticCost = 0, observations = 0;
  for (let p = path; p; p = p.previous) {
    pieces.push(p.reference);
    acousticCost += p.acousticCost;
    if (p.consumes && ++observations === limit) break;
  }
  return { pronunciation: pieces.reverse().join(""), acousticCost, observations };
}
