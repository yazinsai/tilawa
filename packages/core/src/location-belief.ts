import { UNIT } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

/** Candidate-relative features only: neither passage nor speaker identity enters the model. */
export const BELIEF_FEATURES = 12;

export interface LocationBeliefModel {
  version: 1;
  input: number[][];
  recurrent: number[][];
  bias: number[];
  outputBias: number[];
  output: number[];
  /** Logit-to-edit conversion selected on training speakers, never practice. */
  scale: number;
  control?: "nonrecurrent";
}

export interface BeliefFrame {
  lo: number;
  features: number[][];
  costs: Float64Array;
  /** Previous observation's cell, in absolute reference coordinates; not a same-row deletion parent. */
  predecessor: Int32Array;
  logits: Float64Array;
}

export interface BeliefSnapshot { lo: number; hidden: Float64Array; }

/** A bounded candidate memory; its states follow edit predecessors across local run-origin changes. */
export class LocationBelief {
  readonly model: LocationBeliefModel;
  private lo = 0;
  private hidden: Float64Array = new Float64Array(0);

  constructor(model: LocationBeliefModel) {
    this.model = model;
    const h = model.bias.length;
    if (model.version !== 1 || !h || model.input.length !== h || model.recurrent.length !== h
      || model.output.length !== h || model.outputBias.length !== h
      || model.input.some(r => r.length !== BELIEF_FEATURES) || model.recurrent.some(r => r.length !== h)
      || !Number.isFinite(model.scale) || model.scale < 0
      || [...model.input.flat(), ...model.recurrent.flat(), ...model.bias, ...model.outputBias, ...model.output].some(x => !Number.isFinite(x))) {
      throw new Error("Invalid location belief model");
    }
  }

  advance(lo: number, features: number[][], predecessor: Int32Array): Float64Array {
    const m = this.model, h = m.bias.length;
    const next = new Float64Array(features.length * h);
    const logits = new Float64Array(features.length);
    for (let i = 0; i < features.length; i++) {
      const x = features[i];
      if (!x.length) continue; // Unreachable states carry no fabricated belief.
      const p = predecessor[i] - this.lo;
      const inner = m.control === "nonrecurrent"
        ? m.input.map((row, j) => Math.tanh(row.reduce((s, w, k) => s + w * x[k], m.bias[j]))) : null;
      for (let j = 0; j < h; j++) {
        let v = inner ? m.outputBias[j] : m.bias[j];
        if (!inner) for (let k = 0; k < x.length; k++) v += m.input[j][k] * x[k];
        for (let k = 0; k < h; k++) v += m.recurrent[j][k]
          * (inner ? inner[k] : p >= 0 && (p + 1) * h <= this.hidden.length ? this.hidden[p * h + k] : 0);
        const state = Math.tanh(v);
        next[i * h + j] = state;
        logits[i] += m.output[j] * (inner ? state : Math.tanh(state + m.outputBias[j]));
      }
    }
    this.lo = lo;
    this.hidden = next;
    return logits;
  }

  save(): BeliefSnapshot { return { lo: this.lo, hidden: this.hidden }; }
  restore(s: BeliefSnapshot): void { this.lo = s.lo; this.hidden = s.hidden; }
  reset(): void { this.hidden = new Float64Array(0); }
}

/** The same feature extraction serves training, the recurrent model and its instantaneous control. */
export function beliefFeatures(index: ReferenceIndex, lo: number, best: number, cost: Float64Array,
  predecessor: Int32Array, kinds: Int32Array,
  insertion: Float64Array, substitution: Float64Array, deletion: Float64Array): number[][] {
  const min = Math.min(...cost);
  const sourceWord = index.wordAt(Math.max(0, best - 1));
  return Array.from(cost, (c, i) => {
    if (!Number.isFinite(c)) return [];
    const at = lo + i;
    const wi = index.wordAt(Math.max(0, at - 1)), w = index.words[wi];
    return [Math.min(10, (c - min) / UNIT) / 10,
      insertion[i] / UNIT, substitution[i] / UNIT, deletion[i] / UNIT,
      (at - w.start) / Math.max(1, w.end - w.start), Math.log1p(w.end - w.start) / 5,
      Math.tanh((wi - sourceWord) / 8), Math.tanh((at - predecessor[i]) / 16),
      ...[0, 1, 2, 3].map(kind => kinds[i] === kind ? 1 : 0)];
  });
}
