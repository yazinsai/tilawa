import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

export const motionEvents = ["read", "edit", "omit", "repeat"] as const;
export type MotionEvent = typeof motionEvents[number];
export interface EpisodeMotionModel {
  /** The training-selected independent baseline bypasses the augmented local search. */
  dependent?: boolean;
  /** State 0: ordinary reading; state 1: a local mistake/recovery episode. */
  initial: readonly number[];
  /** P(event, next state | previous state), normalized jointly over both axes. */
  joint: readonly (readonly (readonly number[])[])[];
  /** Independent event distribution, the likelihood-ratio denominator. */
  background: readonly number[];
}
const SCALE = UNIT / 3;
const normalize = (a: number[]) => {
  const total = a.reduce((x, y) => x + y, 0);
  return a.map(x => x / total);
};
const add = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
};

export function validateEpisodeModel(model: EpisodeMotionModel): void {
  const distribution = (row: readonly number[]) => row.every(p => Number.isFinite(p) && p > 0) && Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-8;
  if (model.initial.length !== 2 || !distribution(model.initial) || model.background.length !== 4 || !distribution(model.background)
    || model.joint.length !== 2 || model.joint.some(row => row.length !== 4 || row.some(x => x.length !== 2) || !distribution(row.flat()))) {
    throw new Error("Invalid normalized episode motion model");
  }
}

/** Adds dependence evidence, while the caller still supplies every existing acoustic and movement cost. */
export function episodeTransitionCost(model: EpisodeMotionModel, from: number, event: MotionEvent, to: number): number {
  const e = motionEvents.indexOf(event);
  return -SCALE * Math.log(model.joint[from][e][to] / model.background[e]);
}

export interface EpisodeTrainingWord {
  phones: string;
  /** Scripted omissions bypass the complete word without any heard-token consumption. */
  omit?: boolean;
  /** The second copy of a labelled repeat has an explicit repeat entry. */
  repeat?: boolean;
}
export interface EpisodeTrainingRecording {
  words: readonly EpisodeTrainingWord[];
  heard: string;
}
interface Arc { from: number; to: number; event: number; logP: number }
export interface EpisodeTrainingGraph { nodes: number; arcs: readonly Arc[] }

/**
 * Training-only edit DAG. All token/word boundaries are latent. Labelled omissions are obligatory bypasses;
 * repeats are unfolded by the caller, so there are no cycles. Ordinary phone edits keep the deployed prices.
 * At each reference position, diagonal, deletion and insertion choose .8/.1/.1; emission distributions over
 * the complete inventory are normalized. At the end the channel stops with .9 or inserts with .1.
 */
export function episodeTrainingGraph(record: EpisodeTrainingRecording, phones: readonly string[]): EpisodeTrainingGraph {
  const ref = record.words.map(w => w.phones).join("");
  if (!phones.length || [...ref, ...record.heard].some(p => !phones.includes(p))) throw new Error("Training phone outside inventory");
  const omissions = new Map<number, number>();
  const hidden = new Set<number>();
  const repeat = new Set<number>();
  let offset = 0;
  for (const w of record.words) {
    if (!w.phones.length) throw new Error("Empty training word");
    if (w.omit) {
      omissions.set(offset, offset + w.phones.length);
      for (let i = offset + 1; i < offset + w.phones.length; i++) hidden.add(i);
    }
    if (w.repeat) repeat.add(offset);
    offset += w.phones.length;
  }
  const width = record.heard.length + 1;
  const at = (i: number, j: number) => i * width + j;
  const arcs: Arc[] = [];
  const emit = (cost: (p: string) => number, p: string) => {
    const weights = phones.map(h => Math.exp(-cost(h) / SCALE));
    return Math.log(weights[phones.indexOf(p)] / weights.reduce((a, b) => a + b, 0));
  };
  // Emission rows are shared by every heard boundary at the same reference position.
  const diagonal = [...ref].map(r => new Map(phones.map(h => [h, emit(p => subCost(r, p), h)])));
  const inserts = Array.from({ length: ref.length + 1 }, (_, i) => new Map(phones.map(h => [h, emit(p => insertionCost(ref, i, p), h)])));
  for (let i = 0; i <= ref.length; i++) {
    if (hidden.has(i)) continue;
    for (let j = 0; j <= record.heard.length; j++) {
      const from = at(i, j);
      if (j < record.heard.length) arcs.push({ from, to: at(i, j + 1), event: 1, logP: Math.log(.1) + inserts[i].get(record.heard[j])! });
      if (i === ref.length) continue;
      const skip = omissions.get(i);
      if (skip !== undefined) {
        arcs.push({ from, to: at(skip, j), event: 2, logP: Math.log(.9) });
        continue;
      }
      const entryEvent = repeat.has(i) ? 3 : null;
      const deletion = .1 * Math.exp(-deletionCost(ref, i) / SCALE);
      const denominator = .9 + deletion;
      // Normalize the reference-dependent edit choice, including its unchanged deletion cost.
      if (j < record.heard.length) arcs[arcs.length - 1].logP -= Math.log(denominator);
      arcs.push({ from, to: at(i + 1, j), event: entryEvent ?? 1, logP: Math.log(deletion / denominator) });
      if (j < record.heard.length) arcs.push({ from, to: at(i + 1, j + 1), event: entryEvent ?? (ref[i] === record.heard[j] ? 0 : 1), logP: Math.log(.8 / denominator) + diagonal[i].get(record.heard[j])! });
    }
  }
  // Node order is a topological order, including reference-only omission arcs.
  return { nodes: (ref.length + 1) * width, arcs };
}

/** Forward/backward over both event boundaries and recovery states; no best-alignment labels are used. */
function expectation(model: EpisodeMotionModel, graph: EpisodeTrainingGraph, counts?: number[][][], initial?: number[]): number {
  const f = new Float64Array(graph.nodes * 2).fill(-Infinity);
  const b = new Float64Array(f.length).fill(-Infinity);
  for (let s = 0; s < 2; s++) f[s] = Math.log(model.initial[s]);
  const weight = (arc: Arc, s: number, t: number) => arc.logP + Math.log(model.joint[s][arc.event][t]);
  for (const arc of graph.arcs) for (let s = 0; s < 2; s++) for (let t = 0; t < 2; t++) {
    f[arc.to * 2 + t] = add(f[arc.to * 2 + t], f[arc.from * 2 + s] + weight(arc, s, t));
  }
  const end = (graph.nodes - 1) * 2;
  const logP = add(f[end], f[end + 1]) + Math.log(.9);
  if (!Number.isFinite(logP)) throw new Error("No complete training alignment");
  if (!counts || !initial) return logP;
  b[end] = b[end + 1] = Math.log(.9);
  for (let i = graph.arcs.length - 1; i >= 0; i--) {
    const arc = graph.arcs[i];
    for (let s = 0; s < 2; s++) for (let t = 0; t < 2; t++) {
      b[arc.from * 2 + s] = add(b[arc.from * 2 + s], weight(arc, s, t) + b[arc.to * 2 + t]);
    }
  }
  for (let s = 0; s < 2; s++) initial[s] += Math.exp(f[s] + b[s] - logP);
  for (const arc of graph.arcs) for (let s = 0; s < 2; s++) for (let t = 0; t < 2; t++) {
    counts[s][arc.event][t] += Math.exp(f[arc.from * 2 + s] + weight(arc, s, t) + b[arc.to * 2 + t] - logP);
  }
  return logP;
}
export const episodeTrainingLogP = (model: EpisodeMotionModel, graph: EpisodeTrainingGraph) => expectation(model, graph);

/** Hyperparameters and independent/two-state choice must be selected on speaker-grouped training folds. */
export function fitEpisodeMotion(graphs: readonly EpisodeTrainingGraph[], mass: number, dependent = true): EpisodeMotionModel {
  if (!graphs.length || !(mass > 0)) throw new Error("Invalid episode training request");
  let model: EpisodeMotionModel = {
    dependent, initial: [.9, .1], background: [.7, .2, .05, .05],
    joint: [[ [.63, .07], [.04, .16], [.01, .04], [.01, .04] ], [ [.14, .56], [.02, .18], [.005, .045], [.005, .045] ]],
  };
  for (let iteration = 0; iteration < 40; iteration++) {
    const counts = Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => [mass / 8, mass / 8]));
    const initial = [mass / 2, mass / 2];
    for (const graph of graphs) expectation(model, graph, counts, initial);
    const joint = counts.map(row => {
      const flat = normalize(row.flat());
      return Array.from({ length: 4 }, (_, e) => flat.slice(e * 2, e * 2 + 2));
    });
    const background = normalize(Array.from({ length: 4 }, (_, e) => counts.reduce((a, row) => a + row[e][0] + row[e][1], 0)));
    if (!dependent) for (let s = 0; s < 2; s++) for (let e = 0; e < 4; e++) joint[s][e] = background[e] === 0 ? [0, 0] : [background[e] / 2, background[e] / 2];
    const next = { dependent, initial: dependent ? normalize(initial) : [.5, .5], joint, background };
    const change = next.joint.flat(2).reduce((a, p, i) => a + Math.abs(p - model.joint.flat(2)[i]), 0);
    model = next;
    if (change < 1e-7) break;
  }
  // Canonicalize the more read-heavy state to ordinary, without changing its probability model.
  if (model.joint[1][0].reduce((a, b) => a + b, 0) > model.joint[0][0].reduce((a, b) => a + b, 0)) {
    model = { ...model, initial: [...model.initial].reverse(), joint: [...model.joint].reverse().map(row => row.map(pair => [...pair].reverse())) };
  }
  validateEpisodeModel(model);
  return model;
}
