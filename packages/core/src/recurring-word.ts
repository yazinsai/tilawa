import { PosteriorFrames } from "./ctc.js";
import type { TopKPosteriors } from "./evidence.js";

export type PosteriorTrajectory = readonly (readonly number[])[];
export interface WordOccurrence {
  pronunciation: string;
  fromFrame: number;
  toFrame: number;
  trajectory: PosteriorTrajectory;
}
export interface OccurrenceStep {
  /** Global word ordinal, used solely to establish in-order bracketing. */
  word: number;
  pronunciation: string;
  run: string;
  fromFrame: number;
  /** A single committed candidate, with positive fit, outside any hold or relocation. */
  unambiguous: boolean;
  /** Local raw-cost separation may arrive inside the word, after its first committed phoneme. */
  supported?: boolean;
}
export interface OccurrenceCalibration {
  version: 1;
  feature: "posterior-subsequence-dtw-v1";
  trainingOnly: true;
  speakers: number;
  folds: number;
  pairs: number;
  /** Speaker-separated training logistic calibration, in nats per distance. */
  scaleNats: number;
}
export interface OccurrenceComparison {
  fromFrame: number;
  toFrame: number;
  continuation: string;
  restart: string;
  continuationExemplars: number;
  restartExemplars: number;
  continuationDistance: number | null;
  restartDistance: number | null;
  /** Relative log likelihood favouring continuation, or null without frozen calibration. */
  continuationLogLikelihood: number | null;
}

function normalized(row: readonly number[]): number[] {
  if (!row.length || row.some(p => !Number.isFinite(p) || p < 0)) throw new Error("Invalid posterior row");
  const mass = row.reduce((a, b) => a + b, 0);
  if (!(mass > 0)) throw new Error("Empty posterior mass");
  return row.map(p => p / mass);
}

/** Squared Hellinger distance retains every posterior class, including blank. */
function frameDistance(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) throw new Error("Posterior inventories disagree");
  let overlap = 0;
  for (let i = 0; i < a.length; i++) overlap += Math.sqrt(a[i] * b[i]);
  return Math.max(0, 1 - Math.min(1, overlap));
}

/**
 * Match the complete query against a nonempty subsequence of an earlier complete rendition. Free exemplar
 * prefix/suffix permit a streaming word prefix; all query frames are consumed. Cost is the mean along the
 * minimum cumulative-cost DTW path, with path length breaking exact cumulative-cost ties. No phone collapse.
 */
export function subsequenceDtw(query: PosteriorTrajectory, exemplar: PosteriorTrajectory): number | null {
  if (!query.length || !exemplar.length) return null;
  const q = query.map(normalized), r = exemplar.map(normalized);
  const n = r.length;
  let prev = new Float64Array(n + 1);
  let lengths = new Uint32Array(n + 1);
  for (let i = 0; i < q.length; i++) {
    const row = new Float64Array(n + 1).fill(Infinity);
    const lens = new Uint32Array(n + 1);
    for (let j = 1; j <= n; j++) {
      let cost = prev[j - 1], length = lengths[j - 1];
      for (const [c, l] of [[prev[j], lengths[j]], [row[j - 1], lens[j - 1]]]) {
        if (c < cost || (c === cost && l < length)) { cost = c; length = l; }
      }
      row[j] = cost + frameDistance(q[i], r[j - 1]);
      lens[j] = length + 1;
    }
    prev = row;
    lengths = lens;
  }
  let best = 1;
  for (let j = 2; j <= n; j++) if (prev[j] < prev[best] || (prev[j] === prev[best] && lengths[j] < lengths[best])) best = j;
  return prev[best] / lengths[best];
}

export function validateOccurrenceCalibration(model: OccurrenceCalibration): void {
  if (model.version !== 1 || model.feature !== "posterior-subsequence-dtw-v1" || model.trainingOnly !== true
    || !Number.isInteger(model.speakers) || model.speakers < 2 || !Number.isInteger(model.folds) || model.folds < 2
    || model.folds > model.speakers || !Number.isInteger(model.pairs) || model.pairs < 1
    || !Number.isFinite(model.scaleNats) || model.scaleNats < 0) throw new Error("Invalid training-frozen occurrence calibration");
}

/**
 * Session-local memory indexed only by pronunciation. A middle word is saved after a subsequent in-order
 * commit brackets it. Stored rows and comparison results are owned snapshots; live evidence cannot mutate them.
 */
export class WordOccurrenceMemory {
  private readonly frames: PosteriorFrames;
  private readonly occurrences = new Map<string, WordOccurrence[]>();
  private steps: OccurrenceStep[] = [];
  private publishedTo = -1;
  private comparisonFrom: number | null = null;

  constructor(size: number, private readonly onOccurrence?: (occurrence: WordOccurrence) => void) {
    this.frames = new PosteriorFrames(size);
  }

  append(rows: TopKPosteriors): void {
    this.frames.append(rows);
    if (!this.steps.length) this.frames.dropBefore(Math.min(rows.startFrame, this.comparisonFrom ?? rows.startFrame));
  }

  /** The disputed word can have an unclear commit and still needs its entire live acoustic interval. */
  retainFrom(frame: number | null): void { this.comparisonFrom = frame; }

  trajectory(from: number, to: number): PosteriorTrajectory | null {
    if (!Number.isInteger(from) || !Number.isInteger(to) || to <= from
      || from < this.frames.startFrame || to > this.frames.endFrame) return null;
    const out: number[][] = [];
    for (let f = from; f < to; f++) out.push(Array.from(this.frames.row(f), Math.exp));
    return out;
  }

  /** Clear only pending bracketing; already established earlier renditions stay valid after movement. */
  interrupt(): void { this.steps = []; }

  commit(step: OccurrenceStep): void {
    if (!step.unambiguous || !step.pronunciation || !Number.isInteger(step.fromFrame)) { this.interrupt(); return; }
    const last = this.steps.at(-1);
    if (last && step.word === last.word && step.run === last.run) return;
    if (last && (last.run !== step.run || last.word + 1 !== step.word || last.fromFrame >= step.fromFrame)) this.steps = [];
    this.steps.push({ ...step, supported: step.supported ?? true });
    if (this.steps.length > 3) this.steps.shift();
    this.admit();
    const pending = this.steps[0].fromFrame;
    this.frames.dropBefore(Math.min(pending, this.comparisonFrom ?? pending));
  }

  /** Require a separated raw fit inside every bracket word, rather than demanding it on an ambiguous onset. */
  support(word: number, separated: boolean): void {
    const current = this.steps.at(-1);
    if (current?.word !== word || !separated) return;
    current.supported = true;
    this.admit();
  }

  private admit(): void {
    if (this.steps.length !== 3 || !this.steps.every(s => s.supported)) return;
    const [, b, c] = this.steps;
    if (c.fromFrame <= this.publishedTo) return;
    const trajectory = this.trajectory(b.fromFrame, c.fromFrame);
    if (trajectory) {
      const occurrence: WordOccurrence = { pronunciation: b.pronunciation, fromFrame: b.fromFrame, toFrame: c.fromFrame, trajectory };
      const list = this.occurrences.get(b.pronunciation) ?? [];
      list.push(occurrence);
      // Two complete earlier renditions per pronunciation bound retained acoustic memory.
      if (list.length > 2) list.shift();
      this.occurrences.set(b.pronunciation, list);
      this.publishedTo = c.fromFrame;
      this.onOccurrence?.(structuredClone(occurrence));
    }
  }

  get currentStart(): number | null { return this.steps.at(-1)?.fromFrame ?? null; }

  compare(continuation: string, restart: string, fromFrame: number, toFrame: number, model?: OccurrenceCalibration): OccurrenceComparison {
    if (model) validateOccurrenceCalibration(model);
    const query = this.trajectory(fromFrame, toFrame);
    const eligible = (pronunciation: string) => (this.occurrences.get(pronunciation) ?? []).filter(e => e.toFrame <= fromFrame);
    const a = eligible(continuation), b = eligible(restart);
    const result: OccurrenceComparison = { fromFrame, toFrame, continuation, restart,
      continuationExemplars: a.length, restartExemplars: b.length,
      continuationDistance: null, restartDistance: null, continuationLogLikelihood: null };
    if (!query || !a.length || !b.length || continuation === restart) return result;
    const distance = (items: WordOccurrence[]) => items.reduce((sum, e) => sum + subsequenceDtw(query, e.trajectory)!, 0) / items.length;
    result.continuationDistance = distance(a);
    result.restartDistance = distance(b);
    if (model) result.continuationLogLikelihood = model.scaleNats * (result.restartDistance - result.continuationDistance);
    return result;
  }
}
