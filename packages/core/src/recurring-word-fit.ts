import { subsequenceDtw } from "./recurring-word.js";
import type { OccurrenceCalibration, PosteriorTrajectory } from "./recurring-word.js";

export interface TrainingOccurrence {
  speaker: string;
  recording: string;
  pronunciation: string;
  trajectory: PosteriorTrajectory;
  fromFrame?: number;
  toFrame?: number;
}
interface Pair { speaker: string; contrast: number }

function phoneDistance(a: string, b: string): number {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) next[j] = Math.min(next[j - 1] + 1, row[j] + 1, row[j - 1] + Number(a[i - 1] !== b[j - 1]));
    row = next;
  }
  return row[b.length] / Math.max(a.length, b.length);
}

/** Same-speaker history is another recording or a completed earlier interval in this recording. */
function pairs(occurrences: readonly TrainingOccurrence[]): Pair[] {
  const out: Pair[] = [];
  for (const query of occurrences) {
    const history = occurrences.filter(x => x.speaker === query.speaker && (x.recording !== query.recording
      || query.fromFrame !== undefined && x.toFrame !== undefined && x.toFrame <= query.fromFrame));
    const same = history.filter(x => x.pronunciation === query.pronunciation).slice(0, 2);
    if (!same.length) continue;
    const competitors = [...new Set(history.map(x => x.pronunciation))].filter(p => p !== query.pronunciation);
    competitors.sort((a, b) => phoneDistance(query.pronunciation, a) - phoneDistance(query.pronunciation, b) || a.localeCompare(b));
    const other = history.filter(x => x.pronunciation === competitors[0]).slice(0, 2);
    if (!other.length) continue;
    // Streaming prefixes, including the full word, train one shared scale rather than a duration-specific table.
    for (const fraction of [0.25, 0.5, 0.75, 1]) {
      const trajectory = query.trajectory.slice(0, Math.max(1, Math.ceil(query.trajectory.length * fraction)));
      const distance = (examples: TrainingOccurrence[]) => examples.reduce((sum, e) => sum + subsequenceDtw(trajectory, e.trajectory)!, 0) / examples.length;
      const contrast = distance(other) - distance(same);
      if (!Number.isFinite(contrast)) throw new Error("Nonfinite occurrence training feature");
      out.push({ speaker: query.speaker, contrast });
    }
  }
  return out;
}

const softplus = (x: number) => Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
function loss(data: readonly Pair[], scale: number): number {
  const speakers = new Map<string, number[]>();
  for (const p of data) {
    const list = speakers.get(p.speaker) ?? [];
    list.push(softplus(-scale * p.contrast));
    speakers.set(p.speaker, list);
  }
  return [...speakers.values()].reduce((sum, xs) => sum + xs.reduce((a, b) => a + b, 0) / xs.length, 0) / speakers.size;
}
function fit(data: readonly Pair[], variance: number): number {
  const objective = (s: number) => loss(data, s) + s * s / (2 * variance);
  // Convex one-dimensional objective; the L2 prior gives a finite search upper bound.
  let lo = 0, hi = 1;
  while (objective(hi) < objective(hi / 2)) hi *= 2;
  for (let i = 0; i < 80; i++) {
    const a = (2 * lo + hi) / 3, b = (lo + 2 * hi) / 3;
    if (objective(a) < objective(b)) hi = b; else lo = a;
  }
  const scale = (lo + hi) / 2;
  return objective(0) <= objective(scale) ? 0 : scale;
}

/** No practice input is accepted. Each calibration fold contains every take and exemplar from its speakers. */
export function fitOccurrenceCalibration(occurrences: readonly TrainingOccurrence[], authorization: { trainingOnly: true }): {
  model: OccurrenceCalibration;
  validation: { variance: number; nll: number; foldNll: number[] }[];
} {
  if (authorization.trainingOnly !== true) throw new Error("Training authorization required");
  if (occurrences.some(x => !x.speaker || !x.recording || !x.pronunciation || !x.trajectory.length)) throw new Error("Incomplete training occurrence");
  const examples = pairs(occurrences);
  const speakers = [...new Set(examples.map(x => x.speaker))].sort();
  if (speakers.length < 2) throw new Error("Calibration needs repeated pronunciations and competitors for at least two training speakers");
  const folds = Math.min(5, speakers.length);
  const foldOf = new Map(speakers.map((s, i) => [s, i % folds]));
  const validation = [0.25, 1, 4, 16].map(variance => {
    const foldNll: number[] = [];
    for (let fold = 0; fold < folds; fold++) {
      const train = examples.filter(p => foldOf.get(p.speaker) !== fold);
      const held = examples.filter(p => foldOf.get(p.speaker) === fold);
      foldNll.push(loss(held, fit(train, variance)));
    }
    return { variance, foldNll, nll: foldNll.reduce((a, b) => a + b, 0) / folds };
  });
  const selected = [...validation].sort((a, b) => a.nll - b.nll || a.variance - b.variance)[0];
  return { model: { version: 1, feature: "posterior-subsequence-dtw-v1", trainingOnly: true,
    speakers: speakers.length, folds, pairs: examples.length, scaleNats: fit(examples, selected.variance) }, validation };
}
