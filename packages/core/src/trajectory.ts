import { UNIT, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

export const TRAJECTORY_FEATURES = [
  "insert", "consume", "restart", "delete", "acoustic",
  "heardVowel", "heardNasal", "heardEmphatic", "writtenVowel", "writtenNasal", "writtenEmphatic",
  "exact", "soft", "durationRun", "wordStart", "wordEnd", "wordLength",
  "wordMovement", "movementLength", "ayahCrossing", "phoneMovement",
  "restartExact", "restartSoft", "restartNasal",
] as const;

export interface TrajectoryModel {
  version: 1;
  features: readonly string[];
  /** Additive edit-unit potentials, selected exclusively on training speakers. */
  weights: readonly number[];
}

/** One causal graph layer. All four original arc families, including reference-only deletion closure. */
export interface TrajectoryFrame {
  lo: number;
  previousLo: number;
  previousSize: number;
  source: number;
  /** Per destination cell, insert / consume / restart / delete. Null means the arc does not exist. */
  arcs: (null | { cost: number; x: number[] })[][];
}

export interface TrajectoryOptions {
  model?: TrajectoryModel | null;
  observe?: (frame: TrajectoryFrame) => void;
}

export function validateTrajectoryModel(model: TrajectoryModel): void {
  if (model.version !== 1 || model.features.length !== TRAJECTORY_FEATURES.length
    || model.features.some((f, i) => f !== TRAJECTORY_FEATURES[i])
    || model.weights.length !== TRAJECTORY_FEATURES.length || model.weights.some(w => !Number.isFinite(w))) {
    throw new Error("Invalid conditional trajectory model");
  }
}

const vowel = (p: string) => "َُِاۦۥ".includes(p) && !!p;
const nasal = (p: string) => "نمں۾".includes(p) && !!p;
const emphatic = (p: string) => "صضطظق".includes(p) && !!p;

/** Shared phone features and relative movement; no passage, word, recording or speaker identifiers. */
export function trajectoryFeatures(index: ReferenceIndex, kind: number, from: number, to: number,
  heard: string, acousticCost: number): number[] {
  const at = kind === 0 ? Math.min(to, index.phonemes.length - 1) : to - 1;
  const written = index.phonemes[at] ?? "";
  const wi = index.wordAt(at), w = index.words[wi];
  const source = index.words[index.wordAt(Math.max(0, from - 1))];
  const movement = kind === 2 ? wi - index.wordAt(Math.max(0, from - 1)) : 0;
  const exact = Number(kind !== 3 && written === heard);
  const soft = Number(kind !== 3 && !exact && subCost(written, heard) < UNIT);
  const jump = Number(kind === 2);
  return [
    ...[0, 1, 2, 3].map(k => Number(kind === k)), acousticCost / UNIT,
    Number(vowel(heard)), Number(nasal(heard)), Number(emphatic(heard)),
    Number(vowel(written)), Number(nasal(written)), Number(emphatic(written)), exact, soft,
    Number(at > 0 && index.phonemes[at - 1] === written || index.phonemes[at + 1] === written),
    Number(at === w.start), Number(to === w.end), Math.log1p(w.end - w.start) / 5,
    Math.tanh(movement / 8), Math.log1p(Math.abs(movement)) / 5,
    jump * Number(source.surah !== w.surah || source.ayah !== w.ayah),
    jump * Math.tanh((to - 1 - from) / 16), jump * exact, jump * soft, jump * Number(nasal(heard)),
  ];
}

export function trajectoryPotential(model: TrajectoryModel | null | undefined, features: readonly number[]): number {
  return model ? features.reduce((sum, x, i) => sum + x * model.weights[i], 0) * UNIT : 0;
}
