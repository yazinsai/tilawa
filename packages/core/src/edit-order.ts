import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

/** The recurrent state is the preceding edit operation, including reference word boundaries. */
export const EDIT_OPERATIONS = ["match", "sub", "ins", "del", "boundary"] as const;
export type EditOperation = typeof EDIT_OPERATIONS[number];
export const EDIT_SCRIPTS = 16;

export interface EditScript {
  /** Ordered operations are immutable and shared by descendants. */
  readonly previous: EditScript | null;
  readonly operation: EditOperation | null;
  readonly counts: readonly number[];
  readonly transitions: readonly number[];
  readonly last: number;
  readonly length: number;
}

export function emptyEditScript(): EditScript {
  return { previous: null, operation: null, counts: [0, 0, 0, 0, 0], transitions: Array(25).fill(0), last: -1, length: 0 };
}

export function appendEdit(script: EditScript, operation: EditOperation): EditScript {
  const next = EDIT_OPERATIONS.indexOf(operation);
  const counts = [...script.counts], transitions = [...script.transitions];
  counts[next]++;
  if (script.last >= 0) transitions[script.last * 5 + next]++;
  return { previous: script, operation, counts, transitions, last: next, length: script.length + 1 };
}

export function editSequence(script: EditScript): EditOperation[] {
  const result: EditOperation[] = [];
  for (let p: EditScript | null = script; p?.operation; p = p.previous) result.push(p.operation);
  return result.reverse();
}

/** Center on uniform permutations with exactly these counts; totals and length supply no mean advantage. */
export function orderFeatures(script: Pick<EditScript, "counts" | "transitions" | "length">): number[] {
  const n = script.length;
  return script.transitions.map((value, k) => {
    const a = Math.floor(k / 5), b = k % 5;
    const expected = n ? script.counts[a] * (script.counts[b] - (a === b ? 1 : 0)) / n : 0;
    return value - expected;
  });
}

export interface EditOrderModel {
  version: 1;
  /** Same 25-parameter recurrent architecture in both arms. */
  ordered: readonly number[];
  shuffled: readonly number[];
  /** Frozen speaker-separated training calibration, in log likelihood ratio units. */
  scale: number;
}

export function validateEditOrderModel(model: EditOrderModel): void {
  if (model.version !== 1 || model.ordered.length !== 25 || model.shuffled.length !== 25
    || ![...model.ordered, ...model.shuffled, model.scale].every(Number.isFinite) || model.scale < 0)
    throw new Error("Invalid edit-order model");
}

export function editOrderLogRatio(script: EditScript, model: EditOrderModel, mode: "ordered" | "shuffled"): number {
  const features = mode === "ordered" ? orderFeatures(script) : shuffledOrderFeatures(script.counts);
  const weights = mode === "ordered" ? model.ordered : model.shuffled;
  return model.scale * features.reduce((sum, x, k) => sum + x * weights[k], 0);
}

/** Shared count-keyed Fisher-Yates permutation: matched pairs have identical control evidence. */
export function shuffledOrderFeatures(counts: readonly number[]): number[] {
  let seed = 2166136261;
  const sequence: number[] = [];
  counts.forEach((n, k) => { seed = Math.imul(seed ^ n, 16777619) >>> 0; for (let i = 0; i < n; i++) sequence.push(k); });
  for (let i = sequence.length - 1; i > 0; i--) {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    const j = (seed >>> 0) % (i + 1);
    [sequence[i], sequence[j]] = [sequence[j], sequence[i]];
  }
  const transitions = Array(25).fill(0);
  for (let i = 1; i < sequence.length; i++) transitions[sequence[i - 1] * 5 + sequence[i]]++;
  return orderFeatures({ counts, transitions, length: sequence.length });
}

export interface ScoredEditScript { script: EditScript; cost: number }

/** Marginalize the retained ambiguous scripts with normalized original acoustic weights. */
export function marginalOrderLogRatio(paths: readonly ScoredEditScript[], model: EditOrderModel, mode: "ordered" | "shuffled"): number {
  const logsum = (xs: number[]) => {
    const max = Math.max(...xs);
    return max + Math.log(xs.reduce((s, x) => s + Math.exp(x - max), 0));
  };
  if (!paths.length) return 0;
  const base = paths.map(p => -3 * p.cost / UNIT);
  return logsum(paths.map((p, i) => base[i] + editOrderLogRatio(p.script, model, mode))) - logsum(base);
}

/** Bounded k-shortest directed edit scripts; reference and observation endpoints are both explicit. */
export function orderedAlignments(reference: string, heard: string, boundaries: ReadonlySet<number>, limit = EDIT_SCRIPTS): ScoredEditScript[] {
  type Path = ScoredEditScript & { sequence: string };
  const append = (p: Path, op: EditOperation, charge: number, boundary = false): Path => {
    let script = p.script;
    if (boundary) script = appendEdit(script, "boundary");
    script = appendEdit(script, op);
    return { script, cost: p.cost + charge, sequence: p.sequence + (boundary ? "4" : "") + EDIT_OPERATIONS.indexOf(op) };
  };
  const trim = (paths: Path[]): Path[] => {
    paths.sort((a, b) => a.cost - b.cost || a.sequence.localeCompare(b.sequence));
    const seen = new Set<string>();
    return paths.filter(p => { if (seen.has(p.sequence)) return false; seen.add(p.sequence); return true; }).slice(0, limit);
  };
  let row: Path[][] = Array.from({ length: reference.length + 1 }, () => []);
  row[0] = [{ script: emptyEditScript(), cost: 0, sequence: "" }];
  for (let i = 1; i < row.length; i++) row[i] = row[i - 1].map(p => append(p, "del", deletionCost(reference, i - 1), boundaries.has(i - 1)));
  for (let j = 0; j < heard.length; j++) {
    const next: Path[][] = Array.from({ length: row.length }, () => []);
    next[0] = row[0].map(p => append(p, "ins", insertionCost(reference, 0, heard[j])));
    for (let i = 1; i < row.length; i++) {
      const boundary = boundaries.has(i - 1);
      next[i] = trim([
        ...row[i].map(p => append(p, "ins", insertionCost(reference, i, heard[j]))),
        ...row[i - 1].map(p => append(p, reference[i - 1] === heard[j] ? "match" : "sub", subCost(reference[i - 1], heard[j]), boundary)),
        ...next[i - 1].map(p => append(p, "del", deletionCost(reference, i - 1), boundary)),
      ]);
    }
    row = next;
  }
  return row.at(-1)!;
}

export interface EditOrderComparison {
  previous: number;
  weightedSelected: number;
  selected: number;
  observations: number;
  candidates: { offset: number; origin: number; cost: number; ordered: number; shuffled: number; scripts: number; counts: readonly number[] }[];
}
export interface LocalEditOrderOptions {
  model: EditOrderModel;
  mode: "audit" | "ordered" | "shuffled";
  onComparison?: (comparison: EditOrderComparison) => void;
}
