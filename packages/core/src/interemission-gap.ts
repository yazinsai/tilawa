import type { DecodedPhoneme } from "./evidence.js";
import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Logarithmic bins on the listener's frame grid; the last bin includes every longer interval. */
export const GAP_BINS = 8;
export interface GapContext { left: string; right: string; between: string; boundary: boolean }
export interface GapObservation extends GapContext { frames: number; weight: number }
export interface GapCounts { pooled: number[]; contexts: Record<string, number[]>; mass: number }
const vowel = new Set("َُِاۦۥ۪ٲ");
const sonorant = new Set("لممنں۾وي");
const kind = (p: string) => !p ? "_" : vowel.has(p) ? "V" : sonorant.has(p) ? "S" : "C";
export const gapBin = (frames: number) => Math.min(GAP_BINS - 1, Math.floor(Math.log2(Math.max(0, frames) + 1)));

/** Hierarchy: erased-phone composition/boundary, flank classes, exact flanks, exact erased sequence. */
export function gapKeys(c: GapContext): string[] {
  const erased = c.between.length > 3 ? "long" : [...c.between].map(kind).join("");
  const base = `${c.boundary ? 1 : 0}/${erased}`;
  return [base, `${base}/${kind(c.left)}:${kind(c.right)}`, `${base}/${c.left}:${c.right}`,
    `${base}/${c.left}:${c.right}/${c.between.length > 3 ? "long" : c.between}`];
}

export function fitGapCounts(observations: readonly GapObservation[], mass: number): GapCounts {
  const out: GapCounts = { pooled: Array(GAP_BINS).fill(0), contexts: {}, mass };
  for (const o of observations) {
    const b = gapBin(o.frames);
    out.pooled[b] += o.weight;
    for (const k of new Set(gapKeys(o))) (out.contexts[k] ??= Array(GAP_BINS).fill(0))[b] += o.weight;
  }
  return out;
}

/** Normalized hierarchical prediction, expressed relative to the pooled gap law. */
export function gapLogRatio(model: GapCounts, c: GapContext, frames: number): number {
  if (!Number.isFinite(frames) || frames < 0) return 0;
  const b = gapBin(frames);
  const total = model.pooled.reduce((a, x) => a + x, 0);
  let p = ((model.pooled[b] ?? 0) + 0.5) / (total + GAP_BINS * 0.5);
  const pooled = p;
  for (const k of new Set(gapKeys(c))) {
    const row = model.contexts[k];
    if (!row) continue;
    const n = row.reduce((a, x) => a + x, 0);
    p = (row[b] + model.mass * p) / (n + model.mass);
  }
  return Math.log(p / pooled);
}

export const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
};
/** Existing tracker conversion: three nats carry the weight of one ordinary edit. */
const EDIT_NATS = 3 / UNIT;

/**
 * A reference endpoint immediately after an emission (including insertion). Deletions are traversed before the
 * next emission, making all their placements explicit. No duration symbol is collapsed or forgiven.
 */
export function gapContext(reference: string, boundaries: ReadonlySet<number>, from: number, to: number, insertion: boolean): GapContext {
  return { left: reference[from - 1] ?? "", right: insertion ? "" : reference[to] ?? "", between: reference.slice(from, to),
    boundary: [...boundaries].some(b => b >= from && b <= to) };
}

interface Arc { from: number; to: number; logWeight: number; context: GapContext }
function arcs(reference: string, boundaries: ReadonlySet<number>, heard: string): Arc[] {
  const out: Arc[] = [];
  for (let from = 0; from <= reference.length; from++) {
    let deleted = 0;
    for (let to = from; to <= reference.length; to++) {
      const ins = gapContext(reference, boundaries, from, to, true);
      out.push({ from, to, logWeight: -(deleted + insertionCost(reference, to, heard)) * EDIT_NATS, context: ins });
      if (to < reference.length) {
        out.push({ from, to: to + 1, logWeight: -(deleted + subCost(reference[to], heard)) * EDIT_NATS,
          context: gapContext(reference, boundaries, from, to, false) });
        deleted += deletionCost(reference, to);
      }
    }
  }
  return out;
}

/** Gap likelihood ratio marginalized over the same edit placements in numerator and denominator. */
export function gapAlignmentRatio(reference: string, boundaries: ReadonlySet<number>, heard: readonly DecodedPhoneme[], model: GapCounts,
  endpoint = reference.length): number {
  const width = reference.length + 1;
  let plain = new Float64Array(width).fill(-Infinity), joint = plain.slice();
  plain[0] = joint[0] = 0;
  const byPhone = new Map<string, Arc[]>();
  for (let i = 0; i < heard.length; i++) {
    const nextPlain = new Float64Array(width).fill(-Infinity), nextJoint = nextPlain.slice();
    const gap = i ? heard[i].start - heard[i - 1].end - 1 : NaN;
    const p = heard[i].p;
    if (!byPhone.has(p)) byPhone.set(p, arcs(reference, boundaries, p));
    for (const a of byPhone.get(p)!) {
      nextPlain[a.to] = logAdd(nextPlain[a.to], plain[a.from] + a.logWeight);
      nextJoint[a.to] = logAdd(nextJoint[a.to], joint[a.from] + a.logWeight + gapLogRatio(model, a.context, gap));
    }
    plain = nextPlain; joint = nextJoint;
  }
  // Reference-only terminal transitions still pay their existing costs, but generate no additional gap.
  let deleted = 0, plainEnd = -Infinity, jointEnd = -Infinity;
  for (let j = endpoint; j >= 0; j--) {
    plainEnd = logAdd(plainEnd, plain[j] - deleted * EDIT_NATS);
    jointEnd = logAdd(jointEnd, joint[j] - deleted * EDIT_NATS);
    if (j > 0) deleted += deletionCost(reference, j - 1);
  }
  return jointEnd - plainEnd;
}

/** Training-only edit posterior. Each physical gap receives exactly one unit of expected context count. */
export function gapTrainingObservations(reference: string, boundaries: ReadonlySet<number>, heard: readonly DecodedPhoneme[]): GapObservation[] {
  const width = reference.length + 1;
  const rows = [new Float64Array(width).fill(-Infinity)];
  rows[0][0] = 0;
  const byPhone = new Map<string, Arc[]>();
  const get = (p: string) => { if (!byPhone.has(p)) byPhone.set(p, arcs(reference, boundaries, p)); return byPhone.get(p)!; };
  for (const h of heard) {
    const prev = rows.at(-1)!;
    const row = new Float64Array(width).fill(-Infinity);
    for (const a of get(h.p)) row[a.to] = logAdd(row[a.to], prev[a.from] + a.logWeight);
    rows.push(row);
  }
  // Training clips are full ayahs: trailing phones also incur the existing deletion costs.
  let back = new Float64Array(width);
  for (let j = reference.length - 1; j >= 0; j--) back[j] = back[j + 1] - deletionCost(reference, j) * EDIT_NATS;
  let z = -Infinity;
  for (let j = 0; j < width; j++) z = logAdd(z, rows.at(-1)![j] + back[j]);
  const out = new Map<string, GapObservation>();
  for (let i = heard.length - 1; i >= 0; i--) {
    const before = new Float64Array(width).fill(-Infinity);
    const frames = i ? heard[i].start - heard[i - 1].end - 1 : -1;
    for (const a of get(heard[i].p)) {
      const suffix = a.logWeight + back[a.to];
      before[a.from] = logAdd(before[a.from], suffix);
      if (frames >= 0) {
        const weight = Math.exp(rows[i][a.from] + suffix - z);
        // Discard only numerical underflow, never an edit route with a finite contribution.
        if (weight > 0) {
          const key = `${gapKeys(a.context).at(-1)}/${gapBin(frames)}`;
          const existing = out.get(key);
          if (existing) existing.weight += weight;
          else out.set(key, { ...a.context, frames, weight });
        }
      }
    }
    back = before;
  }
  return [...out.values()];
}
