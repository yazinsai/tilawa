import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { editParameterCounter, parameterTrajectoryDecision } from "./edit-parameter.js";
import type { EditParameterPosterior, EditStatistics } from "./edit-parameter.js";

/**
 * Jumping to another word start: back for a repeat or self-correction, ahead for a skip. Both cost the same, so where
 * a landing word's opening also occurs earlier in the window the audio decides, not the direction.
 */
const JUMP_COST = 2 * UNIT;
/** Per phoneme of jump distance; small enough to only break ties, in favour of the nearer landing. */
const JUMP_DISTANCE = 1e-3;
/** The window spans this many ayahs before and after the bookmark's ayah. */
const BACK_AYAHS = 2;
const AHEAD_AYAHS = 2;
/**
 * Entering the next surah: reciters say a basmala between surahs, which the tracker hands to global search, so what
 * reaches a surah's end here (a takbir, a new passage) shouldn't slide into the next surah's opening.
 */
const SURAH_ENTRY = 3 * UNIT;
/** Recent cost is averaged over this many observed phonemes. */
const RECENT = 16;

export interface LocalParameterTrajectory {
  offset: number;
  origin: number;
  edits: number;
  /** Raw existing weighted cost over the whole interval since this local search began. */
  acousticCost: number;
  statistics: EditStatistics;
}

export interface LocalParameterComparison {
  previous: { offset: number; origin: number };
  weightedSelected: number;
  trajectories: LocalParameterTrajectory[];
  decision: NonNullable<ReturnType<typeof parameterTrajectoryDecision>>;
}

export interface LocalParameterOptions {
  posterior: EditParameterPosterior;
  /** Audit preserves the original search; mean isolates the same channel without parameter integration. */
  mode?: "audit" | "mean" | "marginal";
  onComparison?: (comparison: LocalParameterComparison) => void;
}

export interface LocalPosition {
  /** Reference phonemes consumed up to here (exclusive end of the last matched phoneme). */
  offset: number;
  /** Where the current contiguous run began; resets on a jump. */
  origin: number;
  /** Unweighted edits along the run, for its PER. */
  edits: number;
  /** Weighted cost per observed phoneme over the last `RECENT` phonemes, in plain edits. */
  recentCost: number;
  /** Weighted cost of the last observed phoneme, in plain edits. */
  stepCost: number;
}

export interface LocalSearch {
  /** Consume one observed phoneme. */
  step(p: string): void;
  position(): LocalPosition;
  /** The search state, to rewind to with `restore`. */
  save(): unknown;
  restore(state: unknown): void;
}

/**
 * Streaming alignment around a bookmark: one DP column over a window of nearby ayahs, advanced one observed phoneme at
 * a time, with jump arcs from the best state to any word start in the window. Work per phoneme is bounded by the window.
 */
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">,
  parameters?: LocalParameterOptions): LocalSearch {
  const ref = index.phonemes;
  const ayahOfWord: number[] = [];
  const ayahStart: number[] = [];
  const ayahEnd: number[] = [];
  index.words.forEach((w, i) => {
    const prev = index.words[i - 1];
    if (!prev || prev.surah !== w.surah || prev.ayah !== w.ayah) ayahStart.push(w.start);
    ayahOfWord.push(ayahStart.length - 1);
    ayahEnd[ayahStart.length - 1] = w.end;
  });
  const wordStarts = new Set(index.words.map((w) => w.start));
  // At-Tawba is the one surah entered without a basmala.
  const surahFirst = new Set(index.words.filter((w, i) => i > 0 && w.surah !== index.words[i - 1].surah && w.surah !== 9).map((w) => w.start));
  const surahOf = (offset: number) => index.words[index.wordAt(offset)].surah;
  const entry = (offset: number) => (surahFirst.has(offset) ? SURAH_ENTRY : 0);
  const crossing = (offset: number) => (surahOf(offset) === surahOf(Math.max(0, best - 1)) ? 0 : SURAH_ENTRY);
  const ayahOf = (offset: number) => ayahOfWord[index.wordAt(Math.max(0, offset - 1))];

  let lo = 0;
  let cost = new Float64Array(0);
  let origin = new Int32Array(0);
  let edits = new Int32Array(0);
  // Conditional on the original weighted representative at each cell, not a marginal over edit alignments.
  let statistics: (EditStatistics | undefined)[] = [];
  const parameterPosterior = parameters ? { ...parameters.posterior, phones: [...parameters.posterior.phones],
    operations: parameters.posterior.operations.map(row => [...row]), insertion: [...parameters.posterior.insertion] as [number, number],
    substituted: parameters.posterior.substituted.map(row => [...row]), inserted: [...parameters.posterior.inserted] } : null;
  const counter = parameterPosterior ? editParameterCounter(parameterPosterior) : null;
  let accumulated = 0;
  let best = start.offset;
  let ayah = -1;
  const recent: number[] = [];

  const recentre = (offset: number) => {
    const a = ayahOf(offset);
    if (a === ayah) return;
    ayah = a;
    const newLo = ayahStart[Math.max(0, a - BACK_AYAHS)];
    const newHi = ayahEnd[Math.min(ayahEnd.length - 1, a + AHEAD_AYAHS)];
    const c = new Float64Array(newHi - newLo + 1).fill(Infinity);
    const o = new Int32Array(c.length);
    const e = new Int32Array(c.length);
    const t: typeof statistics = counter ? Array(c.length) : [];
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        if (counter) t[i] = statistics[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    statistics = t;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;
  if (counter) statistics[start.offset - lo] = counter.empty();


  return {
    step(p) {
      const n = cost.length;
      const base = cost[best - lo];
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const t: typeof statistics = counter ? Array(n) : [];
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        let bt = counter && statistics[i] ? counter.extend(statistics[i]!,
          { kind: "insert", heard: p, duration: insertionCost(ref, at, p) === 2 }) : undefined;
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            bt = counter && statistics[i - 1] ? counter.extend(statistics[i - 1]!,
              { kind: "diagonal", reference: ref[at - 1], heard: p }, entry(at - 1)) : undefined;
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              bt = counter && statistics[best - lo] ? counter.extend(statistics[best - lo]!,
                { kind: "diagonal", reference: ref[at - 1], heard: p }, j - base - subCost(ref[at - 1], p)) : undefined;
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            bt = counter && t[i - 1] ? counter.extend(t[i - 1]!,
              { kind: "delete", reference: ref[at - 1], duration: deletionCost(ref, at - 1) === 2 }, entry(at - 1)) : undefined;
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        if (counter) t[i] = bt;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      if (parameters && o[k] !== origin[best - lo]) {
        const trajectories: LocalParameterTrajectory[] = [];
        const cells: number[] = [];
        for (let i = 0; i < n; i++) if (Number.isFinite(c[i]) && t[i]) {
          cells.push(i);
          trajectories.push({ offset: lo + i, origin: o[i], edits: e[i], acousticCost: c[i] + accumulated, statistics: t[i]! });
        }
        const decision = parameterTrajectoryDecision(parameterPosterior!, trajectories.map(p => p.statistics));
        if (decision) {
          const comparison = { previous: { offset: best, origin: origin[best - lo] }, weightedSelected: cells.indexOf(k), trajectories, decision };
          if (parameters.mode !== "audit") k = cells[(parameters.mode === "mean" ? decision.mean : decision.marginal).selected];
          // The hook receives owned statistics, so retained DP history cannot be changed by an audit consumer.
          parameters.onComparison?.({ ...comparison, trajectories: trajectories.map(p => ({ ...p,
            statistics: { ...p.statistics, counts: [...p.statistics.counts] } })) });
        }
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      statistics = t;
      accumulated += min;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, statistics, accumulated, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits;
        statistics: typeof statistics; accumulated: number; best: number; ayah: number; recent: number[] };
      ({ lo, cost, origin, edits, statistics, accumulated, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
