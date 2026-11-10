import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { addLogProbability, pathChannelLogs, positionSequenceMedoid } from "./path-risk.js";
import type { PathChannel, PositionTrajectory } from "./path-risk.js";

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

interface RetainedTrajectory extends PositionTrajectory {
  /** Cheapest existing edit alignment representing this position sequence at this cell. */
  acousticCost: number;
  origin: number;
  edits: number;
}
type TrajectoryCell = Map<string, RetainedTrajectory>;

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
  /** Complete paths retained at every finite local cell, on the common interval since this search began. */
  trajectories(): (PositionTrajectory & { offset: number; origin: number })[];
}

/**
 * Streaming alignment around a bookmark: one DP column over a window of nearby ayahs, advanced one observed phoneme at
 * a time, with jump arcs from the best state to any word start in the window. The ordinary acoustic recurrence is
 * bounded by the window. An optional path channel additionally retains complete position sequences; its work grows
 * with the number of distinct histories, and must be measured before activating it on long streams.
 */
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, channel?: PathChannel | null): LocalSearch {
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
  let paths: (TrajectoryCell | undefined)[] = [];
  const logs = channel ? pathChannelLogs(channel) : null;
  let selected: { offset: number; origin: number; edits: number } | null = null;
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
    const t: typeof paths = Array(c.length);
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        t[i] = paths[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    paths = t;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;
  if (logs) paths[start.offset - lo] = new Map([["[]", { words: [], logProbability: 0, acousticCost: 0, origin: start.origin, edits: start.edits }]]);

  const extend = (target: TrajectoryCell, source: TrajectoryCell | undefined, logProbability: number, acousticCost: number,
    edit: number, word?: number, restart?: number) => {
    if (!source) return;
    for (const path of source.values()) {
      const words = word === undefined || (restart === undefined && path.words.at(-1) === word) ? path.words : [...path.words, word];
      const key = JSON.stringify(words);
      const next: RetainedTrajectory = { words, logProbability: path.logProbability + logProbability,
        acousticCost: path.acousticCost + acousticCost, origin: restart ?? path.origin,
        edits: restart === undefined ? path.edits + edit : edit };
      const previous = target.get(key);
      if (previous) {
        const representative = next.acousticCost < previous.acousticCost ? next : previous;
        target.set(key, { ...representative, logProbability: addLogProbability(previous.logProbability, next.logProbability) });
      } else target.set(key, next);
    }
  };
  const trajectories = () => paths.flatMap((cell, i) => cell ? [...cell.values()].map(path => ({ ...path, offset: lo + i })) : []);


  return {
    step(p) {
      const n = cost.length;
      const base = cost[best - lo];
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const t: typeof paths = Array(n);
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        const bt: TrajectoryCell | undefined = logs ? new Map() : undefined;
        if (logs) extend(bt!, paths[i], logs.insert(p), insertionCost(ref, at, p), 1);
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
          }
          if (logs) extend(bt!, paths[i - 1], logs.match(ref[at - 1], p) + logs.movement(entry(at - 1)), s, edit, index.wordAt(at - 1));
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
            }
            if (logs) extend(bt!, paths[best - lo], logs.match(ref[at - 1], p) + logs.movement(j - base - subCost(ref[at - 1], p)), j - base, edit, index.wordAt(at - 1), from);
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
          }
          if (logs) extend(bt!, t[i - 1], logs.deletion + logs.movement(entry(at - 1)), deletionCost(ref, at - 1) + entry(at - 1), 1);
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        t[i] = bt;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      // Keep the acoustic recurrence, movement edges, recent-cost gate and comparison interval unchanged.
      // Retain every distinct history, including histories dominated on acoustic cost at the same cell.
      // The medoid supplies an existing trajectory to the event layer; it does not change the search edges.
      const previousOrigin = selected?.origin ?? origin[best - lo];
      selected = null;
      if (logs && o[k] !== previousOrigin) {
        const candidates = t.flatMap((cell, i) => cell ? [...cell.values()].map(path => ({ ...path, offset: lo + i })) : []);
        const medoid = positionSequenceMedoid(candidates);
        if (medoid) {
          const path = candidates[medoid.representative];
          selected = { offset: path.offset, origin: path.origin, edits: path.edits };
        }
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      paths = t;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { ...(selected ?? { offset: best, origin: origin[i], edits: edits[i] }), recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    trajectories,
    save: () => ({ lo, cost, origin, edits, paths, selected, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; paths: typeof paths; selected: typeof selected; best: number; ayah: number; recent: number[] };
      ({ lo, cost, origin, edits, paths, selected, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
