import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { fragmentationCosts } from "./fragmentation.js";
import type { FragmentationModel } from "./fragmentation.js";

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
  /** Diagnostic ownership of the latest multi-token recognition event, when a channel is supplied. */
  fragment?: { referenceOffset: number; heard: string; fromObservation: number; toObservation: number };
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
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, model?: FragmentationModel): LocalSearch {
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
  const channel = model ? fragmentationCosts(model, ref, wordStarts) : undefined;
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
  let best = start.offset;
  let ayah = -1;
  const recent: number[] = [];
  type Witness = LocalPosition["fragment"];
  let witnesses: Witness[] = [];
  type Row = { lo: number; cost: Float64Array; origin: Int32Array; edits: Int32Array; best: number; total: number; witnesses: Witness[] };
  let history: Row[] = [];
  let heard = "";
  let total = 0;
  let observations = 0;

  const recentre = (offset: number) => {
    const a = ayahOf(offset);
    if (a === ayah) return;
    ayah = a;
    const newLo = ayahStart[Math.max(0, a - BACK_AYAHS)];
    const newHi = ayahEnd[Math.min(ayahEnd.length - 1, a + AHEAD_AYAHS)];
    const c = new Float64Array(newHi - newLo + 1).fill(Infinity);
    const o = new Int32Array(c.length);
    const e = new Int32Array(c.length);
    const f: Witness[] = Array(c.length);
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        f[i] = witnesses[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    witnesses = f;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;


  return {
    step(p) {
      const n = cost.length;
      const base = cost[best - lo];
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const f: Witness[] = channel ? Array(n) : [];
      if (channel) {
        history.push({ lo, cost, origin, edits, best, total, witnesses });
        if (history.length > 3) history.shift();
        heard = (heard + p).slice(-3);
        observations++;
      }
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        let bf = witnesses[i];
        if (i > 0) {
          const ordinary = channel?.ordinary(at - 1) ?? 0;
          const s = subCost(ref[at - 1], p) + ordinary + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            bf = witnesses[i - 1];
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              bf = undefined;
            }
          }
          // A bounded observation-span arc consumes one reference symbol and every fragment exactly once.
          // Historical rows keep their normalization totals, so continuation and restart charge the same interval.
          if (channel) for (const length of [2, 3]) {
            if (history.length < length) continue;
            const past = history[history.length - length];
            const pi = from - past.lo;
            // A recentered window cannot create a destination that was unavailable at the event's onset.
            if (pi < 0 || pi + 1 >= past.cost.length) continue;
            const sequence = heard.slice(-length);
            const emission = channel.fragmented(from, sequence) + entry(from);
            if (!Number.isFinite(emission)) continue;
            const continuation = past.cost[pi] + past.total - total + emission;
            if (continuation < bc) {
              bc = continuation;
              bo = past.origin[pi];
              be = past.edits[pi] + 1;
              bf = { referenceOffset: from, heard: sequence, fromObservation: observations - length, toObservation: observations };
            }
            if (from !== past.best && wordStarts.has(from)) {
              const cross = surahOf(from) === surahOf(Math.max(0, past.best - 1)) ? 0 : SURAH_ENTRY;
              const restart = past.cost[past.best - past.lo] + past.total - total + JUMP_COST
                + Math.abs(from - past.best) * JUMP_DISTANCE + emission + (entry(from) ? 0 : cross);
              if (restart < bc) {
                bc = restart;
                bo = from;
                be = 1;
                bf = { referenceOffset: from, heard: sequence, fromObservation: observations - length, toObservation: observations };
              }
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + ordinary + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            bf = f[i - 1];
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        if (channel) f[i] = bf;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      if (channel) total += min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      witnesses = f;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT,
        ...(channel && witnesses[i] ? { fragment: witnesses[i] } : {}) };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, best, ayah, recent: [...recent], witnesses, history: [...history], heard, total, observations }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; best: number; ayah: number; recent: number[];
        witnesses: Witness[]; history: Row[]; heard: string; total: number; observations: number };
      ({ lo, cost, origin, edits, best, ayah, witnesses, heard, total, observations } = s);
      history = [...s.history];
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
