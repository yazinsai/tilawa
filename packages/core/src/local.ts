import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { completionDepartureCost } from "./completion-survival.js";
import type { CompletionModel, WordPrefix } from "./completion-survival.js";

export interface LocalSnapshot {
  token: number;
  selected: WordPrefix | null;
  candidates: readonly { offset: number; origin: number; cost: number; prefix: WordPrefix | null }[];
}
export interface LocalOptions {
  completion?: CompletionModel;
  /** Immutable candidate summaries for training on actual causal tracker runs. */
  observe?: (snapshot: LocalSnapshot) => void;
}

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
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, options: LocalOptions = {}): LocalSearch {
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
  let prefixes: (WordPrefix | null)[] = [];
  let token = 0;
  const retainPrefixes = !!(options.completion || options.observe);
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
    const ps: (WordPrefix | null)[] = Array(c.length).fill(null);
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        ps[i] = prefixes[old] ?? null;
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    prefixes = ps;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;

  const extend = (old: WordPrefix | null, at: number, amount: number, consuming: boolean, matched: boolean): WordPrefix | null => {
    if (!retainPrefixes) return null;
    const wi = index.wordAt(Math.max(0, at - 1)), w = index.words[wi];
    const same = old?.word === wi;
    return { word: wi, entry: same ? old!.entry : token, progress: at - w.start, length: w.end - w.start,
      observations: (same ? old!.observations : 0) + (consuming ? 1 : 0),
      matches: (same ? old!.matches : 0) + (matched ? 1 : 0), editCost: (same ? old!.editCost : 0) + amount,
      lastCost: consuming ? amount : (same ? old!.lastCost : 0),
      stalled: consuming ? (matched ? 0 : (same ? old!.stalled : 0) + 1) : (same ? old!.stalled : 0) };
  };


  return {
    step(p) {
      token++;
      const n = cost.length;
      const base = cost[best - lo];
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const ps: (WordPrefix | null)[] = Array(n).fill(null);
      const source = prefixes[best - lo];
      let suffixCost = 0;
      if (source && options.completion) {
        const w = index.words[source.word];
        for (let at = w.start + source.progress; at < w.end; at++) suffixCost += deletionCost(ref, at);
      }
      const departure = source && options.completion ? completionDepartureCost(options.completion, source, suffixCost) : 0;
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        let bp = Number.isFinite(bc) ? extend(prefixes[i], at, insertionCost(ref, at, p), true, false) : null;
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            bp = Number.isFinite(bc) ? extend(prefixes[i - 1], at, s - entry(at - 1), true, edit === 0) : null;
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from)) + departure;
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              bp = extend(null, at, s - entry(at - 1), true, edit === 0);
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            bp = extend(ps[i - 1], at, deletionCost(ref, at - 1), false, false);
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        ps[i] = bp;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      prefixes = ps;
      best = lo + k;
      if (options.observe) options.observe({ token, selected: prefixes[k], candidates: Array.from(c, (v, i) =>
        ({ offset: lo + i, origin: o[i], cost: v, prefix: ps[i] })).filter(x => Number.isFinite(x.cost)) });
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, prefixes, token, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; prefixes: typeof prefixes; token: number; best: number; ayah: number; recent: number[] };
      ({ lo, cost, origin, edits, prefixes, token, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
