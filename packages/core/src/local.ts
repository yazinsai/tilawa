import { CONSONANTS, UNIT, deletionCost, insertionCost, isConfusable, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

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
  /**
   * Phonemes of the current word (the one holding `offset - 1`) heard exactly as written along the best path, counted
   * from that word's start; 0 at a word the run entered by a jump arc's deletion or has not matched yet.
   */
  wordMatches: number;
  /**
   * The current word's consonants heard exactly as written along the best path, a doubled consonant (a shadda: a sun
   * letter, idgham) once; 0 at a word not matched yet. Counted only with `countConsonants`.
   */
  consonantMatches: number;
  /**
   * refinements.begunByConsonants: as `consonantMatches`, a consonant heard as its engine-confusable partner
   * (`isConfusable`) counted as heard. Counted only with `countNear` (and `countConsonants`); 0 otherwise.
   */
  consonantNear: number;
  /**
   * Phonemes of the word before the current one heard exactly as written along the best path; 0 when the path entered
   * the current word by a jump arc, or left the word before out whole. Counted only with `countConsonants` (and
   * `countMatches`).
   */
  previousMatches: number;
  /**
   * Phonemes the search had consumed when the current run began: the run's first phoneme is the `since`-th the search
   * consumed, counting from 0 (0 for the run the search started on). Kept only with `countSince`.
   */
  since: number;
  /** Phonemes the search has consumed (`step`), less any a `restore` took back. */
  steps: number;
}

/** What the search counts beyond its position, and where a declared passage bounds it. */
export interface LocalSearchOptions {
  /** `false` leaves `wordMatches` at 0 and skips counting it (the tracker reads it only for `confirmAdvance`). */
  countMatches?: boolean;
  /** Count `consonantMatches` and `previousMatches` (the tracker reads them only for `consonantConfirm` and `rivalHold`). */
  countConsonants?: boolean;
  /**
   * refinements.passageWindow: the declared passage, as the phonemes from its first word's start to its last word's
   * end. While the search's position is inside it, no jump arc lands past its end: the search leaves the passage
   * forward only by reading on, in order, from its last word. Its window, its steps, a jump back (before the passage's
   * start too) and a search whose position is past the end, where the reciter read on, are as without it.
   */
  passage?: { from: number; to: number };
  /**
   * refinements.restartNoEntry: a jump arc back to the first word of the surah the search is in costs a plain jump: a
   * restart, not an entry into the surah, which pays `SURAH_ENTRY`.
   */
  restartNoEntry?: boolean;
  /** Keep `since` (refinements.withdrawBegun: where a jump began, to judge the word it left as it was then). */
  countSince?: boolean;
  /** Count `consonantNear` too (refinements.begunByConsonants). */
  countNear?: boolean;
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
export function createLocalSearch(
  index: ReferenceIndex,
  start: Omit<LocalPosition, "recentCost" | "stepCost" | "wordMatches" | "consonantMatches" | "consonantNear" | "previousMatches" | "since" | "steps"> & {
    wordMatches?: number;
    consonantMatches?: number;
    consonantNear?: number;
  },
  options: LocalSearchOptions = {},
): LocalSearch {
  const counting = options.countMatches !== false;
  const consonants = options.countConsonants === true && counting;
  const near = options.countNear === true && consonants;
  const tracking = options.countSince === true;
  const passage = options.passage;
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
  /** Exact matches of each state's current word along its best path (`LocalPosition.wordMatches`). */
  let matches = new Int32Array(0);
  /** Exact consonant matches of each state's current word along its best path (`LocalPosition.consonantMatches`). */
  let consonantHits = new Int32Array(0);
  /** Consonant matches, a confusable partner counted, of each state's current word (`LocalPosition.consonantNear`). */
  let nearHits = new Int32Array(0);
  /** Exact matches of the word before each state's current word along its best path (`LocalPosition.previousMatches`). */
  let previous = new Int32Array(0);
  /** Each state's run's first phoneme, as the count of phonemes consumed before it (`LocalPosition.since`). */
  let since = new Int32Array(0);
  let steps = 0;
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
    const m = new Int32Array(counting ? c.length : 0);
    const q = new Int32Array(consonants ? c.length : 0);
    const r = new Int32Array(consonants ? c.length : 0);
    const h = new Int32Array(near ? c.length : 0);
    const b = new Int32Array(tracking ? c.length : 0);
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        if (tracking) b[i] = since[old];
        if (counting) m[i] = matches[old];
        if (consonants) {
          q[i] = consonantHits[old];
          r[i] = previous[old];
        }
        if (near) h[i] = nearHits[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    matches = m;
    consonantHits = q;
    previous = r;
    nearHits = h;
    since = b;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;
  if (counting) matches[start.offset - lo] = start.wordMatches ?? 0;
  if (consonants) consonantHits[start.offset - lo] = start.consonantMatches ?? 0;
  if (near) nearHits[start.offset - lo] = start.consonantNear ?? start.consonantMatches ?? 0;


  return {
    step(p) {
      const n = cost.length;
      const base = cost[best - lo];
      // refinements.passageWindow: from inside the declared passage, no jump lands past its end.
      const jumpsTo = passage && best >= passage.from && best <= passage.to ? passage.to : Infinity;
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const m = new Int32Array(counting ? n : 0);
      const q = new Int32Array(consonants ? n : 0);
      const r = new Int32Array(consonants ? n : 0);
      const h = new Int32Array(near ? n : 0);
      const b = new Int32Array(tracking ? n : 0);
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let bs = tracking ? since[i] : 0;
        let be = edits[i] + 1;
        let bm = counting ? matches[i] : 0;
        let bq = consonants ? consonantHits[i] : 0;
        let br = consonants ? previous[i] : 0;
        let bh = near ? nearHits[i] : 0;
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // Consuming a word's first phoneme starts counting that word's matches afresh.
          const opens = wordStarts.has(at - 1);
          // A consonant heard as written counts once however long it is held: a doubled one is one consonant.
          const hit = consonants && edit === 0 && CONSONANTS.has(p) && (opens || ref[at - 2] !== p) ? 1 : 0;
          const written = ref[at - 1];
          const close = near && CONSONANTS.has(written) && (p === written || isConfusable(written, p)) && (opens || ref[at - 2] !== written) ? 1 : 0;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            if (tracking) bs = since[i - 1];
            be = edits[i - 1] + edit;
            if (counting) bm = (opens ? 0 : matches[i - 1]) + 1 - edit;
            if (consonants) {
              bq = (opens ? 0 : consonantHits[i - 1]) + hit;
              br = opens ? matches[i - 1] : previous[i - 1];
            }
            if (near) bh = (opens ? 0 : nearHits[i - 1]) + close;
          }
          const from = at - 1;
          if (from !== best && from < jumpsTo && wordStarts.has(from)) {
            // refinements.restartNoEntry: back to the first word of the surah the search is in is a restart, no entry.
            const enter = entry(at - 1) && !(options.restartNoEntry && !crossing(from)) ? entry(at - 1) : 0;
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s - entry(at - 1) + enter + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              bs = steps;
              be = edit;
              bm = 1 - edit;
              bq = hit;
              br = 0;
              bh = close;
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            if (tracking) bs = b[i - 1];
            be = e[i - 1] + 1;
            if (counting) bm = opens ? 0 : m[i - 1];
            if (consonants) {
              bq = opens ? 0 : q[i - 1];
              br = opens ? m[i - 1] : r[i - 1];
            }
            if (near) bh = opens ? 0 : h[i - 1];
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        if (tracking) b[i] = bs;
        if (counting) m[i] = bm;
        if (consonants) {
          q[i] = bq;
          r[i] = br;
        }
        if (near) h[i] = bh;
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
      matches = m;
      consonantHits = q;
      previous = r;
      nearHits = h;
      since = b;
      steps++;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return {
        offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT,
        wordMatches: counting ? matches[i] : 0, consonantMatches: consonants ? consonantHits[i] : 0, previousMatches: consonants ? previous[i] : 0,
        consonantNear: near ? nearHits[i] : 0,
        since: tracking ? since[i] : 0, steps,
      };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, matches, consonantHits, previous, nearHits, since, steps, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as {
        lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; matches: typeof matches;
        consonantHits: typeof consonantHits; previous: typeof previous; nearHits: typeof nearHits; since: typeof since;
        steps: number; best: number; ayah: number; recent: number[];
      };
      ({ lo, cost, origin, edits, matches, consonantHits, previous, nearHits, since, steps, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
