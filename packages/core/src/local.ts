import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { acousticPathSuffix, appendAcousticPath } from "./acoustic-run.js";
import type { AcousticPath } from "./acoustic-run.js";

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

export interface AcousticCandidate {
  offset: number;
  origin: number;
  cost: number;
  pronunciation: string;
  acousticCost: number;
  observations: number;
  rankedCost?: number;
}

export interface LocalAcousticOptions {
  /** Return a calibrated acoustic cost on the identical recent observation interval. */
  score?: (pronunciation: string, observations: number) => number | null;
  /** Restrict expensive inference to competitors inside the unchanged tracker commitment margin. */
  margin?: (observations: number) => number;
  /** Training/auditing receives candidates from the original search without changing its decisions. */
  inspect?: (candidates: AcousticCandidate[], original: number, selected: number) => void;
  training?: boolean;
  audit?: boolean;
}

/**
 * Streaming alignment around a bookmark: one DP column over a window of nearby ayahs, advanced one observed phoneme at
 * a time, with jump arcs from the best state to any word start in the window. Work per phoneme is bounded by the window.
 */
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, options?: LocalAcousticOptions): LocalSearch {
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
  let paths: (AcousticPath | null)[] = [];
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
    const p: (AcousticPath | null)[] = options ? new Array(c.length).fill(null) : [];
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        if (options) p[i] = paths[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    paths = p;
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
      const nextPaths: (AcousticPath | null)[] = options ? new Array(n).fill(null) : [];
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        let path = options ? appendAcousticPath(paths[i], "", insertionCost(ref, at, p), true) : null;
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            if (options) path = appendAcousticPath(paths[i - 1], ref[at - 1], subCost(ref[at - 1], p), true);
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              if (options) path = appendAcousticPath(paths[best - lo], ref[at - 1], subCost(ref[at - 1], p), true);
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            if (options) path = appendAcousticPath(nextPaths[i - 1], ref[at - 1], deletionCost(ref, at - 1), false);
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        if (options) nextPaths[i] = path;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      // Only a competing run-origin change calls the learned comparison. Movement charges remain in c.
      if (options && (options.training || o[k] !== origin[best - lo])) {
        const candidates: AcousticCandidate[] = [];
        let selected = k, lowest = Infinity;
        let count: number | undefined;
        let complete = true;
        const horizon = acousticPathSuffix(nextPaths[k]).observations;
        const margin = options.margin?.(horizon) ?? Infinity;
        for (let i = 0; i < n; i++) {
          if (!Number.isFinite(c[i])) continue;
          if (!options.training && c[i] > c[k] + margin) continue;
          const suffix = acousticPathSuffix(nextPaths[i]);
          count ??= suffix.observations;
          if (count !== suffix.observations) throw new Error("Acoustic comparison intervals disagree");
          const candidate: AcousticCandidate = { offset: lo + i, origin: o[i], cost: c[i], ...suffix };
          candidates.push(candidate);
          if (!options.score) continue;
          // Reference-free insertion states have no competing pronunciation.
          if (!suffix.pronunciation) continue;
          const acoustic = options.score(suffix.pronunciation, suffix.observations);
          if (acoustic === null) { complete = false; continue; }
          const ranked = c[i] - suffix.acousticCost + acoustic;
          candidate.rankedCost = ranked;
          if (ranked < lowest || (ranked === lowest && Math.abs(lo + i - best - 1) < Math.abs(lo + selected - best - 1))) {
            lowest = ranked;
            selected = i;
          }
        }
        if (!complete || !acousticPathSuffix(nextPaths[k]).pronunciation) selected = k;
        options.inspect?.(candidates, lo + k, lo + selected);
        if (options.score && !options.audit) k = selected;
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      paths = nextPaths;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, paths, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; paths: typeof paths; best: number; ayah: number; recent: number[] };
      ({ lo, cost, origin, edits, paths, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
