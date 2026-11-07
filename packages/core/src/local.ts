import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { editAlignmentCosts } from "./edit-alignment.js";
import type { EditAlignmentModel } from "./edit-alignment.js";

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
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, model?: EditAlignmentModel): LocalSearch {
  const channel = model ? editAlignmentCosts(model) : null;
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
  // Plain edit-cost witness for confidence: marginal probability ranks locations,
  // while the existing confirmation and misfit thresholds retain their units.
  let plain = new Float64Array(0);
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
    const v = new Float64Array(c.length).fill(Infinity);
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        v[i] = plain[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    plain = v;
  };

  recentre(start.offset);
  cost[start.offset - lo] = 0;
  origin[start.offset - lo] = start.origin;
  edits[start.offset - lo] = start.edits;
  plain[start.offset - lo] = 0;
  if (channel) {
    for (let i = start.offset - lo + 1; i < cost.length; i++) {
      cost[i] = cost[i - 1] + channel.deletion(ref[lo + i - 1]) + entry(lo + i - 1);
      plain[i] = plain[i - 1] + deletionCost(ref, lo + i - 1) + entry(lo + i - 1);
      origin[i] = start.origin;
      edits[i] = edits[i - 1] + 1;
    }
  }


  return {
    step(p) {
      const n = cost.length;
      const base = cost[best - lo];
      const plainBase = plain[best - lo];
      const c = new Float64Array(n);
      const o = new Int32Array(n);
      const e = new Int32Array(n);
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        const insertion = insertionCost(ref, at, p);
        let bc = cost[i] + (channel ? channel.insertion(p) : insertion);
        let bv = plain[i] + insertion;
        let bo = origin[i];
        let be = edits[i] + 1;
        if (i > 0) {
          const s = (channel ? channel.emission(ref[at - 1], p) : subCost(ref[at - 1], p)) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          const diagonal = cost[i - 1] + s;
          const acousticInsertion = bc;
          if (diagonal <= bc) {
            bc = diagonal;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            bv = plain[i - 1] + subCost(ref[at - 1], p) + entry(at - 1);
          }
          // Sum acoustic alignments to this destination, including reference-only
          // deletions. Each route carries the normalized channel probability.
          const deletion = c[i - 1] + (channel ? channel.deletion(ref[at - 1]) : deletionCost(ref, at - 1)) + entry(at - 1);
          if (channel) {
            if (deletion < bc) { bo = o[i - 1]; be = e[i - 1] + 1; bv = v[i - 1] + deletionCost(ref, at - 1) + entry(at - 1); }
            bc = channel.sum(channel.sum(acousticInsertion, diagonal), deletion);
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              bv = plainBase + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + subCost(ref[from], p)
                + entry(from) + (entry(from) ? 0 : crossing(from));
            }
          }
          if (!channel && deletion < bc) {
            bc = deletion;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            bv = v[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        v[i] = bv;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      const plainMin = v[k];
      for (let i = 0; i < n; i++) v[i] -= plainMin;
      recent.push(channel ? plainMin - plainBase : min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      plain = v;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, plain, best, ayah, recent: [...recent] }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; plain: typeof plain; best: number; ayah: number; recent: number[] };
      ({ lo, cost, origin, edits, plain, best, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
