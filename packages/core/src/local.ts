import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";
import { channelLogLikelihood, channelReference } from "./morpheme-channel.js";
import type { MorphemeChannel, MorphemeRole } from "./morpheme-channel.js";

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

export interface LocalObservationCandidate {
  offset: number;
  origin: number;
  cost: number;
  /** Traversed surface-phone occurrences; whole-word movement bypasses are absent. */
  referenceOffsets: number[];
  movementCost: number;
}
export interface LocalObservationComparison {
  heard: string;
  continuation: LocalObservationCandidate;
  restart: LocalObservationCandidate;
  /** Morphology-versus-phonetic residual costs in the existing edit units. */
  continuationCorrection?: number;
  restartCorrection?: number;
}
export interface LocalObservationOptions {
  /** Identical-data frozen channels. The phonetic channel is the matched observation control. */
  channels?: { morphology: MorphemeChannel; phonetic: MorphemeChannel };
  /** Read-only accounting for validating the common observation interval and movement prices. */
  audit?: (comparison: LocalObservationComparison) => void;
}
interface ObservationTrail {
  parent: ObservationTrail | null;
  referenceOffset: number;
  movementCost: number;
}

/**
 * Streaming alignment around a bookmark: one DP column over a window of nearby ayahs, advanced one observed phoneme at
 * a time, with jump arcs from the best state to any word start in the window. Work per phoneme is bounded by the window.
 */
export function createLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">, observation: LocalObservationOptions = {}): LocalSearch {
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
  let best = start.offset;
  let ayah = -1;
  const recent: number[] = [];
  const trackObservations = !!(observation.channels || observation.audit);
  let trails: (ObservationTrail | null)[] = [];
  let heard = "";
  let cumulative = 0;
  const surface = observation.channels ? channelReference(index, 0, ref.length) : null;
  const addTrail = (parent: ObservationTrail | null, referenceOffset: number, movementCost = 0): ObservationTrail | null =>
    trackObservations ? { parent, referenceOffset, movementCost } : null;

  const recentre = (offset: number) => {
    const a = ayahOf(offset);
    if (a === ayah) return;
    ayah = a;
    const newLo = ayahStart[Math.max(0, a - BACK_AYAHS)];
    const newHi = ayahEnd[Math.min(ayahEnd.length - 1, a + AHEAD_AYAHS)];
    const c = new Float64Array(newHi - newLo + 1).fill(Infinity);
    const o = new Int32Array(c.length);
    const e = new Int32Array(c.length);
    const t: (ObservationTrail | null)[] = trackObservations ? Array(c.length).fill(null) : [];
    for (let i = 0; i < c.length; i++) {
      const old = newLo + i - lo;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old];
        o[i] = origin[old];
        e[i] = edits[old];
        if (trackObservations) t[i] = trails[old];
      }
    }
    lo = newLo;
    cost = c;
    origin = o;
    edits = e;
    trails = t;
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
      const t: (ObservationTrail | null)[] = trackObservations ? Array(n).fill(null) : [];
      const previousOrigin = origin[best - lo];
      if (trackObservations) heard += p;
      for (let i = 0; i < n; i++) {
        const at = lo + i;
        let bc = cost[i] + insertionCost(ref, at, p);
        let bo = origin[i];
        let be = edits[i] + 1;
        let bt = trackObservations ? trails[i] : null;
        if (i > 0) {
          const s = subCost(ref[at - 1], p) + entry(at - 1);
          const edit = ref[at - 1] === p ? 0 : 1;
          // On a tie, consuming the reference beats an insertion, so a correct recitation keeps advancing.
          if (cost[i - 1] + s <= bc) {
            bc = cost[i - 1] + s;
            bo = origin[i - 1];
            be = edits[i - 1] + edit;
            bt = addTrail(trackObservations ? trails[i - 1] : null, at - 1, entry(at - 1));
          }
          const from = at - 1;
          if (from !== best && wordStarts.has(from)) {
            const j = base + JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + s + (entry(at - 1) ? 0 : crossing(from));
            if (j < bc) {
              bc = j;
              bo = from;
              be = edit;
              bt = addTrail(trackObservations ? trails[best - lo] : null, at - 1,
                JUMP_COST + Math.abs(from - best) * JUMP_DISTANCE + entry(at - 1) + (entry(at - 1) ? 0 : crossing(from)));
            }
          }
          const d = c[i - 1] + deletionCost(ref, at - 1) + entry(at - 1);
          if (d < bc) {
            bc = d;
            bo = o[i - 1];
            be = e[i - 1] + 1;
            bt = addTrail(trackObservations ? t[i - 1] : null, at - 1, entry(at - 1));
          }
        }
        c[i] = bc;
        o[i] = bo;
        e[i] = be;
        if (trackObservations) t[i] = bt;
      }

      // Lowest cost wins; ties go to the state nearest plain forward progress.
      let k = -1;
      for (let i = 0; i < n; i++) {
        if (k === -1 || c[i] < c[k] || (c[i] === c[k] && Math.abs(lo + i - best - 1) < Math.abs(lo + k - best - 1))) k = i;
      }
      // Only competing local run origins use the additional channel. The control subtracts the phonetic
      // contribution, so the original movement and phone/duration edit prices remain in the path cost.
      // Both complete trajectories consume every token since this local search was initialized.
      if (trackObservations && o[k] !== previousOrigin) {
        let near = -1;
        for (let i = 0; i < n; i++) if (o[i] === previousOrigin && Number.isFinite(c[i]) && (near === -1 || c[i] < c[near])) near = i;
        if (near !== -1) {
          const candidate = (i: number): LocalObservationCandidate => {
            const referenceOffsets: number[] = [];
            let movementCost = 0;
            for (let node = t[i]; node; node = node.parent) {
              referenceOffsets.push(node.referenceOffset);
              movementCost += node.movementCost;
            }
            referenceOffsets.reverse();
            return { offset: lo + i, origin: o[i], cost: c[i] + cumulative, referenceOffsets, movementCost };
          };
          const comparison: LocalObservationComparison = { heard, continuation: candidate(near), restart: candidate(k) };
          if (observation.channels && surface) {
            const correction = (path: LocalObservationCandidate) => {
              const reference = { phones: path.referenceOffsets.map(at => ref[at]).join(""),
                roles: path.referenceOffsets.map(at => surface.roles[at] as MorphemeRole) };
              const morphology = channelLogLikelihood(reference, heard, observation.channels!.morphology);
              const phonetic = channelLogLikelihood(reference, heard, observation.channels!.phonetic);
              // Same nats per edit as tracker posterior ranking; no newly selected commitment margin.
              return morphology === phonetic ? 0 : -(morphology - phonetic) * UNIT / 3;
            };
            comparison.continuationCorrection = correction(comparison.continuation);
            comparison.restartCorrection = correction(comparison.restart);
            if (c[near] + comparison.continuationCorrection < c[k] + comparison.restartCorrection) k = near;
          }
          observation.audit?.(comparison);
        }
      }
      const min = c[k];
      for (let i = 0; i < n; i++) c[i] -= min;
      recent.push(min - base);
      if (recent.length > RECENT) recent.shift();
      cost = c;
      origin = o;
      edits = e;
      trails = t;
      cumulative += min;
      best = lo + k;
      recentre(best);
    },
    position() {
      const i = best - lo;
      const recentCost = recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0;
      return { offset: best, origin: origin[i], edits: edits[i], recentCost, stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    // Steps replace the arrays rather than writing into them, so a saved state can share them.
    save: () => ({ lo, cost, origin, edits, best, ayah, recent: [...recent], trails, heard, cumulative }),
    restore(state) {
      const s = state as { lo: number; cost: typeof cost; origin: typeof origin; edits: typeof edits; best: number; ayah: number; recent: number[]; trails: typeof trails; heard: string; cumulative: number };
      ({ lo, cost, origin, edits, best, ayah, trails, heard, cumulative } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
