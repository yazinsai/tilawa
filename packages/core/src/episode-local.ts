import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import { episodeTransitionCost, validateEpisodeModel } from "./episode-motion.js";
import type { EpisodeMotionModel, MotionEvent } from "./episode-motion.js";
import type { LocalPosition, LocalSearch } from "./local.js";
import type { ReferenceIndex } from "./reference.js";

interface MovementPolicy { jump: number; distance: number; back: number; ahead: number; surahEntry: number; recent: number }

/** Same local edges and acoustic costs, with ordinary and recovery states retained at every reference cell. */
export function createEpisodeLocalSearch(index: ReferenceIndex, start: Omit<LocalPosition, "recentCost" | "stepCost">,
  model: EpisodeMotionModel, policy: MovementPolicy): LocalSearch {
  validateEpisodeModel(model);
  const ref = index.phonemes;
  const starts: number[] = [], ends: number[] = [], wordAyah: number[] = [];
  index.words.forEach((w, i) => {
    const prev = index.words[i - 1];
    if (!prev || prev.surah !== w.surah || prev.ayah !== w.ayah) starts.push(w.start);
    wordAyah.push(starts.length - 1);
    ends[starts.length - 1] = w.end;
  });
  const wordStarts = new Set(index.words.map(w => w.start));
  const surahFirst = new Set(index.words.filter((w, i) => i > 0 && w.surah !== index.words[i - 1].surah && w.surah !== 9).map(w => w.start));
  const surahOf = (offset: number) => index.words[index.wordAt(offset)].surah;
  const entry = (offset: number) => surahFirst.has(offset) ? policy.surahEntry : 0;
  const ayahOf = (offset: number) => wordAyah[index.wordAt(Math.max(0, offset - 1))];
  let lo = 0, ayah = -1, best = start.offset, state = 0;
  let cost = new Float64Array(0), acoustic = new Float64Array(0);
  let origin = new Int32Array(0), edits = new Int32Array(0);
  const recent: number[] = [];
  const recentre = (offset: number) => {
    const a = ayahOf(offset);
    if (a === ayah) return;
    ayah = a;
    const newLo = starts[Math.max(0, a - policy.back)];
    const hi = ends[Math.min(ends.length - 1, a + policy.ahead)];
    const c = new Float64Array((hi - newLo + 1) * 2).fill(Infinity);
    const ac = new Float64Array(c.length).fill(Infinity);
    const o = new Int32Array(c.length), e = new Int32Array(c.length);
    for (let i = 0; i < c.length; i++) {
      const old = (newLo - lo) * 2 + i;
      if (old >= 0 && old < cost.length) {
        c[i] = cost[old]; ac[i] = acoustic[old]; o[i] = origin[old]; e[i] = edits[old];
      }
    }
    lo = newLo; cost = c; acoustic = ac; origin = o; edits = e;
  };
  recentre(best);
  for (let s = 0; s < 2; s++) {
    const k = (best - lo) * 2 + s;
    cost[k] = -UNIT / 3 * Math.log(model.initial[s]);
    acoustic[k] = 0; origin[k] = start.origin; edits[k] = start.edits;
  }
  state = model.initial[1] > model.initial[0] ? 1 : 0;
  const transition = Array.from({ length: 2 }, (_, s) => Object.fromEntries(
    (["read", "edit", "omit", "repeat"] as MotionEvent[]).map(event => [event, [0, 1].map(t => episodeTransitionCost(model, s, event, t))]),
  ) as Record<MotionEvent, number[]>);

  return {
    step(p) {
      const base = acoustic[(best - lo) * 2 + state];
      const c = new Float64Array(cost.length).fill(Infinity), ac = new Float64Array(c.length).fill(Infinity);
      const o = new Int32Array(c.length), e = new Int32Array(c.length);
      const cells = c.length / 2;
      for (let i = 0; i < cells; i++) {
        const at = lo + i;
        for (let t = 0; t < 2; t++) {
          const k = i * 2 + t;
          const take = (source: number, charge: number, event: MotionEvent | null, sameRow: boolean, reset = false, nEdits = 1, consume = false) => {
            const scores = sameRow ? c : cost, raw = sameRow ? ac : acoustic;
            const origins = sameRow ? o : origin, errors = sameRow ? e : edits;
            for (let s = 0; s < 2; s++) {
              // Passage relocation has no single-ayah supervision: its prior and state propagation are unchanged.
              if (event === null && s !== t) continue;
              const prev = source * 2 + s;
              const v = scores[prev] + charge + (event === null ? 0 : transition[s][event][t]);
              if (v < c[k] || (consume && v === c[k])) {
                c[k] = v; ac[k] = raw[prev] + charge;
                o[k] = reset ? at - 1 : origins[prev];
                e[k] = (reset ? 0 : errors[prev]) + nEdits;
              }
            }
          };
          take(i, insertionCost(ref, at, p), "edit", false);
          if (i > 0) {
            const mismatch = ref[at - 1] === p ? 0 : 1;
            const s = subCost(ref[at - 1], p) + entry(at - 1);
            take(i - 1, s, mismatch ? "edit" : "read", false, false, mismatch, true);
            const from = at - 1;
            if (from !== best && wordStarts.has(from)) {
              const crossing = surahOf(from) === surahOf(Math.max(0, best - 1)) ? 0 : policy.surahEntry;
              const charge = policy.jump + Math.abs(from - best) * policy.distance + s + (entry(from) ? 0 : crossing);
              const nextWord = index.wordAt(best);
              const destination = index.wordAt(from);
              // Only one complete word omitted, or a repeat inside this ayah, is supervised by these clips.
              const event: MotionEvent | null = destination === nextWord + 1 && !crossing ? "omit"
                : from < best && ayahOf(from + 1) === ayahOf(best) && !crossing ? "repeat" : null;
              take(best - lo, charge, event, false, true, mismatch);
            }
            take(i - 1, deletionCost(ref, at - 1) + entry(at - 1), "edit", true);
          }
        }
      }
      let k = 0;
      for (let i = 1; i < c.length; i++) {
        const distance = Math.abs(lo + Math.floor(i / 2) - best - 1);
        const current = Math.abs(lo + Math.floor(k / 2) - best - 1);
        if (c[i] < c[k] || (c[i] === c[k] && distance < current)) k = i;
      }
      const min = c[k];
      for (let i = 0; i < c.length; i++) c[i] -= min;
      recent.push(ac[k] - base);
      if (recent.length > policy.recent) recent.shift();
      cost = c; acoustic = ac; origin = o; edits = e;
      best = lo + Math.floor(k / 2); state = k % 2;
      recentre(best);
    },
    position() {
      const i = (best - lo) * 2 + state;
      return { offset: best, origin: origin[i], edits: edits[i],
        recentCost: recent.length ? recent.reduce((a, b) => a + b, 0) / UNIT / recent.length : 0,
        stepCost: (recent.at(-1) ?? 0) / UNIT };
    },
    save: () => ({ lo, cost, acoustic, origin, edits, best, state, ayah, recent: [...recent] }),
    restore(saved) {
      const s = saved as { lo: number; cost: Float64Array; acoustic: Float64Array; origin: Int32Array; edits: Int32Array; best: number; state: number; ayah: number; recent: number[] };
      ({ lo, cost, acoustic, origin, edits, best, state, ayah } = s);
      recent.splice(0, recent.length, ...s.recent);
    },
  };
}
