import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import { letterSkeleton } from "./parallel-phrases.js";
import type { ReferenceIndex } from "./reference.js";

/**
 * A run of ayahs, four words or longer, copied elsewhere in the mushaf with at most one different word in each
 * ayah and at least one such slot. Exact copies are not spans: nothing differs, so nothing is held. Built from
 * letter skeletons only.
 */
export interface SlotSpan {
  /** Word index of each member's first word. */
  starts: readonly number[];
  /** Words in the run. The same for every member. */
  length: number;
  /** Word offsets that differ between members. */
  slots: readonly number[];
}

const cache = new WeakMap<ReferenceIndex, readonly SlotSpan[]>();

interface Ayah {
  surah: number;
  ayah: number;
  lo: number;
  n: number;
  sk: string[];
}

/** Ayahs of equal length whose skeletons differ in at most one word. The differing offsets, or null. */
function compat(ayahs: Ayah[], i: number, j: number): number[] | null {
  const a = ayahs[i];
  const b = ayahs[j];
  if (!a || !b || a.n !== b.n || a.n === 0) return null;
  const diffs: number[] = [];
  for (let k = 0; k < a.n; k++) if (a.sk[k] !== b.sk[k]) diffs.push(k);
  return diffs.length <= 1 ? diffs : null;
}

function consecutive(ayahs: Ayah[], prev: number, next: number): boolean {
  const a = ayahs[prev];
  const b = ayahs[next];
  return !!a && !!b && a.surah === b.surah && b.ayah === a.ayah + 1;
}

export function spanId(span: SlotSpan): string {
  return `${span.starts.join(",")}:${span.length}`;
}

/** The longest span that contains `word`, if several copies of a refrain cover it. */
export function longestSpanAt(spans: readonly SlotSpan[], word: number): SlotSpan | null {
  let best: SlotSpan | null = null;
  for (const span of spans) {
    if (!span.starts.some((st) => word >= st && word < st + span.length)) continue;
    if (!best || span.length > best.length) best = span;
  }
  return best;
}

export function slotSpans(index: ReferenceIndex): readonly SlotSpan[] {
  const hit = cache.get(index);
  if (hit) return hit;
  const text = index.wordText;
  if (!text || text.length !== index.words.length) {
    cache.set(index, []);
    return [];
  }
  const ayahs: Ayah[] = [];
  let lo = 0;
  for (let i = 0; i < index.words.length; ) {
    const w = index.words[i];
    let j = i + 1;
    while (j < index.words.length && index.words[j].surah === w.surah && index.words[j].ayah === w.ayah) j++;
    const sk: string[] = [];
    for (let k = i; k < j; k++) sk.push(letterSkeleton(text[k] ?? ""));
    ayahs.push({ surah: w.surah, ayah: w.ayah, lo, n: j - i, sk });
    lo = j;
    i = j;
  }

  const groups = new Map<string, number[]>();
  ayahs.forEach((a, i) => {
    for (let k = 0; k < a.n; k++) {
      const key = `${a.n}\0${k}\0${a.sk.slice(0, k).join("\0")}\0${a.sk.slice(k + 1).join("\0")}`;
      const list = groups.get(key);
      if (list) list.push(i);
      else groups.set(key, [i]);
    }
  });

  const runs = new Map<string, SlotSpan>();
  for (const idxs of groups.values()) {
    const uniq = [...new Set(idxs)].sort((a, b) => a - b);
    for (let a = 0; a < uniq.length; a++) {
      for (let b = a + 1; b < uniq.length; b++) {
        const diff = compat(ayahs, uniq[a], uniq[b]);
        if (!diff || diff.length !== 1) continue;
        let i = uniq[a];
        let j = uniq[b];
        while (i > 0 && j > 0 && consecutive(ayahs, i - 1, i) && consecutive(ayahs, j - 1, j) && compat(ayahs, i - 1, j - 1)) {
          i--;
          j--;
        }
        const slots: number[] = [];
        let words = 0;
        let ii = i;
        let jj = j;
        let ayahsN = 0;
        while (ii < ayahs.length && jj < ayahs.length) {
          if (ayahsN && (!consecutive(ayahs, ii - 1, ii) || !consecutive(ayahs, jj - 1, jj))) break;
          const d = compat(ayahs, ii, jj);
          if (!d) break;
          for (const k of d) slots.push(words + k);
          words += ayahs[ii].n;
          ayahsN++;
          ii++;
          jj++;
        }
        if (words < 4 || !slots.length) continue;
        const starts = [ayahs[i].lo, ayahs[j].lo].sort((x, y) => x - y);
        const span: SlotSpan = { starts, length: words, slots };
        runs.set(spanId(span), span);
      }
    }
  }
  const out = [...runs.values()];
  cache.set(index, out);
  return out;
}

/**
 * Min plain-edit cost of aligning the whole of `word` to any substring of `heard`, using the tracker's word costs.
 * `phones` is how much of `heard` that alignment consumed. `end` is the exclusive index in `heard` where it finished.
 */
export function wordSubstringCost(word: string, heard: string, tail = false): { edits: number; phones: number; end: number } | null {
  if (!word || !heard) return null;
  const n = word.length;
  const m = heard.length;
  let best = Infinity;
  let phones = 0;
  let end = 0;
  for (let s = 0; s < m; s++) {
    const len = m - s;
    let prev = new Float64Array(len + 1).fill(Infinity);
    prev[0] = 0;
    for (let i = 1; i <= n; i++) {
      const cur = new Float64Array(len + 1).fill(Infinity);
      cur[0] = prev[0] + deletionCost(word, i - 1);
      for (let j = 1; j <= len; j++) {
        const h = heard[s + j - 1];
        const sub = prev[j - 1] + subCost(word[i - 1], h);
        const del = prev[j] + deletionCost(word, i - 1);
        const ins = cur[j - 1] + insertionCost(word, i, h, n);
        cur[j] = Math.min(sub, del, ins);
      }
      prev = cur;
    }
    for (let j = 1; j <= len; j++) {
      if ((!tail || s + j >= m - 2) && prev[j] < best) {
        best = prev[j];
        phones = j;
        end = s + j;
      }
    }
  }
  if (!Number.isFinite(best)) return null;
  return { edits: best / UNIT, phones, end };
}

export interface ContinuationPick {
  id: number;
  edits: number;
  phones: number;
}

/**
 * Which member's next word the heard phones are. The current member wins when it fits and nobody beats it by
 * `margin`. Another member wins only when the current word does not fit and that one member does, ahead by the
 * margin. Otherwise wait: the word is not finished, or the copies still tie.
 */
export function continuationDecision(
  heard: string,
  candidates: readonly { id: number; phonemes: string }[],
  current: number,
  margin: (n: number) => number,
  maxCost: number,
): ContinuationPick | null {
  if (heard.length < 4 || candidates.length < 2) return null;
  const scored: { id: number; edits: number; phones: number; per: number }[] = [];
  for (const c of candidates) {
    if (!c.phonemes) return null;
    const fit = wordSubstringCost(c.phonemes, heard, true);
    if (!fit) return null;
    scored.push({ id: c.id, edits: fit.edits, phones: fit.phones, per: fit.edits / c.phonemes.length });
  }
  const cur = scored.find((s) => s.id === current);
  if (!cur) return null;
  const beats = (a: (typeof scored)[number], b: (typeof scored)[number]) => a.edits + margin(Math.max(1, a.phones)) <= b.edits;
  if (cur.per <= maxCost && scored.every((s) => s.id === current || !beats(s, cur))) {
    return { id: current, edits: cur.edits, phones: cur.phones };
  }
  const winners = scored.filter((s) => s.id !== current && s.per <= maxCost && beats(s, cur) && scored.every((o) => o.id === s.id || beats(s, o)));
  if (cur.per > maxCost && winners.length === 1) return { id: winners[0].id, edits: winners[0].edits, phones: winners[0].phones };
  return null;
}
