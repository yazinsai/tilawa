import { collapseHeld } from "./fuzzy.js";

/** Held madd and ghunna collapse, so two timings of the same word still share a key. */
export const anchorForm = (s: string) => collapseHeld([...s]).join("");

export interface DuplicateSpans {
  /** Collapsed phonemes of each word. Empty when the word has no phonemes of its own. */
  form: string[];
  /** The form occurs once in the word's surah, so a hearing of it has only one place to sit. */
  unique: boolean[];
  /** Inside a word sequence of at least three that occurs again in the surah: a refrain, not a common word. */
  refrain: boolean[];
}

/**
 * Duplicate-span index of the mushaf. A word whose phonemes occur again in its surah cannot anchor a skip: a refrain
 * (`فاتقوا الله وأطيعون`, `فبأي آلاء`) fits every copy at the same cost. Unique words are the only anchors.
 */
export function duplicateSpans(words: { surah: number; start: number; end: number }[], phonemes: string): DuplicateSpans {
  const form = words.map((w) => (w.end > w.start ? anchorForm(phonemes.slice(w.start, w.end)) : ""));
  const unique = words.map(() => false);
  const refrain = words.map(() => false);
  let from = 0;
  for (let to = 1; to <= words.length; to++) {
    if (to < words.length && words[to].surah === words[from].surah) continue;
    const count = new Map<string, number>();
    for (let i = from; i < to; i++) if (form[i]) count.set(form[i], (count.get(form[i]) ?? 0) + 1);
    for (let i = from; i < to; i++) unique[i] = !!form[i] && count.get(form[i]) === 1;
    const grams = new Map<string, number[]>();
    for (let i = from; i + 2 < to; i++) {
      if (!form[i] || !form[i + 1] || !form[i + 2]) continue;
      const key = `${form[i]}\0${form[i + 1]}\0${form[i + 2]}`;
      const list = grams.get(key);
      if (list) list.push(i);
      else grams.set(key, [i]);
    }
    for (const starts of grams.values()) if (starts.length >= 2) for (const s of starts) {
      refrain[s] = refrain[s + 1] = refrain[s + 2] = true;
    }
    from = to;
  }
  return { form, unique, refrain };
}

export interface AnchorVisit {
  word: number;
  obsFrom: number;
  obsTo: number;
  counts: boolean;
  wrong: boolean;
}

export interface AnchorRun {
  word: number;
  last: number;
}

/**
 * Skip runs between consecutive unique anchors. Heard duplicate words float onto the copies that keep recitation
 * order. A run is emitted only when it still contains text that was not heard, so repeating a refrain — the same
 * words again, with the unique anchors still contiguous — is not a skip. `runs` empty means the stretch was heard.
 * A pair that cannot be placed is omitted, and the caller keeps the alignment's own gap there.
 */
export function anchorSkipRuns(o: {
  form: string[];
  unique: boolean[];
  refrain?: boolean[];
  ayahStart: boolean[];
  visits: AnchorVisit[];
}): { from: number; to: number; runs: AnchorRun[] }[] {
  const refrain = o.refrain ?? o.form.map(() => true);
  const anchors: AnchorVisit[] = [];
  let prevWord = -1;
  let prevObs = -1;
  for (const v of o.visits) {
    if (!v.counts || v.wrong || !o.unique[v.word] || v.obsFrom < 0) continue;
    if (v.word <= prevWord || v.obsFrom < prevObs) continue;
    anchors.push(v);
    prevWord = v.word;
    prevObs = v.obsFrom;
  }
  const out: { from: number; to: number; runs: AnchorRun[] }[] = [];
  for (let i = 1; i < anchors.length; i++) {
    const left = anchors[i - 1];
    const right = anchors[i];
    if (right.word <= left.word || right.obsFrom < left.obsTo) continue;
    const heard = o.visits
      .filter((v) => v.counts && v.obsFrom >= left.obsTo && v.obsFrom < right.obsFrom)
      .map((v) => o.form[v.word]);
    const runs = place(o, left.word, right.word, heard);
    if (!runs) continue;
    // Ordinary skips stay with the alignment. This pass only re-places a gap that touches a repeated span.
    if (runs.length && !refrain.slice(left.word + 1, right.word).some(Boolean)) continue;
    out.push({ from: left.word, to: right.word, runs });
  }
  return out;
}

/** Lay `heard` forms onto words `(from, to)` in order. Null when some hearing matches nothing there. */
function place(o: { form: string[]; unique: boolean[]; ayahStart: boolean[] }, from: number, to: number, heard: string[]): AnchorRun[] | null {
  const span: number[] = [];
  for (let i = from + 1; i < to; i++) if (o.form[i]) span.push(i);
  const matched = new Set<number>();
  let cursor = 0;
  for (const key of heard) {
    if (!key) return null;
    let found = -1;
    for (let i = cursor; i < span.length; i++) if (o.form[span[i]] === key) { found = i; break; }
    if (found >= 0) {
      matched.add(span[found]);
      cursor = found + 1;
      continue;
    }
    let repeat = false;
    for (let i = 0; i < cursor; i++) if (!o.unique[span[i]] && o.form[span[i]] === key) { repeat = true; break; }
    if (!repeat) return null;
  }
  slideToAyahStart(o, span, matched);
  const heardKey = new Set<string>();
  for (const i of matched) heardKey.add(o.form[i]);
  const runs: AnchorRun[] = [];
  let run: number[] = [];
  const flush = () => {
    if (!run.length) return;
    // The other copy of a refrain already heard is not a skip. Text that was never heard is.
    if (!run.every((i) => !o.unique[i] && heardKey.has(o.form[i]))) runs.push({ word: run[0], last: run[run.length - 1] });
    run = [];
  };
  for (let i = from + 1; i < to; i++) {
    if (!o.form[i]) continue;
    if (matched.has(i)) flush();
    else run.push(i);
  }
  flush();
  return runs;
}

/**
 * The alignment can spend a whole window on the first copy of a refrain, so the later unique word is in the phonemes
 * but not in the visits. Find anchors by the phoneme string itself: a word heard in order, whose text occurs once.
 * `heard` is the raw phoneme string. `raw` is each word's phonemes (empty when it has none).
 */
export function phonemeSkipRuns(o: {
  raw: string[];
  unique: boolean[];
  refrain: boolean[];
  ayahStart: boolean[];
  from: number;
  to: number;
  heard: string;
}): { from: number; to: number; runs: AnchorRun[] }[] {
  const anchors: { wi: number; end: number }[] = [];
  let pos = 0;
  for (let wi = o.from; wi <= o.to && wi < o.raw.length; wi++) {
    const text = o.raw[wi];
    if (!o.unique[wi] || text.length < 4) continue;
    const at = o.heard.indexOf(text, pos);
    if (at < 0) continue;
    anchors.push({ wi, end: at + text.length });
    pos = at + text.length;
  }
  const out: { from: number; to: number; runs: AnchorRun[] }[] = [];
  for (let i = 1; i < anchors.length; i++) {
    const left = anchors[i - 1];
    const right = anchors[i];
    if (right.wi <= left.wi) continue;
    const mid = o.heard.slice(left.end, right.end - o.raw[right.wi].length);
    if (!o.refrain.slice(left.wi + 1, right.wi).some(Boolean)) continue;
    const runs = consume(o.raw, o.unique, o.ayahStart, left.wi, right.wi, mid);
    if (runs) out.push({ from: left.wi, to: right.wi, runs });
  }
  return out;
}

/** Match `mid` to the words between two anchors. Null when phonemes are left over (added speech, not a clean skip). */
function consume(raw: string[], unique: boolean[], ayahStart: boolean[], from: number, to: number, mid: string): AnchorRun[] | null {
  let p = 0;
  let prev = -1;
  const unmatched: number[] = [];
  for (let wi = from + 1; wi < to; wi++) {
    const text = raw[wi];
    if (!text) continue;
    while (prev >= 0 && !unique[prev] && mid.startsWith(raw[prev], p)) p += raw[prev].length;
    if (mid.startsWith(text, p)) { p += text.length; prev = wi; continue; }
    unmatched.push(wi);
  }
  while (prev >= 0 && !unique[prev] && mid.startsWith(raw[prev], p)) p += raw[prev].length;
  if (p !== mid.length) return null;
  // ولا heard after the skipped ayah was the ayah's own first word to the left-to-right pass. Put the gap on the ayah start.
  if (unmatched.length && !ayahStart[unmatched[0]]) {
    const prevWord = unmatched[0] - 1;
    if (prevWord > from && ayahStart[prevWord] && !unique[prevWord] && raw[prevWord] && unmatched.some((wi) => raw[wi] === raw[prevWord])) {
      unmatched.unshift(prevWord);
    }
  }
  // No refrain in the stretch: the alignment's own gap is the seam. This pass only re-places skips around duplicate text.
  if (![...Array(to - from - 1)].some((_, k) => raw[from + 1 + k] && !unique[from + 1 + k])) return null;
  const heard = new Set<string>();
  for (let wi = from + 1; wi < to; wi++) if (raw[wi] && !unmatched.includes(wi)) heard.add(raw[wi]);
  const runs: AnchorRun[] = [];
  let run: number[] = [];
  const flush = () => {
    if (!run.length) return;
    if (!run.every((i) => !unique[i] && heard.has(raw[i]))) runs.push({ word: run[0], last: run[run.length - 1] });
    run = [];
  };
  for (const wi of unmatched) {
    if (run.length && wi !== run[run.length - 1] + 1 && raw.slice(run[run.length - 1] + 1, wi).some(Boolean)) flush();
    run.push(wi);
  }
  flush();
  return runs;
}

/**
 * The alignment can pin a repeated boundary word on either copy (`ولا` before the skipped ayah, or after it).
 * When the word just before the gap is itself an ayah start, keep that copy inside the gap.
 */
function slideToAyahStart(o: { form: string[]; unique: boolean[]; ayahStart: boolean[] }, span: number[], matched: Set<number>) {
  let i = 0;
  while (i < span.length) {
    while (i < span.length && matched.has(span[i])) i++;
    if (i >= span.length) break;
    const start = i;
    while (i < span.length && !matched.has(span[i])) i++;
    const end = i;
    const prev = start - 1;
    if (prev < 0 || !matched.has(span[prev]) || o.unique[span[prev]] || o.ayahStart[span[start]] || !o.ayahStart[span[prev]]) continue;
    const key = o.form[span[prev]];
    for (let j = end - 1; j >= start; j--) if (o.form[span[j]] === key) {
      matched.delete(span[prev]);
      matched.add(span[j]);
      break;
    }
  }
}
