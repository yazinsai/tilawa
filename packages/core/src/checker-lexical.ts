import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
const form = (s: string) => collapseHeld([...s]).join("");
/**
 * A grammatical ending (ـما, a case vowel on top of it) stays at the tail. Longer than this, the divergence is the
 * stem and the ordinary consonant rules own it.
 */
const END = 4;

/** Common stem, then two short tails. The stem has to outlast either tail, so a short shared prefix is not an ending. */
function residual(a: string, b: string): { stem: string; sa: string; sb: string } | null {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  const sa = a.slice(i), sb = b.slice(i);
  if (!i || sa.length > END || sb.length > END || i <= sa.length || i <= sb.length) return null;
  return { stem: a.slice(0, i), sa, sb };
}

function consonantsOf(s: string): string {
  return [...s].filter(c => CONSONANTS.has(c)).join("");
}

/** One side adds or drops consonants and the other side is only vowels, madd or a waqf mark. A substitution is neither. */
function endSegment(sa: string, sb: string): string | null {
  const ca = consonantsOf(sa), cb = consonantsOf(sb);
  if ((ca.length === 0) === (cb.length === 0)) return null;
  return ca || cb;
}

/** Cost of the tails only. The matched stem stays put, so a following vowel cannot be rewritten into the consonant. */
function endingCost(stem: string, sa: string, sb: string): number {
  const ref = stem + sa, base = stem.length, n = sa.length, m = sb.length;
  let row = new Float64Array(m + 1);
  for (let j = 1; j <= m; j++) row[j] = row[j - 1] + insertionCost(ref, base, sb[j - 1], ref.length);
  for (let i = 1; i <= n; i++) {
    const next = new Float64Array(m + 1);
    next[0] = row[0] + deletionCost(ref, base + i - 1);
    for (let j = 1; j <= m; j++) {
      next[j] = Math.min(
        row[j - 1] + subCost(sa[i - 1], sb[j - 1]),
        row[j] + deletionCost(ref, base + i - 1),
        next[j - 1] + insertionCost(ref, base + i, sb[j - 1], ref.length),
      );
    }
    row = next;
  }
  return row[m];
}

/**
 * Align individual consonants independently of vowels, using the existing costs. Hard edits sharing an original
 * contiguous run on either side count as one independent change. This keeps a doubled identity change together
 * without inventing edits when vowels split a run, or erasing separate changes on both sides of a vowel.
 */
export function hardConsonantEdits(expected: string, observed: string): number {
  const a = form(expected), b = form(observed);
  const consonants = (s: string) => {
    const out: { at: number; run: number }[] = [];
    let run = -1;
    for (let i = 0; i < s.length; i++) if (CONSONANTS.has(s[i])) {
      if (!i || s[i - 1] !== s[i]) run++;
      out.push({ at: i, run });
    }
    return out;
  };
  const r = consonants(a), o = consonants(b), width = o.length + 1;
  // Only the current reference/observed runs can receive future edits. Preserve five frontier states so a later
  // substitution can join their components before they close: none, R, O, R/O separate, R/O joined.
  const cell = (i: number, j: number, state: number) => (i * width + j) * 5 + state;
  const cost = new Float64Array((r.length + 1) * width * 5).fill(Infinity);
  const closed = new Float64Array(cost.length).fill(Infinity);
  cost[0] = closed[0] = 0;
  for (let i = 0; i <= r.length; i++) for (let j = 0; j <= o.length; j++) for (let state = 0; state < 5; state++) {
    const from = cell(i, j, state);
    if (!Number.isFinite(cost[from])) continue;
    const step = (di: number, dj: number, charge: number) => {
      let activeR = state === 1 || state >= 3, activeO = state === 2 || state >= 3;
      let joined = state === 4;
      if (charge >= UNIT) {
        if (di) activeR = true;
        if (dj) activeO = true;
        if (di && dj) joined = true;
      }
      const ni = i + di, nj = j + dj;
      const keepR = activeR && ni < r.length && r[ni].run === r[i]?.run;
      const keepO = activeO && nj < o.length && o[nj].run === o[j]?.run;
      const finished = joined ? Number(!keepR && !keepO) : Number(activeR && !keepR) + Number(activeO && !keepO);
      const next = keepR && keepO ? (joined ? 4 : 3) : keepR ? 1 : keepO ? 2 : 0;
      const to = cell(ni, nj, next), c = cost[from] + charge, h = closed[from] + finished;
      if (c < cost[to] || (c === cost[to] && h < closed[to])) { cost[to] = c; closed[to] = h; }
    };
    if (i < r.length && j < o.length) step(1, 1, subCost(a[r[i].at], b[o[j].at]));
    if (i < r.length) step(1, 0, deletionCost(a, r[i].at));
    if (j < o.length) step(0, 1, insertionCost(a, r[i]?.at ?? a.length, b[o[j].at]));
  }
  return closed[cell(r.length, o.length, 0)];
}

/** Shared by checker instances; only supported reference pronunciations enter the whole-word vocabulary. */
const cache = new WeakMap<ReferenceIndex, ReturnType<typeof build>>();
function build(index: ReferenceIndex) {
  const forms = index.words.map((w, i) => [...new Set([
    index.phonemes.slice(w.start, w.end), ...(index.wordForms?.[i] ?? []),
  ].filter(Boolean).map(form))]);
  const isolated = new Set((index.isolatedWordForms ?? []).filter(Boolean).map(form));
  const connected = new Map<string, Set<string>>();
  const sameAyah = (a: number, b: number) => b >= 0 && b < index.words.length &&
    index.words[a].surah === index.words[b].surah && index.words[a].ayah === index.words[b].ayah;
  const neighbor = (wi: number, direction: number) => {
    let at = wi + direction;
    while (sameAyah(wi, at) && index.words[at].start === index.words[at].end) at += direction;
    return sameAyah(wi, at) ? at : -1;
  };
  const word = (wi: number) => index.phonemes.slice(index.words[wi].start, index.words[wi].end);
  const boundary = (text: string, before: boolean) => before ? form(text).at(-1) : form(text)[0];
  const contextKey = (before?: string, after?: string) => `${before ?? "^"}|${after ?? "$"}`;
  index.words.forEach((w, wi) => {
    if (w.start === w.end) return;
    const before = neighbor(wi, -1), after = neighbor(wi, 1);
    const key = contextKey(before < 0 ? undefined : boundary(word(before), true),
      after < 0 ? undefined : boundary(word(after), false));
    const text = form(word(wi));
    const contexts = connected.get(text) ?? new Set<string>();
    contexts.add(key);
    connected.set(text, contexts);
  });
  const attested = new Set<string>();
  for (const list of forms) for (const f of list) if (f) attested.add(f);
  const rawByForm = new Map<string, string[]>();
  for (const list of index.wordForms ?? []) for (const raw of list) if (raw) {
    const key = form(raw);
    const have = rawByForm.get(key);
    if (have) { if (!have.includes(raw)) have.push(raw); }
    else rawByForm.set(key, [raw]);
  }
  // Forms that share a stem and differ by a short tail. Queried from the written word's own tail, never from a verse.
  const byStem = new Map<string, string[]>();
  for (const f of attested) for (let n = 0; n <= END && n < f.length; n++) {
    const stem = f.slice(0, f.length - n);
    if (stem.length <= n) continue;
    const list = byStem.get(stem);
    if (list) list.push(f);
    else byStem.set(stem, [f]);
  }
  const matches = (wi: number, heard: string) => forms[wi].includes(form(heard));
  /**
   * The heard word is another mushaf form that the written word reaches by adding or dropping an ending, and that
   * ending carries a consonant. Waqf and a held vowel change only vowels, tanween or a pause mark, which is one of
   * this word's own forms or has no consonant to anchor.
   */
  const ending = (wi: number, heardRaw: string): { observedConsonant: boolean } | null => {
    const heard = form(heardRaw);
    const own = forms[wi];
    if (!heard || !own.length || own.includes(heard)) return null;
    // A vowel-only or pause-mark tail of any own pronunciation is this word, not a different one.
    if (own.some(f => {
      const r = residual(f, heard);
      return !!r && !endSegment(r.sa, r.sb);
    })) return null;
    let written: { f: string; cost: number } | null = null;
    for (const f of own) {
      const r = residual(f, heard);
      if (!r || !endSegment(r.sa, r.sb)) continue;
      const cost = endingCost(r.stem, r.sa, r.sb);
      if (!written || cost < written.cost) written = { f, cost };
    }
    if (!written || written.cost < UNIT) return null;
    let best = Infinity;
    const stems = new Set<string>();
    for (let n = 0; n <= END && n < written.f.length; n++) {
      const stem = written.f.slice(0, written.f.length - n);
      if (stem.length > n) stems.add(stem);
    }
    for (const stem of stems) for (const alt of byStem.get(stem) ?? []) {
      if (own.includes(alt)) continue;
      const fromWritten = residual(written.f, alt);
      if (!fromWritten || !endSegment(fromWritten.sa, fromWritten.sb)) continue;
      const fromHeard = residual(alt, heard);
      if (!fromHeard) continue;
      best = Math.min(best, endingCost(fromHeard.stem, fromHeard.sa, fromHeard.sb));
    }
    // The other form has to beat the written word by a consonant, not by a cheaper way to spell the same tail.
    if (written.cost - best < UNIT) return null;
    const difference = residual(written.f, heard);
    return difference ? { observedConsonant: consonantsOf(difference.sb).length > 0 } : null;
  };
  /**
   * Attested pronunciations reached by dropping a consonant-bearing ending from this word. A pause or a held vowel
   * drops no consonant, so waqf and اذهبا→اذهب are not here. The raw strings keep madd and ghunna runs; the checker
   * aligns them with the next word so one heard consonant cannot pay for both.
   */
  const droppedSuffixes = (wi: number): string[] => {
    const written = form(word(wi));
    const own = forms[wi];
    if (!written || !own.length) return [];
    const stems = new Set<string>();
    for (let n = 0; n <= END && n < written.length; n++) {
      const stem = written.slice(0, written.length - n);
      if (stem.length > n) stems.add(stem);
    }
    const out: string[] = [];
    const seen = new Set<string>();
    for (const stem of stems) for (const alt of byStem.get(stem) ?? []) {
      if (own.includes(alt) || alt.length >= written.length) continue;
      const r = residual(written, alt);
      if (!r || !consonantsOf(r.sa).length || consonantsOf(r.sb).length) continue;
      for (const raw of rawByForm.get(alt) ?? []) {
        if (seen.has(raw)) continue;
        seen.add(raw);
        out.push(raw);
      }
    }
    return out;
  };
  // Sukun and pause marks. They are not consonants; a spelled word stays the same word without them.
  const MARK = new Set(["ڇ", "ۜ", "ؙ"]);
  const norm = (s: string) => collapseHeld([...s].filter(c => !MARK.has(c))).join("");
  const skel = (s: string) => [...s].filter(c => CONSONANTS.has(c)).join("");
  const wordNorms = forms.map(list => [...new Set(list.map(norm).filter(Boolean))]);
  const bySkel = new Map<string, string[]>();
  for (const list of wordNorms) for (const n of list) {
    const sk = skel(n);
    if (sk.length < 2) continue;
    const have = bySkel.get(sk);
    if (!have) bySkel.set(sk, [n]);
    else if (!have.includes(n)) have.push(n);
  };
  // Same consonants, and vowels only where the sounds differ. A shorter string is not that word.
  const same = (a: string, b: string) => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] === b[i]) continue;
      if (CONSONANTS.has(a[i]) || CONSONANTS.has(b[i]) || subCost(a[i], b[i]) >= UNIT) return false;
    }
    return true;
  };
  const nearIds = (wi: number) => {
    const before: number[] = [], after: number[] = [];
    for (let i = wi - 1; i >= 0 && before.length < 3; i--) if (index.words[i].start !== index.words[i].end) before.push(i);
    for (let i = wi + 1; i < index.words.length && after.length < 3; i++) if (index.words[i].start !== index.words[i].end) after.push(i);
    const ids = [...before.reverse(), wi, ...after];
    return index.words[wi].start === index.words[wi].end ? ids.filter(i => i !== wi) : ids;
  };
  /**
   * The span is a word, a prefix, or a contiguous run of words within ±3 of `wi`. Restarts and false starts
   * spell that neighbourhood; an added word does not.
   */
  const explainsSeam = (wi: number, heard: string, fragments = false) => {
    const n = norm(heard);
    if (!n) return true;
    const ids = nearIds(wi);
    const lists = ids.map(i => fragments ? [...new Set([
      ...wordNorms[i], norm(index.isolatedWordForms?.[i] ?? ""),
    ].filter(Boolean))] : wordNorms[i]);
    const matches = (f: string) => {
      if (fragments) {
        for (let a = 0; a + n.length <= f.length; a++) if (same(n, f.slice(a, a + n.length))) return true;
        return false;
      }
      return same(n, f) || (n.length < f.length && same(n, f.slice(0, n.length)));
    };
    for (const list of lists) for (const f of list) if (matches(f)) return true;
    for (let a = 0; a < ids.length; a++) {
      let combos = [""];
      for (let b = a; b < ids.length && b < a + 4; b++) {
        const next: string[] = [];
        for (const c of combos) for (const f of lists[b]) next.push(c + f);
        const uniq = [...new Set(next)];
        if (b > a) for (const acc of uniq) {
          if (matches(acc)) return true;
        }
        combos = uniq.slice(0, 8);
      }
    }
    return false;
  };
  /**
   * A whole mushaf word of at least two consonants, equal under soft costs only: no hard consonant edit and no
   * dropped or added phone. Null when the span is a neighbour, a neighbour prefix, or a run within ±3.
   */
  const surplusWord = (wi: number, heard: string): string | null => {
    const n = norm(heard);
    const sk = skel(n);
    if (sk.length < 2) return null;
    for (const f of bySkel.get(sk) ?? []) if (same(n, f)) return explainsSeam(wi, heard) ? null : f;
    return null;
  };
  return {
    matches,
    ending,
    droppedSuffixes,
    surplusWord,
    /** A substring of the nearby text can leave enough of its word to beat total omission.
     * Excision must not call that fragment surplus, even when it is itself an attested word. */
    fragmentOfSeam: (wi: number, heard: string) => explainsSeam(wi, heard, true),
    alternative(wi: number, heard: string, context?: { before?: string; after?: string }): boolean {
      const normalized = form(heard);
      const key = context && contextKey(context.before === undefined ? undefined : boundary(context.before, true),
        context.after === undefined ? undefined : boundary(context.after, false));
      const known = isolated.has(normalized) || (key !== undefined && connected.get(normalized)?.has(key));
      return !!known && !matches(wi, heard) &&
        forms[wi].length > 0 && forms[wi].every(expected => hardConsonantEdits(expected, normalized) >= 2);
    },
  };
}

export function lexicalWords(index: ReferenceIndex) {
  let words = cache.get(index);
  if (!words) { words = build(index); cache.set(index, words); }
  return words;
}
