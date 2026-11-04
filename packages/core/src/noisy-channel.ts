import { collapseHeld, deletionCost, insertionCost, subCost } from "./fuzzy.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";

/**
 * P(heard token | written phone) of the shipped listener, counted by {@link accumulateChannel} on
 * gold-excluded decodes. Additive smoothing is the estimator, not a fitted gate. A deletion is the
 * written phone producing nothing (the blank outcome of that phone). An insertion is a heard token
 * with no written phone.
 */
export interface ChannelCounts {
  phones: string[];
  /** Substitution counts, row = written phone, column = heard token, pooled over word position. */
  sub: number[][];
  /** Deletion counts per written phone. */
  del: number[];
  /** Insertion counts per heard token. */
  ins: number[];
  /** Written phones that were aligned (substitutions + deletions). */
  nWritten: number;
  /**
   * Highest log-odds by which a non-neighbour word beat the best competitor on a clean
   * gold-excluded phone or studio seam, under the same span score. Set by the estimator.
   */
  cut: number;
}

const ALPHA = 0.5;
const CONSONANTS = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
/** One consonant substitution or deletion still reaches the full alignment. Two do not. */
const MAX_CONSONANT_MISS = 1;

export interface Channel {
  phones: string[];
  index: Map<string, number>;
  /** log P(heard | written), including the match. Rows sum with `del` over the simplex. */
  logSub: Float64Array[];
  logDel: Float64Array;
  /** log probability of inserting this heard token, scaled per written phone. */
  logIns: Float64Array;
  cut: number;
}

export function channelFromCounts(c: ChannelCounts): Channel {
  const n = c.phones.length;
  const index = new Map(c.phones.map((p, i) => [p, i]));
  const logSub: Float64Array[] = [];
  const logDel = new Float64Array(n);
  for (let w = 0; w < n; w++) {
    let total = c.del[w];
    for (let h = 0; h < n; h++) total += c.sub[w][h];
    const z = total + ALPHA * (n + 1);
    const row = new Float64Array(n);
    for (let h = 0; h < n; h++) row[h] = Math.log((c.sub[w][h] + ALPHA) / z);
    logDel[w] = Math.log((c.del[w] + ALPHA) / z);
    logSub.push(row);
  }
  const logIns = new Float64Array(n);
  const zIns = Math.max(1, c.nWritten) + ALPHA * n;
  for (let h = 0; h < n; h++) logIns[h] = Math.log((c.ins[h] + ALPHA) / zIns);
  return { phones: c.phones, index, logSub, logDel, logIns, cut: c.cut };
}

const NEG = -1e9;

/** Viterbi log P(heard | written) under the channel. */
export function spanLogP(heard: readonly string[], written: readonly string[], ch: Channel): number {
  const n = heard.length;
  const m = written.length;
  const hi = heard.map((p) => ch.index.get(p) ?? -1);
  const wi = written.map((p) => ch.index.get(p) ?? -1);
  const w = m + 1;
  const dp = new Float64Array((n + 1) * w).fill(NEG);
  const at = (i: number, j: number) => i * w + j;
  dp[0] = 0;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= m; j++) {
      const cur = dp[at(i, j)];
      if (cur <= NEG / 2) continue;
      if (i < n) {
        const h = hi[i];
        const add = h < 0 ? Math.log(ALPHA / (ALPHA * ch.phones.length)) : ch.logIns[h];
        const to = at(i + 1, j);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
      if (j < m) {
        const wr = wi[j];
        const add = wr < 0 ? Math.log(ALPHA / (ALPHA * (ch.phones.length + 1))) : ch.logDel[wr];
        const to = at(i, j + 1);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
      if (i < n && j < m) {
        const h = hi[i];
        const wr = wi[j];
        const add = h < 0 || wr < 0 ? Math.log(ALPHA / (ALPHA * (ch.phones.length + 1))) : ch.logSub[wr][h];
        const to = at(i + 1, j + 1);
        if (cur + add > dp[to]) dp[to] = cur + add;
      }
    }
  }
  return dp[at(n, m)];
}

/**
 * Count listener confusions along one minimum-cost alignment of a gold-excluded decode to its
 * canonical phones. Costs are the checker's existing edit costs, so the path is not fit here.
 */
export function accumulateChannel(counts: ChannelCounts, writtenRaw: string, heardRaw: string): void {
  const index = new Map(counts.phones.map((p, i) => [p, i]));
  const written = collapseHeld([...writtenRaw]);
  const heard = collapseHeld([...heardRaw]);
  const n = written.length;
  const m = heard.length;
  const w = m + 1;
  const dp = new Float64Array((n + 1) * w);
  const bt = new Uint8Array((n + 1) * w);
  const at = (i: number, j: number) => i * w + j;
  for (let j = 1; j <= m; j++) {
    dp[j] = dp[j - 1] + insertionCost("", 0, heard[j - 1]);
    bt[j] = 1;
  }
  for (let i = 1; i <= n; i++) {
    dp[at(i, 0)] = dp[at(i - 1, 0)] + deletionCost(written.join(""), i - 1);
    bt[at(i, 0)] = 2;
    for (let j = 1; j <= m; j++) {
      const sub = dp[at(i - 1, j - 1)] + subCost(written[i - 1], heard[j - 1]);
      const ins = dp[at(i, j - 1)] + insertionCost(written.join(""), i, heard[j - 1]);
      const del = dp[at(i - 1, j)] + deletionCost(written.join(""), i - 1);
      let best = sub;
      let how = 0;
      if (ins < best) { best = ins; how = 1; }
      if (del < best) { best = del; how = 2; }
      dp[at(i, j)] = best;
      bt[at(i, j)] = how;
    }
  }
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    const how = bt[at(i, j)];
    if (how === 0 && i > 0 && j > 0) {
      const wr = index.get(written[i - 1]);
      const h = index.get(heard[j - 1]);
      if (wr !== undefined && h !== undefined) counts.sub[wr][h] += 1;
      if (wr !== undefined) counts.nWritten += 1;
      i--;
      j--;
    } else if (how === 1 && j > 0) {
      const h = index.get(heard[j - 1]);
      if (h !== undefined) counts.ins[h] += 1;
      j--;
    } else if (i > 0) {
      const wr = index.get(written[i - 1]);
      if (wr !== undefined) { counts.del[wr] += 1; counts.nWritten += 1; }
      i--;
    } else break;
  }
}

export function emptyCounts(phones: readonly string[], cut = 0): ChannelCounts {
  const n = phones.length;
  return {
    phones: [...phones],
    sub: Array.from({ length: n }, () => Array(n).fill(0)),
    del: Array(n).fill(0),
    ins: Array(n).fill(0),
    nWritten: 0,
    cut,
  };
}

const consonantsOf = (s: readonly string[]) => s.filter((p) => CONSONANTS.has(p)).join("");

/** Consonant edit distance, aborting past `max`. */
function editAtMost(a: string, b: string, max: number): boolean {
  if (Math.abs(a.length - b.length) > max) return false;
  const row = new Uint16Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) row[j] = j;
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    let best = row[0];
    for (let j = 1; j <= b.length; j++) {
      const cur = row[j];
      const sub = prev + (a[i - 1] === b[j - 1] ? 0 : 1);
      const del = row[j] + 1;
      const ins = row[j - 1] + 1;
      const next = sub < del ? (sub < ins ? sub : ins) : (del < ins ? del : ins);
      prev = cur;
      row[j] = next;
      if (next < best) best = next;
    }
    if (best > max) return false;
  }
  return row[b.length] <= max;
}

/**
 * The word's consonants sit in some window of the span with at most one edit. A blurred consonant
 * still passes; a word that is not in the span does not reach the full alignment.
 */
export function consonantWindow(word: readonly string[], heardCons: string): boolean {
  const a = consonantsOf(word);
  if (a.length < 2 || a.length > heardCons.length + MAX_CONSONANT_MISS) return false;
  const lo = Math.max(1, a.length - MAX_CONSONANT_MISS);
  const hi = Math.min(heardCons.length, a.length + MAX_CONSONANT_MISS);
  for (let n = lo; n <= hi; n++) {
    for (let i = 0; i + n <= heardCons.length; i++) {
      if (editAtMost(a, heardCons.slice(i, i + n), MAX_CONSONANT_MISS)) return true;
    }
  }
  return false;
}

export interface SeamRank {
  /** Best whole mushaf word that is not a neighbour, prefix, back-run or phrase. */
  word: string | null;
  wordLogP: number;
  competitor: string;
  competitorLogP: number;
  /** wordLogP − competitorLogP. Positive means the extra word explains the span better. */
  logOdds: number;
}

export interface LexEntry {
  form: string;
  phones: string[];
  cons: string;
}

export interface SeamLexicon {
  forms: string[][];
  /** Collapsed form -> original, one representative. */
  byForm: Map<string, string[]>;
  entries: LexEntry[];
}

const lexCache = new WeakMap<ReferenceIndex, SeamLexicon>();

export function seamLexicon(index: ReferenceIndex): SeamLexicon {
  const hit = lexCache.get(index);
  if (hit) return hit;
  const byForm = new Map<string, string[]>();
  const forms = index.words.map((w, i) => {
    const list = [...new Set([
      index.phonemes.slice(w.start, w.end),
      ...(index.wordForms?.[i] ?? []),
      index.isolatedWordForms?.[i] ?? "",
    ].filter(Boolean))];
    for (const raw of list) {
      const key = collapseHeld([...raw]).join("");
      const have = byForm.get(key);
      if (have) { if (!have.includes(raw)) have.push(raw); }
      else byForm.set(key, [raw]);
    }
    return list;
  });
  const entries: LexEntry[] = [];
  for (const form of byForm.keys()) {
    const phones = [...form];
    const cons = consonantsOf(phones);
    if (cons.length >= 2) entries.push({ form, phones, cons });
  }
  const built = { forms, byForm, entries };
  lexCache.set(index, built);
  return built;
}

function tokens(s: string): string[] {
  return collapseHeld([...s]);
}

/**
 * Rank an extra mushaf word inside the seam. The span runs from the previous ayah's last word through
 * word 1, so every hypothesis keeps those two words and the candidates differ by the one word between
 * them. Competitors are that written pair with nothing inserted, each neighbour word and prefix within
 * ±3, a contiguous back-run ending at the previous word, non-Quran phrases, and a no-word insertion model.
 */
export function rankSeam(
  heardRaw: string,
  word1: number,
  index: ReferenceIndex,
  phrases: readonly string[],
  ch: Channel,
): SeamRank {
  const heard = tokens(heardRaw);
  const heardCons = consonantsOf(heard);
  const lex = seamLexicon(index);
  const prev = (() => {
    let k = word1 - 1;
    while (k > 0 && index.words[k].end === index.words[k].start) k--;
    return k;
  })();
  const near: number[] = [];
  for (let i = word1; i >= 0 && near.length < 4; i--) if (index.words[i].start !== index.words[i].end) near.push(i);
  near.reverse();
  for (let i = word1 + 1; i < index.words.length && near.length < 7; i++) if (index.words[i].start !== index.words[i].end) near.push(i);
  const formsOf = (wi: number) => (wi >= 0 ? lex.forms[wi] ?? [] : []);
  let bestPrev: string[] = [];
  let bestNext: string[] = [];
  let writtenLogP = NEG;
  for (const a of formsOf(prev)) for (const b of formsOf(word1)) {
    const pa = tokens(a);
    const pb = tokens(b);
    const logP = spanLogP(heard, [...pa, ...pb], ch);
    if (logP > writtenLogP) { writtenLogP = logP; bestPrev = pa; bestNext = pb; }
  }
  const competitorForms: { name: string; phones: string[] }[] = [{ name: "written", phones: [...bestPrev, ...bestNext] }];
  const blocked = new Set<string>();
  const addBlocked = (raw: string, sole: boolean) => {
    const t = tokens(raw);
    if (t.length < 2) return;
    blocked.add(t.join(""));
    if (sole) competitorForms.push({ name: t.join(""), phones: t });
    for (let k = 2; k < t.length; k++) {
      const pre = t.slice(0, k);
      blocked.add(pre.join(""));
      if (sole) competitorForms.push({ name: `prefix:${pre.join("")}`, phones: pre });
    }
  };
  for (const wi of near) for (const f of formsOf(wi)) addBlocked(f, true);
  if (prev >= 0) {
    let run = "";
    let n = 0;
    for (let i = prev; i >= 0 && n < 4; i--) {
      if (index.words[i].start === index.words[i].end) continue;
      run = index.phonemes.slice(index.words[i].start, index.words[i].end) + run;
      n++;
      if (n >= 2) addBlocked(run, true);
    }
  }
  for (const p of phrases) addBlocked(p, true);
  let bestC = { name: "noise", logP: spanLogP(heard, [], ch) };
  const seenC = new Set<string>();
  for (const c of competitorForms) {
    const key = c.name === "written" ? "written" : c.phones.join("");
    if (seenC.has(key)) continue;
    seenC.add(key);
    const logP = c.name === "written" ? writtenLogP : spanLogP(heard, c.phones, ch);
    if (logP > bestC.logP) bestC = { name: c.name, logP };
  }
  const present = new Set(heardCons);
  let bestW: { form: string; logP: number } | null = null;
  for (const entry of lex.entries) {
    if (blocked.has(entry.form)) continue;
    let miss = 0;
    for (const c of entry.cons) if (!present.has(c)) miss++;
    if (miss > MAX_CONSONANT_MISS || !consonantWindow(entry.phones, heardCons)) continue;
    const logP = spanLogP(heard, [...bestPrev, ...entry.phones, ...bestNext], ch);
    if (!bestW || logP > bestW.logP) bestW = { form: entry.form, logP };
  }
  const wordLogP = bestW?.logP ?? NEG;
  return {
    word: bestW?.form ?? null,
    wordLogP,
    competitor: bestC.name,
    competitorLogP: bestC.logP,
    logOdds: wordLogP - bestC.logP,
  };
}

export function phrasePhones(phrases: PhraseAsset | undefined): string[] {
  return phrases?.phrases.map((p) => p.p) ?? [];
}

/** A tracker step onto a word: the time the locked word becomes `wi`. */
export interface TrackerStep { wi: number; timeMs: number }

/**
 * Ayah seams the tracker actually walked: from its step onto the previous ayah's last word to its
 * step onto word 2. The pair has to be one walk. A re-lock from much earlier is not this seam.
 */
const SEAM_WALK_MS = 20_000;

export function ayahSeamSpans(steps: readonly TrackerStep[], index: ReferenceIndex): { word1: number; t0: number; t1: number }[] {
  const words = index.words;
  const ayahFirst: number[] = [];
  const ayahOf: number[] = [];
  words.forEach((w, i) => {
    const prev = words[i - 1];
    if (!prev || prev.surah !== w.surah || prev.ayah !== w.ayah) ayahFirst.push(i);
    ayahOf.push(ayahFirst.length - 1);
  });
  const content = (wi: number, dir: number) => {
    let k = wi;
    while (k >= 0 && k < words.length && words[k].end === words[k].start) k += dir;
    return k;
  };
  const out: { word1: number; t0: number; t1: number }[] = [];
  const seen = new Set<number>();
  const lastTime = new Map<number, number>();
  for (const s of steps) {
    lastTime.set(s.wi, s.timeMs);
    const w1 = ayahFirst[ayahOf[s.wi] ?? 0];
    const word2 = content(w1 + 1, 1);
    if (s.wi !== word2 || seen.has(w1)) continue;
    const prev = content(w1 - 1, -1);
    if (prev < 0 || ayahOf[prev] === ayahOf[w1]) continue;
    const t0 = lastTime.get(prev);
    if (t0 === undefined || s.timeMs <= t0 || s.timeMs - t0 > SEAM_WALK_MS) continue;
    seen.add(w1);
    out.push({ word1: w1, t0, t1: s.timeMs });
  }
  return out;
}

/** Greedy tokens whose onset lies in `[t0, t1)`. */
export function spanTokens(phonemes: readonly { p: string; start: number }[], frameMs: number, t0: number, t1: number): string {
  let s = "";
  for (const p of phonemes) {
    const t = p.start * frameMs;
    if (t >= t0 && t < t1) s += p.p;
  }
  return s;
}
