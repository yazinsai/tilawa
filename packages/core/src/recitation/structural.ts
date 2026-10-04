/**
 * Structural correction rules over the free CTC decode. Quran text only, no
 * learned weights. Port of the lab prototypes `lab/scripts/ayah_order.py` and
 * `lab/scripts/similar_verse.py` (frozen settings).
 *
 * - **Ayah order** (`possible_skipped_ayah`): cut the token stream at pauses,
 *   match each segment to a word span of a local window of ayahs, and compare
 *   an in-order chain of the segments with one that may jump whole ayahs.
 * - **Similar verse** (`possible_substitution` / `possible_omission`): align
 *   the heard phonemes to the passage, and to the passage with one region of
 *   one ayah swapped for a look-alike ayah's wording or one interior word
 *   dropped.
 *
 * Pure functions over the corpus, the bundled look-alike index and timed
 * tokens; the session decides when to run them.
 */
import type { QuranCorpus } from "./corpus.js";
import { costTable, TABLE_SIZE } from "./phonemeCost.js";
import { BASMALA, ISTIADHA } from "./search.js";

/** A free-decode token: symbol, absolute encoder frame (counted across decoder
 * resets), and the audio time in seconds when it was decoded. */
export interface TimedToken {
  sym: string;
  frame: number;
  t: number;
}

/** `structural-index.json`: look-alike pairs (`"s:a" -> [[ms, ma, tag, i1, i2, j1, j2, ...]]`,
 * tag 0 replace / 1 delete / 2 insert) and the ayahs ayah-order never flags. */
export interface StructuralIndexJson {
  v: number;
  pairs: Record<string, number[][]>;
  ident: string[];
}

export interface AyahOrderParams {
  /** Blank frames between two tokens that cut a segment (12 = 0.48 s). */
  gap: number;
  /** Shorter segments are merged into the previous one. */
  minChars: number;
  /** Cost per char of unexplained audio. */
  garbage: number;
  /** Cost per skipped char inside the in-order chain. */
  wcost: number;
  /** Cost of going back (restart / repeat). */
  restart: number;
  /** Cost of a forward jump over a whole ayah. */
  jump: number;
  /** Without a passage, the window is [first − back .. first + ahead]. */
  back: number;
  ahead: number;
}

export const AYAH_ORDER_PARAMS: Readonly<AyahOrderParams> = {
  gap: 12, minChars: 3, garbage: 0.5, wcost: 0.5, restart: 2.0, jump: 0.5, back: 1, ahead: 4,
};

export interface AyahOrderRule {
  /** Absolute margin (cost units) that flags on its own. */
  margin: number;
  /** Or: margin per char of the segment after the jump. */
  rel: number;
  /** Both flanking segments must fit at or below this cost per char. */
  fit: number;
  /** Max chars of unexplained audio between the flanking segments. */
  between: number;
  minPost: number;
  /** Never flag an ayah whose skip reads as in-order or repeated text. */
  noIdent: boolean;
  /** Drop a jump after a restart segment that also fits the skipped ayah's ending. */
  noRestartPre?: boolean;
  restartAlt?: number;
  /** Max unexplained chars as a fraction of the skipped ayah's length. */
  betweenRel?: number;
}

/** Frozen on a0w dev (acted dev skip_ayah + help clean dev + v1). */
export const AYAH_ORDER_RULE: Readonly<AyahOrderRule> = {
  margin: 99, rel: 0.04, fit: 0.3, between: 8, minPost: 0, noIdent: true,
};
/** Adds the two guards frozen for the shipped emissions. */
export const AYAH_ORDER_RULE_GUARDED: Readonly<AyahOrderRule> = {
  ...AYAH_ORDER_RULE, noRestartPre: true, restartAlt: 0.3, betweenRel: 0.2,
};

export interface SimilarVerseRule {
  /** Look-alike swap: margin over the canonical text (cost units). */
  lookalike: number;
  /** One- or two-word drop: margin over the canonical text. */
  drop: number;
  /** Max words a variant may replace or drop. */
  maxSpan: number;
  /** Region ± one word must align at or below this cost per char. */
  loc: number;
  /** The whole ayah must align at or below this cost per char. */
  ayahFit: number;
}

/** Frozen on a0w dev (look-alike + drop, one-word regions). */
export const SIMILAR_VERSE_RULE: Readonly<SimilarVerseRule> = {
  lookalike: 3.5, drop: 10, maxSpan: 1, loc: 0.25, ayahFit: 0.5,
};

export interface AyahOrderJump {
  surah: number;
  ayah: number;
  fit0: number;
  fit1: number;
  between: number;
  readLater: boolean;
  /** Seconds: when the segment after the jump had been decoded. */
  at: number;
  postChars: number;
  preChars: number;
  preRestart: boolean;
  preOnSkipped: number;
  skipChars: number;
  ident: boolean;
}

export interface AyahOrderCandidate {
  margin: number;
  jumps: AyahOrderJump[];
  segs: number;
}

export interface StructuralFlag {
  kind: "possible_skipped_ayah" | "possible_substitution" | "possible_omission";
  surah: number;
  ayah: number;
  word: number;
  /** Seconds into the audio. */
  atSeconds: number;
  source: "ayah_order" | "similar_verse";
}

export interface SimilarVerseCandidate {
  slot: number;
  surah: number;
  ayah: number;
  word: number;
  kind: "possible_substitution" | "possible_omission";
  family: "lookalike" | "drop";
  span: number;
  from: [number, number];
  margin: number;
  marginAll: number;
  known: boolean;
  at: number | null;
  loc: number | null;
  ayahD: number | null;
}

interface Look {
  m: [number, number];
  regions: Array<[number, number, number, number, number]>;
}

interface Variant {
  words: string[];
  word: number;
  end: number;
  kind: SimilarVerseCandidate["kind"];
  family: SimilarVerseCandidate["family"];
  span: number;
  from: [number, number];
  /** Slot-local word range the variant changes: words[0..i1) and [i2..) are the ayah's own. */
  i1: number;
  i2: number;
  mid: string[];
}

const PREAMBLES = [ISTIADHA, BASMALA, ISTIADHA + BASMALA];
const REPLACE = 0;
const INSERT = 2;
const SKIP = 0.5;
const COST = costTable().matrix;

/** Python's `round(x, nd)`: round half to even on the exact binary value. */
export function pyRound(x: number, nd: number): number {
  if (!Number.isFinite(x)) return x;
  const exact = Math.abs(x).toFixed(30);
  const dot = exact.indexOf(".");
  const tail = exact.slice(dot + 1 + nd);
  let r = Number(Math.abs(x).toFixed(nd));
  if (tail[0] === "5" && /^50*$/.test(tail)) {
    const digits = exact.slice(0, dot) + exact.slice(dot + 1, dot + 1 + nd);
    const even = Number(digits[digits.length - 1]) % 2 === 0;
    r = (Number(digits) + (even ? 0 : 1)) / 10 ** nd;
  }
  return x < 0 ? -r : r;
}

const encode = (text: string): Uint8Array => costTable().encode(text);

/** C[i * nw + e]: edit cost of the whole query against words i..e of `ref`. */
export function spanCosts(q: Uint8Array, ref: Uint8Array, ws: Int32Array, maxRatio = 2.0, slack = 12.0): Float64Array {
  const nw = ws.length - 1;
  const n = q.length;
  const out = new Float64Array(nw * nw).fill(Infinity);
  const prev = new Float64Array(ref.length + 1);
  const cur = new Float64Array(ref.length + 1);
  const lim = Math.trunc(n * maxRatio + slack);
  for (let i = 0; i < nw; i++) {
    const r0 = ws[i]!;
    const m = Math.min(ref.length - r0, lim);
    if (m <= 0) continue;
    for (let j = 0; j <= m; j++) prev[j] = j;
    for (let a = 1; a <= n; a++) {
      cur[0] = a;
      const row = q[a - 1]! * TABLE_SIZE;
      for (let j = 1; j <= m; j++) {
        let c = prev[j - 1]! + COST[row + ref[r0 + j - 1]!]!;
        const u = prev[j]! + 1;
        if (u < c) c = u;
        const lf = cur[j - 1]! + 1;
        if (lf < c) c = lf;
        cur[j] = c;
      }
      for (let j = 0; j <= m; j++) prev[j] = cur[j]!;
    }
    for (let e = i; e < nw; e++) {
      const L = ws[e + 1]! - r0;
      if (L > m) break;
      out[i * nw + e] = prev[L]!;
    }
  }
  return out;
}

/** Plain edit cost of `q` against `r`. */
export function globalCost(q: Uint8Array, r: Uint8Array): number {
  const m = r.length;
  let prev = new Float64Array(m + 1);
  let cur = new Float64Array(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;
  for (let a = 1; a <= q.length; a++) {
    cur[0] = a;
    const row = q[a - 1]! * TABLE_SIZE;
    for (let j = 1; j <= m; j++) {
      let c = prev[j - 1]! + COST[row + r[j - 1]!]!;
      const u = prev[j]! + 1;
      if (u < c) c = u;
      const lf = cur[j - 1]! + 1;
      if (lf < c) c = lf;
      cur[j] = c;
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m]!;
}

export interface Segment {
  text: string;
  f0: number;
  f1: number;
  t: number;
}

/** Token stream -> pause-delimited segments. */
export function segmentTokens(tokens: readonly TimedToken[], gap: number, minChars: number): Segment[] {
  const toks = [...tokens].sort((a, b) => a.frame - b.frame);
  const segs: Segment[] = [];
  for (const { sym, frame, t } of toks) {
    const last = segs[segs.length - 1];
    if (last && frame - last.f1 < gap) {
      last.text += sym;
      last.f1 = frame;
      last.t = Math.max(last.t, t);
    } else segs.push({ text: sym, f0: frame, f1: frame, t });
  }
  const out: Segment[] = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && s.text.length < minChars) {
      last.text += s.text;
      last.f1 = s.f1;
      last.t = Math.max(last.t, s.t);
    } else out.push(s);
  }
  if (out.length > 1 && out[0]!.text.length < minChars) {
    out[1]!.text = out[0]!.text + out[1]!.text;
    out[1]!.f0 = out[0]!.f0;
    out.shift();
  }
  return out;
}

type Step = { kind: "m"; i: number; e: number } | { kind: "g" } | { kind: "p" };

interface ChainResult {
  best: number;
  path: Step[];
  skipped: Map<number, number[]>;
}

interface ChainInput {
  costs: Float64Array[];
  pre: number[];
  segLen: number[];
  wordsAyah: number[];
  wlenCum: number[];
  ayahFirst: number[];
  ayahLast: number[];
  p: AyahOrderParams;
}

/** DP over segments; state = last word read (nw = nothing yet). */
function chain(inp: ChainInput, free: boolean): ChainResult {
  const { costs, pre, segLen, wordsAyah, wlenCum, ayahFirst, ayahLast, p } = inp;
  const nw = wordsAyah.length;
  const T = new Float64Array((nw + 1) * nw).fill(Infinity);
  for (let i = 0; i < nw; i++) T[nw * nw + i] = 0;
  const skipped = new Map<number, number[]>();
  for (let pe = 0; pe < nw; pe++) {
    const nxt = pe + 1;
    for (let i = 0; i < nw; i++) {
      if (i <= pe) {
        T[pe * nw + i] = p.restart;
        continue;
      }
      const full: number[] = [];
      for (let x = 0; x < ayahFirst.length; x++) if (ayahFirst[x]! >= nxt && ayahLast[x]! < i) full.push(x);
      const gapChars = wlenCum[i]! - wlenCum[nxt]!;
      if (!full.length) T[pe * nw + i] = p.wcost * gapChars;
      else if (free) {
        let fullChars = 0;
        for (const x of full) fullChars += wlenCum[ayahLast[x]! + 1]! - wlenCum[ayahFirst[x]!]!;
        T[pe * nw + i] = p.jump + p.wcost * (gapChars - fullChars);
        skipped.set(pe * nw + i, full);
      }
    }
  }
  let cur = new Float64Array(nw + 1).fill(Infinity);
  cur[nw] = 0;
  const back: Array<Array<{ k: "m"; i: number; s: number } | { k: "g" | "p"; s: number } | null>> = [];
  const A = new Float64Array(nw);
  const Aarg = new Int32Array(nw);
  for (let k = 0; k < costs.length; k++) {
    const C = costs[k]!;
    for (let i = 0; i < nw; i++) {
      let best = cur[0]! + T[i]!;
      let arg = 0;
      for (let s = 1; s <= nw; s++) {
        const v = cur[s]! + T[s * nw + i]!;
        if (v < best) { best = v; arg = s; }
      }
      A[i] = best;
      Aarg[i] = arg;
    }
    const next = new Float64Array(nw + 1).fill(Infinity);
    const bp: Array<{ k: "m"; i: number; s: number } | { k: "g" | "p"; s: number } | null> = new Array(nw + 1).fill(null);
    for (let e = 0; e < nw; e++) {
      let best = A[0]! + C[e]!;
      let bi = 0;
      for (let i = 1; i < nw; i++) {
        const v = A[i]! + C[i * nw + e]!;
        if (v < best) { best = v; bi = i; }
      }
      next[e] = best;
      if (Number.isFinite(best)) bp[e] = { k: "m", i: bi, s: Aarg[bi]! };
    }
    const g = p.garbage * segLen[k]!;
    for (let s = 0; s <= nw; s++) {
      if (cur[s]! + g < next[s]!) {
        next[s] = cur[s]! + g;
        bp[s] = { k: "g", s };
      }
    }
    if (cur[nw]! + pre[k]! < next[nw]!) {
      next[nw] = cur[nw]! + pre[k]!;
      bp[nw] = { k: "p", s: nw };
    }
    back.push(bp);
    cur = next;
  }
  let end = 0;
  for (let s = 1; s <= nw; s++) if (cur[s]! < cur[end]!) end = s;
  const best = cur[end]!;
  const path: Step[] = [];
  let s = end;
  for (let k = costs.length - 1; k >= 0; k--) {
    const b = back[k]![s];
    if (!b) return { best, path: [], skipped };
    if (b.k === "m") {
      path.push({ kind: "m", i: b.i, e: s });
      s = b.s;
    } else {
      path.push({ kind: b.k });
      s = b.s;
    }
  }
  path.reverse();
  return { best, path, skipped };
}

export type Ayah = readonly [number, number];

/** Look-alike index plus the ident guard, bound to a corpus. */
export class StructuralRules {
  private readonly looks = new Map<string, Look[]>();
  private readonly ident: Set<string>;
  private readonly phCache = new Map<string, string[]>();
  private readonly segCache = new Map<string, { costs: Float64Array; pre: number; len: number }>();

  constructor(readonly corpus: QuranCorpus, index: StructuralIndexJson) {
    for (const [key, rows] of Object.entries(index.pairs)) {
      this.looks.set(key, rows.map((r) => {
        const regions: Look["regions"] = [];
        for (let x = 2; x + 4 < r.length; x += 5) regions.push([r[x]!, r[x + 1]!, r[x + 2]!, r[x + 3]!, r[x + 4]!]);
        return { m: [r[0]!, r[1]!], regions };
      }));
    }
    this.ident = new Set(index.ident);
  }

  has(s: number, a: number): boolean {
    return this.corpus.hasAyah(s, a);
  }

  /** Phoneme words of an ayah. */
  ph(s: number, a: number): string[] {
    const key = `${s}:${a}`;
    let w = this.phCache.get(key);
    if (!w) {
      const first = this.corpus.ayahFirstWord(s, a);
      const n = this.corpus.ayahWordCount(s, a);
      w = Array.from({ length: n }, (_, i) => this.corpus.wordPhonemes(first + i));
      this.phCache.set(key, w);
    }
    return w;
  }

  lookalikes(s: number, a: number): readonly Look[] {
    return this.looks.get(`${s}:${a}`) ?? [];
  }

  identGuarded(s: number, a: number): boolean {
    return this.ident.has(`${s}:${a}`);
  }

  /** Window of ayahs for the ayah-order check: the expected passage, or
   * [first − back .. first + ahead] around the tracker's first verse. */
  ayahOrderWindow(expected: { surah: number; ayah: number; ayahEnd?: number } | null, firstVerse: Ayah | null,
    p: AyahOrderParams = AYAH_ORDER_PARAMS): Ayah[] {
    if (expected) {
      const s = expected.surah;
      const out: Ayah[] = [];
      for (let a = expected.ayah; a <= (expected.ayahEnd ?? expected.ayah); a++) if (this.has(s, a)) out.push([s, a]);
      return out;
    }
    if (!firstVerse) return [];
    const [s, a] = firstVerse;
    const out: Ayah[] = [];
    for (let x = Math.max(1, a - p.back); x <= a + p.ahead; x++) if (this.has(s, x)) out.push([s, x]);
    return out;
  }

  /** `ayah_order.detect_take`: candidate jumps with their evidence; thresholds are applied by {@link ayahOrderFlags}. */
  ayahOrderCandidate(tokens: readonly TimedToken[], win: readonly Ayah[],
    p: AyahOrderParams = AYAH_ORDER_PARAMS): AyahOrderCandidate | null {
    if (win.length < 3 || tokens.length < 4) return null;
    const segs = segmentTokens(tokens, p.gap, p.minChars);
    if (segs.length < 2) return null;
    const words: string[] = [];
    const wordsAyah: number[] = [];
    const ayahFirst: number[] = [];
    const ayahLast: number[] = [];
    win.forEach(([s, a], x) => {
      ayahFirst.push(words.length);
      for (const w of this.ph(s, a)) {
        words.push(w);
        wordsAyah.push(x);
      }
      ayahLast.push(words.length - 1);
    });
    const wlenCum = [0];
    for (const w of words) wlenCum.push(wlenCum[wlenCum.length - 1]! + w.length);
    const ws = Int32Array.from(wlenCum);
    const winKey = win.map(([s, a]) => `${s}:${a}`).join(",");
    let ref: Uint8Array | null = null;
    const costs: Float64Array[] = [];
    const pre: number[] = [];
    const segLen: number[] = [];
    for (const s of segs) {
      const key = `${winKey}|${s.text}`;
      let hit = this.segCache.get(key);
      if (!hit) {
        const q = encode(s.text);
        ref ??= encode(words.join(""));
        let best = Infinity;
        for (const ph of PREAMBLES) best = Math.min(best, globalCost(q, encode(ph)));
        hit = { costs: spanCosts(q, ref, ws), pre: best, len: q.length };
        if (this.segCache.size > 4096) this.segCache.clear();
        this.segCache.set(key, hit);
      }
      costs.push(hit.costs);
      pre.push(hit.pre);
      segLen.push(hit.len);
    }
    const inp: ChainInput = { costs, pre, segLen, wordsAyah, wlenCum, ayahFirst, ayahLast, p };
    const cIn = chain(inp, false).best;
    const { best: cFree, path, skipped } = chain(inp, true);
    const out: AyahOrderCandidate = { margin: pyRound(cIn - cFree, 3), jumps: [], segs: segs.length };
    if (!path.length || !Number.isFinite(cFree)) return out;
    const nw = words.length;
    const matched: Array<{ k: number; i: number; e: number }> = [];
    path.forEach((st, k) => { if (st.kind === "m") matched.push({ k, i: st.i, e: st.e }); });
    for (let n = 0; n + 1 < matched.length; n++) {
      const { k: k0, i: i0, e: e0 } = matched[n]!;
      const { k: k1, i: i1, e: e1 } = matched[n + 1]!;
      const full = skipped.get(e0 * nw + i1);
      if (!full) continue;
      let preRestart = false;
      for (let x = 0; x < n; x++) if (matched[x]!.e >= i0) preRestart = true;
      preRestart &&= e0 === ayahLast[wordsAyah[e0]!];
      let between = 0;
      for (let k = k0 + 1; k < k1; k++) between += segLen[k]!;
      const later = new Set<number>();
      for (const m of matched) if (m.k >= k1) for (let w = m.i; w <= m.e; w++) later.add(wordsAyah[w]!);
      const fit0 = costs[k0]![i0 * nw + e0]! / Math.max(1, segLen[k0]!);
      const fit1 = costs[k1]![i1 * nw + e1]! / Math.max(1, segLen[k1]!);
      for (const x of full) {
        let onSkipped = Infinity;
        for (let i = ayahFirst[x]!; i <= ayahLast[x]!; i++) onSkipped = Math.min(onSkipped, costs[k0]![i * nw + ayahLast[x]!]!);
        const [s, a] = win[x]!;
        out.jumps.push({
          surah: s, ayah: a, fit0: pyRound(fit0, 3), fit1: pyRound(fit1, 3), between,
          readLater: later.has(x), at: pyRound(segs[k1]!.t, 2), postChars: segLen[k1]!, preChars: segLen[k0]!,
          preRestart, preOnSkipped: pyRound(onSkipped / Math.max(1, segLen[k0]!), 3),
          skipChars: wlenCum[ayahLast[x]! + 1]! - wlenCum[ayahFirst[x]!]!,
          ident: this.identGuarded(s, a),
        });
      }
    }
    return out;
  }

  /** `similar_verse.detect_take` restricted to the look-alike and drop families. */
  similarVerseCandidates(tokens: readonly TimedToken[], passageIn: readonly Ayah[], known: boolean,
    durationS: number, timesFor?: (c: SimilarVerseCandidate) => boolean): SimilarVerseCandidate[] {
    const passage = passageIn.filter(([s, a]) => this.has(s, a));
    const chars: string[] = [];
    const times: number[] = [];
    for (const tok of tokens) for (const c of tok.sym) { chars.push(c); times.push(tok.t); }
    if (!passage.length || chars.length < 8) return [];
    const q = encode(chars.join(""));
    const n = q.length;
    const slotWords = passage.map(([s, a]) => this.ph(s, a));
    const slotChars = slotWords.map((ws) => ws.join(""));
    const ref = encode(slotChars.join(""));
    // Forward columns at every slot start, backward columns at every slot end.
    const slotStart: number[] = [0];
    for (const c of slotChars) slotStart.push(slotStart[slotStart.length - 1]! + c.length);
    const fwdAt = new Map<number, Float64Array>();
    let col = forwardInit(n);
    fwdAt.set(0, col);
    for (let j = 0; j < ref.length; j++) {
      col = forwardStep(col, q, ref[j]!);
      if (slotStart.includes(j + 1)) fwdAt.set(j + 1, col);
    }
    let base = col[0]! + n * SKIP;
    for (let i = 1; i <= n; i++) {
      const tail = col[i]! + (n - i) * SKIP;
      if (tail < base) base = tail;
    }
    const bwdAt = new Map<number, Float64Array>();
    let bcol = backwardInit(n);
    bwdAt.set(ref.length, bcol);
    for (let j = ref.length - 1; j >= 0; j--) {
      bcol = backwardStep(bcol, q, ref[j]!, ref.length - j);
      if (slotStart.includes(j)) bwdAt.set(j, bcol);
    }
    const out: SimilarVerseCandidate[] = [];
    const meta: Array<{ words: string[]; end: number }> = [];
    passage.forEach(([ks, ka], t) => {
      const k = `${ks}:${ka}`;
      const F = fwdAt.get(slotStart[t]!)!;
      const B = bwdAt.get(slotStart[t + 1]!)!;
      const postLen = ref.length - slotStart[t + 1]!;
      const alts: Ayah[] = known ? [[ks, ka]] : [[ks, ka], ...this.lookalikes(ks, ka).map((c) => c.m)];
      const bounds = new Map<string, { f: Float64Array[]; b: Float64Array[] }>();
      // Forward / backward columns at every word boundary of ayah e read in slot t.
      const boundsOf = (e: Ayah): { f: Float64Array[]; b: Float64Array[] } => {
        const key = `${e[0]}:${e[1]}`;
        let hit = bounds.get(key);
        if (hit) return hit;
        const ws = this.ph(e[0], e[1]);
        const f: Float64Array[] = [F];
        let c = F;
        for (const w of ws) {
          for (const id of encode(w)) c = forwardStep(c, q, id);
          f.push(c);
        }
        const b: Float64Array[] = new Array(ws.length + 1);
        b[ws.length] = B;
        let bc = B;
        let after = postLen;
        for (let wi = ws.length - 1; wi >= 0; wi--) {
          const ids = encode(ws[wi]!);
          for (let x = ids.length - 1; x >= 0; x--) bc = backwardStep(bc, q, ids[x]!, ++after);
          b[wi] = bc;
        }
        hit = { f, b };
        bounds.set(key, hit);
        return hit;
      };
      const pureCost = (e: Ayah): number => {
        const { f, b } = boundsOf(e);
        return combine(f[f.length - 1]!, b[b.length - 1]!);
      };
      const pure = new Map<string, number>([[k, base]]);
      for (const a of alts) {
        const key = `${a[0]}:${a[1]}`;
        if (!pure.has(key)) pure.set(key, pureCost(a));
      }
      for (const e of alts) {
        const ek = `${e[0]}:${e[1]}`;
        const looks = this.lookalikes(e[0], e[1]);
        if (!known) {
          for (const c of looks) {
            const mk = `${c.m[0]}:${c.m[1]}`;
            if (!pure.has(mk)) pure.set(mk, pureCost(c.m));
          }
        }
        const vs: Variant[] = [];
        for (const c of looks) vs.push(...this.variantsFor(e, c));
        if (known || ek === k) vs.push(...this.dropVariants(e));
        const refE = pure.get(ek)!;
        let floor = refE;
        if (!known) for (const v of pure.values()) if (v < floor) floor = v;
        const { f, b } = boundsOf(e);
        for (const v of vs) {
          let c = f[v.i1]!;
          for (const w of v.mid) for (const id of encode(w)) c = forwardStep(c, q, id);
          const cv = combine(c, b[v.i2]!);
          out.push({
            slot: t, surah: e[0], ayah: e[1], word: v.word, kind: v.kind, family: v.family, span: v.span,
            from: v.from, margin: pyRound(refE - cv, 3), marginAll: pyRound(floor - cv, 3), known,
            at: null, loc: null, ayahD: null,
          });
          meta.push({ words: v.words, end: v.end });
        }
      }
    });
    const dur = durationS + 2.0;
    out.forEach((c, x) => {
      if (!(c.margin > 0 && c.marginAll > 0)) return;
      if (timesFor && !timesFor(c)) return;
      const { words: vw, end } = meta[x]!;
      const words: string[] = [];
      slotWords.forEach((sw, t) => words.push(...(t === c.slot ? vw : sw)));
      const starts = [0];
      for (const w of words) starts.push(starts[starts.length - 1]! + w.length);
      let w0 = 0;
      for (let t = 0; t < c.slot; t++) w0 += slotWords[t]!.length;
      const wi = w0 + end;
      const { last, rc } = alignPath(q, encode(words.join("")));
      if (wi < words.length) {
        const qi = last[starts[wi + 1]! - 1]!;
        c.at = qi >= 0 ? pyRound(times[qi]!, 2) : pyRound(dur, 2);
      } else c.at = pyRound(dur, 2);
      const lo = starts[Math.max(w0, w0 + c.word - 1)]!;
      const hi = starts[Math.min(w0 + vw.length, wi + 1)]!;
      let sum = 0;
      for (let j = lo; j < hi; j++) sum += rc[j]!;
      c.loc = pyRound(sum / Math.max(1, hi - lo), 3);
      const a0 = starts[w0]!;
      const a1 = starts[w0 + vw.length]!;
      sum = 0;
      for (let j = a0; j < a1; j++) sum += rc[j]!;
      c.ayahD = pyRound(sum / Math.max(1, a1 - a0), 3);
    });
    return out;
  }

  /** E with one region swapped for look-alike M's wording (merged runs and single ops). */
  private variantsFor(e: Ayah, look: Look): Variant[] {
    const m = look.m;
    const ew = this.ph(e[0], e[1]);
    const mw = this.ph(m[0], m[1]);
    const regs = look.regions.filter((r) => r[0] !== INSERT);
    const runs: Array<[number | "run", number, number, number, number]> = [];
    for (const r of look.regions) {
      const last = runs[runs.length - 1];
      if (last && last[2] === r[1] && last[4] === r[3]) runs[runs.length - 1] = ["run", last[1], r[2], last[3], r[4]];
      else runs.push([...r]);
    }
    const seen = new Set<string>();
    const out: Variant[] = [];
    for (const [tag, i1, i2, j1, j2] of [...regs, ...runs.filter((r) => r[0] === "run")]) {
      if (i2 <= i1) continue;
      const mid = mw.slice(j1, j2);
      const key = `${i1},${i2},${mid.join("\u0001")}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({
        words: [...ew.slice(0, i1), ...mid, ...ew.slice(i2)], word: i1, end: i1 + (j2 - j1),
        kind: j2 <= j1 ? "possible_omission" : "possible_substitution", from: [m[0], m[1]], family: "lookalike",
        span: i2 - i1, i1, i2, mid,
      });
      if (tag === REPLACE && i2 - i1 === j2 - j1 && i2 - i1 > 1) {
        for (let d = 0; d < i2 - i1; d++) {
          const one = [mw[j1 + d]!];
          const key1 = `${i1 + d},${i1 + d + 1},${one[0]}`;
          if (seen.has(key1)) continue;
          seen.add(key1);
          out.push({
            words: [...ew.slice(0, i1 + d), ...one, ...ew.slice(i1 + d + 1)], word: i1 + d, end: i1 + d + 1,
            kind: "possible_substitution", from: [m[0], m[1]], family: "lookalike", span: 1,
            i1: i1 + d, i2: i1 + d + 1, mid: one,
          });
        }
      }
    }
    return out;
  }

  /** Skip ahead: E with one or two consecutive interior words missing. */
  private dropVariants(e: Ayah, maxDrop = 2): Variant[] {
    const ph = this.ph(e[0], e[1]);
    const out: Variant[] = [];
    for (let i = 1; i < ph.length - 1; i++) {
      for (let d = 1; d <= maxDrop; d++) {
        if (i + d < ph.length) {
          out.push({
            words: [...ph.slice(0, i), ...ph.slice(i + d)], word: i, end: i, kind: "possible_omission",
            from: [e[0], e[1]], family: "drop", span: d, i1: i, i2: i + d, mid: [],
          });
        }
      }
    }
    return out;
  }
}

/** `ayah_order.flags_of`. */
export function ayahOrderFlags(cand: AyahOrderCandidate | null, rule: AyahOrderRule = AYAH_ORDER_RULE): StructuralFlag[] {
  if (!cand || !(cand.margin > 0)) return [];
  const out: StructuralFlag[] = [];
  for (const j of cand.jumps) {
    if (cand.margin < rule.margin && cand.margin / Math.max(1, j.postChars) < rule.rel) continue;
    if (j.readLater || j.between > rule.between) continue;
    if (Math.max(j.fit0, j.fit1) > rule.fit) continue;
    if (j.postChars < rule.minPost) continue;
    if (rule.noRestartPre && j.preRestart && j.preOnSkipped <= (rule.restartAlt ?? 99)) continue;
    if (rule.betweenRel !== undefined && j.between > rule.betweenRel * j.skipChars) continue;
    if (rule.noIdent && j.ident) continue;
    out.push({ kind: "possible_skipped_ayah", surah: j.surah, ayah: j.ayah, word: 0, atSeconds: j.at, source: "ayah_order" });
  }
  return out;
}

/** Whether a candidate can pass `rule` before its timing / fit is computed. */
export function similarVerseEligible(c: SimilarVerseCandidate, rule: SimilarVerseRule = SIMILAR_VERSE_RULE): boolean {
  const m = c.known ? c.margin : c.marginAll;
  return m >= (c.family === "drop" ? rule.drop : rule.lookalike) && c.span <= rule.maxSpan && !(c.family === "drop" && c.span > 2);
}

/** `similar_verse_eval.pick`: the best qualifying candidate per passage slot. */
export function similarVersePick(cands: readonly SimilarVerseCandidate[], rule: SimilarVerseRule = SIMILAR_VERSE_RULE): SimilarVerseCandidate[] {
  const best = new Map<number, { c: SimilarVerseCandidate; m: number }>();
  for (const c of cands) {
    if (c.at === null || !similarVerseEligible(c, rule)) continue;
    if (c.loc === null || c.loc > rule.loc) continue;
    if (c.ayahD === null || c.ayahD > rule.ayahFit) continue;
    const m = c.known ? c.margin : c.marginAll;
    const cur = best.get(c.slot);
    if (!cur || m > cur.m) best.set(c.slot, { c, m });
  }
  return [...best.values()].map((b) => b.c);
}

function forwardInit(n: number): Float64Array {
  const col = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) col[i] = i * SKIP;
  return col;
}

/** D[., j] from D[., j − 1] for reference char `r` (query prefix skippable). */
function forwardStep(col: Float64Array, q: Uint8Array, r: number): Float64Array {
  const n = q.length;
  const next = new Float64Array(n + 1);
  next[0] = col[0]! + 1;
  for (let i = 1; i <= n; i++) {
    let c = col[i - 1]! + COST[q[i - 1]! * TABLE_SIZE + r]!;
    const u = next[i - 1]! + 1;
    if (u < c) c = u;
    const lf = col[i]! + 1;
    if (lf < c) c = lf;
    next[i] = c;
  }
  return next;
}

function backwardInit(n: number): Float64Array {
  const col = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) col[i] = (n - i) * SKIP;
  return col;
}

/** E[., j] from E[., j + 1] (query suffix skippable once the reference is consumed). */
function backwardStep(col: Float64Array, q: Uint8Array, r: number, refLeft: number): Float64Array {
  const n = q.length;
  const next = new Float64Array(n + 1);
  next[n] = refLeft;
  for (let i = n - 1; i >= 0; i--) {
    let c = col[i + 1]! + COST[q[i]! * TABLE_SIZE + r]!;
    const u = next[i + 1]! + 1;
    if (u < c) c = u;
    const lf = col[i]! + 1;
    if (lf < c) c = lf;
    next[i] = c;
  }
  return next;
}

function combine(f: Float64Array, b: Float64Array): number {
  let best = Infinity;
  for (let i = 0; i < f.length; i++) {
    const v = f[i]! + b[i]!;
    if (v < best) best = v;
  }
  return best;
}

/** `similar_verse.align_path`: per reference char, the last query index aligned at or before it, and its cost. */
export function alignPath(q: Uint8Array, r: Uint8Array): { last: Int32Array; rc: Float64Array } {
  const n = q.length;
  const m = r.length;
  const W = m + 1;
  const D = new Float64Array((n + 1) * W);
  const T = new Int8Array((n + 1) * W);
  for (let j = 0; j <= m; j++) { D[j] = j; T[j] = 2; }
  for (let i = 1; i <= n; i++) {
    D[i * W] = i * SKIP;
    T[i * W] = 3;
    const row = q[i - 1]! * TABLE_SIZE;
    for (let j = 1; j <= m; j++) {
      let c = D[(i - 1) * W + j - 1]! + COST[row + r[j - 1]!]!;
      let t = 0;
      const u = D[(i - 1) * W + j]! + 1;
      if (u < c) { c = u; t = 1; }
      const lf = D[i * W + j - 1]! + 1;
      if (lf < c) { c = lf; t = 2; }
      D[i * W + j] = c;
      T[i * W + j] = t;
    }
  }
  let bi = 0;
  let bc = D[m]! + n * SKIP;
  for (let i = 1; i <= n; i++) {
    const v = D[i * W + m]! + (n - i) * SKIP;
    if (v < bc) { bc = v; bi = i; }
  }
  const last = new Int32Array(m).fill(-1);
  const rc = new Float64Array(m);
  let i = bi;
  let j = m;
  while (j > 0) {
    let t = T[i * W + j]!;
    if (i === 0) t = 2;
    if (t === 0) {
      last[j - 1] = i - 1;
      rc[j - 1] += COST[q[i - 1]! * TABLE_SIZE + r[j - 1]!]!;
      i--;
      j--;
    } else if (t === 1) {
      rc[j - 1] += 1;
      i--;
    } else if (t === 2) {
      last[j - 1] = i > 0 ? i - 1 : -1;
      rc[j - 1] += 1;
      j--;
    } else break;
  }
  return { last, rc };
}
