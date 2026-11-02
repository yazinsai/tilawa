import { SEAM_WEIGHTS } from "./seam-tagger-weights.js";

/** Inventory order of ref-v2. Embedding row `i` is token `SEAM_TOKENS[i]`. */
export const SEAM_TOKENS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۦۥَُِ۪ـٲڇں۾ؙۜ";

/** Token classes the tagger is trained to emit. The seam score is mean P(added) over the span. */
export const TAG_ADDED = 0;
export const TAG_REPEAT = 1;
export const TAG_NOISE = 2;
export const TAG_OWNED = 3;

/** Who the checker's alignment gave this phone to. */
export const OWN_LEFT = 0;
export const OWN_RIGHT = 1;
export const OWN_REPEAT = 2;
export const OWN_NONE = 3;

const FEAT = 8;
const EPS = 1e-5;

export interface SeamToken {
  /** Index into `SEAM_TOKENS`, or -1 when the phone is outside the inventory. */
  id: number;
  conf: number;
  /** 1 when a labelled pause sits between this phone and the previous one. */
  pause: number;
  /** `OWN_*`. */
  own: number;
  /** 1 on the phone where the tracker stepped onto a new word. */
  step: number;
  /** Phones in the seam span before truncation. Long restarts stay long even if the window clips them. */
  spanLen: number;
  /** Index into the checker's observation buffer. */
  obs: number;
}

export interface SeamGeom {
  /** Word the insertion sits in front of: word 1 of the ayah after the seam. */
  rightWord: number;
  /** Half-open observation range of the phones between the two complete visits. */
  obsFrom: number;
  obsTo: number;
  tokens: SeamToken[];
  /** Half-open range of `tokens` that is the seam span. */
  span: [number, number];
}

export interface SeamStep {
  op: "diag" | "del" | "ins" | "jump" | "lead";
  ref?: number;
  obs?: number;
}

export interface SeamObs {
  p: string;
  conf: number;
  start: number;
  end: number;
  step?: boolean;
}

export interface SeamLayer {
  ln1g: number[];
  ln1b: number[];
  q: number[];
  k: number[];
  v: number[];
  wo: number[];
  ln2g: number[];
  ln2b: number[];
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
}

export interface SeamWeights {
  tokens: string;
  d: number;
  heads: number;
  ff: number;
  maxLen: number;
  /** Confirm only strictly above this. The max score on held-out clean gold-excluded seams. */
  threshold: number;
  embed: number[];
  pos: number[];
  featW: number[];
  featB: number[];
  layers: SeamLayer[];
  headW: number[];
  headB: number[];
}

const tokenIndex = new Map([...SEAM_TOKENS].map((t, i) => [t, i]));

export function seamTokenId(p: string): number {
  return tokenIndex.get(p) ?? -1;
}

export function seamThreshold(): number {
  return SEAM_WEIGHTS?.threshold ?? Infinity;
}

interface Visit {
  word: number;
  covered: number;
  startObs: number;
  endObs: number;
}

/**
 * Ayah seams whose two neighbour words each have a complete in-order lexical visit.
 * The span is the phones strictly between the left visit's last diagonal and the right visit's first.
 */
export function ayahSeamGeoms(args: {
  steps: readonly SeamStep[];
  obs: readonly SeamObs[];
  wordAt: (ref: number) => number;
  wordLen: (wi: number) => number;
  prevWord: (wi: number) => number;
  isAyahStart: (wi: number) => boolean;
  matches: (wi: number, heard: string) => boolean;
  pauseBefore: (obsIndex: number) => boolean;
  maxLen?: number;
}): SeamGeom[] {
  const maxLen = args.maxLen ?? SEAM_WEIGHTS?.maxLen ?? 96;
  const visits: Visit[] = [];
  let cur: Visit | null = null;
  const close = () => {
    if (cur && cur.startObs >= 0) visits.push(cur);
    cur = null;
  };
  for (const s of args.steps) {
    if (s.op === "jump" || s.op === "lead") {
      close();
      continue;
    }
    if (s.op === "ins" || s.ref === undefined) continue;
    const wi = args.wordAt(s.ref);
    if (!cur || cur.word !== wi) {
      close();
      cur = { word: wi, covered: 0, startObs: -1, endObs: -1 };
    }
    cur.covered++;
    if (s.op === "diag" && s.obs !== undefined) {
      if (cur.startObs < 0) cur.startObs = s.obs;
      cur.endObs = s.obs;
    }
  }
  close();
  const complete = visits.filter((v) => {
    const n = args.wordLen(v.word);
    if (n <= 0 || v.covered !== n || v.startObs < 0 || v.endObs < v.startObs) return false;
    const heard = args.obs.slice(v.startObs, v.endObs + 1).map((p) => p.p).join("");
    return args.matches(v.word, heard);
  });
  const diag = new Map<number, number>();
  for (const s of args.steps) if (s.op === "diag" && s.obs !== undefined && s.ref !== undefined) diag.set(s.obs, args.wordAt(s.ref));
  const out: SeamGeom[] = [];
  const seen = new Set<number>();
  for (const right of complete) {
    if (!args.isAyahStart(right.word) || seen.has(right.word)) continue;
    const leftWord = args.prevWord(right.word);
    // The earliest complete reading of the previous word. A later one is the end of a restart, and the
    // phones between the first reading and word 1 are the restart the tagger has to refuse.
    let left: Visit | null = null;
    for (const v of complete) {
      if (v.word !== leftWord || v.endObs >= right.startObs) continue;
      if (!left || v.endObs < left.endObs) left = v;
    }
    if (!left) continue;
    const obsFrom = left.endObs + 1;
    const obsTo = right.startObs;
    if (obsTo <= obsFrom) continue;
    seen.add(right.word);
    const spanLen = obsTo - obsFrom;
    const win = clipWindow(left.startObs, right.endObs + 1, obsFrom, obsTo, maxLen);
    const tokens: SeamToken[] = [];
    for (let i = win.lo; i < win.hi; i++) {
      const phone = args.obs[i];
      if (!phone) continue;
      const who = diag.get(i);
      tokens.push({
        id: seamTokenId(phone.p),
        conf: phone.conf,
        pause: args.pauseBefore(i) ? 1 : 0,
        own: who === undefined ? OWN_NONE : who === left.word ? OWN_LEFT : who === right.word ? OWN_RIGHT : OWN_REPEAT,
        step: phone.step ? 1 : 0,
        spanLen,
        obs: i,
      });
    }
    const span0 = Math.max(0, obsFrom - win.lo);
    const span1 = Math.min(tokens.length, obsTo - win.lo);
    if (span1 <= span0) continue;
    out.push({ rightWord: right.word, obsFrom, obsTo, tokens, span: [span0, span1] });
  }
  return out;
}

function clipWindow(from: number, to: number, spanFrom: number, spanTo: number, maxLen: number): { lo: number; hi: number } {
  let lo = from;
  let hi = to;
  if (hi - lo <= maxLen) return { lo, hi };
  const spanLen = spanTo - spanFrom;
  if (spanLen >= maxLen) {
    const mid = spanFrom + Math.floor(spanLen / 2);
    lo = Math.max(from, mid - Math.floor(maxLen / 2));
    hi = Math.min(to, lo + maxLen);
    if (hi - lo < maxLen) lo = Math.max(from, hi - maxLen);
    return { lo, hi };
  }
  const room = maxLen - spanLen;
  lo = spanFrom - Math.floor(room / 2);
  hi = lo + maxLen;
  if (lo < from) {
    hi += from - lo;
    lo = from;
  }
  if (hi > to) {
    lo -= hi - to;
    hi = to;
  }
  lo = Math.max(from, lo);
  hi = Math.min(to, hi);
  if (lo > spanFrom) lo = spanFrom;
  if (hi < spanTo) hi = spanTo;
  while (hi - lo > maxLen && lo < spanFrom) lo++;
  while (hi - lo > maxLen && hi > spanTo) hi--;
  return { lo, hi };
}

function feat(t: SeamToken): number[] {
  const own = [0, 0, 0, 0];
  if (t.own >= 0 && t.own < 4) own[t.own] = 1;
  return [t.conf, t.pause, own[0], own[1], own[2], own[3], t.step, Math.log1p(t.spanLen) / 4];
}

/** `y = x @ W + b`, W row-major `[in, out]`. */
function linear(x: number[][], w: number[], b: number[] | null, inn: number, out: number): number[][] {
  return x.map((row) => {
    const y = new Array<number>(out).fill(0);
    for (let i = 0; i < inn; i++) {
      const xi = row[i];
      const base = i * out;
      for (let j = 0; j < out; j++) y[j] += xi * w[base + j];
    }
    if (b) for (let j = 0; j < out; j++) y[j] += b[j];
    return y;
  });
}

function layerNorm(x: number[][], g: number[], b: number[]): number[][] {
  const d = g.length;
  return x.map((row) => {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += row[i];
    mean /= d;
    let v = 0;
    for (let i = 0; i < d; i++) {
      const z = row[i] - mean;
      v += z * z;
    }
    const inv = 1 / Math.sqrt(v / d + EPS);
    return row.map((val, i) => (val - mean) * inv * g[i] + b[i]);
  });
}

function gelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const y = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(y));
}

function softmaxRow(row: number[]): number[] {
  let m = -Infinity;
  for (const v of row) if (v > m) m = v;
  const e = row.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function attention(x: number[][], layer: SeamLayer, d: number, heads: number): number[][] {
  const t = x.length;
  const hd = d / heads;
  const q = linear(x, layer.q, null, d, d);
  const k = linear(x, layer.k, null, d, d);
  const v = linear(x, layer.v, null, d, d);
  const mixed: number[][] = x.map(() => new Array<number>(d).fill(0));
  for (let h = 0; h < heads; h++) {
    const scores: number[][] = [];
    for (let i = 0; i < t; i++) {
      const row = new Array<number>(t);
      for (let j = 0; j < t; j++) {
        let s = 0;
        for (let a = 0; a < hd; a++) s += q[i][h * hd + a] * k[j][h * hd + a];
        row[j] = s / Math.sqrt(hd);
      }
      scores.push(softmaxRow(row));
    }
    for (let i = 0; i < t; i++) {
      for (let a = 0; a < hd; a++) {
        let s = 0;
        for (let j = 0; j < t; j++) s += scores[i][j] * v[j][h * hd + a];
        mixed[i][h * hd + a] = s;
      }
    }
  }
  return linear(mixed, layer.wo, null, d, d);
}

function encode(tokens: SeamToken[], w: SeamWeights): number[][] {
  const d = w.d;
  let x = tokens.map((t, pos) => {
    const row = new Array<number>(d).fill(0);
    if (t.id >= 0 && t.id < w.tokens.length) {
      const base = t.id * d;
      for (let i = 0; i < d; i++) row[i] += w.embed[base + i];
    }
    const p = Math.min(pos, w.maxLen - 1);
    const pb = p * d;
    for (let i = 0; i < d; i++) row[i] += w.pos[pb + i];
    const f = feat(t);
    for (let i = 0; i < FEAT; i++) {
      const base = i * d;
      for (let j = 0; j < d; j++) row[j] += f[i] * w.featW[base + j];
    }
    for (let j = 0; j < d; j++) row[j] += w.featB[j];
    return row;
  });
  for (const layer of w.layers) {
    const n1 = layerNorm(x, layer.ln1g, layer.ln1b);
    const att = attention(n1, layer, d, w.heads);
    x = x.map((row, i) => row.map((v, j) => v + att[i][j]));
    const n2 = layerNorm(x, layer.ln2g, layer.ln2b);
    const h = linear(n2, layer.w1, layer.b1, d, w.ff).map((row) => row.map(gelu));
    const ff = linear(h, layer.w2, layer.b2, w.ff, d);
    x = x.map((row, i) => row.map((v, j) => v + ff[i][j]));
  }
  return x;
}

/** Mean P(added) over the span. `-1` when no model is loaded. */
export function scoreSeam(tokens: readonly SeamToken[], span: readonly [number, number], weights: SeamWeights | null = SEAM_WEIGHTS): number {
  if (!weights || span[1] <= span[0] || !tokens.length) return weights ? 0 : -1;
  const x = encode(tokens as SeamToken[], weights);
  const logits = linear(x, weights.headW, weights.headB, weights.d, 4);
  let sum = 0;
  let n = 0;
  for (let i = span[0]; i < span[1] && i < logits.length; i++) {
    sum += softmaxRow(logits[i])[TAG_ADDED];
    n++;
  }
  return n ? sum / n : 0;
}

/** Exposed for the parity check against the training forward. */
export function seamLogits(tokens: readonly SeamToken[], weights: SeamWeights): number[][] {
  return linear(encode(tokens as SeamToken[], weights), weights.headW, weights.headB, weights.d, 4);
}
