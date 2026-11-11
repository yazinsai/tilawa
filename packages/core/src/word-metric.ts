import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Both encoders operate on phone symbols; there are no word or location parameters. */
export interface PhoneEncoder {
  /** [hidden][left/centre/right][phone], with the final phone reserved for boundary padding. */
  convolution: number[][][];
  bias: number[];
  /** [embedding][mean hidden, max hidden]. */
  projection: number[][];
}

export interface WordMetricModel {
  version: 1;
  inventory: string;
  tokens: string;
  written: PhoneEncoder;
  observed: PhoneEncoder;
  /** Training-only least-squares conversion to the tracker's weighted edit units. */
  scale: number;
}

export interface WordMetric {
  /** Null for unsupported or empty strings. Exact observations retain zero identity cost. */
  cost(written: string, observed: string): number | null;
}

function validateEncoder(e: PhoneEncoder, phones: number): void {
  const hidden = e.bias.length;
  const finite = (v: number) => Number.isFinite(v);
  if (!hidden || !e.projection.length || e.convolution.length !== hidden || !e.bias.every(finite)
    || e.convolution.some(row => row.length !== 3 || row.some(k => k.length !== phones + 1 || !k.every(finite)))
    || e.projection.some(row => row.length !== hidden * 2 || !row.every(finite))) {
    throw new Error("Invalid compositional phone encoder");
  }
}

/** Pure CPU inference matching the small training CNN, including normalized mean/max pooling. */
export function phoneEmbedding(e: PhoneEncoder, tokens: string, text: string): number[] | null {
  const ids = [...text].map(p => tokens.indexOf(p));
  if (!ids.length || ids.some(i => i < 0)) return null;
  const hidden = e.bias.length;
  const mean = new Array<number>(hidden).fill(0);
  const max = new Array<number>(hidden).fill(-Infinity);
  for (let t = 0; t < ids.length; t++) {
    for (let h = 0; h < hidden; h++) {
      let x = e.bias[h];
      for (let k = 0; k < 3; k++) x += e.convolution[h][k][ids[t + k - 1] ?? tokens.length];
      const y = Math.tanh(x);
      mean[h] += y / ids.length;
      max[h] = Math.max(max[h], y);
    }
  }
  const pooled = [...mean, ...max];
  const out = e.projection.map(row => row.reduce((n, w, i) => n + w * pooled[i], 0));
  const norm = Math.sqrt(out.reduce((n, v) => n + v * v, 0));
  return norm > 0 ? out.map(v => v / norm) : out.map(() => 0);
}

/** Owns a frozen copy. Bounded caches are keyed by phone strings and never by word/verse IDs. */
export function createWordMetric(source: WordMetricModel, inventory: string): WordMetric {
  const model: WordMetricModel = JSON.parse(JSON.stringify(source));
  if (model.version !== 1 || model.inventory !== inventory || !model.tokens.length
    || new Set(model.tokens).size !== model.tokens.length || !Number.isFinite(model.scale) || model.scale < 0) {
    throw new Error("Incompatible compositional word metric");
  }
  validateEncoder(model.written, model.tokens.length);
  validateEncoder(model.observed, model.tokens.length);
  if (model.written.projection.length !== model.observed.projection.length) throw new Error("Embedding dimensions differ");
  const written = new Map<string, number[]>(), observed = new Map<string, number[]>();
  const embed = (s: string, e: PhoneEncoder, cache: Map<string, number[]>) => {
    const have = cache.get(s);
    if (have) return have;
    const v = phoneEmbedding(e, model.tokens, s);
    if (v) {
      if (cache.size >= 4096) cache.clear();
      cache.set(s, v);
    }
    return v;
  };
  const distance = (a: number[], b: number[]) => a.reduce((n, x, i) => n + (x - b[i]) ** 2, 0);
  return {
    cost(r, o) {
      const a = embed(r, model.written, written), b = embed(o, model.observed, observed);
      const control = embed(r, model.observed, observed);
      if (!a || !b || !control) return null;
      // The identical-string ordinary-reading control removes dual-encoder offset, not duration charges.
      return model.scale * Math.max(0, distance(a, b) - distance(a, control));
    },
  };
}

export interface WordMetricStep {
  /** The exact weighted charge incurred by the existing transition. */
  cost: number;
  /** Absent only on a pure motion edge; such edges never receive a metric correction. */
  word?: number;
  reference?: string;
  observed?: string;
  /** Existing length-change cost, kept separate from word identity. */
  duration?: boolean;
  /** A motion edge ends the current word occurrence even when it returns to that word. */
  move?: boolean;
}

export interface WordMetricPath {
  readonly previous: WordMetricPath | null;
  readonly step: WordMetricStep;
}

/** Classify an existing directed transition without re-pricing it. */
export function wordMetricEdit(
  reference: string, at: number, observed: string | null, kind: "sub" | "ins" | "del", word: number,
): WordMetricStep {
  const r = kind === "ins" ? "" : reference[at];
  const o = kind === "del" ? "" : observed!;
  const cost = kind === "sub" ? subCost(r, o) : kind === "ins" ? insertionCost(reference, at, o) : deletionCost(reference, at);
  // Only the existing 0.2-edit indel is a duration transition; identity substitutions never are.
  return { cost, word, reference: r, observed: o, duration: kind !== "sub" && cost === UNIT / 5 };
}

export interface WordMetricScore {
  raw: number;
  fixed: number;
  identity: number;
  learned: number;
  observations: number;
  words: { written: string; heard: string; original: number; learned: number }[];
}

interface CompletedWord {
  previous: CompletedWord | null;
  score: WordMetricScore["words"][number];
}

interface PrefixScore {
  raw: number;
  fixed: number;
  identity: number;
  learned: number;
  observations: number;
  word: number | undefined;
  written: string;
  heard: string;
  original: number;
  completed: CompletedWord | null;
}

const prefixCaches = new WeakMap<WordMetric, WeakMap<WordMetricPath, PrefixScore>>();

function finishWord(s: PrefixScore, metric: WordMetric): void {
  if (s.word === undefined) return;
  const replacement = metric.cost(s.written, s.heard) ?? s.original;
  s.completed = { previous: s.completed,
    score: { written: s.written, heard: s.heard, original: s.original, learned: replacement } };
  s.identity += s.original;
  s.learned += replacement;
  s.word = undefined;
  s.written = s.heard = "";
  s.original = 0;
}

/**
 * Replace only word-identity charges along one retained complete trajectory. Motion and duration costs survive;
 * an open first/last word uses exactly its consumed reference fragment, without charging an unconsumed suffix.
 * A wholly deleted word has no acoustic match and retains its original charge.
 */
export function wordMetricPathScore(path: WordMetricPath | null, metric: WordMetric): WordMetricScore {
  let cache = prefixCaches.get(metric);
  if (!cache) prefixCaches.set(metric, cache = new WeakMap());
  const pending: WordMetricPath[] = [];
  let p = path;
  while (p && !cache.has(p)) { pending.push(p); p = p.previous; }
  let state: PrefixScore = p ? cache.get(p)! : { raw: 0, fixed: 0, identity: 0, learned: 0,
    observations: 0, word: undefined, written: "", heard: "", original: 0, completed: null };
  // Prefix summaries retain all earlier charges while composing only the open word. Shared
  // predecessors are evaluated once rather than replaying every earlier phone at each restart.
  for (let i = pending.length - 1; i >= 0; i--) {
    const node = pending[i], step = node.step;
    state = { ...state, raw: state.raw + step.cost, observations: state.observations + (step.observed?.length ?? 0) };
    if (step.move || step.word === undefined || step.word !== state.word) finishWord(state, metric);
    if (step.word === undefined) state.fixed += step.cost;
    else {
      state.word = step.word;
      if (step.duration) state.fixed += step.cost;
      else {
        state.written += step.reference ?? "";
        state.heard += step.observed ?? "";
        state.original += step.cost;
      }
    }
    cache.set(node, state);
  }
  const final = { ...state };
  finishWord(final, metric);
  const words: WordMetricScore["words"] = [];
  for (let w = final.completed; w; w = w.previous) words.push({ ...w.score });
  words.reverse();
  return { raw: final.raw, fixed: final.fixed, identity: final.identity,
    learned: final.fixed + final.learned, observations: final.observations, words };
}
