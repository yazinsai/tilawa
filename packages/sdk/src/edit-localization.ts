import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { LexicalChangeEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex,
  WordLocalizationCandidate, WordLocalizationContext } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import type { InferenceBackend, LoadedModel, ModelManifest, Tensor } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export interface FrozenWordFeatures {
  /** Row-major, full-vocabulary log probabilities and frozen pre-CTC features. */
  logp: Float32Array;
  encoder: Float32Array;
  frames: number;
  encoderDim: number;
  vocab: number;
}

export interface EditLocalizer {
  preservation: number;
  independentPreservation: number;
  encode(audio: Float32Array): Promise<FrozenWordFeatures>;
  score(features: Float32Array, words: number): Promise<{ contextual: number[]; independent: number[] }>;
}

const logadd = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

/** Canonical phone occupancy, summed over CTC boundaries, including blank and
 * genuine repeated-label separation. Impossible alignments cannot authorize a veto. */
export function canonicalOccupancy(logp: Float32Array, vocab: number, labels: number[]): Float64Array[] | null {
  const n = logp.length / vocab;
  if (!Number.isInteger(n) || !labels.length || n < labels.length ||
    labels.some(p => !Number.isInteger(p) || p <= 0 || p >= vocab) || Array.from(logp).some(p => !Number.isFinite(p))) return null;
  const states = [0];
  for (const p of labels) states.push(p, 0);
  const k = states.length;
  const skip = (s: number) => s >= 2 && states[s] !== 0 && states[s] !== states[s - 2];
  const alpha: Float64Array[] = Array.from({ length: n }, () => new Float64Array(k).fill(-Infinity));
  alpha[0][0] = logp[0]; alpha[0][1] = logp[labels[0]];
  for (let t = 1; t < n; t++) for (let s = 0; s < k; s++)
    alpha[t][s] = logadd(logadd(alpha[t-1][s], s ? alpha[t-1][s-1] : -Infinity),
      skip(s) ? alpha[t-1][s-2] : -Infinity) + logp[t*vocab+states[s]];
  const z = logadd(alpha[n-1][k-1], alpha[n-1][k-2]);
  if (!Number.isFinite(z)) return null;
  let beta = new Float64Array(k).fill(-Infinity);
  beta[k-1] = beta[k-2] = 0;
  const occupancy = Array.from({ length: labels.length }, () => new Float64Array(n));
  for (let t = n-1; t >= 0; t--) {
    for (let p = 0; p < labels.length; p++) occupancy[p][t] = Math.exp(alpha[t][2*p+1]+beta[2*p+1]-z);
    if (t === 0) break;
    const prev = new Float64Array(k);
    for (let s = 0; s < k; s++) {
      const next = (j: number) => j < k ? beta[j] + logp[t*vocab+states[j]] : -Infinity;
      prev[s] = logadd(logadd(next(s), next(s+1)), s+2 < k && skip(s+2) ? next(s+2) : -Infinity);
    }
    beta = prev;
  }
  return occupancy;
}

/** Shared raw features for the contextual head and independently trained control.
 * No accused-word crop or edit-script diagonal selects the acoustic boundaries. */
export function poolWordFeatures(view: FrozenWordFeatures, pronunciations: string[], tokens: string): Float32Array | null {
  const { frames: n, encoderDim: d, vocab, logp, encoder } = view;
  if (logp.length !== n*vocab || encoder.length !== n*d || vocab !== [...tokens].length+1 ||
    Array.from(encoder).some(x => !Number.isFinite(x))) return null;
  const ids = new Map([...tokens].map((p, i) => [p, i+1]));
  const words = pronunciations.map(s => [...s].map(p => ids.get(p) ?? 0));
  if (words.some(w => !w.length)) return null;
  const occupancy = canonicalOccupancy(logp, vocab, words.flat());
  if (!occupancy) return null;
  const dim = 2*d + 2*vocab + 2, result = new Float32Array(dim*words.length);
  let offset = 0;
  for (let w = 0; w < words.length; w++) {
    const weight = new Float64Array(n);
    for (let p = 0; p < words[w].length; p++) for (let t = 0; t < n; t++) weight[t] += occupancy[offset+p][t];
    offset += words[w].length;
    const total = weight.reduce((s, v) => s+v, 0);
    if (!(total > 0)) return null;
    const base = w*dim;
    for (let j = 0; j < d; j++) {
      let mean = 0, second = 0;
      for (let t = 0; t < n; t++) {
        const x = encoder[t*d+j], a = weight[t]/total;
        mean += a*x; second += a*x*x;
      }
      result[base+j] = mean; result[base+d+j] = Math.sqrt(Math.max(0, second-mean*mean));
    }
    for (let j = 0; j < vocab; j++) {
      let mass = 0;
      for (let t = 0; t < n; t++) mass += weight[t]/total*Math.exp(logp[t*vocab+j]);
      result[base+2*d+j] = mass;
      result[base+2*d+vocab+j] = words[w].filter(p => p === j).length/words[w].length;
    }
    result[base+dim-2] = Math.log1p(total); result[base+dim-1] = Math.log1p(words[w].length);
  }
  return result;
}

/** A separate streaming state reads frozen features. It never replaces primary
 * probabilities, token runs, or the primary model's streaming caches. */
export async function encodeFrozenWords(backend: InferenceBackend, m: ModelManifest, audio: Float32Array): Promise<FrozenWordFeatures> {
  const f = new Fbank().push(audio), vocab = [...m.tokens].length+1;
  const need = Math.max(0, Math.floor((Math.floor((f.length-7)/2)+1)/2));
  let states: Record<string, Tensor> = Object.fromEntries(m.states.map(s => [s.name,
    { type: s.dtype, dims: s.shape, data: s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a,b) => a*b,1)) :
      new Float32Array(s.shape.reduce((a,b) => a*b,1)) }]));
  const enc: number[] = [], lp: number[] = [];
  let d = 0, produced = 0;
  for (let start = 0; produced < need; start += m.shift) {
    const x = new Float32Array(m.window*m.fbank.mels).fill(m.fbank.padValue);
    for (let t = 0; t < m.window && start+t < f.length; t++) x.set(f[start+t],t*m.fbank.mels);
    const out = await backend.run({ x: { type: "float32", data: x, dims: [1,m.window,m.fbank.mels] }, ...states });
    const features = out.encoder_features;
    if (!features || features.dims.length !== 3 || features.dims[0] !== 1) throw new Error("Frozen encoder output must be [1,frames,features]");
    const step = out.log_probs.data as Float32Array, width = features.dims[2];
    if (d && width !== d || step.length/vocab !== features.dims[1]) throw new Error("Frozen encoder clock changed");
    d = width;
    const take = Math.min(need-produced, step.length/vocab);
    if (!Number.isInteger(take) || take <= 0) throw new Error("Frozen encoder made no progress");
    enc.push(...Array.from((features.data as Float32Array).subarray(0,take*d)));
    lp.push(...Array.from(step.subarray(0,take*vocab))); produced += take;
    states = Object.fromEntries(m.states.map(s => [s.name,out[`new_${s.name}`]]));
  }
  return { encoder: Float32Array.from(enc), logp: Float32Array.from(lp), frames: need, encoderDim: d, vocab };
}

export function carryLexicalChangeEvidence(chunks: PhonemeEvidence[], certificates: LexicalChangeEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c,i) => i !== first ? c : { ...c, phonemes: c.phonemes.map((p,j) => j ? p :
    { ...p, lexicalChanges: [...(p.lexicalChanges ?? []), ...certificates] }) });
}

/** Overlapping engine windows belong to one acoustic occurrence. Disjoint
 * re-reads remain separate, even when they name the same ayah. */
export function noteLocalizationContext(contexts: WordLocalizationContext[], c: WordLocalizationContext): void {
  const merged = { ...c, words: [...c.words], coveredWords: [...c.coveredWords] };
  const overlap = (g: WordLocalizationContext) => g.words.length === merged.words.length &&
    g.words.every((w,i) => w === merged.words[i]) && g.fromFrame < merged.toFrame && merged.fromFrame < g.toFrame;
  // A newly observed bridge can join two earlier overlapping views.
  for (let i = contexts.length-1; i >= 0; i--) if (overlap(contexts[i])) {
    const g = contexts.splice(i,1)[0];
    merged.fromFrame = Math.min(merged.fromFrame,g.fromFrame);
    merged.toFrame = Math.max(merged.toFrame,g.toFrame);
    merged.coveredWords = [...new Set([...merged.coveredWords,...g.coveredWords])];
  }
  contexts.push(merged);
}

/** Offline listener-stage enrichment on actual engine-requested occurrences.
 * The complete-ayah head emits separate sigmoid scores, so several nearby words
 * may remain changed. The independent control is diagnostic, never a runtime veto. */
export async function addEditLocalizationEvidence(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  index: ReferenceIndex, phrases?: PhraseAsset,
  diagnostic?: (candidate: WordLocalizationCandidate, contextual: number, independent: number) => void): Promise<PhonemeEvidence[]> {
  const localizer = model.editLocalization;
  if (!localizer || !audio.length) return chunks;
  if (model.manifest.sampleRate !== 16000 || model.manifest.frameMs !== 40) throw new Error("Edit localization requires the shipped acoustic clock");
  const cells = insertionCells(chunks);
  if (!cells.length) return chunks;
  const candidates: WordLocalizationCandidate[] = [], contexts: WordLocalizationContext[] = [];
  const tracker = createTracker(index, { mode: "free", phrases });
  const checker = createChecker(index, { phrases,
    onWordLocalization: c => candidates.push(c), onLocalizationContext: c => noteLocalizationContext(contexts,c) });
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.length) return chunks;
  const view = await localizer.encode(audio);
  const certificates: LexicalChangeEvidence[] = [];
  const cache = new Map<string, { contextual: number[]; independent: number[] } | null>();
  for (const c of candidates) {
    const wi = index.words.findIndex(w => w.surah === c.word.surah && w.ayah === c.word.ayah && w.word === c.word.word);
    // Complete context containing this acoustic occurrence, with the most
    // observed speech; no word-specific fallback when a whole ayah is absent.
    const context = contexts.filter(g => g.words.includes(wi) && g.coveredWords.includes(g.words[0]) &&
      g.coveredWords.includes(g.words.at(-1)!) && g.fromFrame < c.toFrame && g.toFrame > c.fromFrame)
      .sort((a,b) => (b.toFrame-b.fromFrame)-(a.toFrame-a.fromFrame))[0];
    if (!context) continue;
    const a = Math.min(context.fromFrame,c.fromFrame), b = Math.max(context.toFrame,c.toFrame);
    if (a < 0 || b > view.frames) continue;
    const key = `${context.words.join(",")}/${a}/${b}`;
    if (!cache.has(key)) {
      const f = poolWordFeatures({ ...view, frames: b-a, encoder: view.encoder.subarray(a*view.encoderDim,b*view.encoderDim),
        logp: view.logp.subarray(a*view.vocab,b*view.vocab) },
        context.words.map(w => index.phonemes.slice(index.words[w].start,index.words[w].end)), model.manifest.tokens);
      cache.set(key, f ? await localizer.score(f,context.words.length) : null);
    }
    const scores = cache.get(key);
    if (!scores) continue;
    const at = context.words.indexOf(wi), score = scores.contextual[at], independent = scores.independent[at];
    if (![score,independent].every(s => Number.isFinite(s) && s >= 0 && s <= 1)) throw new Error("Invalid lexical-change probability");
    diagnostic?.(c,score,independent);
    certificates.push({ ...c, score, preservation: localizer.preservation });
  }
  return carryLexicalChangeEvidence(chunks,certificates);
}
