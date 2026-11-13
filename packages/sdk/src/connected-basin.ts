import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, ConnectedBasinCandidate, ConnectedBasinEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export const BASIN_ARMS = ["endpoint_a", "endpoint_b", "average", "ordinary", "connected"] as const;
type Arm = typeof BASIN_ARMS[number];
export interface BasinHead { "0.weight": number[][]; "0.bias": number[]; "2.weight": number[][]; "2.bias": number[] }
export interface BasinReadouts {
  schema: 1; inventory: string; tokens: string; primarySha256: string; featureOutput: string;
  width: number; hidden: number; center: number[]; scale: number[]; heads: Record<Arm, BasinHead>;
}
export interface BasinScorer { model: BasinReadouts; backend: InferenceBackend }

export function validateBasinReadouts(value: unknown, manifest: ModelManifest): BasinReadouts {
  const m = value as BasinReadouts;
  const vector = (x: unknown, n: number): x is number[] => Array.isArray(x) && x.length === n && x.every(Number.isFinite);
  const matrix = (x: unknown, n: number, d: number): x is number[][] => Array.isArray(x) && x.length === n && x.every(r => vector(r, d));
  if (!m || m.schema !== 1 || m.inventory !== manifest.inventory || m.tokens !== manifest.tokens ||
      m.primarySha256 !== manifest.sha256 || !Number.isSafeInteger(m.width) || m.width <= 0 ||
      !Number.isSafeInteger(m.hidden) || m.hidden <= 0 || !m.featureOutput ||
      !vector(m.center, m.width) || !vector(m.scale, m.width) || m.scale.some(v => v <= 0))
    throw new Error("Invalid connected-basin readouts");
  for (const arm of BASIN_ARMS) {
    const h = m.heads?.[arm];
    if (!h || !matrix(h["0.weight"], m.hidden, m.width) || !vector(h["0.bias"], m.hidden) ||
        !matrix(h["2.weight"], m.tokens.length+1, m.hidden) || !vector(h["2.bias"], m.tokens.length+1))
      throw new Error("Invalid connected-basin head geometry");
  }
  return m;
}

export function basinLogProbs(m: BasinReadouts, arm: Arm, features: number[][], baseline: number[][]): number[][] {
  const h = m.heads[arm];
  return features.map((row, t) => {
    const x = row.map((v, d) => (v-m.center[d])/m.scale[d]);
    const hidden = h["0.weight"].map((w, i) => Math.tanh(w.reduce((s, v, d) => s+v*x[d], h["0.bias"][i])));
    const z = h["2.weight"].map((w, i) => baseline[t][i]+w.reduce((s, v, d) => s+v*hidden[d], h["2.bias"][i]));
    const peak = Math.max(...z), norm = peak+Math.log(z.reduce((s, v) => s+Math.exp(v-peak), 0));
    return z.map(v => v-norm);
  });
}
const logadd = (a: number, b: number) => {
  const m = Math.max(a, b);
  return Number.isFinite(m) ? m+Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Ordinary complete CTC forward sum. No partial prefix or skipped phone can win. */
export function basinCtc(rows: number[][], ids: number[]): number {
  if (!rows.length || !ids.length || ids.some(i => !Number.isInteger(i) || i <= 0 || i >= rows[0].length)) return -Infinity;
  const labels = [0, ...ids.flatMap(i => [i, 0])];
  let a = labels.map(() => -Infinity);
  a[0] = rows[0][0]; a[1] = rows[0][ids[0]];
  for (let t = 1; t < rows.length; t++) {
    a = labels.map((p, s) => {
      let v = logadd(a[s], s ? a[s-1] : -Infinity);
      if (s >= 2 && p && p !== labels[s-2]) v = logadd(v, a[s-2]);
      return v+rows[t][p];
    });
  }
  return logadd(a[a.length-1], a[a.length-2]);
}

export function basinComparison(m: BasinReadouts, features: number[][], baseline: number[][],
  candidate: ConnectedBasinCandidate): ConnectedBasinEvidence | null {
  const valid = (x: number[][], width: number) => x.length > 0 && x.every(r => r.length === width && r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 || !Number.isInteger(candidate.toFrame) ||
      candidate.toFrame-candidate.fromFrame !== features.length || features.length !== baseline.length ||
      !valid(features, m.width) || !valid(baseline, m.tokens.length+1)) return null;
  // The shipped CTC training transcript represents a held-phone run once.
  // Normalize both complete hypotheses identically, without truncating either.
  const sequences = [candidate.written, candidate.rival].map(s => [...s]
    .filter((p, i, all) => i === 0 || p !== all[i-1]).map(p => m.tokens.indexOf(p)+1));
  if (sequences.some(s => !s.length || s.some(p => p <= 0))) return null;
  const scores = BASIN_ARMS.map(arm => {
    const rows = basinLogProbs(m, arm, features, baseline);
    return sequences.map(ids => basinCtc(rows, ids));
  });
  if (!scores.flat().every(Number.isFinite)) return null;
  return {...candidate, endpointAWritten: scores[0][0], endpointARival: scores[0][1],
    endpointBWritten: scores[1][0], endpointBRival: scores[1][1], averageWritten: scores[2][0], averageRival: scores[2][1],
    ordinaryWritten: scores[3][0], ordinaryRival: scores[3][1], connectedWritten: scores[4][0], connectedRival: scores[4][1]};
}
export function basinPrefersWritten(c: ConnectedBasinEvidence): boolean {
  return [c.endpointAWritten, c.endpointARival, c.endpointBWritten, c.endpointBRival, c.averageWritten,
    c.averageRival, c.ordinaryWritten, c.ordinaryRival, c.connectedWritten, c.connectedRival].every(Number.isFinite) &&
    c.connectedWritten > c.connectedRival;
}
/** The card's premise requires new written preference beyond both endpoints and arithmetic averaging. */
export function basinUniqueGain(c: ConnectedBasinEvidence): boolean {
  return basinPrefersWritten(c) && c.endpointAWritten <= c.endpointARival && c.endpointBWritten <= c.endpointBRival &&
    c.averageWritten <= c.averageRival;
}
export function carryBasinEvidence(chunks: PhonemeEvidence[], certificates: ConnectedBasinEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e, i) => i !== first ? e : {...e, phonemes: e.phonemes.map((p, j) => j ? p : {
    ...p, connectedBasinVetoes: [...(p.connectedBasinVetoes ?? []), ...certificates],
  })});
}
export function basinVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const key = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(key));
  const removed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(key(e))).map(key));
  return live.filter(e => e.errorClass === "wrong_word" && removed.has(key(e)));
}

/** Independent full-recording streaming state retains the original native frame grid and context. */
export async function basinObservations(scorer: BasinScorer, manifest: ModelManifest, audio: Float32Array) {
  const f = new Fbank().push(audio), features: number[][] = [], baseline: number[][] = [];
  const total = Math.max(0, Math.floor((Math.floor((f.length-7)/2)+1)/2));
  let states: Record<string, Tensor> = Object.fromEntries(manifest.states.map(s => [s.name, {type: s.dtype, dims: s.shape,
    data: s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a, b) => a*b, 1)) : new Float32Array(s.shape.reduce((a, b) => a*b, 1))}]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue), vocab = manifest.tokens.length+1;
  for (let at = 0; baseline.length < total; at += manifest.shift) {
    const x = new Float32Array(manifest.window*manifest.fbank.mels);
    for (let t = 0; t < manifest.window; t++) x.set(f[at+t] ?? pad, t*manifest.fbank.mels);
    const out = await scorer.backend.run({x: {type: "float32", dims: [1, manifest.window, manifest.fbank.mels], data: x}, ...states});
    states = Object.fromEntries(manifest.states.map(s => [s.name, out[`new_${s.name}`]]));
    const h = out[scorer.model.featureOutput], lp = out.log_probs, n = lp?.data.length/vocab;
    if (!Number.isSafeInteger(n) || n <= 0 || !h || h.type !== "float32" || lp.type !== "float32" || h.data.length !== n*scorer.model.width)
      throw new Error("Connected-basin extractor has invalid output geometry");
    for (let t = 0; t < n && baseline.length < total; t++) {
      features.push(Array.from(h.data.slice(t*scorer.model.width, (t+1)*scorer.model.width), Number));
      baseline.push(Array.from(lp.data.slice(t*vocab, (t+1)*vocab), Number));
    }
  }
  return {features, baseline};
}

export async function addBasinEvidence(scorer: BasinScorer | undefined, manifest: ModelManifest, audio: Float32Array,
  chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset, options: CheckerOptions = {},
  mode: "free" | "memorization" = "free", inspect?: (c: ConnectedBasinEvidence | null, request: ConnectedBasinCandidate) => void) {
  if (!scorer || !audio.length) return chunks;
  const candidates = new Map<string, ConnectedBasinCandidate>();
  const tracker = createTracker(index, {mode, phrases, acquireMs: 1000});
  const checker = createChecker(index, {...options, phrases, connectedBasin(c) { candidates.set(JSON.stringify(c), c); return true; }});
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const rows = await basinObservations(scorer, manifest, audio), certificates: ConnectedBasinEvidence[] = [];
  for (const c of candidates.values()) {
    const result = c.toFrame <= rows.features.length ? basinComparison(scorer.model,
      rows.features.slice(c.fromFrame, c.toFrame), rows.baseline.slice(c.fromFrame, c.toFrame), c) : null;
    inspect?.(result, c);
    if (result && basinPrefersWritten(result)) certificates.push(result);
  }
  return carryBasinEvidence(chunks, certificates);
}
