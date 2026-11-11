import type { HeldBandEvidence, PhonemeEvidence, WordEvidenceRequest } from "@tilawa/core";

export const HELD_BAND_FILE = "held-band.json";

interface Predictor { weights: number[][]; variance: number[] }
interface Partition {
  visible: number[];
  held: number[];
  background: Predictor;
  phones: Predictor[];
}

/** Small conditional Gaussian bundle, fitted only on training speakers. Row zero is nonspeech. */
export interface HeldBandModel {
  schema: 1;
  inventory: string;
  tokens: string;
  edges: number[];
  mean: number[];
  scale: number[];
  phoneMean: number[][];
  phoneVariance: number[][];
  phoneCount: number[];
  stay: number;
  partitions: [Partition, Partition];
}

const add = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
};

/** Normalized CTC timing prior. Acceptance requires every phone, including genuinely repeated phones. */
export function timingPartition(emissions: readonly Float64Array[], labels: readonly number[], stay: number): number {
  if (!emissions.length || !labels.length) return -Infinity;
  const states = 2 * labels.length + 1;
  const id = (s: number) => s % 2 ? labels[(s - 1) / 2] : 0;
  let prev = new Float64Array(states).fill(-Infinity);
  prev[0] = Math.log(.5); prev[1] = Math.log(.5);
  for (let t = 0; t < emissions.length; t++) {
    const current = Float64Array.from(prev, (p, s) => p + emissions[t][id(s)]);
    if (t === emissions.length - 1) return add(current[states - 1], current[states - 2]);
    const next = new Float64Array(states).fill(-Infinity);
    for (let s = 0; s < states; s++) {
      if (s === states - 1) { next[s] = add(next[s], current[s]); continue; }
      if (s % 2 === 0) {
        next[s] = add(next[s], current[s] + Math.log(.5));
        next[s + 1] = add(next[s + 1], current[s] + Math.log(.5));
      } else {
        const direct = s + 2 < states && id(s) !== id(s + 2);
        const leave = Math.log((1 - stay) / (direct ? 2 : 1));
        next[s] = add(next[s], current[s] + Math.log(stay));
        next[s + 1] = add(next[s + 1], current[s] + leave);
        if (direct) next[s + 2] = add(next[s + 2], current[s] + leave);
      }
    }
    prev = next;
  }
  return -Infinity;
}

const gaussian = (x: number, mean: number, variance: number) =>
  -.5 * (Math.log(2 * Math.PI * variance) + (x - mean) ** 2 / variance);

function conditional(x: Float64Array, p: Predictor, partition: Partition): number {
  let lp = 0;
  for (let j = 0; j < partition.held.length; j++) {
    let prediction = p.weights[0][j];
    for (let i = 0; i < partition.visible.length; i++) prediction += x[partition.visible[i]] * p.weights[i + 1][j];
    lp += gaussian(x[partition.held[j]], prediction, p.variance[j]);
  }
  return lp;
}

export function createHeldBandScorer(model: HeldBandModel) {
  validate(model);
  const ids = new Map([...model.tokens].map((p, i) => [p, i + 1]));
  return {
    edges: model.edges,
    inventory: model.inventory,
    /** Both sequences see identical actual observations. Unknown phones and impossible completions abstain. */
    compare(frames: readonly Float64Array[], request: WordEvidenceRequest, frameMs: number): [number, number] | null {
      const from = request.fromFrame * frameMs / 10, to = request.toFrame * frameMs / 10;
      if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to > frames.length || to <= from) return null;
      const sequences = [request.written, request.observed].map(s => [...s].map(p => ids.get(p)));
      if (sequences.some(s => !s.length || s.some(i => i === undefined || model.phoneCount[i] === 0))) return null;
      const observations = frames.slice(from, to).map(f => Float64Array.from(f, (v, i) => (v - model.mean[i]) / model.scale[i]));
      if (observations.some(f => f.length !== model.mean.length || f.some(v => !Number.isFinite(v)))) return null;
      const margins = model.partitions.map(partition => {
        const visible: Float64Array[] = [], joint: Float64Array[] = [];
        let background = 0;
        for (const x of observations) {
          const v = new Float64Array(model.tokens.length + 1), j = new Float64Array(v.length);
          background += conditional(x, partition.background, partition);
          for (let id = 0; id < v.length; id++) {
            for (const b of partition.visible) v[id] += gaussian(x[b], model.phoneMean[id][b], model.phoneVariance[id][b]);
            j[id] = v[id] + conditional(x, partition.phones[id], partition);
          }
          visible.push(v); joint.push(j);
        }
        // Divide out the visible-only partition: withheld data cannot select the timing prior.
        // Subtract the same pronunciation-independent prediction from each complete likelihood.
        const score = (labels: number[]) => timingPartition(joint, labels, model.stay) -
          timingPartition(visible, labels, model.stay) - background;
        return score(sequences[0] as number[]) - score(sequences[1] as number[]);
      });
      return margins.every(Number.isFinite) ? margins as [number, number] : null;
    },
  };
}

/** Transport only positive acoustic vetoes; phonemes and primary posterior rows retain all their values. */
export function carryHeldBandEvidence(chunks: PhonemeEvidence[], certificates: HeldBandEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(c => c.phonemes.length);
  return chunks.map((c, i) => i !== first ? c : { ...c, phonemes: c.phonemes.map((p, j) => j ? p : {
    ...p, heldBands: [...(p.heldBands ?? []), ...certificates],
  }) });
}

function validate(m: HeldBandModel) {
  const bands = m.edges.length - 1, phones = m.tokens.length + 1;
  const vector = (v: number[], n: number, positive = false) => Array.isArray(v) && v.length === n &&
    v.every(x => Number.isFinite(x) && (!positive || x > 0));
  const matrix = (v: number[][], n: number, k: number, positive = false) => Array.isArray(v) && v.length === n && v.every(r => vector(r, k, positive));
  if (m.schema !== 1 || !m.inventory || !m.tokens || new Set(m.tokens).size !== m.tokens.length ||
    !(m.stay > 0 && m.stay < 1) || bands < 2 || m.edges[0] < 1 || m.edges.at(-1)! > 256 ||
    m.edges.some((v, i) => !Number.isInteger(v) || (i > 0 && v <= m.edges[i - 1])) ||
    !vector(m.mean, bands) || !vector(m.scale, bands, true) || !matrix(m.phoneMean, phones, bands) ||
    !matrix(m.phoneVariance, phones, bands, true) || !vector(m.phoneCount, phones) || m.phoneCount.some(n => n < 0) ||
    !Array.isArray(m.partitions) || m.partitions.length !== 2)
    throw new Error("Invalid held-band acoustic model");
  for (const p of m.partitions) {
    const all = [...p.visible, ...p.held];
    const predictor = (q: Predictor) => matrix(q.weights, p.visible.length + 1, p.held.length) && vector(q.variance, p.held.length, true);
    if (!p.visible.length || !p.held.length || all.length !== bands || new Set(all).size !== bands ||
      all.some(i => !Number.isInteger(i) || i < 0 || i >= bands) || !predictor(p.background) ||
      p.phones.length !== phones || !p.phones.every(predictor)) throw new Error("Invalid conditional band predictors");
  }
  if (m.partitions[0].held.join() !== m.partitions[1].visible.join() ||
    m.partitions[1].held.join() !== m.partitions[0].visible.join()) throw new Error("Held bands must be complementary");
}
