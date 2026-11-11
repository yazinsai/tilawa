import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** Background parameters contain phone statistics only, never recording or reference identities. */
export interface PhoneBackground {
  tokens: string;
  initial: number[];
  next: number[][];
}

export interface BackgroundRecording {
  speaker: string;
  split: string;
  kind: string;
  phones: string;
}

/** Fixed half-count smoothing; all fitting is confined to clean training speakers. */
export function fitPhoneBackground(tokens: string, recordings: readonly BackgroundRecording[]): PhoneBackground {
  if (!tokens.length || new Set(tokens).size !== tokens.length) throw new Error("Invalid phone inventory");
  const held = new Set(recordings.filter(r => r.split !== "train").map(r => r.speaker));
  const clean = recordings.filter(r => r.split === "train" && r.kind === "none" && r.phones.length);
  if (!clean.length || clean.some(r => held.has(r.speaker))) throw new Error("No disjoint clean training speakers");
  const ids = new Map([...tokens].map((p, i) => [p, i]));
  const initial = new Array<number>(tokens.length).fill(0.5);
  const next = Array.from(tokens, () => new Array<number>(tokens.length).fill(0.5));
  for (const r of clean) {
    let prev: number | undefined;
    for (const p of r.phones) {
      const at = ids.get(p);
      if (at === undefined) throw new Error(`Unknown observed phone ${p}`);
      // A local window starts at an arbitrary speech token, not necessarily an ayah onset.
      initial[at]++;
      if (prev !== undefined) next[prev][at]++;
      prev = at;
    }
  }
  const normalize = (row: number[]) => { const sum = row.reduce((a,b) => a+b,0); return row.map(x => x/sum); };
  return { tokens, initial: normalize(initial), next: next.map(normalize) };
}

/** Same directed acoustic edit prices as anchored local matching: fixed entry, open terminal suffix. No movement edges. */
export function minimumAcousticCost(reference: string, heard: string): number {
  let row = new Float64Array(reference.length + 1).fill(Infinity);
  row[0] = 0;
  for (const p of heard) {
    const next = new Float64Array(row.length);
    next[0] = row[0] + insertionCost(reference,0,p);
    for (let i = 1; i < row.length; i++) next[i] = Math.min(
      row[i] + insertionCost(reference,i,p),
      row[i-1] + subCost(reference[i-1],p),
      next[i-1] + deletionCost(reference,i-1),
    );
    row = next;
  }
  return Math.min(...row) / UNIT;
}

/** One fixed pseudo-random stream per sample. Prefixes are shared for every reference and length. */
export function backgroundSamples(model: PhoneBackground, n: number, count = 4096): string[] {
  if (!Number.isInteger(n) || n < 0 || !Number.isInteger(count) || count < 1) throw new Error("Invalid sample dimensions");
  const pick = (row: number[], u: number) => {
    let sum = 0;
    for (let i = 0; i < row.length; i++) { sum += row[i]; if (u < sum) return i; }
    return row.length - 1;
  };
  return Array.from({length:count}, (_,sample) => {
    let state = (0x9e3779b9 ^ Math.imul(sample+1,0x85ebca6b)) >>> 0;
    let prev: number | undefined;
    let out = "";
    for (let j = 0; j < n; j++) {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
      const at = pick(prev === undefined ? model.initial : model.next[prev], (state >>> 0) / 4294967296);
      out += model.tokens[at]; prev = at;
    }
    return out;
  });
}

/** Log mean exp(-minimum cost): each sampled string contributes once, regardless of edit-path multiplicity. */
export function logAcceptanceVolume(reference: string, samples: readonly string[]): number {
  if (!samples.length) throw new Error("No background samples");
  const values = samples.map(s => -minimumAcousticCost(reference,s));
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum,x) => sum + Math.exp(x-max),0) / values.length);
}

/** This cache is indexed solely by pronunciation and observed-token count. */
export function createAcceptanceVolume(model: PhoneBackground, count = 4096): (reference: string,n: number) => number {
  const strings = new Map<number,string[]>();
  const cache = new Map<string,Map<number,number>>();
  return (reference,n) => {
    let lengths = cache.get(reference);
    if (!lengths) cache.set(reference,lengths = new Map());
    const have = lengths.get(n);
    if (have !== undefined) return have;
    let samples = strings.get(n);
    if (!samples) strings.set(n,samples = backgroundSamples(model,n,count));
    const result = logAcceptanceVolume(reference,samples);
    lengths.set(n,result);
    return result;
  };
}
