import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** A normalized recognition transducer: geometric insertions, then one phone or epsilon per input. */
export interface PhoneChannel {
  phones: string[];
  rows: number[][];
  insertion: number[];
  insertionRate: number;
}
export interface SpokenPhoneModel {
  recognition: PhoneChannel;
  /** P(latent spoken phone | reference phone); each row sums to one. */
  pronunciation: number[][];
  /** Identical-data, direct reference-to-observation control. */
  control: PhoneChannel;
}
export interface PhoneRecording {
  written: string;
  heard: string;
  /** Only these reference phones may vary in the pronunciation stage during training. */
  variable: boolean[];
}

export const phoneClass = (p: string): string => {
  for (const [name, letters] of [["vowel", "َُِ۪"], ["long", "اۦۥـ"], ["nasal", "نمں۾"],
    ["stop", "بتدطضقكء"], ["fricative", "ثذزسشصظفخغحهه"], ["liquid", "لروي"], ["pharyngeal", "عٲڇؙۜ"]]) {
    if (letters.includes(p)) return name;
  }
  return "other";
};
const identity = (n: number) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => Number(i === j)));
const zeros = (n: number, m: number) => Array.from({ length: n }, () => Array(m).fill(0) as number[]);
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);
export function logAdd(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
}

/** Analytic composition sums all intermediate phone strings, without a best-latent-phone approximation. */
export function composePhoneChannel(recognition: PhoneChannel, pronunciation: readonly (readonly number[])[]): PhoneChannel {
  const n = recognition.phones.length;
  if (pronunciation.length !== n || pronunciation.some(r => r.length !== n || r.some(p => !Number.isFinite(p) || p < 0) || Math.abs(sum(r) - 1) > 1e-8)) throw new Error("Invalid pronunciation simplex");
  if (!n || new Set(recognition.phones).size !== n || recognition.rows.length !== n
    || recognition.rows.some(r => r.length !== n + 1 || r.some(p => !Number.isFinite(p) || p < 0) || Math.abs(sum(r) - 1) > 1e-8)
    || recognition.insertion.length !== n || recognition.insertion.some(p => !Number.isFinite(p) || p < 0) || Math.abs(sum(recognition.insertion) - 1) > 1e-8
    || !(recognition.insertionRate > 0 && recognition.insertionRate < 1)) throw new Error("Invalid recognition simplex");
  const rows = zeros(n, n + 1);
  for (let r = 0; r < n; r++) for (let s = 0; s < n; s++) for (let h = 0; h <= n; h++) rows[r][h] += pronunciation[r][s] * recognition.rows[s][h];
  return { ...recognition, rows };
}

/** Full-string forward/backward with an explicit terminal insertion state. */
function partitions(channel: PhoneChannel, record: PhoneRecording, pronunciation?: number[][]) {
  const n = channel.phones.length, index = new Map(channel.phones.map((p, i) => [p, i]));
  const w = [...record.written].map(p => index.get(p)), h = [...record.heard].map(p => index.get(p));
  if (record.variable.length !== w.length) throw new Error("Training mask must retain reference-phone ownership");
  if (w.some(i => i === undefined) || h.some(i => i === undefined)) throw new Error("Phone outside training inventory");
  const wi = w as number[], hi = h as number[], width = hi.length + 1;
  const rows = pronunciation ? composePhoneChannel(channel, pronunciation).rows : channel.rows;
  const row = (i: number) => pronunciation && !record.variable[i] ? channel.rows[wi[i]] : rows[wi[i]];
  const a = new Float64Array((wi.length + 1) * width).fill(-Infinity), b = new Float64Array(a.length).fill(-Infinity);
  const stop = Math.log1p(-channel.insertionRate), ins = hi.map(x => Math.log(channel.insertionRate * channel.insertion[x]));
  const at = (i: number, j: number) => i * width + j;
  a[0] = 0;
  for (let i = 0; i <= wi.length; i++) for (let j = 0; j <= hi.length; j++) {
    const cur = a[at(i, j)];
    if (j < hi.length) a[at(i, j + 1)] = logAdd(a[at(i, j + 1)], cur + ins[j]);
    if (i < wi.length) {
      a[at(i + 1, j)] = logAdd(a[at(i + 1, j)], cur + stop + Math.log(row(i)[n]));
      if (j < hi.length) a[at(i + 1, j + 1)] = logAdd(a[at(i + 1, j + 1)], cur + stop + Math.log(row(i)[hi[j]]));
    }
  }
  b[at(wi.length, hi.length)] = stop;
  for (let i = wi.length; i >= 0; i--) for (let j = hi.length; j >= 0; j--) {
    if (i === wi.length && j === hi.length) continue;
    let cur = -Infinity;
    if (j < hi.length) cur = logAdd(cur, ins[j] + b[at(i, j + 1)]);
    if (i < wi.length) {
      cur = logAdd(cur, stop + Math.log(row(i)[n]) + b[at(i + 1, j)]);
      if (j < hi.length) cur = logAdd(cur, stop + Math.log(row(i)[hi[j]]) + b[at(i + 1, j + 1)]);
    }
    b[at(i, j)] = cur;
  }
  return { a, b, wi, hi, at, row, stop, ins, logP: b[0] };
}
export function phoneLogP(channel: PhoneChannel, record: PhoneRecording, pronunciation?: number[][]): number {
  return partitions(channel, record, pronunciation).logP;
}

function smoothRows(counts: number[][], phones: string[], mass: number, pronunciation = false): number[][] {
  const n = phones.length, groups = new Map<string, number>();
  const key = (r: number, h: number) => h === n ? `${phoneClass(phones[r])}:epsilon`
    : r === h ? `${phoneClass(phones[r])}:same` : `${phoneClass(phones[r])}:${phoneClass(phones[h])}`;
  const eligible = (r: number, h: number) => !pronunciation || r === h || phoneClass(phones[r]) === phoneClass(phones[h]) || subCost(phones[r], phones[h]) < UNIT;
  for (let r = 0; r < n; r++) for (let h = 0; h < counts[r].length; h++) if (eligible(r, h)) groups.set(key(r, h), (groups.get(key(r, h)) ?? 0) + counts[r][h]);
  return counts.map((row, r) => {
    const prior = row.map((_, h) => eligible(r, h) ? (groups.get(key(r, h)) ?? 0) + (r === h ? mass : 1) : 0);
    // Each class-pair's shared mass is divided over its possible output symbols.
    for (let h = 0; h < prior.length; h++) {
      const multiplicity = row.filter((_, j) => eligible(r, j) && key(r, h) === key(r, j)).length;
      if (multiplicity) prior[h] /= multiplicity;
    }
    const z = sum(prior), total = sum(row) + mass;
    return row.map((c, h) => (c + mass * prior[h] / z) / total);
  });
}
function initialChannel(phones: string[]): PhoneChannel {
  const n = phones.length;
  return { phones, insertionRate: 0.05, insertion: Array(n).fill(1 / n), rows: phones.map(p => {
    const raw = phones.map(h => Math.exp(-3 * subCost(p, h) / UNIT));
    raw.push(0.08);
    const z = sum(raw);
    return raw.map(x => x / z);
  }) };
}
function fitRecognition(records: readonly PhoneRecording[], phones: string[], mass: number, iterations: number): PhoneChannel {
  let model = initialChannel(phones);
  const n = phones.length;
  for (let it = 0; it < iterations; it++) {
    const counts = zeros(n, n + 1), inserted = Array(n).fill(0) as number[];
    let consumes = 0;
    for (const record of records) {
      const { a, b, wi, hi, at, row, stop, ins, logP } = partitions(model, record);
      consumes += wi.length + 1;
      for (let i = 0; i <= wi.length; i++) for (let j = 0; j <= hi.length; j++) {
        const cur = a[at(i, j)] - logP;
        if (j < hi.length) inserted[hi[j]] += Math.exp(cur + ins[j] + b[at(i, j + 1)]);
        if (i < wi.length) {
          counts[wi[i]][n] += Math.exp(cur + stop + Math.log(row(i)[n]) + b[at(i + 1, j)]);
          if (j < hi.length) counts[wi[i]][hi[j]] += Math.exp(cur + stop + Math.log(row(i)[hi[j]]) + b[at(i + 1, j + 1)]);
        }
      }
    }
    const total = sum(inserted);
    model = { phones, rows: smoothRows(counts, phones, mass), insertionRate: (total + 1) / (total + consumes + 2),
      insertion: inserted.map(c => (c + 1) / (total + n)) };
  }
  return model;
}

/** Recognition is anchored on clean takes and frozen before weak pronunciation supervision. */
export function fitSpokenPhoneModel(records: readonly PhoneRecording[], phones: string[], mass: number, iterations = 20): SpokenPhoneModel {
  const clean = records.filter(r => !r.variable.some(Boolean)), labelled = records.filter(r => r.variable.some(Boolean));
  if (!clean.length || !labelled.length) throw new Error("Need clean anchors and pronunciation labels");
  const recognition = fitRecognition(clean, phones, mass, iterations), n = phones.length;
  // Same recordings, inventory, smoothing family and number of EM iterations as the two-stage fit.
  const control = fitRecognition(records, phones, mass, iterations);
  let pronunciation = smoothRows(identity(n), phones, mass, true);
  for (let it = 0; it < iterations; it++) {
    const counts = zeros(n, n);
    for (const record of labelled) {
      const { a, b, wi, hi, at, row, stop, logP } = partitions(recognition, record, pronunciation);
      for (let i = 0; i < wi.length; i++) if (record.variable[i]) for (let j = 0; j <= hi.length; j++) {
        for (const [h, tail] of [[n, b[at(i + 1, j)]], ...(j < hi.length ? [[hi[j], b[at(i + 1, j + 1)]]] : [])]) {
          const expected = Math.exp(a[at(i, j)] + stop + Math.log(row(i)[h]) + tail - logP);
          for (let s = 0; s < n; s++) counts[wi[i]][s] += expected * pronunciation[wi[i]][s] * recognition.rows[s][h] / row(i)[h];
        }
      }
    }
    pronunciation = smoothRows(counts, phones, mass, true);
  }
  return { recognition, pronunciation, control };
}

/** Cost ratios use a common observed-token baseline and the tracker's existing three nats per edit. */
export function localPhoneCosts(channel: PhoneChannel) {
  composePhoneChannel(channel, identity(channel.phones.length)); // validate frozen parameters
  const index = new Map(channel.phones.map((p, i) => [p, i])), n = channel.phones.length;
  const stop = 1 - channel.insertionRate, scale = UNIT / 3;
  const duration = (ref: string, at: number) => ref[at] === ref[at - 1];
  const baseline = (h: number) => stop * channel.rows[h][h];
  return {
    sub(ref: string, at: number, p: string) {
      const r = index.get(ref[at]), h = index.get(p);
      if (r === undefined || h === undefined || duration(ref, at)) return subCost(ref[at], p);
      if (r === h) return 0;
      return -scale * Math.log(stop * channel.rows[r][h] / baseline(h));
    },
    del(ref: string, at: number) {
      const r = index.get(ref[at]);
      if (r === undefined || duration(ref, at)) return deletionCost(ref, at);
      return -scale * Math.log(stop * channel.rows[r][n]);
    },
    ins(ref: string, at: number, p: string) {
      const h = index.get(p), ordinary = insertionCost(ref, at, p);
      if (h === undefined || ordinary < UNIT) return ordinary;
      return -scale * Math.log(channel.insertionRate * channel.insertion[h] / baseline(h));
    },
  };
}
