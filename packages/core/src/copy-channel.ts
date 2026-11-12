import type { DecodedPhoneme } from "./evidence.js";
import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** The local matcher's existing recent-observation horizon. */
export const COPY_HISTORY = 16;
const NATS_PER_EDIT = 3;
const CLASSES = ["َُِ", "اۦۥ", "نمں۾", "بف", "تدطض", "سزصثذظ", "شج", "كق", "عءهحخغ", "لر", "وي"];
export const copyPhoneClass = (p: string): number => {
  const c = CLASSES.findIndex(s => s.includes(p));
  return c < 0 ? CLASSES.length : c;
};
const NC = CLASSES.length + 1;
const normalize = (xs: readonly number[]): number[] => {
  const total = xs.reduce((a, b) => a + b, 0);
  if (!(total > 0) || xs.some(x => !Number.isFinite(x) || x < 0)) throw new Error("Invalid copy-channel probability row");
  return xs.map(x => x / total);
};

export interface CopyChannelModel {
  version: 1;
  linked: boolean;
  inventory: string[];
  insertion: number;
  copy: number;
  same: number;
  ordinary: number[];
  sourcePrior: number[];
  copyClasses: number[][];
  delay: number[];
  sourceClasses: number[];
}
export interface CopyRecording { written: string; observations: DecodedPhoneme[] }
export interface CopySource { index: number; lag: number; phone: string; weight: number }

/** Only distinct, already-consumed acoustic occurrences can supply a delayed copy. */
export function copySources(model: CopyChannelModel, history: readonly DecodedPhoneme[], observation: DecodedPhoneme): CopySource[] {
  const sources: CopySource[] = [];
  const seen = new Set<string>();
  for (let i = history.length - 1; i >= Math.max(0, history.length - COPY_HISTORY); i--) {
    const p = history[i], lag = history.length - i;
    if (p.end >= observation.start || !model.inventory.includes(p.p)) continue;
    const occurrence = `${p.start}:${p.end}:${p.p}`;
    if (seen.has(occurrence)) continue;
    seen.add(occurrence);
    sources.push({ index: i, lag, phone: p.p,
      weight: model.delay[lag - 1] * model.sourceClasses[copyPhoneClass(p.p)] });
  }
  const total = sources.reduce((a, s) => a + s.weight, 0);
  return total > 0 ? sources.map(s => ({ ...s, weight: s.weight / total })) : [];
}

function ordinary(model: CopyChannelModel, phone: string): number {
  const c = copyPhoneClass(phone), size = model.inventory.filter(p => copyPhoneClass(p) === c).length;
  return size ? model.ordinary[c] / size : 0;
}
function copied(model: CopyChannelModel, source: string, phone: string): number {
  const c = copyPhoneClass(phone), size = model.inventory.filter(p => copyPhoneClass(p) === c).length;
  return (source === phone ? model.same : 0) + (size ? (1 - model.same) * model.copyClasses[copyPhoneClass(source)][c] / size : 0);
}
function sourceOptions(model: CopyChannelModel, history: readonly DecodedPhoneme[], observation: DecodedPhoneme): CopySource[] {
  return model.linked ? copySources(model, history, observation)
    : model.inventory.map((phone, i) => ({ phone, index: -1, lag: 0, weight: model.sourcePrior[i] }));
}
export function copyObservationProbability(model: CopyChannelModel, history: readonly DecodedPhoneme[], observation: DecodedPhoneme): number {
  if (!model.inventory.includes(observation.p)) return 0;
  const sources = sourceOptions(model, history, observation);
  const q = ordinary(model, observation.p);
  if (!sources.length) return q;
  return (1 - model.copy) * q + model.copy * sources.reduce((a, s) => a + s.weight * copied(model, s.phone, observation.p), 0);
}

/** Normalized ordinary/copy alternatives both consume one observation and no reference phone. */
export function copyInsertionCost(model: CopyChannelModel, reference: string, at: number,
  history: readonly DecodedPhoneme[], observation: DecodedPhoneme): number {
  const base = insertionCost(reference, at, observation.p);
  // Existing pronunciation-length charges remain separate from recognition echoes.
  if (base < insertionCost("", 0, observation.p)) return base;
  const q = copyObservationProbability(model, history, observation);
  return q > 0 ? -Math.log(model.insertion * q) * UNIT / NATS_PER_EDIT : base;
}

function emission(model: CopyChannelModel, ref: string, obs: string): number {
  const weights = model.inventory.map(p => Math.exp(-NATS_PER_EDIT * subCost(ref, p) / UNIT));
  return Math.exp(-NATS_PER_EDIT * subCost(ref, obs) / UNIT) / weights.reduce((a, b) => a + b, 0);
}
const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};
interface Stats {
  logP: number; insertions: number; nonInsertions: number; copies: number; ordinary: number[];
  same: number; different: number; copyClasses: number[][]; sourcePrior: number[];
  delayGradient: number[]; sourceGradient: number[];
}
const stats = (model: CopyChannelModel): Stats => ({ logP: 0, insertions: 0, nonInsertions: 0, copies: 0,
  ordinary: new Array(NC).fill(0), same: 0, different: 0,
  copyClasses: Array.from({ length: NC }, () => new Array(NC).fill(0)), sourcePrior: model.inventory.map(() => 0),
  delayGradient: new Array(COPY_HISTORY).fill(0), sourceGradient: new Array(NC).fill(0) });

/** Complete-string forward/backward; edit placement and copy-source identity are latent. */
function expectation(model: CopyChannelModel, record: CopyRecording, collect: boolean): Stats {
  const obs = record.observations, n = record.written.length, m = obs.length, width = m + 1;
  if ([...record.written, ...obs.map(p => p.p)].some(p => !model.inventory.includes(p))) throw new Error("Unknown training phone");
  const ins = obs.map((p, j) => model.insertion * copyObservationProbability(model, obs.slice(Math.max(0, j - COPY_HISTORY), j), p));
  const del = [...record.written].map((_, i) => {
    const odds = Math.exp(-NATS_PER_EDIT * deletionCost(record.written, i) / UNIT);
    return odds / (1 + odds);
  });
  const emit = [...record.written].map(p => obs.map(o => emission(model, p, o.p)));
  const a = new Float64Array((n + 1) * width).fill(-Infinity), b = new Float64Array(a.length).fill(-Infinity);
  a[0] = 0;
  const at = (i: number, j: number) => i * width + j;
  const li = ins.map(Math.log), stop = Math.log(1 - model.insertion);
  const ld = del.map(d => stop + Math.log(d));
  const lm = del.map((d, i) => emit[i].map(q => stop + Math.log(1 - d) + Math.log(q)));
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const x = a[at(i, j)];
    if (j < m) a[at(i, j + 1)] = logAdd(a[at(i, j + 1)], x + li[j]);
    if (i < n) {
      a[at(i + 1, j)] = logAdd(a[at(i + 1, j)], x + ld[i]);
      if (j < m) a[at(i + 1, j + 1)] = logAdd(a[at(i + 1, j + 1)], x + lm[i][j]);
    }
  }
  b[at(n, m)] = stop;
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    let x = b[at(i, j)];
    if (j < m) x = logAdd(x, li[j] + b[at(i, j + 1)]);
    if (i < n) {
      x = logAdd(x, ld[i] + b[at(i + 1, j)]);
      if (j < m) x = logAdd(x, lm[i][j] + b[at(i + 1, j + 1)]);
    }
    b[at(i, j)] = x;
  }
  const out = stats(model), z = a[at(n, m)] + stop;
  out.logP = z;
  if (!collect) return out;
  out.nonInsertions = n + 1; // Every complete path consumes all reference symbols and stops once.
  for (let j = 0; j < m; j++) {
    let count = 0;
    for (let i = 0; i <= n; i++) count += Math.exp(a[at(i, j)] + li[j] + b[at(i, j + 1)] - z);
    out.insertions += count;
    const p = obs[j].p, c = copyPhoneClass(p), history = obs.slice(Math.max(0, j - COPY_HISTORY), j);
    const sources = sourceOptions(model, history, obs[j]), q = ins[j] / model.insertion;
    if (!sources.length) { out.ordinary[c] += count; continue; }
    out.ordinary[c] += count * (1 - model.copy) * ordinary(model, p) / q;
    let echoCount = 0;
    for (const s of sources) {
      const probability = copied(model, s.phone, p);
      const r = count * model.copy * s.weight * probability / q;
      echoCount += r;
      const exact = s.phone === p ? model.same / probability : 0;
      out.same += r * exact;
      out.different += r * (1 - exact);
      out.copyClasses[copyPhoneClass(s.phone)][c] += r * (1 - exact);
      out.sourcePrior[model.inventory.indexOf(s.phone)] += r;
      if (model.linked) { out.delayGradient[s.lag - 1] += r; out.sourceGradient[copyPhoneClass(s.phone)] += r; }
    }
    out.copies += echoCount;
    if (model.linked) for (const s of sources) {
      out.delayGradient[s.lag - 1] -= echoCount * s.weight;
      out.sourceGradient[copyPhoneClass(s.phone)] -= echoCount * s.weight;
    }
  }
  return out;
}

export const copyRecordingLogP = (model: CopyChannelModel, recording: CopyRecording): number => expectation(model, recording, false).logP;

/** Training-only hierarchical EM; pooling is selected by held-speaker predictive likelihood. */
export function fitCopyChannel(records: readonly CopyRecording[], inventory: readonly string[], pooling: number, linked: boolean): CopyChannelModel {
  if (!records.length || inventory.length < 2 || new Set(inventory).size !== inventory.length || !(pooling > 0)) throw new Error("Invalid copy training input");
  const counts = Array.from({ length: NC }, (_, c) => inventory.filter(p => copyPhoneClass(p) === c).length);
  const background = normalize(counts), positiveClasses = counts.map(n => n ? 1 : 0);
  let model: CopyChannelModel = { version: 1, linked, inventory: [...inventory], insertion: 0.1, copy: 0.1, same: 0.8,
    ordinary: background, sourcePrior: inventory.map(() => 1 / inventory.length),
    copyClasses: Array.from({ length: NC }, () => [...background]),
    delay: Array.from({ length: COPY_HISTORY }, () => 1 / COPY_HISTORY), sourceClasses: normalize(positiveClasses) };
  for (let iteration = 0; iteration < 24; iteration++) {
    const total = stats(model);
    for (const record of records) {
      const x = expectation(model, record, true);
      total.insertions += x.insertions; total.nonInsertions += x.nonInsertions; total.copies += x.copies;
      total.same += x.same; total.different += x.different;
      for (let c = 0; c < NC; c++) {
        total.ordinary[c] += x.ordinary[c]; total.sourceGradient[c] += x.sourceGradient[c];
        for (let d = 0; d < NC; d++) total.copyClasses[c][d] += x.copyClasses[c][d];
      }
      x.sourcePrior.forEach((v, i) => total.sourcePrior[i] += v);
      x.delayGradient.forEach((v, i) => total.delayGradient[i] += v);
    }
    const poolCopy = normalize(background.map((p, c) => p * pooling + total.copyClasses.reduce((a, row) => a + row[c], 0)));
    const updateWeights = (old: number[], gradient: number[], prior: number[]) => normalize(old.map((v, i) =>
      prior[i] ? v * Math.exp((gradient[i] + pooling * (prior[i] - v)) / Math.max(pooling, total.copies)) : 0));
    model = { ...model,
      insertion: (total.insertions + 1) / (total.insertions + total.nonInsertions + 10),
      copy: (total.copies + 1) / (total.insertions + 10),
      same: (total.same + 4) / (total.same + total.different + 5),
      ordinary: normalize(total.ordinary.map((v, c) => v + pooling * background[c])),
      sourcePrior: normalize(total.sourcePrior.map(v => v + pooling / inventory.length)),
      copyClasses: total.copyClasses.map(row => normalize(row.map((v, c) => v + pooling * poolCopy[c]))),
      delay: updateWeights(model.delay, total.delayGradient, model.delay.map(() => 1 / COPY_HISTORY)),
      sourceClasses: updateWeights(model.sourceClasses, total.sourceGradient, normalize(positiveClasses)),
    };
  }
  return model;
}
