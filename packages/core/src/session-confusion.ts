import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";

/** A recording draws one mapping; it is shared by every word in that recording. */
export interface ConfusionModel {
  phones: string[];
  prior: number[];
  /** Normalized P(heard phone | written phone, mapping). */
  emissions: number[][][];
  /** Normalized channel before observing the recording-specific mapping. */
  pooled: number[][];
  mass: number;
}

export interface ConfusionRecording {
  /** Expected written/heard pairs, marginalized over the ordinary edit alignment. */
  pairs: number[][];
}

/** The engine's existing acoustic likelihood conversion, also used by tracker.ts. */
const NATS_PER_EDIT = 3;
const logAdd = (a: number, b: number): number => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};
const normalize = (row: number[]) => {
  const total = row.reduce((a, b) => a + b, 0);
  return row.map(x => x / total);
};
const logNormalize = (row: number[]) => {
  const z = row.reduce(logAdd, -Infinity);
  return row.map(x => Math.exp(x - z));
};

/** Feature sharing is independent of word identity, neighbouring phones and location. */
export function phoneFeatures(p: string): string[] {
  const group = (groups: string[]) => {
    const at = groups.findIndex(g => g.includes(p));
    return at < 0 ? "other" : String(at);
  };
  return [
    group(["َُِاۦۥ۪ٲ", "ءبتثجحخدذرزسشصضطظعغفقكلمنهوينں۾ڇؙۜ"]),
    group(["ءبتدطضقك", "ثحخذزسشصظغفهڇ", "ج", "منں۾", "رل", "وي", "َُِاۦۥ۪ٲ"]),
    group(["بمفو۾", "تثدذرزسصضطظلنں", "جشكيڇ", "خغق", "حع", "ءه", "َُِاۦۥ۪ٲ"]),
    group(["بجدذرزضظعغرلمنويں۾", "ءتثحخسشصطفقكهڇ", "َُِاۦۥ۪ٲ"]),
    group(["َاٲ", "ُۥو", "ِۦ۪ي", "ں۾نم", "ءبتثجحخدذرزسشصضطظعغفقكلهڇؙۜ"]),
  ];
}

const blankPairs = (n: number) => Array.from({ length: n }, () => Array(n).fill(0) as number[]);

/**
 * Forward/backward expected diagonal counts. Duration deletions and insertions keep their existing prices;
 * neither teaches an identity confusion. This does not collapse distinct articulations or expanded durations.
 */
export function confusionRecording(written: string, heard: string, phones: readonly string[]): ConfusionRecording {
  const n = written.length, m = heard.length, width = m + 1;
  const at = (i: number, j: number) => i * width + j;
  const f = new Float64Array((n + 1) * width).fill(-Infinity);
  const b = new Float64Array(f.length).fill(-Infinity);
  const sub = (i: number, j: number) => -NATS_PER_EDIT * subCost(written[i], heard[j]) / UNIT;
  const ins = (i: number, j: number) => -NATS_PER_EDIT * insertionCost(written, i, heard[j]) / UNIT;
  const del = (i: number) => -NATS_PER_EDIT * deletionCost(written, i) / UNIT;
  f[0] = 0;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= m; j++) {
    const cur = f[at(i, j)];
    if (i < n && j < m) f[at(i + 1, j + 1)] = logAdd(f[at(i + 1, j + 1)], cur + sub(i, j));
    if (i < n) f[at(i + 1, j)] = logAdd(f[at(i + 1, j)], cur + del(i));
    if (j < m) f[at(i, j + 1)] = logAdd(f[at(i, j + 1)], cur + ins(i, j));
  }
  b[at(n, m)] = 0;
  for (let i = n; i >= 0; i--) for (let j = m; j >= 0; j--) {
    if (i < n && j < m) b[at(i, j)] = logAdd(b[at(i, j)], sub(i, j) + b[at(i + 1, j + 1)]);
    if (i < n) b[at(i, j)] = logAdd(b[at(i, j)], del(i) + b[at(i + 1, j)]);
    if (j < m) b[at(i, j)] = logAdd(b[at(i, j)], ins(i, j) + b[at(i, j + 1)]);
  }
  const ids = new Map(phones.map((p, i) => [p, i]));
  const pairs = blankPairs(phones.length);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    const wi = ids.get(written[i]), hi = ids.get(heard[j]);
    if (wi !== undefined && hi !== undefined) pairs[wi][hi] += Math.exp(f[at(i, j)] + sub(i, j) + b[at(i + 1, j + 1)] - f[at(n, m)]);
  }
  return { pairs };
}

type FeatureCounts = Map<string, Map<string, number>>;
const featureKey = (axis: number, value: string) => `${axis}/${value}`;
function featuresOfPairs(pairs: number[][], features: string[][]): FeatureCounts {
  const out: FeatureCounts = new Map();
  for (let w = 0; w < pairs.length; w++) for (let h = 0; h < pairs.length; h++) {
    const weight = pairs[w][h];
    if (!weight) continue;
    for (let a = 0; a < features[w].length; a++) {
      const key = featureKey(a, features[w][a]);
      let row = out.get(key);
      if (!row) out.set(key, row = new Map());
      row.set(features[h][a], (row.get(features[h][a]) ?? 0) + weight);
    }
  }
  return out;
}
function featureProbability(counts: FeatureCounts, key: string, value: string, values: string[], parent?: FeatureCounts, mass = 0): number {
  const row = counts.get(key) ?? new Map<string, number>();
  const total = [...row.values()].reduce((a, b) => a + b, 0);
  const prior = parent ? featureProbability(parent, key, value, values) : 1 / values.length;
  const strength = parent ? mass : 0.5 * values.length;
  return ((row.get(value) ?? 0) + strength * prior) / (total + strength);
}

/** Fractional pair counts are a fixed edit-posterior training statistic, not reference-context features. */
export function fitConfusionModel(records: readonly ConfusionRecording[], phones: readonly string[], mappings: number, mass: number): ConfusionModel {
  if (!records.length || !phones.length || !Number.isInteger(mappings) || mappings < 1 || mappings > records.length || !(mass > 0)) throw new Error("Invalid confusion fit");
  const n = phones.length;
  const features = phones.map(phoneFeatures);
  const values = features[0].map((_, axis) => [...new Set(features.map(f => f[axis]))]);
  const pooledCounts = blankPairs(n);
  for (const r of records) for (let w = 0; w < n; w++) for (let h = 0; h < n; h++) pooledCounts[w][h] += r.pairs[w][h];
  const pooled = pooledCounts.map((row, w) => {
    const prior = normalize(phones.map(p => Math.exp(-NATS_PER_EDIT * subCost(phones[w], p) / UNIT)));
    return normalize(row.map((v, h) => v + mass * prior[h]));
  });
  const parent = featuresOfPairs(pooledCounts, features);
  let responsibilities = records.map((_, i) => Array.from({ length: mappings }, (__, k) => k === i % mappings ? 1 : 0));
  let model: ConfusionModel = { phones: [...phones], prior: Array(mappings).fill(1 / mappings), emissions: [], pooled, mass };
  for (let iteration = 0; iteration < 40; iteration++) {
    const priorCounts = Array(mappings).fill(0.5) as number[];
    model.emissions = Array.from({ length: mappings }, (_, k) => {
      const counts = blankPairs(n);
      for (let i = 0; i < records.length; i++) {
        const weight = responsibilities[i][k];
        priorCounts[k] += weight;
        for (let w = 0; w < n; w++) for (let h = 0; h < n; h++) counts[w][h] += weight * records[i].pairs[w][h];
      }
      const child = featuresOfPairs(counts, features);
      return pooled.map((row, w) => normalize(row.map((p, h) => {
        let ratio = 1;
        for (let a = 0; a < features[w].length; a++) {
          const key = featureKey(a, features[w][a]), value = features[h][a];
          ratio *= featureProbability(child, key, value, values[a], parent, mass) / featureProbability(parent, key, value, values[a]);
        }
        return p * ratio;
      })));
    });
    model.prior = normalize(priorCounts);
    const next = records.map(r => confusionPosterior(model, [r]));
    const change = next.reduce((sum, row, i) => sum + row.reduce((s, p, k) => s + Math.abs(p - responsibilities[i][k]), 0), 0);
    responsibilities = next;
    if (change < 1e-7) break;
  }
  return model;
}

function recordingLogP(model: ConfusionModel, record: ConfusionRecording, k: number): number {
  let logP = 0;
  for (let w = 0; w < model.phones.length; w++) for (let h = 0; h < model.phones.length; h++) {
    logP += record.pairs[w][h] * Math.log(model.emissions[k][w][h]);
  }
  return logP;
}

/** One shared mapping is inferred from all earlier eligible stretches; the prior prevents unsupported adaptation. */
export function confusionPosterior(model: ConfusionModel, earlier: readonly ConfusionRecording[]): number[] {
  return logNormalize(model.prior.map((p, k) => Math.log(p) + earlier.reduce((sum, r) => sum + recordingLogP(model, r, k), 0)));
}

export function confusionPredictiveLogP(model: ConfusionModel, record: ConfusionRecording): number {
  return model.prior.map((p, k) => Math.log(p) + recordingLogP(model, record, k)).reduce(logAdd, -Infinity);
}

export interface FrozenConfusion {
  readonly posterior: readonly number[];
  readonly mappings: number;
  /** Identity likelihood relative to the pre-existing edit model. Indel/duration prices stay unchanged. */
  substitution(written: string, heard: string, mapping: number): number;
  /** Marginalize complete candidate costs over the same frozen recording posterior. */
  marginal(costs: readonly number[]): number;
}

export function freezeConfusion(model: ConfusionModel, posterior: readonly number[]): FrozenConfusion {
  if (posterior.length !== model.prior.length || posterior.some(p => !Number.isFinite(p) || p < 0) ||
    Math.abs(posterior.reduce((a, b) => a + b, 0) - 1) > 1e-6) throw new Error("Invalid confusion posterior");
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  // Own the snapshot: neither later adaptation nor the caller's arrays can alter this comparison.
  const snapshot = Object.freeze([...posterior]);
  const referenceChannel = model.phones.map(w => normalize(model.phones.map(h => Math.exp(-NATS_PER_EDIT * subCost(w, h) / UNIT))));
  const adjustments = model.emissions.map(rows => rows.map((row, w) => row.map((p, h) => -UNIT / NATS_PER_EDIT * Math.log(p / referenceChannel[w][h]))));
  return {
    posterior: snapshot,
    mappings: snapshot.length,
    substitution(written, heard, k) {
      const w = ids.get(written), h = ids.get(heard);
      return subCost(written, heard) + (w === undefined || h === undefined ? 0 : adjustments[k][w][h]);
    },
    marginal(costs) {
      if (costs.length !== snapshot.length) throw new Error("Candidate mappings differ");
      return -UNIT / NATS_PER_EDIT * costs.map((c, k) => Math.log(snapshot[k]) - NATS_PER_EDIT * c / UNIT).reduce(logAdd, -Infinity);
    },
  };
}

/**
 * Fixed-start, open-terminal comparison prototype. Every heard token is consumed. The mapping stays fixed
 * along each complete edit path, then its likelihood is marginalized once at the endpoint. Leading reference
 * phones pay deletions, while an unfinished terminal word remains open, as in local tracking.
 */
export function confusionSpanCost(written: string, heard: string, frozen: FrozenConfusion): number {
  const rows = Array.from({ length: frozen.mappings }, () => {
    const row = new Float64Array(written.length + 1);
    for (let j = 1; j < row.length; j++) row[j] = row[j - 1] + deletionCost(written, j - 1);
    return row;
  });
  for (const p of heard) for (let k = 0; k < frozen.mappings; k++) {
    const previous = rows[k];
    const current = new Float64Array(previous.length);
    current[0] = previous[0] + insertionCost(written, 0, p);
    for (let j = 1; j < current.length; j++) current[j] = Math.min(
      previous[j] + insertionCost(written, j, p),
      previous[j - 1] + frozen.substitution(written[j - 1], p, k),
      current[j - 1] + deletionCost(written, j - 1),
    );
    rows[k] = current;
  }
  let best = Infinity;
  for (let j = 0; j <= written.length; j++) best = Math.min(best, frozen.marginal(rows.map(row => row[j])));
  return best;
}

export interface CommittedConfusionStretch {
  /** Half-open heard-token interval; eligible only if it ends before the comparison starts. */
  from: number;
  to: number;
  /** Generic reference-word indices, used solely to exclude checker accusations. */
  words: readonly number[];
  recording: ConfusionRecording;
}

/**
 * Read-only adaptation ledger. Callers supply actually committed stretches and checker feedback. A stretch
 * is excluded if any of its words has ever been accused (including uncertain and resolved issues). Acknowledgment
 * ensures evidence reaches the checker before it is used for inference; no guessed clean label is substituted.
 */
export function committedConfusionHistory(model: ConfusionModel) {
  const stretches: CommittedConfusionStretch[] = [];
  const accused = new Set<number>();
  let checkedThrough = 0;
  return {
    commit(stretch: CommittedConfusionStretch) {
      if (stretch.from < 0 || stretch.to <= stretch.from || !stretch.words.length) throw new Error("Invalid committed stretch");
      const previous = stretches.at(-1);
      if (previous && stretch.from < previous.to) throw new Error("Adaptation observations must not overlap");
      stretches.push({ ...stretch, words: [...stretch.words], recording: { pairs: stretch.recording.pairs.map(row => [...row]) } });
    },
    feedback(through: number, words: readonly number[]) {
      checkedThrough = Math.max(checkedThrough, through);
      for (const word of words) accused.add(word);
    },
    freezeBefore(comparisonFrom: number): FrozenConfusion {
      const earlier = stretches.filter(s => s.to <= Math.min(comparisonFrom, checkedThrough) && !s.words.some(w => accused.has(w)));
      return freezeConfusion(model, confusionPosterior(model, earlier.map(s => s.recording)));
    },
  };
}
