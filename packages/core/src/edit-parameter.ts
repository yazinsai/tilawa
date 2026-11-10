import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** One draw of each row is shared by every edit in a complete trajectory. */
export interface EditParameterPosterior {
  phones: string[];
  /** Dirichlet concentrations for exact recognition, substitution and deletion, by written phone. */
  operations: number[][];
  /** Beta concentrations for inserting a token and advancing the reference. */
  insertion: [number, number];
  /** Conditional identities given substitution/insertion. These nuisance distributions are frozen. */
  substituted: number[][];
  inserted: number[];
  priorStrength: number;
}

export interface EditStatistics {
  /** Flattened [match, substitution, deletion] counts for each reference phone, then [insert, advance]. */
  counts: number[];
  emissionLogP: number;
  /** Existing movement and duration costs, in UNITs, outside the uncertain channel. */
  fixedCost: number;
  observations: number;
}

export type TokenEdit =
  | { kind: "diagonal"; reference: string; heard: string }
  | { kind: "insert"; heard: string; duration: boolean }
  | { kind: "delete"; reference: string; duration: boolean };

const sum = (row: readonly number[]) => row.reduce((a, b) => a + b, 0);
const normalize = (row: number[]) => { const total = sum(row); return row.map(x => x / total); };

/** Integer counts permit an exact rising-factorial calculation without quadrature or gamma cancellation. */
export function dirichletLogPredictive(alpha: readonly number[], counts: readonly number[], mean = false): number {
  if (alpha.length !== counts.length || !alpha.length || alpha.some(x => !(x > 0) || !Number.isFinite(x))
    || counts.some(x => !Number.isSafeInteger(x) || x < 0)) throw new Error("Invalid Dirichlet statistic");
  const total = sum(alpha), n = sum(counts);
  let logP = 0;
  for (let i = 0; i < alpha.length; i++) {
    if (mean) logP += counts[i] * Math.log(alpha[i] / total);
    else for (let k = 0; k < counts[i]; k++) logP += Math.log(alpha[i] + k);
  }
  if (!mean) for (let k = 0; k < n; k++) logP -= Math.log(total + k);
  return logP;
}

/** Both scoring arms consume exactly the same path and fixed movement/duration costs. */
export function editParameterLogP(model: EditParameterPosterior, stats: EditStatistics, mean = false): number {
  if (stats.counts.length !== model.phones.length * 3 + 2) throw new Error("Wrong edit statistic inventory");
  let logP = stats.emissionLogP - 3 * stats.fixedCost / UNIT;
  for (let i = 0; i < model.phones.length; i++) {
    logP += dirichletLogPredictive(model.operations[i], stats.counts.slice(3 * i, 3 * i + 3), mean);
  }
  return logP + dirichletLogPredictive(model.insertion, stats.counts.slice(-2), mean);
}

/** Rerank existing complete trajectories, with the posterior-mean arm using precisely the same channel. */
export function parameterTrajectoryDecision(model: EditParameterPosterior, paths: readonly EditStatistics[]) {
  if (!paths.length) return null;
  const n = paths[0].observations;
  if (n < 1 || paths.some(p => p.observations !== n)) throw new Error("Trajectory comparison needs a common heard interval");
  const marginal = paths.map(p => editParameterLogP(model, p));
  const mean = paths.map(p => editParameterLogP(model, p, true));
  const best = (scores: number[]) => {
    let at = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[at]) at = i;
    return at;
  };
  return { marginal: { selected: best(marginal), logP: marginal }, mean: { selected: best(mean), logP: mean } };
}

export function editParameterCounter(model: EditParameterPosterior) {
  const ids = new Map(model.phones.map((p, i) => [p, i]));
  const id = (p: string) => { const at = ids.get(p); if (at === undefined) throw new Error("Phone outside edit channel"); return at; };
  const empty = (): EditStatistics => ({ counts: Array(model.phones.length * 3 + 2).fill(0), emissionLogP: 0, fixedCost: 0, observations: 0 });
  const extend = (before: EditStatistics, edit: TokenEdit, movementCost = 0): EditStatistics => {
    const next = { ...before, counts: [...before.counts], fixedCost: before.fixedCost + movementCost };
    const insertionAt = model.phones.length * 3;
    if (edit.kind !== "delete") next.observations++;
    if (edit.kind === "diagonal") {
      const wi = id(edit.reference), hi = id(edit.heard), same = wi === hi;
      next.counts[3 * wi + (same ? 0 : 1)]++;
      next.counts[insertionAt + 1]++;
      if (!same) next.emissionLogP += Math.log(model.substituted[wi][hi]);
    } else if (edit.duration) {
      // A duration-expanded symbol remains a fixed length charge; it does not teach an identity edit rate.
      next.fixedCost += 2;
    } else if (edit.kind === "insert") {
      next.counts[insertionAt]++;
      next.emissionLogP += Math.log(model.inserted[id(edit.heard)]);
    } else {
      next.counts[3 * id(edit.reference) + 2]++;
      next.counts[insertionAt + 1]++;
    }
    return next;
  };
  return { empty, extend };
}

export interface ParameterRecording { written: string; heard: string }

/** Fixed directed alignments provide conjugate observations; this does not claim a latent-alignment posterior. */
export function parameterEdits({ written, heard }: ParameterRecording): TokenEdit[] {
  const width = heard.length + 1;
  const cost = new Float64Array((written.length + 1) * width);
  const back = new Uint8Array(cost.length);
  for (let j = 1; j <= heard.length; j++) {
    cost[j] = cost[j - 1] + insertionCost(written, 0, heard[j - 1]); back[j] = 1;
  }
  for (let i = 1; i <= written.length; i++) {
    cost[i * width] = cost[(i - 1) * width] + deletionCost(written, i - 1); back[i * width] = 2;
    for (let j = 1; j <= heard.length; j++) {
      const at = i * width + j;
      const options = [cost[at - width - 1] + subCost(written[i - 1], heard[j - 1]),
        cost[at - 1] + insertionCost(written, i, heard[j - 1]), cost[at - width] + deletionCost(written, i - 1)];
      const how = options.indexOf(Math.min(...options)); cost[at] = options[how]; back[at] = how;
    }
  }
  const edits: TokenEdit[] = [];
  let i = written.length, j = heard.length;
  while (i || j) {
    const how = back[i * width + j];
    if (how === 0) edits.push({ kind: "diagonal", reference: written[--i], heard: heard[--j] });
    else if (how === 1) {
      const p = heard[--j]; edits.push({ kind: "insert", heard: p, duration: insertionCost(written, i, p) === 2 });
    } else {
      i--; edits.push({ kind: "delete", reference: written[i], duration: deletionCost(written, i) === 2 });
    }
  }
  return edits.reverse();
}

/** Empirical-Bayes row pooling; the pooled mean and all counts use only the supplied training partition. */
export function fitEditParameters(records: readonly ParameterRecording[], phones: readonly string[], priorStrength: number): EditParameterPosterior {
  if (!records.length || phones.length < 2 || !(priorStrength > 0)) throw new Error("Invalid edit parameter training input");
  const ids = new Map(phones.map((p, i) => [p, i]));
  const operations = phones.map(() => [0, 0, 0]);
  const insertion: [number, number] = [0, 0];
  const substituted = phones.map(() => phones.map(() => 0));
  const inserted = phones.map(() => 0);
  for (const record of records) for (const edit of parameterEdits(record)) {
    const wi = edit.kind !== "insert" ? ids.get(edit.reference) : undefined;
    const hi = edit.kind !== "delete" ? ids.get(edit.heard) : undefined;
    if ((edit.kind !== "insert" && wi === undefined) || (edit.kind !== "delete" && hi === undefined)) throw new Error("Unknown training phone");
    if (edit.kind === "diagonal") {
      operations[wi!][wi === hi ? 0 : 1]++; insertion[1]++;
      if (wi !== hi) substituted[wi!][hi!]++;
    } else if (!edit.duration) {
      if (edit.kind === "insert") { insertion[0]++; inserted[hi!]++; }
      else { operations[wi!][2]++; insertion[1]++; }
    }
  }
  const pooled = normalize([0, 1, 2].map(k => 0.5 + operations.reduce((n, row) => n + row[k], 0)));
  const insertMean = normalize(insertion.map(x => x + 0.5));
  return {
    phones: [...phones], priorStrength,
    operations: operations.map(row => row.map((x, k) => x + priorStrength * pooled[k])),
    insertion: insertion.map((x, k) => x + priorStrength * insertMean[k]) as [number, number],
    substituted: substituted.map((row, wi) => {
      const prior = normalize(phones.map((p, hi) => hi === wi ? 0 : Math.exp(-3 * subCost(phones[wi], p) / UNIT)));
      return normalize(row.map((x, hi) => x + priorStrength * prior[hi]));
    }),
    inserted: normalize(inserted.map(x => x + 0.5)),
  };
}

/** No combinatorial count multiplicity: the probability is of this ordered edit path. */
export function parameterRecordingLogP(model: EditParameterPosterior, record: ParameterRecording, mean = false): number {
  const counter = editParameterCounter(model);
  return editParameterLogP(model, parameterEdits(record).reduce((s, e) => counter.extend(s, e), counter.empty()), mean);
}
