import { UNIT, deletionCost, insertionCost, subCost } from "./fuzzy.js";

/** An independent, normalized token edit channel, estimated on clean training recordings. */
export interface PathChannel {
  phones: string[];
  /** Probabilities of substitution/match, insertion, and deletion; sum to one. */
  operations: [number, number, number];
  /** P(heard | written, match operation), each row sums to one. */
  identity: number[][];
  /** P(heard | insertion), sums to one. */
  insertion: number[];
}

export interface PathRecording { written: string; heard: string }
const normalize = (row: number[]) => {
  const sum = row.reduce((a, b) => a + b, 0);
  return row.map(x => x / sum);
};

/**
 * Jeffreys-smoothed counts on the existing directed edit alignment. Reference length repetitions are retained,
 * as in local matching. No practice observations or location features enter this estimator.
 */
export function fitPathChannel(records: readonly PathRecording[], phones: readonly string[]): PathChannel {
  if (!records.length || !phones.length) throw new Error("A path channel needs clean training recordings and an inventory");
  const ids = new Map(phones.map((p, i) => [p, i]));
  const operations = [0.5, 0.5, 0.5];
  const identity = phones.map(() => phones.map(() => 0.5));
  const insertion = phones.map(() => 0.5);
  for (const { written, heard } of records) {
    if ([...written, ...heard].some(p => !ids.has(p))) throw new Error("Training phone outside inventory");
    const width = heard.length + 1;
    const dp = new Float64Array((written.length + 1) * width);
    const back = new Uint8Array(dp.length);
    for (let j = 1; j <= heard.length; j++) {
      dp[j] = dp[j - 1] + insertionCost(written, 0, heard[j - 1]);
      back[j] = 1;
    }
    for (let i = 1; i <= written.length; i++) {
      dp[i * width] = dp[(i - 1) * width] + deletionCost(written, i - 1);
      back[i * width] = 2;
      for (let j = 1; j <= heard.length; j++) {
        const at = i * width + j;
        const candidates = [dp[at - width - 1] + subCost(written[i - 1], heard[j - 1]),
          dp[at - 1] + insertionCost(written, i, heard[j - 1]), dp[at - width] + deletionCost(written, i - 1)];
        const how = candidates.indexOf(Math.min(...candidates));
        dp[at] = candidates[how];
        back[at] = how;
      }
    }
    let i = written.length, j = heard.length;
    while (i || j) {
      const how = back[i * width + j];
      operations[how]++;
      if (how === 0) identity[ids.get(written[--i])!][ids.get(heard[--j])!]++;
      else if (how === 1) insertion[ids.get(heard[--j])!]++;
      else i--;
    }
  }
  return { phones: [...phones], operations: normalize(operations) as PathChannel["operations"], identity: identity.map(normalize), insertion: normalize(insertion) };
}

/** Probabilities accompany the existing edit paths; they never replace their acoustic costs. */
export function pathChannelLogs(channel: PathChannel) {
  const ids = new Map(channel.phones.map((p, i) => [p, i]));
  return {
    match: (r: string, h: string) => {
      const ri = ids.get(r), hi = ids.get(h);
      return ri === undefined || hi === undefined ? -Infinity : Math.log(channel.operations[0] * channel.identity[ri][hi]);
    },
    insert: (h: string) => {
      const hi = ids.get(h);
      return hi === undefined ? -Infinity : Math.log(channel.operations[1] * channel.insertion[hi]);
    },
    deletion: Math.log(channel.operations[2]),
    /** The original movement potential, expressed using the tracker's existing three nats per edit. */
    movement: (cost: number) => -3 * cost / UNIT,
  };
}

export interface PositionTrajectory {
  /** Ordered reference-word visits. A repeated occurrence is another entry, even at the same position. */
  words: readonly number[];
  logProbability: number;
}

export function positionSequenceDistance(a: readonly number[], b: readonly number[]): number {
  let row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 0; i < a.length; i++) {
    const next = [i + 1];
    for (let j = 0; j < b.length; j++) next.push(Math.min(next[j] + 1, row[j + 1] + 1, row[j] + (a[i] === b[j] ? 0 : 1)));
    row = next;
  }
  return row[b.length];
}

export const addLogProbability = (a: number, b: number) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
};

/** Merge identical complete sequences before normalizing and calculating posterior expected edit risk. */
export function positionSequenceMedoid(paths: readonly PositionTrajectory[]) {
  const groups = new Map<string, { words: readonly number[]; logProbability: number; representative: number }>();
  paths.forEach((path, representative) => {
    if (!Number.isFinite(path.logProbability)) return;
    const key = JSON.stringify(path.words), previous = groups.get(key);
    if (previous) {
      previous.logProbability = addLogProbability(previous.logProbability, path.logProbability);
      if (path.logProbability > paths[previous.representative].logProbability) previous.representative = representative;
    } else groups.set(key, { ...path, representative });
  });
  const trajectories = [...groups.values()];
  if (!trajectories.length) return null;
  const z = trajectories.reduce((sum, t) => addLogProbability(sum, t.logProbability), -Infinity);
  const probabilities = trajectories.map(t => Math.exp(t.logProbability - z));
  const risks = trajectories.map(() => 0);
  for (let i = 0; i < trajectories.length; i++) for (let j = i + 1; j < trajectories.length; j++) {
    const distance = positionSequenceDistance(trajectories[i].words, trajectories[j].words);
    risks[i] += probabilities[j] * distance;
    risks[j] += probabilities[i] * distance;
  }
  let best = 0;
  for (let i = 1; i < risks.length; i++) if (risks[i] < risks[best] || (risks[i] === risks[best] && probabilities[i] > probabilities[best])) best = i;
  return { representative: trajectories[best].representative, risk: risks[best],
    trajectories: trajectories.map((t, i) => ({ ...t, probability: probabilities[i], risk: risks[i] })) };
}
