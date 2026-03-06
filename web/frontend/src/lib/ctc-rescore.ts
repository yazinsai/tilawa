export interface AcousticEvidence {
  logprobs: Float32Array;
  timeSteps: number;
  vocabSize: number;
  blankId: number;
}

export interface CtcCandidate<T = unknown> {
  ids: number[];
  meta: T;
  priorScore?: number;
}

export interface ScoredCtcCandidate<T = unknown> extends CtcCandidate<T> {
  acousticScore: number;
  feasible: boolean;
  minFramesRequired: number;
}

const NEG_INF = Number.NEGATIVE_INFINITY;
const IMPOSSIBLE_SCORE = 1e9;

function logAddExp(a: number, b: number): number {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return hi + Math.log1p(Math.exp(lo - hi));
}

export function minFramesRequired(ids: readonly number[]): number {
  if (ids.length === 0) return 1;

  let repeats = 0;
  for (let i = 1; i < ids.length; i++) {
    if (ids[i] === ids[i - 1]) {
      repeats++;
    }
  }

  return ids.length + repeats;
}

export function scoreCtcSequence(
  evidence: AcousticEvidence,
  ids: readonly number[],
): number {
  const { logprobs, timeSteps, vocabSize, blankId } = evidence;
  const targetLength = ids.length;

  if (targetLength === 0) {
    return IMPOSSIBLE_SCORE;
  }

  if (minFramesRequired(ids) > timeSteps) {
    return IMPOSSIBLE_SCORE;
  }

  const stateCount = targetLength * 2 + 1;
  const states = new Int32Array(stateCount);
  for (let s = 0; s < stateCount; s++) {
    states[s] = s % 2 === 0 ? blankId : ids[(s - 1) >> 1];
  }

  let prev = new Float64Array(stateCount);
  let curr = new Float64Array(stateCount);
  prev.fill(NEG_INF);
  curr.fill(NEG_INF);

  prev[0] = logprobs[blankId];
  if (stateCount > 1) {
    prev[1] = logprobs[states[1]];
  }

  for (let t = 1; t < timeSteps; t++) {
    curr.fill(NEG_INF);
    const frameOffset = t * vocabSize;

    for (let s = 0; s < stateCount; s++) {
      let total = prev[s];
      if (s > 0) {
        total = logAddExp(total, prev[s - 1]);
      }
      if (
        s > 1 &&
        states[s] !== blankId &&
        states[s] !== states[s - 2]
      ) {
        total = logAddExp(total, prev[s - 2]);
      }

      if (total !== NEG_INF) {
        curr[s] = total + logprobs[frameOffset + states[s]];
      }
    }

    const tmp = prev;
    prev = curr;
    curr = tmp;
  }

  let finalScore = prev[stateCount - 1];
  if (stateCount > 1) {
    finalScore = logAddExp(finalScore, prev[stateCount - 2]);
  }
  if (!Number.isFinite(finalScore)) {
    return IMPOSSIBLE_SCORE;
  }

  return -finalScore / targetLength;
}

export function scoreCtcCandidates<T>(
  evidence: AcousticEvidence,
  candidates: readonly CtcCandidate<T>[],
): ScoredCtcCandidate<T>[] {
  return candidates
    .map((candidate) => {
      const minFrames = minFramesRequired(candidate.ids);
      const acousticScore = scoreCtcSequence(evidence, candidate.ids);
      return {
        ...candidate,
        acousticScore,
        feasible: Number.isFinite(acousticScore) && acousticScore < IMPOSSIBLE_SCORE,
        minFramesRequired: minFrames,
      };
    })
    .sort((a, b) => {
      if (a.acousticScore !== b.acousticScore) {
        return a.acousticScore - b.acousticScore;
      }
      return (b.priorScore ?? 0) - (a.priorScore ?? 0);
    });
}

export function chooseLongestStablePrefix<T>(
  scored: readonly ScoredCtcCandidate<T>[],
  tolerance = 0.12,
): ScoredCtcCandidate<T> | null {
  if (!scored.length) return null;

  const feasible = scored.filter((candidate) => candidate.feasible);
  if (!feasible.length) return null;

  const bestScore = feasible[0].acousticScore;
  let best = feasible[0];
  for (const candidate of feasible) {
    if (candidate.acousticScore > bestScore + tolerance) {
      break;
    }
    if (candidate.ids.length >= best.ids.length) {
      best = candidate;
    }
  }
  return best;
}
