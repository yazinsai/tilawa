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

export interface CtcAlignment {
  feasible: boolean;
  score: number;
  startFrame: number;
  endFrame: number;
  tokenFrames: number[];
}

export interface CtcPrefixAlignment extends CtcAlignment {
  tokensAligned: number;
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

function impossibleAlignment(): CtcAlignment {
  return {
    feasible: false,
    score: IMPOSSIBLE_SCORE,
    startFrame: -1,
    endFrame: -1,
    tokenFrames: [],
  };
}

function impossiblePrefixAlignment(tokensAligned = 0): CtcPrefixAlignment {
  return {
    ...impossibleAlignment(),
    tokensAligned,
  };
}

export function alignCtcSequence(
  evidence: AcousticEvidence,
  targetIds: readonly number[],
): CtcAlignment {
  const { logprobs, timeSteps, vocabSize, blankId } = evidence;
  const targetLength = targetIds.length;

  if (targetLength === 0 || minFramesRequired(targetIds) > timeSteps) {
    return impossibleAlignment();
  }

  const stateCount = targetLength * 2 + 1;
  const states = new Int32Array(stateCount);
  for (let s = 0; s < stateCount; s++) {
    states[s] = s % 2 === 0 ? blankId : targetIds[(s - 1) >> 1];
  }

  let prev = new Float64Array(stateCount);
  let curr = new Float64Array(stateCount);
  prev.fill(NEG_INF);
  curr.fill(NEG_INF);

  // Backpointer stores the previous state chosen for each (time, state).
  // -1 means unreachable/uninitialized.
  const back = new Int32Array(timeSteps * stateCount);
  back.fill(-1);

  prev[0] = logprobs[blankId];
  back[0] = 0;
  if (stateCount > 1) {
    prev[1] = logprobs[states[1]];
    back[1] = 1;
  }

  for (let t = 1; t < timeSteps; t++) {
    curr.fill(NEG_INF);
    const frameOffset = t * vocabSize;
    const backOffset = t * stateCount;

    for (let s = 0; s < stateCount; s++) {
      let best = prev[s];
      let bestPrev = s;

      if (s > 0 && prev[s - 1] > best) {
        best = prev[s - 1];
        bestPrev = s - 1;
      }
      if (
        s > 1 &&
        states[s] !== blankId &&
        states[s] !== states[s - 2] &&
        prev[s - 2] > best
      ) {
        best = prev[s - 2];
        bestPrev = s - 2;
      }

      if (best !== NEG_INF) {
        curr[s] = best + logprobs[frameOffset + states[s]];
        back[backOffset + s] = bestPrev;
      }
    }

    const tmp = prev;
    prev = curr;
    curr = tmp;
  }

  let finalState = stateCount - 1;
  let bestLogProb = prev[finalState];
  if (stateCount > 1 && prev[stateCount - 2] > bestLogProb) {
    finalState = stateCount - 2;
    bestLogProb = prev[finalState];
  }

  if (!Number.isFinite(bestLogProb)) {
    return impossibleAlignment();
  }

  const path = new Int32Array(timeSteps);
  let state = finalState;
  for (let t = timeSteps - 1; t >= 0; t--) {
    path[t] = state;
    const prevState = back[t * stateCount + state];
    if (t > 0 && prevState < 0) {
      return impossibleAlignment();
    }
    state = prevState;
  }

  const firstTokenFrame = new Int32Array(targetLength);
  firstTokenFrame.fill(-1);
  let startFrame = -1;
  let endFrame = -1;
  for (let t = 0; t < timeSteps; t++) {
    const s = path[t];
    if (s % 2 === 1) {
      const tokenIndex = (s - 1) >> 1;
      if (firstTokenFrame[tokenIndex] < 0) {
        firstTokenFrame[tokenIndex] = t;
      }
      if (startFrame < 0) startFrame = t;
      endFrame = t;
    }
  }

  if (startFrame < 0 || endFrame < 0) {
    return impossibleAlignment();
  }

  return {
    feasible: true,
    score: -bestLogProb / targetLength,
    startFrame,
    endFrame,
    tokenFrames: Array.from(firstTokenFrame),
  };
}

export function alignCtcPrefix(
  evidence: AcousticEvidence,
  targetIds: readonly number[],
): CtcPrefixAlignment {
  const { logprobs, timeSteps, vocabSize, blankId } = evidence;
  if (targetIds.length === 0 || timeSteps <= 0) {
    return impossiblePrefixAlignment();
  }

  const stateCount = targetIds.length * 2 + 1;
  const states = new Int32Array(stateCount);
  for (let s = 0; s < stateCount; s++) {
    states[s] = s % 2 === 0 ? blankId : targetIds[(s - 1) >> 1];
  }

  let prev = new Float64Array(stateCount);
  let curr = new Float64Array(stateCount);
  prev.fill(NEG_INF);
  curr.fill(NEG_INF);
  const back = new Int32Array(timeSteps * stateCount);
  back.fill(-1);

  prev[0] = logprobs[blankId];
  back[0] = 0;
  if (stateCount > 1) {
    prev[1] = logprobs[states[1]];
    back[1] = 1;
  }

  for (let t = 1; t < timeSteps; t++) {
    curr.fill(NEG_INF);
    const frameOffset = t * vocabSize;
    const backOffset = t * stateCount;

    for (let s = 0; s < stateCount; s++) {
      let best = prev[s];
      let bestPrev = s;
      if (s > 0 && prev[s - 1] > best) {
        best = prev[s - 1];
        bestPrev = s - 1;
      }
      if (
        s > 1 &&
        states[s] !== blankId &&
        states[s] !== states[s - 2] &&
        prev[s - 2] > best
      ) {
        best = prev[s - 2];
        bestPrev = s - 2;
      }
      if (best !== NEG_INF) {
        curr[s] = best + logprobs[frameOffset + states[s]];
        back[backOffset + s] = bestPrev;
      }
    }

    const tmp = prev;
    prev = curr;
    curr = tmp;
  }

  let finalState = -1;
  let bestLogProb = NEG_INF;
  for (let s = 1; s < stateCount; s++) {
    const tokensAligned = Math.floor((s + 1) / 2);
    if (tokensAligned <= 0) continue;
    const normalized = prev[s] / tokensAligned;
    if (Number.isFinite(normalized) && normalized > bestLogProb) {
      bestLogProb = normalized;
      finalState = s;
    }
  }

  if (finalState < 0 || !Number.isFinite(bestLogProb)) {
    return impossiblePrefixAlignment();
  }

  const path = new Int32Array(timeSteps);
  let state = finalState;
  for (let t = timeSteps - 1; t >= 0; t--) {
    path[t] = state;
    const prevState = back[t * stateCount + state];
    if (t > 0 && prevState < 0) {
      return impossiblePrefixAlignment();
    }
    state = prevState;
  }

  const tokensAligned = Math.floor((finalState + 1) / 2);
  const firstTokenFrame = new Int32Array(tokensAligned);
  firstTokenFrame.fill(-1);
  let startFrame = -1;
  let endFrame = -1;
  let pathLogProb = 0;
  for (let t = 0; t < timeSteps; t++) {
    const s = path[t];
    pathLogProb += logprobs[t * vocabSize + states[s]];
    if (s % 2 === 1) {
      const tokenIndex = (s - 1) >> 1;
      if (tokenIndex < tokensAligned && firstTokenFrame[tokenIndex] < 0) {
        firstTokenFrame[tokenIndex] = t;
      }
      if (startFrame < 0) startFrame = t;
      endFrame = t;
    }
  }

  if (startFrame < 0 || endFrame < 0) {
    return impossiblePrefixAlignment(tokensAligned);
  }

  return {
    feasible: true,
    score: -pathLogProb / tokensAligned,
    startFrame,
    endFrame,
    tokenFrames: Array.from(firstTokenFrame),
    tokensAligned,
  };
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
