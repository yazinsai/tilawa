/** Posteriorgram subsequence DTW. All query frames participate; reference edges are free. */
export function subsequenceDtw(query: readonly ArrayLike<number>[], reference: readonly ArrayLike<number>[]): number {
  if (!query.length || !reference.length) return Infinity;
  const width = query[0].length;
  if (!width || query.some(r => r.length !== width) || reference.some(r => r.length !== width)) return Infinity;
  let prev = new Float64Array(reference.length + 1);
  for (const q of query) {
    const next = new Float64Array(reference.length + 1).fill(Infinity);
    for (let j = 1; j <= reference.length; j++) {
      const r = reference[j - 1];
      let inner = 0;
      for (let k = 0; k < width; k++) inner += q[k] * r[k];
      if (!(inner > 0) || inner > 1 + 1e-6) continue;
      next[j] = -Math.log(Math.min(1, inner)) + Math.min(prev[j - 1], prev[j], next[j - 1]);
    }
    prev = next;
  }
  return Math.min(...prev.subarray(1)) / query.length;
}

/** Missing competitors and ties abstain; the decision has no fitted distance threshold. */
export function candidateRanksFirst(candidate: readonly number[], competitors: readonly (readonly number[])[]): boolean {
  if (!candidate.length || !competitors.length || competitors.some(r => !r.length)) return false;
  const own = Math.min(...candidate);
  return Number.isFinite(own) && competitors.every(r => {
    const nearest = Math.min(...r);
    return Number.isFinite(nearest) && own < nearest;
  });
}

/** Prefix cuts from a closed CTC Viterbi reading of the known reference word, without a confidence cut. */
export function referencePrefixEnds(rows: readonly ArrayLike<number>[], labels: readonly number[]): number[] {
  if (!rows.length || labels.length < 2) return [];
  const states = 2 * labels.length + 1;
  const label = (s: number) => s % 2 ? labels[(s - 1) >> 1] : 0;
  const back = new Uint8Array(rows.length * states);
  let prev = new Float64Array(states).fill(-Infinity);
  prev[0] = Math.log(rows[0][0]);
  prev[1] = Math.log(rows[0][labels[0]]);
  for (let t = 1; t < rows.length; t++) {
    const next = new Float64Array(states);
    for (let s = 0; s < states; s++) {
      let best = prev[s], jump = 0;
      if (s && prev[s - 1] > best) { best = prev[s - 1]; jump = 1; }
      if (s > 1 && s % 2 && label(s) !== label(s - 2) && prev[s - 2] > best) { best = prev[s - 2]; jump = 2; }
      next[s] = best + Math.log(rows[t][label(s)]);
      back[t * states + s] = jump;
    }
    prev = next;
  }
  let s = prev[states - 1] > prev[states - 2] ? states - 1 : states - 2;
  if (!Number.isFinite(prev[s])) return [];
  const path = new Int32Array(rows.length);
  for (let t = rows.length - 1; t >= 0; t--) { path[t] = s; s -= back[t * states + s]; }
  return Array.from({ length: labels.length - 1 }, (_, k) => path.findIndex(s => s >= 2 * (k + 1) + 1));
}
