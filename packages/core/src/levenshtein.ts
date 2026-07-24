// Scratch rows reused across calls. These functions run tens of thousands of times
// per recognition cycle and on an interpreter (Hermes) the per-call typed-array
// allocation dominates. `distance` and `semiGlobalDistance` never nest and never
// yield, so a shared pair per function is safe; they get separate pairs so that
// stays true if a caller ever nests them.
//
// Every cell read in `0..m` is written before it is read (`prev` by the seed loop,
// `curr[0]` each row and `curr[1..m]` by the inner loop), so stale data left beyond
// `m` by an earlier, larger call is never observed. Results are identical.
let _distPrev = new Uint16Array(0);
let _distCurr = new Uint16Array(0);

/**
 * Levenshtein edit distance between two strings.
 * Uses a single-row DP approach for O(min(m,n)) space.
 */
export function distance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  // Ensure a is the shorter string for space efficiency
  if (a.length > b.length) [a, b] = [b, a];

  const m = a.length;
  const n = b.length;
  if (_distPrev.length < m + 1) {
    _distPrev = new Uint16Array(m + 1);
    _distCurr = new Uint16Array(m + 1);
  }
  let prev = _distPrev;
  let curr = _distCurr;

  for (let i = 0; i <= m; i++) prev[i] = i;

  for (let j = 1; j <= n; j++) {
    curr[0] = j;
    for (let i = 1; i <= m; i++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[i] = Math.min(
        prev[i] + 1, // deletion
        curr[i - 1] + 1, // insertion
        prev[i - 1] + cost, // substitution
      );
    }
    const tmp = prev;
    prev = curr;
    curr = tmp;
  }

  return prev[m];
}

/**
 * Normalized Levenshtein similarity ratio.
 * Returns 1.0 for identical strings, 0.0 for completely different.
 * Matches python-Levenshtein's `ratio()` behavior:
 *   ratio = (len(a) + len(b) - distance) / (len(a) + len(b))
 */
export function ratio(a: string, b: string): number {
  const lenSum = a.length + b.length;
  if (lenSum === 0) return 1.0;
  return (lenSum - distance(a, b)) / lenSum;
}

// Scratch rows for `semiGlobalDistance` — see the note on `_distPrev` above.
let _sgPrev = new Uint16Array(0);
let _sgCurr = new Uint16Array(0);

/**
 * Semi-global edit distance: finds the minimum edit distance to align
 * the entire query against any substring of ref.
 * Free gaps at start and end of ref (row 0 initialized to 0, take min of last row).
 * Use case: "how well does this transcript fragment match somewhere inside this verse?"
 */
export function semiGlobalDistance(query: string, ref: string): number {
  if (query.length === 0) return 0;
  if (ref.length === 0) return query.length;
  const m = query.length;
  const n = ref.length;
  if (_sgPrev.length < m + 1) {
    _sgPrev = new Uint16Array(m + 1);
    _sgCurr = new Uint16Array(m + 1);
  }
  let prev = _sgPrev;
  let curr = _sgCurr;
  for (let i = 0; i <= m; i++) prev[i] = i;
  let best = prev[m];
  for (let j = 1; j <= n; j++) {
    curr[0] = 0; // Free to start anywhere in ref
    for (let i = 1; i <= m; i++) {
      const cost = query[i - 1] === ref[j - 1] ? 0 : 1;
      curr[i] = Math.min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost);
    }
    best = Math.min(best, curr[m]); // Free to end anywhere in ref
    const tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return best;
}

/**
 * Fragment score: how well does the query match as a fragment of ref?
 * Returns 0.0-1.0. Score of 1.0 means query is an exact substring of ref.
 * Directional: measures "how much of the query does the ref explain?"
 */
export function fragmentScore(query: string, ref: string): number {
  if (query.length === 0) return 1.0;
  return Math.max(0, 1 - semiGlobalDistance(query, ref) / query.length);
}
