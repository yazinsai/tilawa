export type AlignmentOp =
  | { op: "match"; ref: string; refAt: number; obsAt: number }
  | { op: "sub"; ref: string; obs: string; refAt: number; obsAt: number }
  | { op: "del"; ref: string; refAt: number }
  | { op: "ins"; obs: string; obsAt: number };

/** Unit-cost Levenshtein alignment of observed phonemes against a reference span. */
export function align(ref: string, obs: string): AlignmentOp[] {
  const n = ref.length;
  const m = obs.length;
  const d = new Uint32Array((n + 1) * (m + 1));
  const at = (i: number, j: number) => i * (m + 1) + j;
  for (let i = 0; i <= n; i++) d[at(i, 0)] = i;
  for (let j = 0; j <= m; j++) d[at(0, j)] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const diag = d[at(i - 1, j - 1)] + (ref[i - 1] === obs[j - 1] ? 0 : 1);
      d[at(i, j)] = Math.min(diag, d[at(i - 1, j)] + 1, d[at(i, j - 1)] + 1);
    }
  }
  const ops: AlignmentOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[at(i, j)] === d[at(i - 1, j - 1)] + (ref[i - 1] === obs[j - 1] ? 0 : 1)) {
      ops.push(
        ref[i - 1] === obs[j - 1]
          ? { op: "match", ref: ref[i - 1], refAt: i - 1, obsAt: j - 1 }
          : { op: "sub", ref: ref[i - 1], obs: obs[j - 1], refAt: i - 1, obsAt: j - 1 },
      );
      i--;
      j--;
    } else if (i > 0 && d[at(i, j)] === d[at(i - 1, j)] + 1) {
      ops.push({ op: "del", ref: ref[i - 1], refAt: i - 1 });
      i--;
    } else {
      ops.push({ op: "ins", obs: obs[j - 1], obsAt: j - 1 });
      j--;
    }
  }
  return ops.reverse();
}
