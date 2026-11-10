import { collapseHeld, subCost, UNIT } from "./fuzzy.js";

export interface UtteranceSplit { from: number; to: number }

/** Token offsets within one existing pause-bounded utterance. Only its edges
 * may be assigned to neighbours; the nonempty middle is still a whole-word
 * candidate and must pass the ordinary spelling and acoustic checks. */
export function openEdgeUtteranceSplits(
  observed: string, before: readonly string[], after: readonly string[],
): UtteranceSplit[] {
  const norm = (s: string) => collapseHeld([...s]).join("");
  const same = (a: string, b: string) => {
    a = norm(a); b = norm(b);
    return a.length === b.length && [...a].every((p, i) => subCost(p, b[i]) < UNIT);
  };
  // Never split a held sound into two copies of the same acoustic realization.
  const edge = (at: number) => {
    if (at === 0 || at === observed.length || observed[at - 1] !== observed[at]) return true;
    let from = at - 1, to = at + 1;
    while (from > 0 && observed[from - 1] === observed[at]) from--;
    while (to < observed.length && observed[to] === observed[at]) to++;
    return collapseHeld([...observed.slice(from, to)]).length === to - from;
  };
  const left = [0], right = [observed.length];
  for (let at = 1; at < observed.length; at++) {
    if (!edge(at)) continue;
    if (before.some(f => [...f].some((_, i) => i > 0 && same(f.slice(i), observed.slice(0, at))))) left.push(at);
    if (after.some(f => [...f].some((_, i) => i > 0 && same(f.slice(0, i), observed.slice(at))))) right.push(at);
  }
  const splits: UtteranceSplit[] = [];
  for (const from of left) for (const to of right) {
    if (from >= to || (from === 0 && to === observed.length)) continue;
    splits.push({ from, to });
  }
  return splits.sort((a, b) => (b.to - b.from) - (a.to - a.from) || a.from - b.from);
}
