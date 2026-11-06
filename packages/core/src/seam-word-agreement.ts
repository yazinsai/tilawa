import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import type { ReferenceIndex } from "./reference.js";

const normalize = (s: string) => collapseHeld([...s].filter(p => !"ڇؙۜ".includes(p))).join("");

/** Complete-word alignment. Soft edits retain their existing costs; equal-cost
 * paths prefer fewer hard edits. Neither end of the proposed word is free. */
export function seamWordFit(expected: string, observed: string, maxHard?: number): { cost: number; hard: number } {
  const a = normalize(expected), b = normalize(observed), width = b.length + 1;
  const states = maxHard === undefined ? 1 : maxHard + 1;
  const costs = new Float64Array((a.length + 1) * width * states).fill(Infinity);
  const hard = new Float64Array(costs.length).fill(Infinity);
  costs[0] = hard[0] = 0;
  for (let i = 0; i <= a.length; i++) for (let j = 0; j <= b.length; j++) for (let state = 0; state < states; state++) {
    const at = (i * width + j) * states + state;
    const step = (di: number, dj: number, charge: number) => {
      const h = hard[at] + Number(charge >= UNIT);
      if (maxHard !== undefined && h > maxHard) return;
      const to = ((i + di) * width + j + dj) * states + (maxHard === undefined ? 0 : h), c = costs[at] + charge;
      if (c < costs[to] || (c === costs[to] && h < hard[to])) { costs[to] = c; hard[to] = h; }
    };
    if (i < a.length && j < b.length) step(1, 1, subCost(a[i], b[j]));
    if (i < a.length) step(1, 0, deletionCost(a, i));
    if (j < b.length) step(0, 1, insertionCost(a, i, b[j]));
  }
  let best = costs.length - states;
  for (let s = best + 1; s < costs.length; s++) if (costs[s] < costs[best] ||
    (costs[s] === costs[best] && hard[s] < hard[best])) best = s;
  return { cost: costs[best], hard: hard[best] };
}

/** Local words, every pronunciation prefix, and contiguous back-runs ending
 * immediately before the seam. These are alternatives to an added word. */
export function seamWordCompetitors(index: ReferenceIndex, word: number): string[] {
  const forms = (wi: number) => [...new Set([
    index.phonemes.slice(index.words[wi].start, index.words[wi].end),
    ...(index.wordForms?.[wi] ?? []), index.isolatedWordForms?.[wi] ?? "",
  ].map(normalize).filter(Boolean))];
  const previous: number[] = [], following: number[] = [];
  for (let i = word - 1; i >= 0 && previous.length < 3; i--)
    if (index.words[i].end > index.words[i].start) previous.unshift(i);
  for (let i = word; i < index.words.length && following.length < 3; i++)
    if (index.words[i].end > index.words[i].start) following.push(i);
  const out = new Set<string>();
  const prefixes = (s: string) => { for (let n = 1; n <= s.length; n++) out.add(s.slice(0, n)); };
  for (const wi of [...previous, ...following]) for (const f of forms(wi)) prefixes(f);
  let runs = [""];
  for (let i = previous.length - 1; i >= 0; i--) {
    runs = [...new Set(forms(previous[i]).flatMap(f => runs.map(r => f + r)))];
    for (const run of runs) prefixes(run);
  }
  return [...out];
}

/** The independently proposed word must beat every local explanation on the
 * same SSL token interval. A tie abstains; no confidence or fitted cut enters. */
export function seamWordAgreement(candidate: string, observed: string, competitors: readonly string[]) {
  const fit = seamWordFit(candidate, observed, 1);
  const rival = Math.min(...competitors.map(c => seamWordFit(c, observed).cost));
  return { ...fit, rival, agrees: !!normalize(observed) && fit.hard <= 1 && fit.cost < rival };
}
