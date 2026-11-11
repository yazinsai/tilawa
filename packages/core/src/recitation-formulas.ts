import { deletionCost, insertionCost, subCost, UNIT } from "./fuzzy.js";
import type { PhraseAsset, ReferenceIndex } from "./reference.js";

/** The checker's existing maximum edit cost per formula phone. */
export const PHRASE_MAX_COST = 0.3;

/** The same permitted material used by the checker, including the basmala. */
export function recitationFormulas(index: ReferenceIndex, asset?: PhraseAsset): string[] {
  const fatiha = index.words.filter(w => w.surah === 1 && w.ayah === 1);
  return [...(asset?.phrases.map(p => p.p) ?? []),
    index.phonemes.slice(fatiha[0].start, fatiha.at(-1)!.end)];
}

/** A compositional segment must account for its complete observed interval.
 * The formula's spelling costs and tolerance are unchanged. */
export function completeRecitationFormula(observed: string, formula: string): boolean {
  if (!observed || !formula || Math.abs(formula.length - observed.length) > Math.ceil(formula.length * 0.6) + 4) return false;
  // Same direction and edit costs as phraseSpan: observed material is the
  // searched reference. Close both boundaries rather than letting a substring
  // alignment silently leave leading or trailing observed phones unconsumed.
  let row = new Float64Array(formula.length + 1);
  for (let j = 1; j <= formula.length; j++) row[j] = row[j - 1] + insertionCost(observed, 0, formula[j - 1]);
  for (let i = 1; i <= observed.length; i++) {
    const next = new Float64Array(formula.length + 1);
    next[0] = row[0] + deletionCost(observed, i - 1);
    for (let j = 1; j <= formula.length; j++) next[j] = Math.min(
      row[j - 1] + subCost(observed[i - 1], formula[j - 1]),
      row[j] + deletionCost(observed, i - 1),
      next[j - 1] + insertionCost(observed, i, formula[j - 1]),
    );
    row = next;
  }
  return row[formula.length] / UNIT / formula.length <= PHRASE_MAX_COST;
}

/** Reachable boundaries of complete, nonempty formula sequences. Shared with
 * certificate replay so a residual's complementary material is checked again. */
export function recitationFormulaBoundaries(raw: string, formulas: readonly string[]) {
  const n = raw.length, vocabulary = [...new Set(formulas.filter(Boolean))];
  const edges: number[][] = Array.from({ length: n + 1 }, () => []);
  for (let a = 0; a < n; a++) for (const formula of vocabulary) {
    const slack = Math.ceil(formula.length * 0.6) + 4;
    for (let b = a + Math.max(1, formula.length - slack); b <= Math.min(n, a + formula.length + slack); b++) {
      if (completeRecitationFormula(raw.slice(a, b), formula)) edges[a].push(b);
    }
  }
  const before = Array<boolean>(n + 1).fill(false), after = Array<boolean>(n + 1).fill(false);
  before[0] = after[n] = true;
  for (let a = 0; a < n; a++) if (before[a]) for (const b of edges[a]) before[b] = true;
  for (let a = n - 1; a >= 0; a--) after[a] = edges[a].some(b => after[b]);
  return { before, after };
}
