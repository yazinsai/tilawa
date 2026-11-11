import { recitationFormulaBoundaries } from "@tilawa/core";

export interface FormulaResidue { from: number; to: number }

/** Parse one existing pause-bounded utterance as formula* word formula*.
 * Nonempty edges consume disjoint token intervals and must cover all tokens.
 * A formula-only reading retains its exemption. No new pause or pronunciation
 * is introduced, and the caller supplies the unchanged lexical predicate. */
export function formulaResidues(raw: string, formulas: readonly string[],
  isWord: (observed: string) => boolean): FormulaResidue[] {
  const n = raw.length;
  const { before, after } = recitationFormulaBoundaries(raw, formulas);
  if (before[n]) return []; // Every token already has a permitted explanation.
  const out: FormulaResidue[] = [];
  // Longer residual words first, retaining every valid split for verification.
  for (let length = n - 1; length > 0; length--) for (let a = 0; a + length <= n; a++) {
    const b = a + length;
    if (before[a] && after[b] && isWord(raw.slice(a, b))) out.push({ from: a, to: b });
  }
  return out;
}
