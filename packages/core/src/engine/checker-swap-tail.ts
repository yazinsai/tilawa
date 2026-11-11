import type { MemorizationErrorClass } from "./events.js";

interface Span { obsFrom: number; obsTo: number }
interface Accusation extends Span {
  cls: MemorizationErrorClass;
  word: number;
  last: number;
  lexicon?: boolean;
}

/**
 * A split substitution can leave an added-word tail across an ordinary inter-phone blank. Adjacency is
 * structural: no labelled pause or counting word visit between the accusations, and together they spell
 * one whole non-neighbour word. Spell-rule additions keep their independent accusation.
 */
export function swapTailUnion(
  wrong: Accusation | undefined,
  added: Accusation,
  context: {
    nextWord(wi: number): number;
    visits: readonly Span[];
    labelledPause(prev: number): boolean;
    heard(span: Span): string;
    spellsWord(wi: number, heard: string): string | null;
  },
): { obsFrom: number; obsTo: number; observed: string } | null {
  if (!wrong || wrong.cls !== "wrong_word" || added.cls !== "added_word" || added.lexicon ||
    context.nextWord(wrong.last) !== added.word || wrong.obsFrom < 0 || wrong.obsTo <= wrong.obsFrom ||
    added.obsTo <= added.obsFrom || wrong.obsTo > added.obsFrom) return null;
  if (context.visits.some(v => v.obsFrom < added.obsFrom && v.obsTo > wrong.obsTo)) return null;
  for (let k = wrong.obsTo - 1; k < added.obsFrom; k++) if (context.labelledPause(k)) return null;
  const span = { obsFrom: wrong.obsFrom, obsTo: added.obsTo };
  const observed = context.heard(span);
  return context.spellsWord(wrong.word, observed) === null ? null : { ...span, observed };
}
