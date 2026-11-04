import type { LocalPosition } from "./local.js";
import type { AyahRef, ReferenceIndex } from "./reference.js";

/** Coverage from next-word steps the local path actually took, never from an alternative alignment or a jump. */
export function createSteppedAyahs(index: ReferenceIndex) {
  let previousWord: number | null = null;
  const covered = new Set<string>();
  let pending: AyahRef[] = [];

  const reset = () => { previousWord = null; };

  return {
    reset,
    step(before: LocalPosition, after: LocalPosition) {
      const from = index.wordAt(Math.max(0, before.offset - 1));
      const wi = index.wordAt(Math.max(0, after.offset - 1));
      if (wi === from && after.origin === before.origin) return;
      const word = index.words[wi];
      // Only the ordinary, zero-edit step onto the next word's first phoneme witnesses that word.
      if (wi !== from + 1 || before.offset !== word.start || after.offset !== word.start + 1
        || after.origin !== before.origin || after.edits !== before.edits || after.stepCost !== 0) {
        reset();
        return;
      }
      if (previousWord !== null) {
        const prev = index.words[previousWord];
        const id = `${word.surah}:${word.ayah}`;
        if (wi === previousWord + 1 && prev.surah === word.surah && prev.ayah === word.ayah && !covered.has(id)) {
          covered.add(id);
          pending.push({ surah: word.surah, ayah: word.ayah });
        }
      }
      previousWord = wi;
    },
    /** Each ayah is exported once, on the next existing position event. */
    take(): AyahRef[] {
      const out = pending;
      pending = [];
      return out;
    },
  };
}
