import type { PositionEvent } from "./events.js";
import type { DecodedPhoneme } from "./evidence.js";
import type { LocalPosition } from "./local.js";
import type { ReferenceIndex } from "./reference.js";

export interface CommittedWordSpan { word: number; frameFrom: number; frameTo: number }
// Private metadata: serialized position events and the tracker's decisions remain identical.
const spans = new WeakMap<PositionEvent, CommittedWordSpan>();
export const committedWordSpan = (event: PositionEvent): CommittedWordSpan | undefined => spans.get(event);

/** A middle word's span closes only after three neighboring words actually commit in order in one local run. */
export function createWordRun(index: ReferenceIndex) {
  let previous: { word: number; frame: number } | null = null;
  let current: { word: number; frame: number } | null = null;
  const reset = () => { previous = null; current = null; };
  const wordAt = (p: LocalPosition) => index.wordAt(Math.max(0, p.offset - 1));
  return {
    reset,
    observe(before: LocalPosition, after: LocalPosition) {
      if (before.origin !== after.origin || after.offset < before.offset || wordAt(after) > wordAt(before) + 1) reset();
    },
    commit(event: PositionEvent, before: LocalPosition, after: LocalPosition, phone: DecodedPhoneme) {
      const wi = wordAt(after);
      if (event.state !== "locked" || event.hold || event.relocated || before.origin !== after.origin || wi !== wordAt(before) + 1) {
        reset();
        return;
      }
      if (current && (wi !== current.word + 1 || phone.start <= current.frame)) reset();
      if (previous && current && current.word === previous.word + 1) {
        spans.set(event, { word: current.word, frameFrom: current.frame, frameTo: phone.start });
      }
      previous = current;
      current = { word: wi, frame: phone.start };
    },
  };
}
