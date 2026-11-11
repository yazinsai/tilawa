import type { AcousticTokenIdentity, CheckerState, DiscontinuityIdentity, MemorizationErrorClass, TransitionHistory } from "./events.js";

export const acousticIdentity = (token: AcousticTokenIdentity): AcousticTokenIdentity =>
  Object.freeze({ p: token.p, start: token.start, end: token.end });

export function discontinuityIdentity(
  generation: number, id: number, source: AcousticTokenIdentity | null, resume: AcousticTokenIdentity,
  from: number, to: number, parent: DiscontinuityIdentity | null,
): DiscontinuityIdentity {
  return Object.freeze({ generation, id, source: source ? acousticIdentity(source) : null, resume: acousticIdentity(resume), from, to, parent });
}

const sameToken = (a: AcousticTokenIdentity | null, b: AcousticTokenIdentity | null) =>
  a !== null && b !== null && a.p === b.p && a.start === b.start && a.end === b.end;

/** Compare values as well as IDs, so transporting an event as JSON preserves the contract. */
export function sameDiscontinuity(a: DiscontinuityIdentity | null, b: DiscontinuityIdentity | null): boolean {
  if (!a || !b) return a === b;
  while (a && b) {
    if (a.generation !== b.generation || a.id !== b.id || a.from !== b.from || a.to !== b.to ||
      !sameToken(a.source, b.source) || !sameToken(a.resume, b.resume)) return false;
    a = a.parent;
    b = b.parent;
  }
  return a === null && b === null;
}

/** All accepted jumps between two anchors; null means their ancestry cannot be reconciled. */
export function transitionsBetween(left: TransitionHistory | undefined, right: TransitionHistory | undefined): DiscontinuityIdentity[] | null {
  if (!left || !right || left.generation !== right.generation) return null;
  const out: DiscontinuityIdentity[] = [];
  let at = right.head;
  while (at && !(left.head && at.id === left.head.id && at.generation === left.head.generation)) {
    out.push(at);
    at = at.parent;
  }
  return sameDiscontinuity(at, left.head) ? out.reverse() : null;
}

/** A word edit owns the gap around its original acoustic span, never a nearby token or time tolerance. */
export function transitionForEdit(history: TransitionHistory | undefined, source: AcousticTokenIdentity | undefined,
  resume: AcousticTokenIdentity | undefined): DiscontinuityIdentity | undefined {
  if (!history || !source || !resume) return;
  let found: DiscontinuityIdentity | undefined;
  for (let at = history.head; at; at = at.parent) {
    if (!sameToken(at.source, source) || !sameToken(at.resume, resume)) continue;
    if (found) return; // Two distinct accepted transitions are independent evidence.
    found = at;
  }
  return found;
}

/** Retains confirmed word descriptions so a singleton ledger description of that event can resolve. */
export function createTransitionExclusivity() {
  const confirmed = new Map<string, { word: number; last: number; identity: DiscontinuityIdentity }>();
  const key = (identity: DiscontinuityIdentity) => `${identity.generation}/${identity.id}`;
  return {
    record(state: CheckerState, cls: MemorizationErrorClass, word: number, last: number, identity: DiscontinuityIdentity | undefined) {
      if (state !== "confirmed" || cls === "skipped_passage" || !identity?.source) return;
      confirmed.set(key(identity), { word, last, identity });
    },
    alternative(word: number, last: number, support: readonly DiscontinuityIdentity[] | null | undefined): boolean {
      if (!support || support.length !== 1) return false;
      const edit = confirmed.get(key(support[0]));
      return !!edit && edit.word >= word && edit.last <= last && sameDiscontinuity(edit.identity, support[0]);
    },
  };
}
