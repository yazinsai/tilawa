/** Training-only admission of labelled learner errors; never used by runtime confirmation. */
import type { CheckerEvent, WordRef } from "@tilawa/core";

export interface LearnerLabel {
  id: string; speaker: string; split: string; device?: string; scored: boolean;
  excluded?: string; exclude?: string; kind: string; wav: string; ayahs: string[];
  events: { errorClass: string; state: string; location: WordRef; observed?: string; words?: number }[];
  non_errors: { kind: string; location: WordRef }[];
  unsupported?: unknown[];
}
export const locationKey = (w: WordRef) => `${w.surah}:${w.ayah}:${w.word}`;

export function eligibleLearner(l: LearnerLabel): boolean {
  return l.split === "train" && (l.device === "phone" || l.events.some(e => e.errorClass === "wrong_word")) && l.scored && !l.excluded && !l.exclude &&
    !(l.unsupported?.length) && l.events.every(e =>
      ["wrong_word", "omitted_word", "skipped_passage"].includes(e.errorClass) &&
      (e.errorClass !== "wrong_word" || !!e.observed)) &&
    l.non_errors.every(e => e.kind === "repeat");
}

/** Correctness is scoped to the labelled slot, including multiword skips and repeats. */
export function correctSlot(l: LearnerLabel, w: WordRef): boolean {
  return l.ayahs.includes(`${w.surah}:${w.ayah}`) &&
    ![...l.events, ...l.non_errors].some(e =>
      e.location.surah === w.surah && e.location.ayah === w.ayah &&
      w.word >= e.location.word && w.word < e.location.word + ("words" in e ? e.words ?? 1 : 1));
}

export function minedAccusations(l: LearnerLabel, events: CheckerEvent[]): CheckerEvent[] {
  if (!eligibleLearner(l) || l.device !== "phone") return [];
  const latest = new Map<string, CheckerEvent>();
  for (const e of events) latest.set(e.id, e);
  return [...latest.values()].filter(e => e.errorClass === "wrong_word" && e.state === "confirmed" && correctSlot(l, e.location));
}
