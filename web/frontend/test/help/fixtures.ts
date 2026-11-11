import type { Incident } from "@tilawa/core/report";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../../help/src/types";

export const word = (ayah: number, w: number, surah = 1): WordRef => ({ surah, ayah, word: w });

export function pos(timeMs: number, at: WordRef | null, state: PositionEvent["state"] = "locked", extra: Partial<PositionEvent> = {}): PositionEvent {
  return {
    type: "position",
    v: 1,
    timeMs,
    referenceVersion: "ref-v2",
    expected: null,
    observed: at ? [{ word: at, span: { start: at, end: at }, confidence: 0.9, cost: 0, per: 0 }] : [],
    state,
    ...extra,
  };
}

export function check(timeMs: number, id: string, state: CheckerEvent["state"], at: WordRef = word(1, 1)): CheckerEvent {
  return { type: "checker", v: 1, id, timeMs, state, errorClass: "wrong_word", location: at, confidence: 0.8 };
}

export function bookmark(id: string, sessionMs: number): Incident {
  return { id, kind: "stumble", startMs: sessionMs, endMs: sessionMs, eventIds: [], markedAt: { sessionMs, wallMs: 1_700_000_000_000 + sessionMs }, claim: { answer: null } };
}

/** Positions every 500 ms across `ms`, walking word by word through ayah 1..n. */
export function steadyReading(ms: number): EngineEvent[] {
  const out: EngineEvent[] = [];
  for (let t = 500, i = 0; t <= ms; t += 500, i++) out.push(pos(t, word(1 + Math.floor(i / 8), 1 + (i % 8))));
  return out;
}

/** Park-Miller, warmed up so small seeds don't all start near 0. */
export const seeded = (seed = 1) => {
  const next = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (let i = 0; i < 5; i++) next();
  return next;
};
