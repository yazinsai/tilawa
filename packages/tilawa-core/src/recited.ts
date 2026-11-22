import type { AyahRef, EngineEvent as ResearchEvent } from "./engine/index.js";
import type { EngineEvent } from "./session.js";

/**
 * The ayahs recited, in order, from a session's position events: the best observed word of every `locked` position,
 * with consecutive repeats of the same ayah collapsed. Replaces 0.x `final_sequence`.
 */
export function ayahsRecited(events: Iterable<EngineEvent | ResearchEvent>): AyahRef[] {
  const out: AyahRef[] = [];
  for (const e of events) {
    if (e.type !== "position" || e.state !== "locked" || !e.observed[0]) continue;
    const { surah, ayah } = e.observed[0].word;
    const last = out.at(-1);
    if (!last || last.surah !== surah || last.ayah !== ayah) out.push({ surah, ayah });
  }
  return out;
}
