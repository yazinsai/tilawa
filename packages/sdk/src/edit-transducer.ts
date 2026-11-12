import type { DecodedPhoneme } from "@tilawa/core";

export interface SourceEvent { id: number; start: number; end: number }
const HOLD = 0, DELETE = 1, COPY = 2, SUBSTITUTE = 3, INSERT = 4;

/** Collapse the shipped decoder before editing. A carried run is consumed once. */
export function editSource(ids: readonly number[], previous = 0): SourceEvent[] {
  const events: SourceEvent[] = [];
  ids.forEach((id, t) => {
    if (id && id !== previous) events.push({ id, start: t, end: t });
    else if (id && events.at(-1)?.id === id) events.at(-1)!.end = t;
    previous = id;
  });
  return events;
}

function logSumExp(values: number[]): number {
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum, x) => sum + Math.exp(x - max), 0));
}

/**
 * Viterbi over (acoustic frame, consumed phone events). Every source-consuming
 * action lies inside its event's span. Inserts need their own acoustic frame;
 * deletes consume an event without moving the clock. No text enters the graph.
 * Scores use the same locally normalized actions/payloads as training's forward
 * sum. Source + target script alternatives are not greedily pruned.
 */
export function decodeEdits(
  actions: Float32Array, phones: Float32Array, source: readonly SourceEvent[],
  frames: number, tokens: string, frameOffset = 0, validFrames = frames,
): DecodedPhoneme[] {
  const cursors = frames + 1, vocabulary = tokens.length;
  if (!Number.isInteger(frames) || frames < 1 || !Number.isInteger(validFrames) || validFrames < 1 || validFrames > frames ||
      actions.length !== frames * cursors * 5 || phones.length !== frames * cursors * vocabulary ||
      source.length > validFrames || !vocabulary ||
      source.some((s, i) => s.id < 1 || s.id > vocabulary || !Number.isInteger(s.id) ||
        s.start < 0 || s.end < s.start || s.end >= validFrames || !Number.isInteger(s.start) ||
        !Number.isInteger(s.end) || (i > 0 && s.start <= source[i - 1].end)) ||
      !actions.every(Number.isFinite) || !phones.every(Number.isFinite))
    throw new Error("Invalid acoustic edit output or source clock");
  interface Path { score: number; emitted: DecodedPhoneme[] }
  const paths: (Path | undefined)[][] = Array.from({ length: validFrames + 1 }, () => []);
  paths[0][0] = { score: 0, emitted: [] };
  for (let t = 0; t < validFrames; t++) for (let i = 0; i <= source.length; i++) {
    const path = paths[t][i];
    if (!path) continue;
    const event = source[i];
    const allowed = (!event || t < event.start ? [HOLD, INSERT] : t > event.end ? [] :
      t === event.end ? [DELETE, COPY, SUBSTITUTE] : [HOLD, DELETE, COPY, SUBSTITUTE, INSERT])
      .filter(a => vocabulary > 1 || a !== SUBSTITUTE);
    if (!allowed.length) continue;
    const a = (t * cursors + i) * 5, p = (t * cursors + i) * vocabulary;
    const norm = logSumExp(allowed.map(k => actions[a + k]));
    for (const action of allowed) {
      let id = 0, score = actions[a + action] - norm;
      if (action === COPY) id = event.id;
      if (action === INSERT || action === SUBSTITUTE) {
        const candidates = Array.from({ length: vocabulary }, (_, k) => k + 1)
          .filter(k => action === INSERT || k !== event.id);
        if (!candidates.length) continue;
        id = candidates.reduce((best, k) => phones[p + k - 1] > phones[p + best - 1] ? k : best);
        score += phones[p + id - 1] - logSumExp(candidates.map(k => phones[p + k - 1]));
      }
      const nextTime = t + (action !== DELETE ? 1 : 0);
      const nextCursor = i + ([DELETE, COPY, SUBSTITUTE].includes(action) ? 1 : 0);
      const total = path.score + score;
      if (!paths[nextTime][nextCursor] || total > paths[nextTime][nextCursor]!.score) {
        const emitted = id ? path.emitted.map(p => ({ ...p })) : path.emitted;
        if (id) {
          // Preserve copied acoustic duration, shortening it when a later edit
          // occupies one of those frames. Paths must not mutate shared tokens.
          const last = emitted.at(-1);
          if (last && last.end >= frameOffset + t) last.end = frameOffset + t - 1;
          emitted.push({
            p: tokens[id - 1], start: frameOffset + t,
            end: frameOffset + (action === COPY || action === SUBSTITUTE ? event.end : t),
            conf: Math.round(Math.exp(score) * 1e4) / 1e4,
          });
        }
        paths[nextTime][nextCursor] = { score: total, emitted };
      }
    }
  }
  const best = paths[validFrames][source.length];
  if (!best) throw new Error("No complete acoustic edit script");
  return best.emitted;
}
