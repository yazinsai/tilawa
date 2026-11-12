import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../engine/index.js";
import type { Session } from "./session.js";

/**
 * Event latency of a session fed audio in real time, from measured compute: push `i` arrives at `(i + 1) * pushMs`
 * of audio, starts once it has arrived and the previous push is done, and delivers its events when it finishes.
 *
 * - Word highlight: a `locked` position event moving to a new word, from the evidence time it carries (the end of the
 *   phoneme that moved it there) to delivery.
 * - Suspected: an error's first `suspected` event, from the onset of the first word after the error (the first position
 *   event past it, about when the wrong or skipped word ended) to delivery; 0 when the error was suspected earlier.
 */
export interface LatencyRun {
  audioMs: number;
  computeMs: number;
  highlight: number[];
  suspected: number[];
  events: EngineEvent[];
}

export async function measureLatency(
  session: Session,
  samples: Float32Array,
  sampleRate: number,
  pushMs = 320,
  /** Wait for each push's arrival time on the wall clock, as a microphone would (battery and thermal runs). */
  realtime = false,
): Promise<LatencyRun> {
  const push = Math.round((pushMs * sampleRate) / 1000);
  const delivered: { e: EngineEvent; at: number }[] = [];
  let done = 0;
  let computeMs = 0;
  const start = performance.now();
  const run = async (arrive: number, f: () => Promise<EngineEvent[]>) => {
    const wait = start + arrive - performance.now();
    if (realtime && wait > 0) await new Promise((r) => setTimeout(r, wait));
    const t0 = performance.now();
    const events = await f();
    const ms = performance.now() - t0;
    computeMs += ms;
    done = Math.max(arrive, done) + ms;
    for (const e of events) delivered.push({ e, at: done });
  };
  for (let a = 0; a < samples.length; a += push) {
    const chunk = samples.subarray(a, a + push);
    await run(((a + chunk.length) / sampleRate) * 1000, () => session.pushAudio(chunk, sampleRate));
  }
  const audioMs = (samples.length / sampleRate) * 1000;
  await run(audioMs, () => session.end());

  const order = (r: WordRef) => r.surah * 1e6 + r.ayah * 1e3 + r.word;
  const positions = delivered.filter((d): d is { e: PositionEvent; at: number } => d.e.type === "position" && d.e.state === "locked");
  const highlight: number[] = [];
  let word = "";
  for (const { e, at } of positions) {
    const w = e.observed[0]?.word;
    if (!w || `${w.surah}:${w.ayah}:${w.word}` === word) continue;
    word = `${w.surah}:${w.ayah}:${w.word}`;
    if (e.timeMs < audioMs) highlight.push(at - e.timeMs);
  }
  const suspected: number[] = [];
  const seen = new Set<string>();
  for (const { e, at } of delivered) {
    if (e.type !== "checker" || (e as CheckerEvent).state !== "suspected" || seen.has((e as CheckerEvent).id)) continue;
    const c = e as CheckerEvent;
    seen.add(c.id);
    const after = positions.find((p) => p.e.observed[0] && order(p.e.observed[0].word) > order(c.location));
    if (after) suspected.push(Math.max(0, at - after.e.timeMs));
  }
  return { audioMs, computeMs, highlight, suspected, events: delivered.map((d) => d.e) };
}

export function percentile(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)];
}
