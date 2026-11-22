import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../src/index.js";
import type { PositionEvent, ReferenceAsset, WordRef } from "../src/index.js";
import { benchRecording, chunk, summarise, wordsToUnique } from "../src/bench/index.js";
import type { QudRecording } from "../src/bench/index.js";
import { perfGate, perfReplay } from "../src/bench/perf.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });

/** A recording of `from`..`to` with exact truth: each reference word timed by its synthetic phonemes. */
const recording = (from: WordRef, to: WordRef): QudRecording => {
  const f = withPosteriors(index, syntheticFixture(index, { id: "r", from, to }).evidence[0]);
  const frameMs = 20;
  const ws = index.words.slice(
    index.words.findIndex((x) => x.surah === from.surah && x.ayah === from.ayah && x.word === from.word),
    index.words.findIndex((x) => x.surah === to.surah && x.ayah === to.ayah && x.word === to.word) + 1,
  );
  const base = ws[0].start;
  const time = (offset: number) => (f.phonemes[Math.min(f.phonemes.length - 1, offset - base)].start * frameMs) / 1000;
  const p = f.posteriors!;
  return {
    id: "synthetic",
    meta: { style: "murattal", content: "quran_only", noisy: false, passages: "", duration_s: 0, repeat_events: 0 },
    inventory: f.inventory,
    frameMs,
    nFrames: p.nFrames,
    phonemes: f.phonemes,
    posteriors: { k: p.k, ids: [...p.ids], probs: [...p.probs] },
    truth: {
      words: ws.filter((x) => x.end > x.start).map((x) => ({ word: `${x.surah}:${x.ayah}:${x.word}`, start_s: time(x.start), end_s: time(x.end - 1) + 0.04 })),
      non_quran: [],
    },
  };
};

describe("chunking", () => {
  it("keeps every phoneme whole and inside its chunk's posterior frames", () => {
    const rec = recording(w(36, 1, 1), w(36, 5, 3));
    const chunks = chunk(rec, 0.3, true);
    expect(chunks.flatMap((c) => c.phonemes)).toEqual(rec.phonemes);
    for (const c of chunks) {
      for (const ph of c.phonemes) {
        expect(ph.start).toBeGreaterThanOrEqual(c.posteriors!.startFrame);
        expect(ph.end).toBeLessThan(c.posteriors!.startFrame + c.posteriors!.nFrames);
      }
    }
    // The tracker accepts the stream with posteriors.
    const t = createTracker(index);
    expect(() => chunks.forEach((c) => t.push(c))).not.toThrow();
  });
});

describe("scoring", () => {
  it("scores a clean recitation as located, with no false locks or jumps", () => {
    const r = benchRecording(index, recording(w(36, 1, 1), w(36, 12, 4)), { posteriors: false });
    expect(r.starts).toHaveLength(1);
    expect(r.starts[0].latencyS).not.toBeNull();
    expect(r.starts[0].latencyS!).toBeLessThan(5);
    expect(r.starts[0].falseLock).toBe(false);
    expect(r.wrongLockedSamples).toBe(0);
    expect(r.falseJumps).toBe(0);
    const s = summarise([r]);
    expect(s.within5s).toBe(1);
    expect(s.unresolved).toBe(0);
  });

  it("measures how many words a shared opening needs", () => {
    expect(wordsToUnique(index, w(36, 1, 1))).toBeLessThanOrEqual(2);
    // Al-Baqara opens like five other surahs.
    expect(wordsToUnique(index, w(2, 1, 1))).toBeGreaterThan(1);
  });
});

describe("performance gate", () => {
  it("replays every tracker phase", () => {
    const { evidence } = perfReplay(index);
    const t = createTracker(index);
    const events = [...evidence.flatMap((e) => t.push(e)), ...t.end()].filter((e): e is PositionEvent => e.type === "position");
    const states = new Set(events.map((e) => e.state));
    expect(states.has("locked")).toBe(true);
    expect(states.has("lost")).toBe(true);
    const surahs = events.filter((e) => e.state === "locked").map((e) => e.observed[0].word.surah);
    expect(surahs[0]).toBe(2);
    expect(surahs).toContain(36);
    expect(surahs.at(-1)).toBe(67);
  });

  it("fails when replay falls below real time at the speed factor", { timeout: 30_000 }, () => {
    expect(perfGate(index, 1, 1).pass).toBe(true);
    expect(perfGate(index, 1e6, 1).pass).toBe(false);
  });
});
