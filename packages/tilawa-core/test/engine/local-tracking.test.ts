import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { EngineEvent, PhonemeEvidence, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
/** Phoneme offset of `word`'s start within a fixture beginning at `from`. */
const at = (from: WordRef, word: WordRef) => find(word).start - find(from).start;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;

const positions = (events: EngineEvent[]) => events.filter((e): e is PositionEvent => e.type === "position");

/** Stream a synthetic recitation `chunkSize` phonemes at a time; returns every event. */
const stream = (from: WordRef, to: WordRef, edits: Edit[] = [], chunkSize = 1) => {
  const f = syntheticFixture(index, { id: "l", from, to, edits, chunkSize });
  const t = createTracker(index);
  return positions([...f.evidence.flatMap((e) => t.push(e)), ...t.end()]);
};

/**
 * `next` follows `prev` in `words`, allowing one transient word between them: a word whose opening phonemes the next
 * word shares can show briefly before the audio tells them apart.
 */
const followedBy = (words: string[], prev: string, next: string[]) => {
  const i = words.indexOf(prev);
  return i !== -1 && [1, 2].some((d) => JSON.stringify(words.slice(i + d, i + d + next.length)) === JSON.stringify(next));
};

/** Words reported while locked, in order, with consecutive duplicates removed. */
const lockedWords = (events: PositionEvent[]) => {
  const out: string[] = [];
  for (const e of events) if (e.state === "locked" && key(e.observed[0].word) !== out.at(-1)) out.push(key(e.observed[0].word));
  return out;
};

describe("forward progress", () => {
  it("advances word by word through a multi-ayah recitation", () => {
    const events = stream(w(2, 255, 1), w(2, 257, lastWord(2, 257)));
    const words = lockedWords(events);
    const all = index.words.filter((x) => x.surah === 2 && x.ayah >= 255 && x.ayah <= 257 && x.end > x.start).map(key);
    // Every word after the lock appears, in order.
    expect(all.slice(all.indexOf(words[0]))).toEqual(words);
    expect(words.length).toBeGreaterThan(all.length - 15);
    expect(events.at(-1)!.observed[0].word).toEqual(w(2, 257, lastWord(2, 257)));
    const lock = events.findIndex((e) => e.state === "locked");
    expect(events.slice(lock).every((e) => e.state === "locked")).toBe(true);
  });

  it("follows into the next surah", () => {
    const events = stream(w(100, 9, 1), w(101, 3, lastWord(101, 3)));
    expect(events.at(-1)!.state).toBe("locked");
    expect(events.at(-1)!.observed[0].word).toEqual(w(101, 3, lastWord(101, 3)));
  });
});

describe("errors", () => {
  it("does not jump on a mispronounced word", () => {
    const from = w(2, 255, 1);
    const word = at(from, w(2, 255, 15));
    const events = stream(from, w(2, 255, 25), [
      { kind: "sub", at: word, to: "ق" },
      { kind: "sub", at: word + 2, to: "ز" },
    ]);
    const order = lockedWords(events).map((x) => index.words.findIndex((r) => key(r) === x));
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(key(events.at(-1)!.observed[0].word)).toBe("2:255:25");
    expect(key(events.at(-1)!.observed[0].span.start)).toBe("2:255:1");
  });
});

describe("repeat", () => {
  const from = w(2, 255, 1);
  const to = w(2, 255, 20);

  it("goes back to the repeated phrase, then on, without leaving the ayah", () => {
    // Recite words 8..12, then repeat 10..12.
    const events = stream(from, to, [{ kind: "repeat", from: at(from, w(2, 255, 10)), to: at(from, w(2, 255, 13)) }]);
    const words = lockedWords(events);
    expect(followedBy(words, "2:255:12", ["2:255:10", "2:255:11", "2:255:12"])).toBe(true);
    expect(words.at(-1)).toBe("2:255:20");
    for (const e of events.filter((e) => e.state === "locked")) expect(e.observed[0].word.ayah).toBe(255);
  });

  it("keeps the position on a single repeated word", () => {
    const events = stream(from, to, [{ kind: "repeat", from: at(from, w(2, 255, 11)), to: at(from, w(2, 255, 12)) }]);
    const words = lockedWords(events);
    expect(followedBy(words, "2:255:11", ["2:255:11", "2:255:12"])).toBe(true);
    // The second 11 is a new run starting at the repeated word, not a jump elsewhere.
    const again = events.filter((e) => e.state === "locked" && key(e.observed[0].word) === "2:255:11").at(-1)!;
    expect(key(again.observed[0].span.start)).toBe("2:255:11");
    expect(words.at(-1)).toBe("2:255:20");
    for (const e of events.filter((e) => e.state === "locked")) expect(e.observed[0].word.ayah).toBe(255);
  });
});

describe("backward restart", () => {
  it("restarts from an earlier word in the ayah", () => {
    const from = w(2, 255, 1);
    const events = stream(from, w(2, 255, 30), [{ kind: "repeat", from: at(from, w(2, 255, 14)), to: at(from, w(2, 255, 22)) }]);
    const words = lockedWords(events);
    expect(followedBy(words, "2:255:21", ["2:255:14", "2:255:15", "2:255:16"])).toBe(true);
    expect(words.at(-1)).toBe("2:255:30");
    expect(key(events.at(-1)!.observed[0].span.start)).toBe("2:255:14");
  });

  it("restarts from the previous ayah", () => {
    const from = w(2, 255, 1);
    const events = stream(from, w(2, 256, 8), [{ kind: "repeat", from: at(from, w(2, 255, 40)), to: at(from, w(2, 256, 4)) }]);
    const words = lockedWords(events);
    expect(followedBy(words, "2:256:3", ["2:255:40", "2:255:41"])).toBe(true);
    expect(words.at(-1)).toBe("2:256:8");
  });
});

describe("skip", () => {
  it("follows a skip of a few words and shows it", () => {
    const from = w(2, 255, 1);
    const skipped = at(from, w(2, 255, 14)) - at(from, w(2, 255, 11));
    const events = stream(from, w(2, 255, 25), [{ kind: "del", at: at(from, w(2, 255, 11)), count: skipped }]);
    const words = lockedWords(events);
    expect(words).not.toContain("2:255:12");
    expect(words).not.toContain("2:255:13");
    expect(followedBy(words, "2:255:10", ["2:255:14", "2:255:15"])).toBe(true);
    expect(words.at(-1)).toBe("2:255:25");
    // The run restarts at the landing word, so the skip is visible in the span too.
    const landed = events.find((e) => e.state === "locked" && key(e.observed[0].word) === "2:255:14")!;
    expect(key(landed.observed[0].span.start)).toBe("2:255:14");
  });

  it("follows a skipped ayah", () => {
    const from = w(2, 255, 30);
    const start = at(from, w(2, 256, 1));
    const count = at(from, w(2, 257, 1)) - start;
    const events = stream(from, w(2, 257, 10), [{ kind: "del", at: start, count }]);
    const words = lockedWords(events);
    expect(words.filter((x) => x.startsWith("2:256:")).length).toBeLessThanOrEqual(1);
    expect(followedBy(words, `2:255:${lastWord(2, 255)}`, ["2:257:1", "2:257:2"])).toBe(true);
    expect(key(events.at(-1)!.observed[0].span.start)).toBe("2:257:1");
    expect(words.at(-1)).toBe("2:257:10");
  });
});

describe("pauses", () => {
  it("does not move or change state over a long silence", () => {
    const from = w(2, 255, 1);
    const gapAt = at(from, w(2, 255, 15));
    const f = syntheticFixture(index, { id: "g", from, to: w(2, 255, 25), chunkSize: 1, edits: [{ kind: "gap", at: gapAt, frames: 50 * 60 * 30 }] });
    const t = createTracker(index);
    const events = positions(f.evidence.slice(0, gapAt).flatMap((e) => t.push(e)));
    const before = events.at(-1)!;
    expect(before.state).toBe("locked");
    // Silence arrives as empty chunks.
    const silent: PhonemeEvidence = { inventory: index.inventoryId, frameMs: 20, phonemes: [] };
    for (let i = 0; i < 100; i++) expect(t.push(silent)).toEqual([]);
    const after = positions(f.evidence.slice(gapAt).flatMap((e) => t.push(e)));
    expect(after[0].state).toBe("locked");
    expect(after[0].observed[0].word).toEqual(w(2, 255, 15));
    expect(after.at(-1)!.observed[0].word).toEqual(w(2, 255, 25));
  });
});

describe("streaming equivalence", () => {
  it("reports the same word and state at shared boundaries for every chunk size", { timeout: 30_000 }, () => {
    const from = w(2, 255, 1);
    // Kept short: one whole-clip global search grows with the square of the clip length.
    const to = w(2, 255, 22);
    const edits: Edit[] = [
      { kind: "repeat", from: at(from, w(2, 255, 10)), to: at(from, w(2, 255, 13)) },
      { kind: "sub", at: at(from, w(2, 255, 18)) + 1, to: "ق" },
    ];
    const phonemes = syntheticFixture(index, { id: "e", from, to, edits }).evidence[0].phonemes;
    const total = phonemes.length;
    /** State after each boundary a chunking reaches: the last event emitted so far. */
    const run = (size: number) => {
      const f = syntheticFixture(index, { id: "e", from, to, edits, chunkSize: size });
      const t = createTracker(index);
      const after = new Map<number, PositionEvent>();
      let last: PositionEvent | undefined;
      let seen = 0;
      for (const e of f.evidence) {
        last = positions(t.push(e)).at(-1) ?? last;
        seen += e.phonemes.length;
        if (last) after.set(seen, last);
      }
      return after;
    };
    const summary = (e: PositionEvent) => ({ state: e.state, word: e.observed[0]?.word });
    const reference = run(1);
    for (const size of [2, 3, 5, 8, 13, 21, 64, total]) {
      const got = run(size);
      for (const [boundary, e] of got) {
        const want = reference.get(boundary)!;
        // Both locked, or both still acquiring on identical evidence: either way they must agree on the word.
        if (want.state === "locked" && e.state === "locked") expect(summary(e), `size ${size} at ${boundary}`).toEqual(summary(want));
        else if (e.state !== "locked") expect(summary(e), `size ${size} at ${boundary}`).toEqual(summary(want));
      }
      expect(summary(got.get(total)!)).toEqual(summary(reference.get(total)!));
    }
  });
});

describe("long session", () => {
  it("keeps time per push flat", { timeout: 60_000 }, () => {
    const f = syntheticFixture(index, { id: "long", from: w(2, 2, 1), to: w(2, 120, lastWord(2, 120)), chunkSize: 8 });
    const t = createTracker(index);
    const times: number[] = [];
    for (const e of f.evidence) {
      const t0 = performance.now();
      t.push(e);
      times.push(performance.now() - t0);
    }
    const final = positions(t.end())[0];
    expect(final.state).toBe("locked");
    expect(final.observed[0].word).toEqual(w(2, 120, lastWord(2, 120)));
    const n = times.length;
    expect(n).toBeGreaterThan(1000);
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    // Skip the acquisition pushes at the start.
    const early = mean(times.slice(Math.floor(n * 0.1), Math.floor(n * 0.3)));
    const late = mean(times.slice(Math.floor(n * 0.8)));
    expect(late).toBeLessThan(early * 3 + 0.5);
  });
});
