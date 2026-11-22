import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { DecodedPhoneme, PhonemeEvidence, PositionEvent, ReferenceAsset, TrackerOptions, WordRef } from "../../src/engine/index.js";
import { jumpReport, syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const at = (from: WordRef, word: WordRef) => find(word).start - find(from).start;
const key = (r: WordRef | null) => (r ? `${r.surah}:${r.ayah}:${r.word}` : "-");

interface Segment {
  from: WordRef;
  to: WordRef;
  edits?: Edit[];
}

/** Recite segments back to back, frames continuing; returns the phonemes and where each segment starts. */
const concat = (segments: Segment[]) => {
  const phonemes: DecodedPhoneme[] = [];
  const starts: number[] = [];
  let frame = 0;
  for (const s of segments) {
    starts.push(phonemes.length);
    const f = syntheticFixture(index, { id: "s", from: s.from, to: s.to, edits: s.edits });
    for (const ph of f.evidence[0]?.phonemes ?? []) phonemes.push({ ...ph, start: ph.start + frame, end: ph.end + frame });
    frame = (phonemes.at(-1)?.end ?? frame) + 2;
  }
  return { phonemes, starts };
};

const chunked = (phonemes: DecodedPhoneme[], size: number): PhonemeEvidence[] => {
  const out: PhonemeEvidence[] = [];
  for (let i = 0; i < phonemes.length; i += size) out.push({ inventory: index.inventoryId, frameMs: 20, phonemes: phonemes.slice(i, i + size) });
  return out;
};

const run = (segments: Segment[], options: TrackerOptions = {}, size = 1) => {
  const { phonemes, starts } = concat(segments);
  const t = createTracker(index, options);
  const events = [...chunked(phonemes, size).flatMap((e) => t.push(e)), ...t.end()].filter((e): e is PositionEvent => e.type === "position");
  return { events, starts, phonemes };
};

const locked = (events: PositionEvent[]) => events.filter((e) => e.state === "locked");

describe("a wrong word", () => {
  it("that matches another verse does not commit a jump", () => {
    const from = w(2, 255, 1);
    const stray = index.phonemes.slice(find(w(36, 2, 2)).start, find(w(36, 2, 2)).end);
    const word = find(w(2, 255, 15));
    const edits: Edit[] = [
      { kind: "del", at: at(from, w(2, 255, 15)), count: word.end - word.start },
      { kind: "ins", at: at(from, w(2, 255, 15)), text: stray },
    ];
    const { events } = run([{ from, to: w(2, 255, 30), edits }]);
    for (const e of locked(events)) expect(e.observed[0].word.surah).toBe(2);
    expect(events.at(-1)!.state).toBe("locked");
    expect(key(events.at(-1)!.observed[0].word)).toBe("2:255:30");
  });
});

describe("a distant jump", () => {
  const segments: Segment[] = [
    { from: w(2, 255, 1), to: w(2, 255, 20) },
    { from: w(36, 1, 1), to: w(36, 9, 1) },
  ];

  it("commits in free mode and moves both bookmarks", () => {
    const { events } = run(segments, { mode: "free" });
    const first = events.findIndex((e) => e.state === "locked" && e.observed[0].word.surah === 36);
    expect(first).toBeGreaterThan(-1);
    const after = events.slice(first);
    expect(after.every((e) => e.state === "locked" && e.observed[0].word.surah === 36)).toBe(true);
    expect(after.every((e) => e.expected?.surah === 36)).toBe(true);
    expect(events.at(-1)!.observed[0].word).toEqual(w(36, 9, 1));
    expect(events.at(-1)!.expected).toEqual(w(36, 9, 1));
  });

  it("commits in memorization mode and leaves the expected bookmark before the jump", () => {
    const { events } = run(segments, { mode: "memorization" });
    const first = events.findIndex((e) => e.state === "locked" && e.observed[0].word.surah === 36);
    expect(first).toBeGreaterThan(-1);
    const expected = events[first].expected!;
    expect(expected.surah).toBe(2);
    expect(expected.ayah).toBe(255);
    expect(expected.word).toBeGreaterThanOrEqual(18);
    for (const e of events.slice(first)) expect(e.expected).toEqual(expected);
    expect(events.at(-1)!.observed[0].word).toEqual(w(36, 9, 1));
  });

  it("waits for the configured persistence span", () => {
    const latency = (jumpPhonemes: number) => {
      const { events, starts } = run(segments, { jumpPhonemes });
      return jumpReport(events, { onsetMs: starts[1] * 3 * 20, target: (r) => r.surah === 36, path: (r) => r.surah === 2 || r.surah === 36 });
    };
    const short = latency(4);
    const long = latency(20);
    expect(short.committedMs).not.toBeNull();
    expect(long.committedMs).not.toBeNull();
    expect(long.committedMs! - short.committedMs!).toBeGreaterThanOrEqual(16 * 3 * 20);
  });

  it("commits the same way for any chunk size", () => {
    const final = (size: number) => run(segments, { mode: "free" }, size).events.at(-1)!;
    for (const size of [1, 7, 32]) expect(final(size).observed[0].word).toEqual(w(36, 9, 1));
  });
});

describe("a misfit on the same passage", () => {
  it("holds the lock through the search and relocks without emitting lost", () => {
    const from = w(2, 255, 1);
    const at15 = find(w(2, 255, 15)).start - find(from).start;
    // Long enough to exhaust the lost window. The passage continues, so the anchor stays locked.
    const { events } = run([{ from, to: w(2, 255, 30), edits: [{ kind: "noise", at: at15, length: 60, seed: 3 }] }]);
    expect(events.some((e) => e.state === "locked")).toBe(true);
    expect(events.some((e) => e.state === "lost")).toBe(false);
    for (const e of events) if (e.state === "locked" && e.observed[0]) {
      expect(e.observed[0].word.surah).toBe(2);
      expect(Math.abs(e.observed[0].word.ayah - 255)).toBeLessThanOrEqual(2);
    }
    expect(events.at(-1)!.state).toBe("locked");
    expect(key(events.at(-1)!.observed[0].word)).toBe("2:255:30");
  });
});

describe("after lost", () => {
  it("re-acquires elsewhere without a restart", () => {
    const { events } = run([
      { from: w(2, 255, 1), to: w(2, 255, 20) },
      { from: w(2, 255, 20), to: w(2, 255, 20), edits: [{ kind: "noise", at: 0, length: 60, seed: 3 }, { kind: "del", at: 0, count: 99 }] },
      { from: w(36, 1, 1), to: w(36, 9, 1) },
    ]);
    const lost = events.findIndex((e) => e.state === "lost");
    expect(lost).toBeGreaterThan(-1);
    const back = events.findIndex((e, i) => i > lost && e.state === "locked");
    expect(back).toBeGreaterThan(-1);
    expect(events[back].observed[0].word.surah).toBe(36);
    expect(events.at(-1)!.state).toBe("locked");
    expect(events.at(-1)!.observed[0].word).toEqual(w(36, 9, 1));
    // Lost is not a jump: the expected bookmark stays where it was.
    expect(events[lost].expected?.surah).toBe(2);
    expect(events[lost].expected?.ayah).toBe(255);
  });
});

describe("jump report", () => {
  it("reports commit latency and false jumps per fixture", () => {
    const cases: { id: string; segments: Segment[]; target: (r: WordRef) => boolean; path: (r: WordRef) => boolean; jump: boolean }[] = [
      {
        id: "2:255 → 36:1 jump",
        segments: [{ from: w(2, 255, 1), to: w(2, 255, 20) }, { from: w(36, 1, 1), to: w(36, 9, 1) }],
        target: (r) => r.surah === 36,
        path: (r) => (r.surah === 2 && r.ayah === 255) || r.surah === 36,
        jump: true,
      },
      {
        id: "18:1 → 67:1 jump",
        segments: [{ from: w(18, 1, 1), to: w(18, 3, 2) }, { from: w(67, 1, 1), to: w(67, 3, 4) }],
        target: (r) => r.surah === 67,
        path: (r) => r.surah === 18 || r.surah === 67,
        jump: true,
      },
      {
        id: "2:255 mispronounced",
        segments: [{ from: w(2, 255, 1), to: w(2, 255, 40), edits: [{ kind: "sub", at: 60, to: "ق" }, { kind: "sub", at: 62, to: "ز" }, { kind: "sub", at: 64, to: "ط" }] }],
        target: () => false,
        path: (r) => r.surah === 2 && r.ayah === 255,
        jump: false,
      },
    ];
    // Within the local search's window of the recited ayahs.
    const near = (c: (typeof cases)[number]) => (r: WordRef) =>
      [-2, -1, 1, 2].some((d) => c.path({ ...r, ayah: r.ayah + d }) && c.segments.some((s) => s.from.surah === r.surah));
    const rows = cases.map((c) => {
      const { events, starts } = run(c.segments);
      const r = jumpReport(events, { onsetMs: (starts[1] ?? 0) * 3 * 20, target: c.target, path: c.path, near: near(c) });
      return { id: c.id, latencyMs: r.latencyMs, falseJumps: r.falseJumps, localFlashes: r.localFlashes, jump: c.jump };
    });
    console.table(rows);
    for (const r of rows) {
      expect(r.falseJumps, r.id).toBe(0);
      if (r.jump) expect(r.latencyMs, r.id).not.toBeNull();
      else expect(r.latencyMs, r.id).toBeNull();
    }
  });
});
