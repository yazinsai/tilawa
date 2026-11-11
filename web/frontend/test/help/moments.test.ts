import { describe, expect, test } from "vitest";
import { clipAround, foldWarnings, markedMoment, selectMoments, wordAt } from "../../help/src/moments";
import { bookmark, check, pos, seeded, steadyReading, word } from "./fixtures";

const sorted = (xs: { clip: { startMs: number } }[]) => xs.every((m, i) => i === 0 || xs[i - 1].clip.startMs <= m.clip.startMs);
const overlaps = (a: { startMs: number; endMs: number }, b: { startMs: number; endMs: number }) => a.startMs < b.endMs && b.startMs < a.endMs;

describe("selectMoments", () => {
  test("a clean take gets two random unflagged samples, never assumed clean", () => {
    const events = steadyReading(60_000);
    const m = selectMoments(events, [], { durationMs: 60_000, rng: seeded(3) });
    expect(m.map((x) => x.kind)).toEqual(["sample", "sample"]);
    for (const s of m) {
      expect(s.question).toBe("reading_correct");
      expect(s.clip.endMs - s.clip.startMs).toBe(8000);
      expect(s.location).not.toBeNull();
    }
    expect(overlaps(m[0].clip, m[1].clip)).toBe(false);
  });

  test("samples are random: different seeds pick different regions", () => {
    const events = steadyReading(120_000);
    const starts = new Set([1, 2, 3, 4, 5].map((s) => selectMoments(events, [], { durationMs: 120_000, rng: seeded(s) })[0].clip.startMs));
    expect(starts.size).toBeGreaterThan(1);
  });

  test("priority: confirmed warnings, bookmarks, suspected, position; capped at 5 with one sample", () => {
    const events = [
      ...steadyReading(200_000),
      check(20_000, "s1", "suspected"),
      check(40_000, "c1", "suspected"),
      check(41_000, "c1", "confirmed", word(3, 2)),
      check(60_000, "r1", "suspected"),
      check(61_000, "r1", "resolved"),
      check(80_000, "u1", "suspected"),
      check(81_500, "u1", "uncertain"),
      check(100_000, "c2", "confirmed"),
      pos(150_000, word(9, 1), "ambiguous"),
    ];
    const marks = [bookmark("b1", 120_000), bookmark("b2", 130_000)];
    const m = selectMoments(events, marks, { durationMs: 200_000, rng: seeded(7) });
    expect(m).toHaveLength(5);
    expect(sorted(m)).toBe(true);
    const kinds = m.map((x) => x.kind).sort();
    // 2 confirmed + 2 bookmarks fill 4 flagged slots; suspected/position are cut; one bias sample stays.
    expect(kinds).toEqual(["sample", "stumble", "stumble", "warning", "warning"]);
    expect(m.filter((x) => x.kind === "warning").every((x) => x.confirmed)).toBe(true);
    expect(m.flatMap((x) => x.eventIds)).not.toContain("r1"); // resolved: retracted, not asked
    expect(m.find((x) => x.eventIds.includes("c1"))!.location).toEqual(word(3, 2));
  });

  test("a bookmark next to a warning is folded into it, asked once", () => {
    const events = [...steadyReading(60_000), check(30_000, "w", "confirmed")];
    const m = selectMoments(events, [bookmark("b", 31_500)], { durationMs: 60_000, rng: seeded(1) });
    const flagged = m.filter((x) => x.kind !== "sample");
    expect(flagged).toHaveLength(1);
    expect(flagged[0]).toMatchObject({ kind: "warning", eventIds: ["w"], bookmarkIds: ["b"], question: "made_mistake" });
  });

  test("samples keep clear of every warning (resolved too) and bookmark", () => {
    const events = [...steadyReading(40_000), check(10_000, "r", "suspected"), check(10_500, "r", "resolved")];
    const m = selectMoments(events, [bookmark("b", 30_000)], { durationMs: 40_000, rng: seeded(2) });
    for (const s of m.filter((x) => x.kind === "sample")) {
      expect(overlaps(s.clip, { startMs: 10_000 - 6000 - 3000, endMs: 10_500 + 2000 + 3000 })).toBe(false);
      expect(overlaps(s.clip, { startMs: 30_000 - 9000, endMs: 30_000 + 5000 })).toBe(false);
    }
  });

  test("position trouble asks 'wrong ayah' about what we showed", () => {
    const events = [...steadyReading(20_000), pos(25_000, word(7, 1), "locked", { relocated: { from: word(2, 3), to: word(7, 1) } }), ...steadyReading(60_000).slice(50)];
    const m = selectMoments(events, [], { durationMs: 60_000, rng: seeded(1) });
    const p = m.find((x) => x.kind === "position")!;
    expect(p.question).toBe("wrong_ayah");
    expect(p.location).toEqual(word(2, 3));
  });

  test("clips stay inside the take", () => {
    const events = [...steadyReading(10_000), check(1_000, "a", "confirmed"), check(9_900, "b", "confirmed")];
    for (const m of selectMoments(events, [], { durationMs: 10_000, rng: seeded(1) })) {
      expect(m.clip.startMs).toBeGreaterThanOrEqual(0);
      expect(m.clip.endMs).toBeLessThanOrEqual(10_000);
      expect(m.clip.endMs).toBeGreaterThan(m.clip.startMs);
    }
  });

  test("short or empty takes", () => {
    expect(selectMoments([], [], { durationMs: 0 })).toEqual([]);
    expect(selectMoments([], [], { durationMs: 1500 })).toEqual([]); // too short for a sample
    const m = selectMoments([], [], { durationMs: 5000, rng: seeded(1) });
    expect(m).toHaveLength(1); // the engine never placed the reader: still sampled, with no ayah
    expect(m[0].location).toBeNull();
  });

  test("ids are unique and stable in order", () => {
    const m = selectMoments(steadyReading(60_000), [bookmark("b", 20_000)], { durationMs: 60_000, rng: seeded(9) });
    expect(m.map((x) => x.id)).toEqual(m.map((_, i) => `m${i + 1}`));
  });
});

describe("helpers", () => {
  test("foldWarnings keeps first time and final state", () => {
    expect(foldWarnings([check(5, "a", "suspected"), check(9, "a", "confirmed")])).toMatchObject([{ id: "a", firstMs: 5, final: "confirmed" }]);
  });
  test("wordAt: latest placed position, else the first after", () => {
    const ev = [pos(1000, null, "acquiring"), pos(2000, word(1, 2)), pos(3000, word(1, 3))];
    expect(wordAt(ev, 2500)).toEqual(word(1, 2));
    expect(wordAt(ev, 500)).toEqual(word(1, 2));
    expect(wordAt([], 500)).toBeNull();
  });
  test("clipAround clamps", () => {
    expect(clipAround(1000, 5000)).toEqual({ startMs: 0, endMs: 3000 });
    expect(clipAround(4900, 5000)).toEqual({ startMs: 0, endMs: 5000 });
  });
  test("markedMoment asks 'made_mistake'", () => {
    expect(markedMoment("x1", 12_000, steadyReading(20_000), 20_000)).toMatchObject({ kind: "marked", question: "made_mistake", clip: { startMs: 6000, endMs: 14_000 } });
  });
});
