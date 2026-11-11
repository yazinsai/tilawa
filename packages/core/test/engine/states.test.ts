import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { PositionEvent, ReferenceAsset, TrackerOptions, WordRef } from "../../src/engine/index.js";
import { finalPosition, replay, syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;

const recite = (from: WordRef, to: WordRef, edits: Edit[] = [], options: TrackerOptions = {}) =>
  finalPosition(replay(index, syntheticFixture(index, { id: "t", from, to, edits }), options))!;

describe("deduplication", () => {
  it("does not call a repeated phrase within one ayah ambiguous", () => {
    const e = recite(w(2, 255, 3), w(2, 255, 12), [{ kind: "repeat", from: 10, to: 30 }]);
    expect(e.state).toBe("locked");
    expect(e.observed).toHaveLength(1);
  });

  it("does not call a clip ambiguous because its span can shift by a word", () => {
    const e = recite(w(2, 255, 3), w(2, 255, 12), [{ kind: "del", at: 0, count: 3 }]);
    expect(e.state).toBe("locked");
    expect(e.observed).toHaveLength(1);
  });
});

describe("ambiguous", () => {
  it("lists every surah that opens with the same ayah", () => {
    const e = recite(w(2, 1, 1), w(2, 1, lastWord(2, 1)));
    expect(e.state).toBe("ambiguous");
    expect(e.expected).toBeNull();
    expect(e.observed.map((c) => c.span.start.surah).sort((a, b) => a - b)).toEqual([2, 3, 7, 13, 29, 30, 31, 32]);
  });

  it("treats a lone basmala as ambiguous between al-Fatiha and an-Naml", () => {
    const e = recite(w(1, 1, 1), w(1, 1, 4));
    expect(e.state).toBe("ambiguous");
    expect(e.observed.map((c) => `${c.span.start.surah}:${c.span.start.ayah}`).sort()).toEqual(["1:1", "27:30"]);
  });
});

describe("lost", () => {
  it("reports seeded noise as lost", () => {
    const e = recite(w(2, 255, 3), w(2, 255, 3), [{ kind: "noise", at: 0, length: 40, seed: 7 }, { kind: "del", at: 0, count: 99 }]);
    expect(e.state).toBe("lost");
    expect(e.observed).toEqual([]);
  });
});

describe("sufficient evidence", () => {
  it("does not lock on a short phrase even when it is unique", () => {
    const e = recite(w(2, 70, 8), w(2, 70, 9));
    expect(e.observed[0].span.start).toEqual(w(2, 70, 8));
    expect(e.state).not.toBe("locked");
    expect(e.expected).toBeNull();
  });

  it("does not lock on a unique but poorly fitting clip", () => {
    const e = recite(w(2, 255, 3), w(2, 255, 6), [
      { kind: "sub", at: 2, to: "ق" },
      { kind: "sub", at: 6, to: "ك" },
      { kind: "sub", at: 10, to: "ز" },
      { kind: "sub", at: 14, to: "ط" },
      { kind: "sub", at: 18, to: "غ" },
    ]);
    expect(e.state).not.toBe("locked");
  });
});

describe("prior", () => {
  it("does not lock on the prior ayah when the audio fits another", () => {
    const e = recite(w(100, 1, 1), w(100, 1, 2), [], { prior: { surah: 100, ayah: 3, weight: 1 } });
    expect(e.state).toBe("locked");
    expect(e.observed[0].span.start).toEqual(w(100, 1, 1));
  });

  it("cannot break a genuine tie on its own", () => {
    const e = recite(w(2, 1, 1), w(2, 1, lastWord(2, 1)), [], { prior: { surah: 3, ayah: 1, weight: 1 } });
    expect(e.state).toBe("ambiguous");
  });

  it("ranks the prior's passage first among tied candidates", () => {
    const e = recite(w(2, 1, 1), w(2, 1, lastWord(2, 1)), [], { prior: { surah: 31, ayah: 1, weight: 1 } });
    expect(e.observed[0].span.start).toEqual(w(31, 1, 1));
  });
});

describe("range prior", () => {
  const opening = (priorRange?: TrackerOptions["priorRange"]) => recite(w(2, 1, 1), w(2, 1, lastWord(2, 1)), [], { priorRange });

  it("ranks a candidate inside the range first, but cannot break the tie", () => {
    const e = opening({ ranges: [{ from: { surah: 31, ayah: 1 }, to: { surah: 31, ayah: 34 } }], weight: 1 });
    expect(e.observed[0].span.start).toEqual(w(31, 1, 1));
    expect(e.state).toBe("ambiguous");
  });

  it("gives nothing to candidates outside the range", () => {
    const none = opening();
    const outside = opening({ ranges: [{ from: { surah: 31, ayah: 2 }, to: { surah: 32, ayah: 0 } }, { from: { surah: 4, ayah: 1 }, to: { surah: 6, ayah: 9 } }], weight: 1 });
    expect(outside.observed.map((c) => c.span.start)).toEqual(none.observed.map((c) => c.span.start));
  });

  it("spans surah boundaries", () => {
    const e = opening({ ranges: [{ from: { surah: 29, ayah: 60 }, to: { surah: 30, ayah: 3 } }], weight: 1 });
    expect(e.observed[0].span.start).toEqual(w(30, 1, 1));
  });
});

describe("wrong word that matches another verse", () => {
  // 10:1 and 12:1 differ only in word 5. Reciting 12:1 while meaning 10:1 is a memorisation error,
  // but the tracker reports where the audio is; flagging it against the intended passage is the checker's job.
  it("locks to the verse actually recited", () => {
    const e = recite(w(12, 1, 1), w(12, 1, lastWord(12, 1)), [], { prior: { surah: 10, ayah: 1, weight: 1 } });
    expect(e.state).toBe("locked");
    expect(e.observed[0].span.start).toEqual(w(12, 1, 1));
  });
});

describe("confidence", () => {
  it("falls as the fit worsens", () => {
    const exact = recite(w(2, 255, 3), w(2, 255, 12));
    const one = recite(w(2, 255, 3), w(2, 255, 12), [{ kind: "sub", at: 12, to: "ق" }]);
    const three = recite(w(2, 255, 3), w(2, 255, 12), [
      { kind: "sub", at: 12, to: "ق" },
      { kind: "sub", at: 25, to: "ك" },
      { kind: "sub", at: 38, to: "ز" },
    ]);
    expect(exact.observed[0].confidence).toBe(1);
    expect(one.observed[0].confidence).toBeLessThan(exact.observed[0].confidence);
    expect(three.observed[0].confidence).toBeLessThan(one.observed[0].confidence);
  });

  it("is lower for tied candidates than for a separated one", () => {
    const tied = recite(w(2, 1, 1), w(2, 1, lastWord(2, 1)));
    const unique = recite(w(2, 255, 3), w(2, 255, 12));
    for (const c of tied.observed) expect(c.confidence).toBeLessThanOrEqual(0.5);
    expect(unique.observed[0].confidence).toBe(1);
  });

  it("never locks a streamed clip before it has enough evidence", () => {
    const t = createTracker(index);
    const f = syntheticFixture(index, { id: "s", from: w(2, 255, 3), to: w(2, 255, 12), chunkSize: 3 });
    const early = f.evidence.slice(0, 3).flatMap((e) => t.push(e)) as PositionEvent[];
    for (const e of early) expect(e.state).not.toBe("locked");
    const rest = [...f.evidence.slice(3).flatMap((e) => t.push(e)), ...t.end()] as PositionEvent[];
    expect(rest.at(-1)!.state).toBe("locked");
  });
});
