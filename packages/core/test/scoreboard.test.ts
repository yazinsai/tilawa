import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { PhraseAsset, ReferenceAsset } from "../src/index.js";
import { scoreClip } from "../src/bench/acceptance.js";
import { oracleClip, renderScoreboard, scored } from "../src/bench/scoreboard.js";
import type { ScoreRow } from "../src/bench/scoreboard.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const declared = (c: ReturnType<typeof oracleClip>) => scoreClip(index, c!, { arm: "declared", phrases });

describe("oracle evidence from labels", () => {
  it("lets a correct checker catch an omitted word", () => {
    const c = oracleClip(index, asset, {
      id: "x", ayahs: ["94:5", "94:6"], non_errors: [],
      events: [{ errorClass: "omitted_word", state: "confirmed", location: { surah: 94, ayah: 5, word: 2 } }],
    });
    const s = declared(c);
    expect([s.hits, s.misses.length, s.falsePositives.length]).toEqual([1, 0, 0]);
  });

  it("plants a wrong word from its written form", () => {
    // 26:106 with نوح said as هود.
    const c = oracleClip(index, asset, {
      id: "x", ayahs: ["26:106", "26:107", "26:108"], non_errors: [],
      events: [{ errorClass: "wrong_word", state: "confirmed", location: { surah: 26, ayah: 106, word: 5 }, observed: "هُودٌ" }],
    });
    expect(declared(c).hits).toBe(1);
  });

  it("re-recites a planted repeat without raising anything", () => {
    const c = oracleClip(index, asset, { id: "x", ayahs: ["93:3"], events: [], non_errors: [{ kind: "repeat", location: { surah: 93, ayah: 3, word: 3 } }] });
    expect(declared(c).confirmed).toEqual([]);
  });

  it("gives up on a wrong word whose written form is nowhere in the Quran", () => {
    const c = oracleClip(index, asset, {
      id: "x", ayahs: ["26:106"], non_errors: [],
      events: [{ errorClass: "wrong_word", state: "confirmed", location: { surah: 26, ayah: 106, word: 5 }, observed: "قطةقطة" }],
    });
    expect(c).toBeNull();
  });
});

describe("scoreboard rendering", () => {
  const row = (sha: string, precision: number | null): ScoreRow => ({
    sha, date: "2026-10-07", listener: "teacher",
    metrics: [
      scored({ id: "e2e.precision", group: "End to end", name: "Precision", value: precision, unit: "%", bar: { op: ">=", target: 0.9 } }),
      scored({ id: "tracker.tlog", group: "Tracker", name: "Ambiguous tlog", value: null, status: "unmeasurable", note: "no passage truth" }),
    ],
  });

  it("marks each metric against its bar and shows the delta vs the baseline row", () => {
    const md = renderScoreboard([row("aaaaaaa", 0.55), row("bbbbbbb", 0.62)]);
    expect(md).toContain("| End to end | Precision | 62.0% | ≥ 90.0% | fail | +7.0 pp |  |");
    expect(md).toContain("| Tracker | Ambiguous tlog | — |  | unmeasurable |  | no passage truth |");
  });

  it("passes a metric that meets its bar", () => {
    expect(scored({ id: "p", group: "g", name: "n", value: 0.95, unit: "%", bar: { op: ">=", target: 0.9 } }).status).toBe("pass");
    expect(scored({ id: "p", group: "g", name: "n", value: 1.2, bar: { op: "<=", target: 1 } }).status).toBe("fail");
  });
});
