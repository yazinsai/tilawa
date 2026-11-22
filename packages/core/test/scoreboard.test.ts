import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { PhraseAsset, ReferenceAsset } from "../src/index.js";
import { scoreClip } from "../src/bench/acceptance.js";
import { gap, goldPerMetrics, moves, oracleClip, parentOf, renderScoreboard, scored, upsertRow, verdict } from "../src/bench/scoreboard.js";
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

  const gates = [{ id: "e2e.precision", label: "Precision", area: "Product" as const }];
  const counted = (sha: string, listener: string, hit: number, of: number): ScoreRow => ({
    ...row(sha, hit / of), listener, metrics: [scored({ id: "e2e.precision", group: "End to end", name: "Precision", value: hit / of, unit: "%", display: `${hit}/${of} (${hit}/${of})`, bar: { op: ">=", target: 0.9 } })],
  });

  it("ranks the candidate's gates and deltas them against its previous run", () => {
    const md = renderScoreboard([row("aaaaaaa", 0.55), row("bbbbbbb", 0.62)], "teacher", gates, []);
    expect(md).toContain("**Not ready**: 0/1 gates pass, 1 fail");
    expect(md).toContain("| Precision | 62.0% | ≥ 90.0% | 31% short | +7.0 pp |");
    expect(md).toContain("| Tracker | Ambiguous tlog | — |  | unmeasurable |  | no passage truth |");
  });

  it("judges a listener's first run against the candidate and treats two-clip moves as noise", () => {
    const rows = [counted("aaaaaaa", "teacher", 30, 40), counted("bbbbbbb", "student", 36, 40), counted("ccccccc", "student", 35, 40)];
    expect(verdict(moves(rows[0], rows[1], gates))).toBe("better");
    expect(parentOf(rows, 1, "teacher")).toBe(rows[0]);
    expect(parentOf(rows, 2, "teacher")).toBe(rows[1]);
    expect(verdict(moves(rows[1], rows[2], gates))).toBe("noise");
    const md = renderScoreboard(rows, "teacher", gates, []);
    expect(md).toContain("| bbbbbbb | student | teacher aaaaaaa | ▲ better | ▲ Precision 30/40 (30/40) → 36/40 (36/40) |");
  });

  it("is not ready while a blocked requirement has no data", () => {
    const md = renderScoreboard([row("aaaaaaa", 0.95)], "teacher", gates, [{ id: "tracker.tlog", label: "Ambiguous tlog", area: "Tracker" }]);
    expect(md).toContain("**Not ready**: 1/1 gates pass, 0 fail, 0 not measured on this listener, 1 requirements");
    expect(gap(row("x", 0.95).metrics[0])).toBe(0);
  });

  it("replaces the row with the same commit and listener, else appends", () => {
    const b = upsertRow(upsertRow([], row("aaaaaaa", 0.5)), row("bbbbbbb", 0.6));
    const c = upsertRow(b, { ...row("aaaaaaa", 0.7), note: "rerun" });
    expect(c.map((r) => [r.sha, r.metrics[0].value, r.note])).toEqual([["aaaaaaa", 0.7, "rerun"], ["bbbbbbb", 0.6, undefined]]);
  });

  it("passes a metric that meets its bar", () => {
    expect(scored({ id: "p", group: "g", name: "n", value: 0.95, unit: "%", bar: { op: ">=", target: 0.9 } }).status).toBe("pass");
    expect(scored({ id: "p", group: "g", name: "n", value: 1.2, bar: { op: "<=", target: 1 } }).status).toBe("fail");
  });
});

describe("gold PER", () => {
  const per = (canon: number, clips: number) => ({ canon, raw: canon + 0.02, clips, refPhonemes: 1000 });
  const scores = {
    "e2-libri-full": { gold: { clips: 3, overall: per(0.06, 3), byDevice: { tlog: per(0.04, 2), phone: per(0.12, 1) },
      groups: { short_vowels: { per: 0.05, edits: 5, phonemes: 100 } } } },
  };
  const target = { bar: 0.0365, bindingMetric: "declared.precision", teacherGoldPer: 0.0339 };

  it("is the primary listener PER, gated by the calibrated bar", () => {
    const [m] = goldPerMetrics(scores, target, "e2-libri-full");
    expect([m.id, m.value, m.bar, m.status]).toEqual(["listener.goldPer", 0.06, { op: "<=", target: 0.0365 }, "fail"]);
  });

  it("splits by device and phoneme group, with raw PER as info", () => {
    const ms = goldPerMetrics(scores, target, "e2-libri-full");
    const v = Object.fromEntries(ms.map((m) => [m.id, [m.value, m.status]]));
    expect(v["listener.goldPer.phone"]).toEqual([0.12, "info"]);
    expect(v["listener.goldPer.short_vowels"]).toEqual([0.05, "info"]);
    expect(v["listener.goldPerRaw"]).toEqual([0.08, "info"]);
  });

  it("reports the slice never used in training as info", () => {
    const held = { ...scores["e2-libri-full"], goldHeldOut: { ...scores["e2-libri-full"].gold, clips: 1, overall: per(0.09, 1) } };
    const m = goldPerMetrics({ x: held }, target, "x").find((m) => m.id === "listener.goldPerHeldOut")!;
    expect([m.value, m.status, m.display]).toEqual([0.09, "info", "9.00% (1 clips)"]);
  });

  it("is pending for a listener that was never decoded on the gold set", () => {
    const [m] = goldPerMetrics(scores, target, "combo");
    expect([m.value, m.status]).toEqual([null, "pending"]);
  });
});
