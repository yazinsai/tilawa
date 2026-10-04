/**
 * Structural rules on token streams built from the corpus itself (no audio,
 * no ONNX): pause segmentation, the ayah-order chain and its guards, the
 * similar-verse variants, and the session's queue of structural flags.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { requireCorpus } from "./paths";
import { QuranCorpus } from "../../src/recitation/corpus";
import {
  AYAH_ORDER_RULE,
  StructuralRules,
  ayahOrderFlags,
  globalCost,
  pyRound,
  segmentTokens,
  similarVersePick,
  type Ayah,
  type StructuralIndexJson,
  type TimedToken,
} from "../../src/recitation/structural";
import { costTable } from "../../src/recitation/phonemeCost";
import INDEX from "../../src/recitation/structural-index.json" with { type: "json" };

const corpusJson = JSON.parse(readFileSync(requireCorpus(), "utf8"));
const corpus = new QuranCorpus(corpusJson);
const rules = new StructuralRules(corpus, INDEX as StructuralIndexJson);
const table = costTable();

/** One token per char, 2 frames apart; `pause` blank frames between segments. */
function timed(segments: string[], pause = 20): TimedToken[] {
  const out: TimedToken[] = [];
  let frame = 0;
  for (const seg of segments) {
    for (const ch of seg) {
      out.push({ sym: ch, frame, t: frame * 0.04 });
      frame += 2;
    }
    frame += pause;
  }
  return out;
}

const ph = (s: number, a: number) => corpus.ayahPhonemes(s, a);
const read = (s: number, ...ayahs: number[]) => timed(ayahs.map((a) => ph(s, a)));
const flagsOf = (tokens: TimedToken[], win: Ayah[], rule = AYAH_ORDER_RULE) =>
  ayahOrderFlags(rules.ayahOrderCandidate(tokens, win), rule).map((f) => `${f.surah}:${f.ayah}`);
const noPassage = (s: number, first: number) => rules.ayahOrderWindow(null, [s, first]);

describe("pyRound", () => {
  it("rounds half to even on the exact binary value, like Python", () => {
    expect(pyRound(0.0625, 3)).toBe(0.062);
    expect(pyRound(0.0635, 3)).toBe(0.064); // 0.0635 is above the half in binary
    expect(pyRound(2.675, 2)).toBe(2.67); // below the half in binary
    expect(pyRound(-0.0625, 3)).toBe(-0.062);
    expect(pyRound(1.5, 0)).toBe(2);
    expect(pyRound(2.5, 0)).toBe(2);
    expect(pyRound(Infinity, 3)).toBe(Infinity);
  });
});

describe("segmentTokens", () => {
  it("cuts at gaps of at least `gap` frames and merges short segments back", () => {
    const toks = timed(["ءَبَجَد", "هَ", "وَزَي"], 12);
    const segs = segmentTokens(toks, 12, 3);
    expect(segs.map((s) => s.text)).toEqual(["ءَبَجَدهَ", "وَزَي"]);
    expect(segmentTokens(timed(["ءَبَجَد", "وَزَي"], 8), 12, 3)).toHaveLength(1);
    // A short first segment joins the next one.
    expect(segmentTokens(timed(["بِ", "سمِللَااه", "ءَحَد"], 20), 12, 3).map((s) => s.text))
      .toEqual(["بِسمِللَااه", "ءَحَد"]);
  });
});

describe("ayah-order check", () => {
  it("flags an ayah skipped between two read ones, without a passage", () => {
    expect(flagsOf(read(87, 1, 2, 4, 5), noPassage(87, 1))).toEqual(["87:3"]);
  });

  it("flags the same skip inside an expected passage", () => {
    expect(flagsOf(read(87, 1, 2, 4, 5), rules.ayahOrderWindow({ surah: 87, ayah: 1, ayahEnd: 5 }, null)))
      .toEqual(["87:3"]);
  });

  it("does not flag an in-order reading, or one opened with isti'adha and basmala", () => {
    expect(flagsOf(read(87, 1, 2, 3, 4, 5), noPassage(87, 1))).toEqual([]);
    const pre = "ءَعُۥۥذُبِللَااهِمِنَششَييطَاانِررَجِۦۦم";
    const bas = "بِسمِللَااهِررَحمَاانِررَحِۦۦۦۦم";
    expect(flagsOf(timed([pre, bas, ph(87, 1), ph(87, 2), ph(87, 3), ph(87, 4)]), noPassage(87, 1))).toEqual([]);
    expect(flagsOf(timed([pre, bas, ph(87, 1), ph(87, 2), ph(87, 4), ph(87, 5)]), noPassage(87, 1))).toEqual(["87:3"]);
  });

  it("does not flag a skip that is read later, or a restart", () => {
    expect(flagsOf(read(87, 1, 2, 4, 3), noPassage(87, 1))).toEqual([]);
    expect(flagsOf(read(87, 1, 2, 1, 2, 3, 4), noPassage(87, 1))).toEqual([]);
  });

  it("keeps the flag time at the end of the segment after the jump", () => {
    const toks = read(87, 1, 2, 4, 5);
    const cand = rules.ayahOrderCandidate(toks, noPassage(87, 1))!;
    const seg4 = segmentTokens(toks, 12, 3)[2]!;
    expect(cand.jumps.find((x) => x.ayah === 3)?.at).toBe(pyRound(seg4.t, 2));
  });

  it("never flags an ayah whose skip reads as in-order text (exact identical-text guard)", () => {
    expect(rules.identGuarded(55, 17)).toBe(true); // 55:16 == 55:18
    expect(rules.identGuarded(109, 4)).toBe(true); // 109:3 == 109:5
    expect(rules.identGuarded(94, 5)).toBe(false); // 94:6 adds wa-/fa-: near-equal stays flaggable
    expect(rules.identGuarded(55, 16)).toBe(false);
    const toks = read(55, 14, 15, 16, 18, 19);
    const win = noPassage(55, 14);
    expect(flagsOf(toks, win)).not.toContain("55:17");
  });

  it("needs a window of at least three ayahs and two segments", () => {
    expect(rules.ayahOrderCandidate(read(108, 1, 3), rules.ayahOrderWindow({ surah: 108, ayah: 1, ayahEnd: 2 }, null))).toBeNull();
    expect(rules.ayahOrderCandidate(timed([ph(87, 1) + ph(87, 2)]), noPassage(87, 1))).toBeNull();
  });
});

/** A look-alike pair with one one-word `replace` region whose words differ by at least `minCost`. */
function lookalikeSwap(minCost = 5): { e: Ayah; m: Ayah; i: number; j: number } {
  for (const [key, rows] of Object.entries((INDEX as StructuralIndexJson).pairs)) {
    const [s, a] = key.split(":").map(Number) as [number, number];
    for (const r of rows) {
      const regions = [];
      for (let x = 2; x + 4 < r.length; x += 5) regions.push(r.slice(x, x + 5));
      if (regions.length !== 1) continue;
      const [tag, i1, i2, j1, j2] = regions[0]!;
      if (tag !== 0 || i2! - i1! !== 1 || j2! - j1! !== 1 || i1 === 0) continue;
      const ew = rules.ph(s, a);
      const mw = rules.ph(r[0]!, r[1]!);
      if (i1! >= ew.length - 1) continue;
      if (globalCost(table.encode(ew[i1!]!), table.encode(mw[j1!]!)) < minCost) continue;
      return { e: [s, a], m: [r[0]!, r[1]!], i: i1!, j: j1! };
    }
  }
  throw new Error("no look-alike pair");
}

describe("similar-verse check", () => {
  const { e, m, i, j } = lookalikeSwap();
  const ew = rules.ph(e[0], e[1]);
  const mw = rules.ph(m[0], m[1]);
  const words = (ws: string[]) => timed([ws.join("")]);

  it("flags a look-alike word swapped into the expected ayah", () => {
    const heard = [...ew.slice(0, i), mw[j]!, ...ew.slice(i + 1)];
    const picked = similarVersePick(rules.similarVerseCandidates(words(heard), [e], true, 10));
    expect(picked).toHaveLength(1);
    expect(picked[0]).toMatchObject({ surah: e[0], ayah: e[1], word: i, kind: "possible_substitution", from: m });
    expect(picked[0]!.at).not.toBeNull();
  });

  it("does not flag a clean reading, or a correct reading of the look-alike without a passage", () => {
    expect(similarVersePick(rules.similarVerseCandidates(words(ew), [e], true, 10))).toEqual([]);
    expect(similarVersePick(rules.similarVerseCandidates(words(mw), [e], false, 10))).toEqual([]);
  });

  it("offers a drop variant for a skipped interior word", () => {
    const k = ew.findIndex((w, x) => x > 0 && x < ew.length - 1 && w.length >= 3);
    const heard = [...ew.slice(0, k), ...ew.slice(k + 1)];
    const drops = rules.similarVerseCandidates(words(heard), [e], true, 10)
      .filter((c) => c.family === "drop" && c.word === k && c.span === 1);
    expect(drops[0]?.margin).toBeGreaterThan(0);
    expect(drops[0]?.kind).toBe("possible_omission");
  });

  it("matches a full alignment of every variant (forward / backward split is exact)", () => {
    // similar_verse.align_cost: reference global, query ends skippable at 0.5 per char.
    const alignCost = (q: Uint8Array, r: Uint8Array): number => {
      let prev = Float64Array.from({ length: r.length + 1 }, (_, x) => x);
      let best = prev[r.length]! + q.length * 0.5;
      for (let a = 1; a <= q.length; a++) {
        const cur = new Float64Array(r.length + 1);
        cur[0] = a * 0.5;
        for (let b = 1; b <= r.length; b++) {
          cur[b] = Math.min(prev[b - 1]! + table.cost(q[a - 1]!, r[b - 1]!), prev[b]! + 1, cur[b - 1]! + 1);
        }
        best = Math.min(best, cur[r.length]! + (q.length - a) * 0.5);
        prev = cur;
      }
      return best;
    };
    const before = rules.ph(e[0], e[1] > 1 ? e[1] - 1 : e[1] + 1);
    const prevAyah: Ayah = [e[0], e[1] > 1 ? e[1] - 1 : e[1] + 1];
    const heard = [...before, ...ew.slice(0, 2), ...ew.slice(3)];
    const q = table.encode(heard.join(""));
    const passage: Ayah[] = [prevAyah, e];
    const cands = rules.similarVerseCandidates(words(heard), passage, true, 10).filter((c) => c.slot === 1 && c.family === "drop");
    const canon = alignCost(q, table.encode([...before, ...ew].join("")));
    expect(cands.length).toBeGreaterThan(4);
    for (const c of cands) {
      const variant = [...before, ...ew.slice(0, c.word), ...ew.slice(c.word + c.span)];
      expect(c.margin).toBeCloseTo(pyRound(canon - alignCost(q, table.encode(variant.join(""))), 3), 9);
    }
  });
});
