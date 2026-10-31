import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { hardConsonantEdits, lexicalWords } from "../src/checker-lexical.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { ReferenceAsset } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { collapseHeld, deletionCost, insertionCost, subCost, UNIT } from "../src/fuzzy.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const word = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(w => key(w) === key(r))!;

/** Exhaustive oracle: enumerate complete edit paths, then count their hard-edit graph components. No DP pruning
 * by run state is used, so different histories meeting at the same phoneme indices remain distinct paths. */
function exhaustiveHardChanges(expected: string, observed: string): number {
  const a = collapseHeld([...expected]).join(""), b = collapseHeld([...observed]).join("");
  const consonants = new Set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي");
  const positions = (s: string, side: string) => {
    const out: { at: number; node: string }[] = [];
    for (let start = 0; start < s.length;) {
      let end = start + 1;
      while (end < s.length && s[end] === s[start]) end++;
      if (consonants.has(s[start])) for (let at = start; at < end; at++) out.push({ at, node: `${side}:${start}` });
      start = end;
    }
    return out;
  };
  const reference = positions(a, "reference"), heard = positions(b, "observed");
  const components = (edits: string[][]) => {
    const parent = new Map<string, string>();
    const root = (node: string): string => {
      const next = parent.get(node)!;
      return next === node ? node : root(next);
    };
    for (const nodes of edits) {
      for (const node of nodes) if (!parent.has(node)) parent.set(node, node);
      for (const node of nodes.slice(1)) parent.set(root(node), root(nodes[0]));
    }
    return new Set([...parent.keys()].map(root)).size;
  };
  let bestCost = Infinity, bestChanges = Infinity;
  const visit = (i: number, j: number, cost: number, edits: string[][]) => {
    if (cost > bestCost) return;
    if (i === reference.length && j === heard.length) {
      const count = components(edits);
      if (cost < bestCost) { bestCost = cost; bestChanges = count; }
      else bestChanges = Math.min(bestChanges, count);
      return;
    }
    const take = (di: number, dj: number, charge: number, nodes: string[]) =>
      visit(i + di, j + dj, cost + charge, charge >= UNIT ? [...edits, nodes] : edits);
    if (i < reference.length && j < heard.length) take(1, 1,
      subCost(a[reference[i].at], b[heard[j].at]), [reference[i].node, heard[j].node]);
    if (i < reference.length) take(1, 0, deletionCost(a, reference[i].at), [reference[i].node]);
    if (j < heard.length) take(0, 1, insertionCost(a, reference[i]?.at ?? a.length, b[heard[j].at]), [heard[j].node]);
  };
  visit(0, 0, 0, []);
  return bestChanges;
}

describe("independent consonant changes", () => {
  it.each([
    ["ءَننننَ", "ءَمممَ"],
    ["للَهُننننَ", "للَهُمم"],
  ])("counts one geminated identity change: %s to %s", (expected, observed) => {
    expect(hardConsonantEdits(expected, observed)).toBe(1);
  });

  it("counts changing a run's identity and length as one changed sound", () => {
    expect(hardConsonantEdits("ببب", "ت")).toBe(1);
    expect(hardConsonantEdits("ب", "تتت")).toBe(1);
  });

  it("keeps vowel-separated repetitions independent", () => {
    expect(hardConsonantEdits("نَن", "مَم")).toBe(2);
    expect(hardConsonantEdits("نَنَن", "مَمَم")).toBe(3);
  });

  it("does not count a vowel-only split or join of an unchanged consonant run", () => {
    expect(hardConsonantEdits("نن", "نَن")).toBe(0);
    expect(hardConsonantEdits("نَن", "نن")).toBe(0);
    expect(hardConsonantEdits("بَنن", "تَنَن")).toBe(1);
  });

  it("counts a partial consonant-run identity change once", () => {
    expect(hardConsonantEdits("نن", "نم")).toBe(1);
    expect(hardConsonantEdits("نم", "نن")).toBe(1);
  });

  it("counts genuine gemination loss and a different changed run separately", () => {
    expect(hardConsonantEdits("زز", "ز")).toBe(1);
    expect(hardConsonantEdits("بَبب", "تَب")).toBe(2);
  });

  it("preserves confusable discounts and held ghunna duration", () => {
    expect(hardConsonantEdits("ححَق", "ههَك")).toBe(0);
    expect(hardConsonantEdits("نننن", "ننن")).toBe(0);
    expect(hardConsonantEdits("نننن", "ن")).toBe(1);
  });

  it("matches exhaustive hard-edit connectivity on short split, joined, and partial runs", () => {
    const samples = ["", "ن", "م", "نن", "نَن", "مم", "مَم", "نم", "ننم", "نَمَن"];
    const pairs = samples.flatMap(a => samples.map(b => [a, b]));
    pairs.push(["حَق", "هَك"], ["حح", "هَه"], ["نننن", "ممم"], ["بَبب", "تَب"]);
    for (const [a, b] of pairs) {
      expect(hardConsonantEdits(a, b), `${JSON.stringify(a)} → ${JSON.stringify(b)}`)
        .toBe(exhaustiveHardChanges(a, b));
    }
  });
});

describe("the two observed false-alarm patterns", () => {
  it.each([
    { from: word(9, 3, 6), to: word(9, 3, 14), target: word(9, 3, 10), observed: "ءَمممَ" },
    { from: word(24, 60, 15), to: word(24, 60, 23), target: word(24, 60, 20), observed: "للَهُمم" },
  ])("abstains at $target with confident decoded evidence", ({ from, to, target, observed }) => {
    const at = find(target).start - find(from).start;
    const fixture = syntheticFixture(index, { id: "independent-consonants", from, to, chunkSize: 4, edits: [
      { kind: "del", at, count: find(target).end - find(target).start },
      { kind: "ins", at, text: observed },
    ] });
    const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
    for (const evidence of fixture.evidence) checker.push(withPosteriors(index, evidence), []);
    checker.end();
    expect(checker.mistakes().filter(e => key(e.location) === key(target))).toEqual([]);
  });
});

describe("connected pronunciation provenance", () => {
  // The donor's doubled meem belongs to the following meem context; its isolated form has a single meem.
  const phones = ["بَ", "سُمم", "مَا", "جَل"];
  const asset: ReferenceAsset = {
    version: "context-test", text: "test", profile: {}, inventory: { id: "test", tokens: "بَسُمماجل" },
    ayahs: [{ s: 1, a: 1, w: ["before", "donor", "after", "target"], p: phones.join(""),
      o: [0, 2, 6, 9], iso: ["ب", "سُم", "مَا", "جَل"] }],
  };
  const vocabulary = lexicalWords(loadReferenceIndex(asset));
  const target = 3;

  it("accepts a connected form in a compatible boundary context", () => {
    expect(vocabulary.alternative(target, "سُمم", { before: "بَ", after: "مَا" })).toBe(true);
  });

  it("rejects a connected form with the wrong following or preceding sound", () => {
    expect(vocabulary.alternative(target, "سُمم", { before: "بَ", after: "وَا" })).toBe(false);
    expect(vocabulary.alternative(target, "سُمم", { before: "بُ", after: "مَا" })).toBe(false);
  });

  it("requires context for a connected-only form", () => {
    expect(vocabulary.alternative(target, "سُمم")).toBe(false);
  });

  it("compares boundary sounds without requiring identical neighboring words", () => {
    expect(vocabulary.alternative(target, "سُمم", { before: "تَ", after: "مِي" })).toBe(true);
  });

  it("keeps isolated vocabulary forms independent of neighboring context", () => {
    expect(vocabulary.alternative(target, "سُم")).toBe(true);
    expect(vocabulary.alternative(target, "سُم", { before: "بُ", after: "وَا" })).toBe(true);
  });

  it("distinguishes explicit ayah edges from ordinary neighboring sounds", () => {
    const edgeVocabulary = lexicalWords(loadReferenceIndex({ ...asset,
      inventory: { ...asset.inventory, tokens: `${asset.inventory.tokens}د` }, ayahs: [
      { s: 1, a: 1, w: ["initial donor", "after"], p: "سُمممَا", o: [0, 4], iso: ["سُم", "مَا"] },
      { s: 1, a: 2, w: ["before", "final donor"], p: "بَسُدد", o: [0, 2], iso: ["ب", "سُد"] },
      { s: 1, a: 3, w: ["target"], p: "جَل", o: [0], iso: ["جَل"] },
    ] }));
    expect(edgeVocabulary.alternative(4, "سُمم", { after: "مَا" })).toBe(true);
    expect(edgeVocabulary.alternative(4, "سُمم", { before: "بَ", after: "مَا" })).toBe(false);
    expect(edgeVocabulary.alternative(4, "سُدد", { before: "بَ" })).toBe(true);
    expect(edgeVocabulary.alternative(4, "سُدد", { before: "بَ", after: "مَا" })).toBe(false);
    expect(edgeVocabulary.alternative(4, "سُمم")).toBe(false);
  });

  it("accepts a form without context when another word supports it in isolation", () => {
    const sharedVocabulary = lexicalWords(loadReferenceIndex({ ...asset, ayahs: [
      ...asset.ayahs,
      { s: 1, a: 2, w: ["independent form"], p: "سُمم", o: [0], iso: ["سُمم"] },
    ] }));
    expect(sharedVocabulary.alternative(target, "سُمم")).toBe(true);
    expect(sharedVocabulary.alternative(target, "سُمم", { before: "بُ", after: "وَا" })).toBe(true);
  });
});
