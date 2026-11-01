import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { procliticWords } from "../src/checker-proclitics.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import type { ReferenceIndex } from "../src/reference.js";
import type { Edit } from "../src/fixtures/index.js";

type Word = [text: string, phones: string];
function reference(ayahs: Word[][]): ReferenceIndex {
  return loadReferenceIndex({
    version: "test", text: "synthetic", profile: {},
    inventory: { id: "test", tokens: "ءبتثجحخدذرزسشصضطظعغفقكلمنهوياۦۥَُؙِۜ" },
    ayahs: [[ ["حمد", "حَمد"] as Word ], ...ayahs].map((words, i) => {
      let p = "";
      const o = words.map(w => { const at = p.length; p += w[1]; return at; });
      return { s: i ? 2 : 1, a: i || 1, w: words.map(w => w[0]), p, o, iso: words.map(w => w[1]) };
    }),
  });
}

describe("attested surface proclitic contrasts", () => {
  it.each([
    ["المؤمنين", "لمُءمِنِۦۦن", "مؤمنين", "مُءمِنِۦۦن", "ل"],
    ["الشمس", "ششَمس", "شمس", "شَمس", "ش"],
    ["الناس", "ننَااس", "ناس", "نَااس", "ن"],
    ["وقوم", "وَقَوم", "فقوم", "فَقَوم", "و"],
    ["فقوم", "فَقَوم", "وقوم", "وَقَوم", "فَ"],
    ["بقوم", "بِقَوم", "قوم", "قَوم", "بِ"],
    ["كقوم", "كَقَوم", "قوم", "قَوم", "كَ"],
    ["لقوم", "لِقَوم", "قوم", "قَوم", "لِ"],
  ])("restores %s from its prefix and a shared stem", (a, ap, b, bp, residue) => {
    const lex = procliticWords(reference([[[a, ap], [b, bp]]]));
    expect(lex.reattaches(1, 2, residue, ap.slice(residue.length))).toBe(true);
    expect(lex.reattaches(1, 2, "", ap)).toBe(false);
    expect(lex.reattaches(1, 2, residue + "ۜ", ap.slice(residue.length))).toBe(false);
  });

  it("keeps the prefix identity hard and requires the complete source word", () => {
    const lex = procliticWords(reference([[["وقوم", "وَقَوم"], ["فقوم", "فَقَوم"]]]));
    expect(lex.reattaches(1, 2, "فَ", "قَوم")).toBe(false);
    expect(lex.reattaches(1, 2, "وَ", "قَو")).toBe(false);
    expect(lex.reattaches(1, 2, "وَ", "كَوم")).toBe(false);
  });

  it("does not strip a stem letter just because it resembles a prefix", () => {
    const lex = procliticWords(reference([[["كتاب", "كِتَااب"], ["تاب", "تَااب"]]]));
    expect(lex.reattaches(1, 2, "كِ", "تَااب")).toBe(false);
  });

  it("requires matching letter stems even when the sounds coincide", () => {
    const lex = procliticWords(reference([[["وقوم", "وَقَوم"], ["قال", "قَوم"]]]));
    expect(lex.reattaches(1, 2, "وَ", "قَوم")).toBe(false);
  });

  it("abstains when the reference has no word text", () => {
    const index = reference([[["وقوم", "وَقَوم"], ["قوم", "قَوم"]]]);
    delete index.wordText;
    expect(procliticWords(index).reattaches(1, 2, "وَ", "قَوم")).toBe(false);
  });
});

describe("proclitic residue at a proposed skip", () => {
  const index = reference([
    [["كتب", "كَتَبَ"], ["من", "مِنَ"], ["المؤمنين", "لمُءمِنِۦۦن"]],
    [["قال", "قَاالَ"], ["رب", "رَبب"], ["هم", "هُم"], ["مؤمنين", "مُءمِنِۦۦن"]],
    [["جعل", "جَعَلَ"], ["فوق", "فَوقَ"], ["باب", "بَااب"]],
    [["رحيم", "رَحِۦۦم"], ["حكيم", "حَكِۦۦم"]],
  ]);
  const source = index.words[3], skip = index.words[4], after = index.words[8];
  const base = index.words[1].start;
  const omit: Edit = { kind: "del", at: skip.start - base, count: after.start - skip.start };
  function run(edits: Edit[]) {
    const from = index.words[1], to = index.words.at(-1)!;
    const fixture = syntheticFixture(index, { id: "proclitic-skip", from, to, edits, chunkSize: 4 });
    const checker = createChecker(index, { passage: { from, to } });
    for (const evidence of fixture.evidence) checker.push(evidence, []);
    checker.end();
    return checker.mistakes().map(e => ({ cls: e.errorClass, ayah: e.location.ayah, word: e.location.word }));
  }

  it("reunites the article and stem and keeps the whole skipped ayah", () => {
    expect(run([omit])).toEqual([{ cls: "skipped_passage", ayah: 2, word: 1 }]);
  });

  it("preserves a skip that really includes the source edge word", () => {
    expect(run([{ kind: "del", at: source.start - base, count: after.start - source.start }]))
      .toEqual([{ cls: "skipped_passage", ayah: 1, word: 3 }]);
  });

  it("does not infer a missing article from a bare stem", () => {
    expect(run([omit, { kind: "del", at: source.start - base, count: 1 }]))
      .toEqual([{ cls: "skipped_passage", ayah: 1, word: 3 }]);
  });

  it("does not attach across a labelled pause", () => {
    expect(run([omit, { kind: "ins", at: source.start + 1 - base, text: "ۜ" }]))
      .toEqual([{ cls: "skipped_passage", ayah: 1, word: 3 }]);
  });

  it("keeps clean recitation quiet", () => {
    expect(run([])).toEqual([]);
  });

  it("keeps an ordinary single-word omission", () => {
    expect(run([{ kind: "del", at: skip.start - base, count: skip.end - skip.start }]))
      .toEqual([{ cls: "omitted_word", ayah: 2, word: 1 }]);
  });
});
