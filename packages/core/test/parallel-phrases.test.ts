import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { letterSkeleton, parallelPaths } from "../src/parallel-phrases.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { CheckerEvent, WordRef } from "../src/events.js";
import type { PhonemeEvidence } from "../src/evidence.js";
import type { ParallelPath } from "../src/parallel-phrases.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const at = (r: WordRef) => index.words.findIndex((x) => key(x) === key(r));
const phone = (r: WordRef) => index.phonemes.slice(index.words[at(r)].start, index.words[at(r)].end);
const MARKS = new Set("َُِ۪ۦۥاـؙۜ");
const consonants = (s: string) => [...s].filter((c) => !MARKS.has(c)).join("");

/** Heard ن at decoder confidence `conf`, with ل a weaker rival on its frames: ل/ن on a phone band. */
const nasal = (conf: number) => (evidence: PhonemeEvidence) => {
  const e = { ...evidence, phonemes: evidence.phonemes.map((p) => (p.p === "ن" ? { ...p, conf } : p)) };
  return withPosteriors(index, e, (j) => (e.phonemes[j].p === "ن" ? [["ن", 0.6], ["ل", 0.05], ["", 0.3]] : undefined));
};

function run(from: WordRef, to: WordRef, edits: Edit[], acoustics: "decoded" | "neutral" | "off" | ((e: PhonemeEvidence) => PhonemeEvidence)) {
  const fixture = syntheticFixture(index, { id: "parallel-phrase", from, to, edits, chunkSize: 6 });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  const events: CheckerEvent[] = [];
  const neutral: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);
  for (const evidence of fixture.evidence) {
    const chunk = typeof acoustics === "function" ? acoustics(evidence) : acoustics === "off" ? evidence
      : acoustics === "neutral" ? withPosteriors(index, evidence, () => neutral, neutral.length)
        : withPosteriors(index, evidence);
    events.push(...checker.push(chunk, []));
  }
  events.push(...checker.end());
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

/** Recite the phrase, optionally with its interior word spoken as `alt`, through two words past the exit. */
function span(path: ParallelPath, swap: boolean) {
  const from = index.words[path.start];
  const to = index.words[Math.min(index.words.length - 1, path.start + 5)];
  const target = index.words[path.word];
  const edits: Edit[] = swap ? [
    { kind: "del", at: target.start - from.start, count: target.end - target.start },
    { kind: "ins", at: target.start - from.start, text: path.altWord },
  ] : [];
  return { from, to, edits, word: target };
}

describe("parallel phrase index", () => {
  it("folds alef and strips diacritics", () => {
    expect(letterSkeleton("عَلَيْهِمَا")).toBe("عليهما");
    expect(letterSkeleton("ٱلْـَٔاخِرِينَ")).toBe("الاخرين");
  });

  it("links a 4-word phrase to the other attested interior word, and only across a consonant", () => {
    const paths = parallelPaths(index);
    const alayhi = paths.get(at(w(37, 108, 2))) ?? [];
    const longer = alayhi.find((p) => p.altWord === phone(w(37, 119, 2)));
    expect(longer).toMatchObject({ start: at(w(37, 108, 1)) });
    expect(longer!.alt.length).toBeGreaterThan(longer!.written.length);
    const thamud = paths.get(at(w(26, 141, 2))) ?? [];
    expect(thamud.map((p) => p.altWord)).toContain(phone(w(26, 123, 2)));
    let pathsN = 0;
    for (const list of paths.values()) for (const p of list) {
      pathsN++;
      const written = index.phonemes.slice(index.words[p.word].start, index.words[p.word].end);
      expect(consonants(written)).not.toBe(consonants(p.altWord));
      expect(p.alt.startsWith(index.phonemes.slice(index.words[p.start].start, index.words[p.start].end))).toBe(true);
    }
    expect(pathsN).toBeGreaterThan(1000);
  });
});

describe("parallel phrase graph", () => {
  const alayhi = (parallelPaths(index).get(at(w(37, 108, 2))) ?? []).find((p) => p.altWord === phone(w(37, 119, 2)))!;
  const alayhima = (parallelPaths(index).get(at(w(37, 119, 2))) ?? []).find((p) => p.altWord === phone(w(37, 108, 2)))!;
  const thamud = (parallelPaths(index).get(at(w(26, 141, 2))) ?? []).find((p) => p.altWord === phone(w(26, 123, 2)))!;

  it("flags the longer parallel when the posteriors prefer it", () => {
    const s = span(alayhi, true);
    expect(run(s.from, s.to, s.edits, "decoded")).toEqual([`wrong_word ${key(s.word)}`]);
  });

  it("flags the shorter parallel when the posteriors prefer it", () => {
    const s = span(alayhima, true);
    expect(run(s.from, s.to, s.edits, "decoded")).toEqual([`wrong_word ${key(s.word)}`]);
  });

  it("flags a parallel word that is not a suffix", () => {
    const s = span(thamud, true);
    expect(run(s.from, s.to, s.edits, "decoded")).toEqual([`wrong_word ${key(s.word)}`]);
  });

  it("stays quiet when the written phrase is what was said", () => {
    const s = span(alayhi, false);
    expect(run(s.from, s.to, s.edits, "decoded")).toEqual([]);
  });

  it("asks a parallel heard unsure to clear the per-word bar", () => {
    // إِلَّا ٱلَّذِينَ against the attested إِنَّ ٱلَّذِينَ: the posteriors favour إنّ by more than the parallel
    // switch, less than twice the support margin. Heard surely it is a wrong word; at even odds it is not.
    const inna = (parallelPaths(index).get(at(w(84, 25, 1))) ?? []).find((p) => p.altWord === "ءِننننَ" && p.start === at(w(84, 24, 3)))!;
    const s = span(inna, true);
    expect(run(s.from, s.to, s.edits, nasal(0.95))).toEqual([`wrong_word ${key(s.word)}`]);
    expect(run(s.from, s.to, s.edits, nasal(0.6))).toEqual([]);
  });

  it("does not switch on flat posteriors or without them", () => {
    const swapped = span(alayhi, true);
    const straight = span(alayhi, false);
    expect(run(swapped.from, swapped.to, swapped.edits, "neutral")).toEqual([]);
    expect(run(straight.from, straight.to, straight.edits, "neutral")).toEqual([]);
    expect(run(swapped.from, swapped.to, swapped.edits, "off")).toEqual([]);
  });

  it("stays quiet when a fast written phrase has shorter madd/ghunna than its uncollapsed reference", () => {
    // 22:1:1..4 (يَا أَيُّهَا النَّاسُ اتَّقُوا رَبَّكُمْ) vs 2:21 (اعْبُدُوا): shortened madd/ghunna
    // leaves fewer frames than the uncollapsed written sequence requires, while the interior word is clear.
    const fast = "يَاءييهَننَاسُتتَقُۥرَببَكُمءِننننَزَلزَلَتَ";
    const phonemes = [...fast].map((p, i) => ({ p, start: i * 2, end: i * 2, conf: 0.96 }));
    const ev = withPosteriors(index, { inventory: index.inventoryId, frameMs: 40, phonemes });
    const checker = createChecker(index, { passage: { from: w(22, 1, 1), to: w(22, 1, 6) }, policy: { confirm: { after: "next-word" } } });
    checker.push(ev, []);
    checker.end();
    expect(checker.mistakes()).toEqual([]);
  });
});
