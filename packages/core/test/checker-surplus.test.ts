import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhraseAsset } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const amin = phrases.phrases.find((p) => p.name === "amin")!.p;

function check(edits: Edit[], acoustics: "none" | "neutral" = "none") {
  const from = w(44, 9, 1), to = w(44, 11, 5);
  const fixture = syntheticFixture(index, { id: "surplus-seam", from, to, edits, chunkSize: 6 });
  const checker = createChecker(index, { passage: { from, to }, phrases });
  const neutral: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);
  for (const evidence of fixture.evidence) {
    const chunk = acoustics === "neutral" ? withPosteriors(index, evidence, () => neutral, neutral.length) : evidence;
    checker.push(chunk, []);
  }
  checker.end();
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

const at = (r: WordRef) => find(r).start - find(w(44, 9, 1)).start;
const seam = w(44, 10, 1);

describe("surplus speech that spells a mushaf word", () => {
  it("confirms قبلهم before word 1 when posteriors are neutral", () => {
    const said = text(w(2, 118, 15));
    expect(check([{ kind: "ins", at: at(seam), text: said }], "neutral")).toEqual([`added_word ${key(seam)}`]);
  });

  it("stays quiet for a neighbour repeat, a word-1 prefix, amin and a bare waqf mark", () => {
    const neighbour = text(w(44, 9, 5));
    const prefix = text(seam).slice(0, 6);
    expect(check([{ kind: "ins", at: at(seam), text: neighbour }])).toEqual([]);
    expect(check([{ kind: "ins", at: at(seam), text: prefix }])).toEqual([]);
    expect(check([{ kind: "ins", at: at(seam), text: amin }])).toEqual([]);
    expect(check([{ kind: "ins", at: at(seam), text: "ڇ" }])).toEqual([]);
  });

  it("does not bypass the support margin for speech that spells no word", () => {
    expect(check([{ kind: "ins", at: at(seam), text: "غَظَضَقَ" }], "neutral")).toEqual([]);
  });

  it("leaves a held meem on عظيم at the seam alone", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), target = w(2, 7, 12);
    const atWord = (r: WordRef) => find(r).start - find(from).start;
    const fixture = syntheticFixture(index, {
      id: "held-meem", from, to, chunkSize: 6,
      edits: [{ kind: "ins", at: atWord(target) + text(target).length - 1, text: "مم" }],
    });
    const checker = createChecker(index, { passage: { from, to } });
    for (const evidence of fixture.evidence) checker.push(evidence, []);
    checker.end();
    expect(checker.mistakes()).toEqual([]);
  });
});
