import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhraseAsset } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../src/evidence.js";
import type { WordRef } from "../src/events.js";
import { emptyCounts, type ChannelCounts } from "../src/noisy-channel.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const iso = (r: WordRef) => index.isolatedWordForms![index.words.indexOf(find(r))]!;

/** 11:11 ends كبير (pausal ر). 11:12 follows, so a gap there is the ayah-closing pause. */
const from = w(11, 11, 7);
const to = w(11, 12, 4);
const target = w(11, 11, 10);
const at = (r: WordRef) => find(r).start - find(from).start;
const neutral: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);

function soften(phonemes: DecodedPhoneme[], needle: string): DecodedPhoneme[] {
  const s = phonemes.map((p) => p.p).join("");
  const i = s.indexOf(needle);
  if (i < 0) throw new Error(`heard span ${needle} not in ${s}`);
  return phonemes.map((p, j) => (j >= i && j < i + [...needle].length ? { ...p, conf: 0.5 } : p));
}

function run(edits: Edit[], o: { needle?: string; neutral?: boolean; span?: { from: WordRef; to: WordRef }; channel?: ChannelCounts } = {}) {
  const span = o.span ?? { from, to };
  const fx = syntheticFixture(index, { id: "manner", from: span.from, to: span.to, edits });
  const evidence: PhonemeEvidence[] = fx.evidence.map((e) => {
    const phonemes = o.needle ? soften(e.phonemes, o.needle) : e.phonemes;
    return { ...withPosteriors(index, { ...e, phonemes }, o.neutral ? () => neutral : undefined), phonemes };
  });
  const checker = createChecker(index, { passage: span, phrases, channel: o.channel });
  for (const e of evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

const pause: Edit = { kind: "gap", at: at(w(11, 12, 1)), frames: 20 };
const shortPause: Edit = { kind: "gap", at: at(w(11, 12, 1)), frames: 4 };
/** No consonant in common with كبير, final ن against pausal ر. */
const nasal = "ءَلكَوثَن";
/** Same skeleton, final ل against ر: both liquid. */
const liquid = "ءَلكَوثَل";

describe("pausal-final manner", () => {
  it("confirms an ayah-final wrong word below 0.6 when the final manner contradicts", () => {
    expect(iso(target)).toBe("كَبِۦۦر");
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: nasal },
      pause,
    ];
    expect(run(edits, { needle: nasal })).toEqual([`wrong_word ${key(target)}`]);
  });

  it("keeps the 0.6 cut when the final manner matches", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: liquid },
      pause,
    ];
    expect(run(edits, { needle: liquid })).toEqual([]);
  });

  it("keeps the 0.6 cut when the same swap is not before a labelled pause", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: nasal },
      shortPause,
    ];
    expect(run(edits, { needle: nasal })).toEqual([]);
  });

  it("keeps the 0.6 cut on a word that is not ayah-final", () => {
    const mid = w(11, 11, 8);
    const edits: Edit[] = [
      { kind: "del", at: at(mid), count: text(mid).length },
      { kind: "ins", at: at(mid), text: nasal },
      pause,
    ];
    expect(run(edits, { needle: nasal })).toEqual([]);
  });

  it("still confirms a same-manner wrong word the decoder was sure of", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: liquid },
      pause,
    ];
    expect(run(edits)).toEqual([`wrong_word ${key(target)}`]);
  });

  it("does not confirm a manner mismatch the posteriors do not support", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: nasal },
      pause,
    ];
    expect(run(edits, { needle: nasal, neutral: true })).toEqual([]);
  });

  it("rekeys an ayah-final omission when the slot holds contradicting consonant speech", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: "مِن" },
      pause,
    ];
    expect(run(edits, { needle: "مِن" })).toEqual([`wrong_word ${key(target)}`]);
  });

  it("drops both rekey paths without channel evidence", () => {
    for (const heard of [nasal, "مِن"]) {
      const edits: Edit[] = [
        { kind: "del", at: at(target), count: text(target).length },
        { kind: "ins", at: at(target), text: heard },
        pause,
      ];
      expect(run(edits, { needle: heard, channel: emptyCounts([]) })).toEqual([]);
    }
  });

  it("leaves an ayah-final omission when the slot's final manner matches", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: "مِل" },
      pause,
    ];
    expect(run(edits, { needle: "مِل" })).toEqual([`omitted_word ${key(target)}`]);
  });

  it("leaves a spliced-out omission: the slot holds no speech", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      pause,
    ];
    expect(run(edits)).toEqual([`omitted_word ${key(target)}`]);
  });

  it("does not rekey an omission that is not before a labelled pause", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(target), count: text(target).length },
      { kind: "ins", at: at(target), text: "مِن" },
      shortPause,
    ];
    expect(run(edits, { needle: "مِن" })).toEqual([`omitted_word ${key(target)}`]);
  });

  it("stays quiet on a clean ayah end, including a ghunna that matches ن", () => {
    const start = w(1, 1, 1);
    const end = w(1, 4, 1);
    const worlds = find(w(1, 2, 4));
    const off = worlds.start - find(start).start;
    const edits: Edit[] = [
      { kind: "sub", at: off + text(w(1, 2, 4)).length - 1, to: "ں" },
      { kind: "gap", at: find(w(1, 3, 1)).start - find(start).start, frames: 20 },
    ];
    expect(run(edits, { span: { from: start, to: end } })).toEqual([]);
    expect(run([pause])).toEqual([]);
  });

  it("does not treat ف for ك as a manner contradiction", () => {
    const start = w(51, 6, 1);
    const end = w(51, 9, 2);
    const kaf = w(51, 7, 3);
    expect(iso(kaf).endsWith("ك")).toBe(true);
    const heard = "شَطَعَف";
    const off = (r: WordRef) => find(r).start - find(start).start;
    const edits: Edit[] = [
      { kind: "del", at: off(kaf), count: text(kaf).length },
      { kind: "ins", at: off(kaf), text: heard },
      { kind: "gap", at: off(w(51, 8, 1)), frames: 20 },
    ];
    expect(run(edits, { needle: heard, span: { from: start, to: end } })).toEqual([]);
    expect(run(edits, { span: { from: start, to: end } })).toEqual([`wrong_word ${key(kaf)}`]);
  });
});
