import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import { replayChecked, syntheticFixture } from "../src/fixtures/index.js";
import type { Edit, Fixture } from "../src/fixtures/index.js";
import type { CheckerEvent, WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

const start = w(44, 5, 1);
const end = w(44, 5, 6);
const inna = w(44, 5, 4);
const kunna = w(44, 5, 5);
const at = (r: WordRef) => find(r).start - find(start).start;

function flag(f: Fixture, heardAt: number) {
  let i = 0;
  for (const e of f.evidence) for (const p of e.phonemes) {
    if (i === heardAt) {
      p.pauseOnset = true;
      return p.p;
    }
    i++;
  }
  throw new Error(`no phoneme at ${heardAt}`);
}

function confirmed(edits: Edit[], mark?: number) {
  const f = syntheticFixture(index, { id: "pause-onset", from: start, to: end, edits, chunkSize: 8 });
  if (mark !== undefined) expect(flag(f, mark)).toBe("ء");
  const events = replayChecked(index, f, {}, { passage: { from: start, to: end } });
  return events.filter((e: CheckerEvent) => e.state === "confirmed").map((e) => `${e.errorClass} ${key(e.location)}`);
}

describe("pause-onset hamza in omission placement", () => {
  const dropInna: Edit[] = [
    { kind: "del", at: at(inna), count: text(inna).length },
    { kind: "sub", at: at(kunna), to: "ء" },
  ];
  // The substituted onset is كنا's, shifted back by the deleted إنا.
  const fakeHamza = at(kunna) - text(inna).length;

  it("places the omission on كنا while the leading ء still counts as إنا's onset", () => {
    expect(confirmed(dropInna)).toEqual([`omitted_word ${key(kunna)}`]);
  });

  it("places the omission on إنا once that ء is a pause onset, so the vowel decides", () => {
    expect(text(inna)[1]).toBe("ِ");
    expect(text(kunna)[1]).toBe("ُ");
    expect(confirmed(dropInna, fakeHamza)).toEqual([`omitted_word ${key(inna)}`]);
  });

  it("keeps a real post-pause hamza on its own word when the neighbour is the one omitted", () => {
    expect(confirmed([{ kind: "del", at: at(kunna), count: text(kunna).length }], at(inna))).toEqual([`omitted_word ${key(kunna)}`]);
  });

  it("stays quiet when the hamza word and its neighbour were both said", () => {
    expect(confirmed([], at(inna))).toEqual([]);
  });
});
