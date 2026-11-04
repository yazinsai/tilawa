import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replayChecked, syntheticFixture } from "../src/fixtures/index.js";
import { collapseHeld } from "../src/fuzzy.js";
import { channelFromCounts, emptyCounts, phrasePhones, rankSeam } from "../src/noisy-channel.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhraseAsset } from "../src/reference.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function identityChannel(cut: number, blur?: { from: string; to: string }) {
  const phones = [...index.tokens];
  const c = emptyCounts(phones, cut);
  phones.forEach((_, i) => { c.sub[i][i] = 40; });
  c.nWritten = phones.length * 40;
  if (blur) c.sub[phones.indexOf(blur.from)][phones.indexOf(blur.to)] = 16;
  return c;
}

const from = w(44, 9, 1);
const to = w(44, 11, 3);
const seam = w(44, 10, 1);
const at = find(seam).start - find(from).start;

function run(edits: Parameters<typeof syntheticFixture>[1]["edits"], cut: number, blur?: { from: string; to: string }) {
  // One phoneme-scale chunk, as the product streams. A single push never steps word by word.
  const fixture = syntheticFixture(index, { id: "channel-seam", from, to, edits, chunkSize: 2 });
  const events = replayChecked(index, fixture, { mode: "free", phrases }, {
    phrases, channel: identityChannel(cut, blur), policy: { confirm: { after: "next-word" } },
  });
  return events.map((e) => `${e.state} ${e.errorClass} ${key(e.location)} ${e.observed ?? ""}`);
}

describe("tracker-step noisy channel seam", () => {
  it("stays quiet when the span is the written words", () => {
    expect(run(undefined, 0).filter((e) => e.includes("added_word"))).toEqual([]);
  });

  it("confirms a one-consonant blur of a non-neighbour word", () => {
    const said = text(w(2, 118, 15));
    const consonant = [...said].find((p) => "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي".includes(p))!;
    const heardWord = said.replace(consonant, "ك");
    const wi = index.words.findIndex((x) => x.surah === 44 && x.ayah === 10 && x.word === 1);
    const prevPh = index.phonemes.slice(index.words[wi - 1].start, index.words[wi - 1].end);
    const w1 = index.phonemes.slice(index.words[wi].start, index.words[wi].end);
    const ch = channelFromCounts(identityChannel(0, { from: consonant, to: "ك" }));
    const rank = rankSeam(prevPh + heardWord + w1, wi, index, phrasePhones(phrases), ch);
    expect(rank.word).toBe(collapseHeld([...said]).join(""));
    expect(rank.logOdds).toBeGreaterThan(0);
    const added = run([{ kind: "ins", at, text: heardWord }], 0, { from: consonant, to: "ك" });
    expect(added.some((e) => e.startsWith(`confirmed added_word ${key(seam)} ${collapseHeld([...said]).join("")}`))).toBe(true);
  });
});
