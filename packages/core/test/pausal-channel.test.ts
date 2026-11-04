import { describe, expect, it } from "vitest";
import { channelFromCounts, emptyCounts } from "../src/noisy-channel.js";
import { pausalChannelChoice } from "../src/pausal-channel.js";
import { loadReferenceIndex } from "../src/reference.js";

const manner = (raw: string) => {
  const consonants = [...raw].filter(p => "كرلمنب".includes(p));
  const last = consonants.at(-1);
  return last && "من".includes(last) ? "nasal" : last && "رل".includes(last) ? "liquid" : last ? "obstruent" : null;
};
function reference({ written = "كَر", rival = "مِن", near = "بَب", extra = "لِل" } = {}) {
  return loadReferenceIndex({
    version: "pausal-channel-test", text: "synthetic", profile: {},
    inventory: { id: "pausal-channel-test", tokens: "كرلمنبَِۦںۜ" },
    ayahs: ["بَب", near, "بَب", "بَب", written, "بَب", "بَب", "بَب", rival, extra, "َِ"]
      .map((p, i) => ({ s: 1, a: i + 1, w: [p], p, o: [0], iso: [p] })),
  });
}
const counts = emptyCounts([...reference().tokens]);
for (let i = 0; i < counts.phones.length; i++) counts.sub[i][i] = 100;
counts.nWritten = 100 * counts.phones.length;
const channel = channelFromCounts(counts);
const choose = (index: ReturnType<typeof reference>, heard: string, ch = channel) =>
  pausalChannelChoice(index, 4, heard, manner, ch);

describe("channel evidence after pausal manner rekey", () => {
  it("keeps a strict rival win and rejects a written-word win", () => {
    const win = choose(reference(), "مِن");
    expect(win.rival).toBe("مِن");
    expect(win.rivalLogP).toBeGreaterThan(win.writtenLogP);
    const written = choose(reference(), "كَر");
    expect(written.rivalLogP).toBeLessThan(written.writtenLogP);
  });

  it("leaves a tie without affirmative evidence", () => {
    const tied = channelFromCounts(emptyCounts([...reference().tokens]));
    const choice = choose(reference(), "مِن", tied);
    expect(choice.rivalLogP).toBe(choice.writtenLogP);
    expect(choice.rivalLogP > choice.writtenLogP).toBe(false);
  });

  it("rejects a manner contradiction the measured channel explains as the written word", () => {
    const blur = emptyCounts([...reference().tokens]);
    for (const [written, heard] of [["ك", "م"], ["َ", "ِ"], ["ر", "ن"]]) {
      blur.sub[blur.phones.indexOf(written!)][blur.phones.indexOf(heard!)] = 1000;
      blur.nWritten += 1000;
    }
    const choice = choose(reference(), "مِن", channelFromCounts(blur));
    expect(choice.rival).toBe("مِن");
    expect(choice.rivalLogP).toBeLessThan(choice.writtenLogP);
  });

  it("scores the best attested rival with the heard final manner", () => {
    const choice = choose(reference({ rival: "مِم", extra: "مِن" }), "مِن");
    expect(choice.rival).toBe("مِن");
    const wrongManner = choose(reference({ rival: "لِل" }), "مِن");
    expect(wrongManner.rival).toBe(null);
  });

  it("excludes own forms, neighbours and neighbour prefixes", () => {
    expect(choose(reference({ rival: "كَر", extra: "بَب" }), "كَر").rival).toBe(null);
    expect(choose(reference({ near: "مِن" }), "مِن").rival).toBe(null);
    expect(choose(reference({ near: "مِنَب" }), "مِن").rival).toBe(null);
    const index = reference();
    const own = { ...index, wordForms: index.wordForms!.map((forms, i) => i === 4 ? [...forms, "مِن"] : forms) };
    expect(choose(own, "مِن").rival).toBe(null);
  });

  it("allows a one-consonant rival and excludes vowel-only forms", () => {
    const choice = choose(reference({ rival: "م", extra: "َِ" }), "م");
    expect(choice.rival).toBe("م");
    expect(choice.rivalLogP).toBeGreaterThan(choice.writtenLogP);
  });

  it("normalizes held sounds and waqf edges identically", () => {
    const base = choose(reference(), "مِن");
    const held = choose(reference({ rival: "مِۦۦنںں" }), "ۜمِۦۦنںں");
    const equivalent = choose(reference({ rival: "مِۦن" }), "مِۦن");
    expect(held).toEqual(equivalent);
    expect(base.rival).toBe("مِن");
  });

  it("abstains without attested pause forms, speech or a channel", () => {
    expect(choose({ ...reference(), isolatedWordForms: undefined }, "مِن").rival).toBe(null);
    expect(choose(reference(), "ۜں").rival).toBe(null);
    expect(choose(reference(), "مِن", channelFromCounts(emptyCounts([]))).rival).toBe(null);
  });
});
