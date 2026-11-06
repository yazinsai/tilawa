import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, sameWrongWord } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { WrongWordAccusation, WrongWordAgreement } from "../src/evidence.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 2, ayah: 255, word: 18 }, to = { surah: 2, ayah: 255, word: 30 };
const target = { surah: 2, ayah: 255, word: 23 };
const at = index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === 23)!;
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!.start;
const rival = index.words.find(w => w.surah === 108 && w.ayah === 1 && w.word === 3)!;
const fixture = syntheticFixture(index, { id: "side-agreement", from, to, chunkSize: 4, edits: [
  { kind: "del", at: at.start - start, count: at.end - at.start },
  { kind: "ins", at: at.start - start, text: index.phonemes.slice(rival.start, rival.end) },
] });
const chunks = fixture.evidence.map(e => withPosteriors(index, e));
function check(certificates: WrongWordAgreement[] = [], clock = 0) {
  const observed: WrongWordAccusation[] = [];
  const checker = createChecker(index, { passage: { from, to }, agreementClock: clock,
    policy: { confirm: { after: "next-word" } }, wrongWord: (c, _id, graph) => { if (!graph) observed.push(c); } });
  const evidence = structuredClone(chunks);
  if (certificates.length) evidence[0].phonemes[0].wrongWordAgreements = JSON.parse(JSON.stringify(certificates));
  for (const e of evidence) checker.push(e, []);
  checker.end();
  return { observed, mistakes: checker.mistakes() };
}

describe("wrong-word side agreement", () => {
  it("observes lexical confirmations without changing the default checker", () => {
    const result = check();
    expect(result.mistakes.some(e => e.errorClass === "wrong_word" && e.location.word === target.word)).toBe(true);
    expect(result.observed).toHaveLength(1);
    expect(result.observed[0].accusation).toHaveProperty("rival");
  });
  it("vetoes a stored disagreement and retains the same rival", () => {
    const primary = check().observed[0];
    const c = { acquireMs: 0, primary, rejudged: [], evidence: chunks };
    expect(check([c]).mistakes).toEqual([]);
    expect(check([{ ...c, rejudged: [primary] }]).mistakes).toEqual(check().mistakes);
    expect(check([c], 1000).mistakes).toEqual(check().mistakes);
  });
  it("does not apply a certificate for a different visit", () => {
    const primary = check().observed[0];
    const c = { acquireMs: 0, primary: { ...primary, fromFrame: primary.fromFrame + 1 }, rejudged: [], evidence: chunks };
    expect(check([c]).mistakes).toEqual(check().mistakes);
  });
  it("requires the same rival or the same bar on the overlapping slot", () => {
    const a = check().observed[0];
    expect(sameWrongWord(a, { ...a, accusation: { rival: "other" } })).toBe(false);
    const bar = { ...a, accusation: { bar: 20 } };
    expect(sameWrongWord(bar, bar)).toBe(true);
    expect(sameWrongWord(bar, { ...bar, accusation: { bar: 10 } })).toBe(false);
    expect(sameWrongWord(bar, { ...bar, fromFrame: bar.toFrame, toFrame: bar.toFrame + 1 })).toBe(false);
  });
});
