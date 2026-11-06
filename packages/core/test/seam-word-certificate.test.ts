import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { SeamWordAgreementEvidence } from "../src/evidence.js";
import { validSeamWordAgreement } from "../src/evidence.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const certificate: SeamWordAgreementEvidence = {
  word: { surah: 1, ayah: 2, word: 1 }, observed: "كَرِۦم", fromFrame: 10, toFrame: 20, cost: 5, hard: 0, rival: 20,
};
const run = (certificates: SeamWordAgreementEvidence[]) => {
  const checker = createChecker(index);
  checker.push({ inventory: index.inventoryId, frameMs: 40,
    phonemes: [{ p: "ب", start: 0, end: 0, conf: 1, seamWordAgreements: certificates }] }, []);
  const events = checker.end();
  return { checker, events };
};
it("emits a whole independent word once after token-preserving JSON transport", () => {
  const { checker, events } = run(JSON.parse(JSON.stringify([certificate, certificate])));
  expect(checker.mistakes()).toMatchObject([{ errorClass: "added_word", state: "confirmed",
    location: certificate.word, observed: certificate.observed, timeMs: 800 }]);
  expect(events).toEqual(checker.mistakes());
  expect(checker.end()).toEqual([]);
});
it("abstains on absent or invalid rank certificates", () => {
  for (const change of [{ cost: 20 }, { hard: 2 }, { rival: NaN }, { cost: -1 }, { fromFrame: -1 }, { toFrame: 10 }]) {
    const c = { ...certificate, ...change };
    expect(validSeamWordAgreement(c)).toBe(false);
    expect(run([c]).checker.mistakes()).toEqual([]);
  }
  expect(run([]).checker.mistakes()).toEqual([]);
});
it("rechecks whole-word, neighbour and eligible seam guards", () => {
  for (const c of [
    { ...certificate, observed: "اَلحَمدُ" },
    { ...certificate, observed: "مخترع" },
    { ...certificate, word: { surah: 1, ayah: 2, word: 2 } },
    { ...certificate, word: { surah: 999, ayah: 2, word: 1 } },
  ]) expect(run([c]).checker.mistakes()).toEqual([]);
});
