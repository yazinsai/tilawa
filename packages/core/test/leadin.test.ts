import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { PhraseAsset, ReferenceAsset, WordRef } from "../src/index.js";
import { scoreClip } from "../src/bench/acceptance.js";
import type { AcceptanceClip, ClipScore } from "../src/bench/acceptance.js";
import { inClip, withLeadIn } from "../src/bench/leadin.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

const evidence = (from: WordRef, to: WordRef, edits: Edit[] = [], posteriors = true): AcceptanceClip["evidence"] => {
  const e = syntheticFixture(index, { id: "c", from, to, edits }).evidence[0];
  const p = withPosteriors(index, e).posteriors!;
  if (!posteriors) return { inventory: e.inventory, frameMs: e.frameMs, nFrames: p.nFrames, phonemes: e.phonemes } as AcceptanceClip["evidence"];
  return { inventory: e.inventory, frameMs: e.frameMs, nFrames: p.nFrames, phonemes: e.phonemes, posteriors: { k: p.k, ids: [...p.ids], probs: [...p.probs] } };
};

// 26:124 with هود said as صالح is exactly 26:142: alone, free recitation follows it there.
const hud = w(26, 124, 5);
const slip = (posteriors = true): AcceptanceClip => {
  const at = find(hud).start - find(w(26, 124, 1)).start;
  return {
    id: "c", ayahs: ["26:124"], nonErrors: [],
    events: [{ errorClass: "wrong_word", state: "confirmed", location: hud, forms_other_ayah: true }],
    evidence: evidence(w(26, 124, 1), w(26, 124, 7), [{ kind: "del", at, count: text(hud).length }, { kind: "ins", at, text: text(w(26, 142, 5)) }], posteriors),
  };
};
const lead = (posteriors = true) => ({ source: "lead", ayahs: ["26:122", "26:123"], evidence: evidence(w(26, 122, 1), w(26, 123, 3), [], posteriors) });

describe("lead-in", () => {
  it("lets free recitation place a slip onto a twin ayah once the preceding ayahs were recited", () => {
    const at = (s: ClipScore) => s.final.filter((e) => e.errorClass === "wrong_word").map((e) => e.location);
    expect(at(scoreClip(index, slip(), { arm: "free", phrases }))).toEqual([]);
    const c = withLeadIn(index, slip(), lead(), 600);
    expect(c.clip.ayahs).toEqual(["26:122", "26:123", "26:124"]);
    expect(at(scoreClip(index, c.clip, { arm: "free", phrases }))).toEqual([hud]);
  });

  it("stitches decodes that carry no posteriors", () => {
    const c = withLeadIn(index, slip(false), lead(false), 600);
    expect(c.clip.evidence.posteriors).toBeUndefined();
    expect(c.clip.evidence.nFrames).toBe(lead().evidence.nFrames + 600 / slip().evidence.frameMs + slip().evidence.nFrames);
  });

  it("splits false positives between the lead-in and the original clip", () => {
    const c = withLeadIn(index, slip(), lead(), 600);
    const fp = (location: WordRef) => ({ errorClass: "wrong_word", location }) as ClipScore["falsePositives"][number];
    const both = [fp(w(26, 123, 2)), fp(w(26, 124, 3))];
    const score = { hits: 1, misses: [], falsePositives: both, confirmed: both, final: both, repeatFalsePositives: 0 } as unknown as ClipScore;
    const s = inClip(score, c.leadAyahs);
    expect(s.score.falsePositives.map((e) => e.location)).toEqual([w(26, 124, 3)]);
    expect(s.score.confirmed.map((e) => e.location)).toEqual([w(26, 124, 3)]);
    expect(s.score.final.map((e) => e.location)).toEqual([w(26, 124, 3)]);
    expect(s.leadInWarnings).toBe(1);
  });
});
