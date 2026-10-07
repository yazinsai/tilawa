import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { PhraseAsset, ReferenceAsset, WordRef } from "../src/index.js";
import { scoreSession, stitchSession } from "../src/bench/jumps.js";
import type { SessionPart } from "../src/bench/jumps.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const words = (surah: number, ayah: number) => index.words.filter((x) => x.surah === surah && x.ayah === ayah && x.end > x.start).length;

/** One clip reciting `said`, labelled as the ayah `label` (the recited one unless the label is wrong). */
const part = (said: [number, number], gapBeforeMs = 800, label = said): SessionPart => {
  const e = withPosteriors(index, syntheticFixture(index, { id: "c", from: w(said[0], said[1], 1), to: w(said[0], said[1], words(...said)) }).evidence[0]);
  const p = e.posteriors!;
  return {
    source: `${said[0]}:${said[1]}`,
    ayahs: [`${label[0]}:${label[1]}`],
    gapBeforeMs,
    evidence: { inventory: e.inventory, frameMs: e.frameMs, nFrames: p.nFrames, phonemes: e.phonemes, posteriors: { k: p.k, ids: [...p.ids], probs: [...p.probs] } },
  };
};
const run = (surah: number, from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => part([surah, from + i]));
const score = (parts: SessionPart[]) => scoreSession(index, stitchSession(index, parts), { phrases });

describe("stitched session scoring", () => {
  it("finds no false jump in consecutive ayahs and locks the start within 5 s", () => {
    const s = score(run(2, 255, 260));
    expect(s).toMatchObject({ startWithin5s: true, falseJumps: 0, planted: 0 });
  });

  it("follows a planted jump to another surah", () => {
    const s = score([...run(2, 255, 257), ...run(36, 1, 6)]);
    expect(s.planted).toBe(1);
    expect(s.followed).toBe(1);
    expect(s.falseJumps).toBe(0);
    expect(s.followLatencyMs[0]).toBeLessThan(5000);
  });

  it("counts a lock on a far passage nobody is reciting as a false jump", () => {
    const s = score([...run(2, 255, 257), part([36, 9], 800, [2, 258]), ...run(2, 259, 261)]);
    expect(s.planted).toBe(0);
    expect(s.falseJumpDetail).toMatchObject([{ to: "36:9", truth: expect.arrayContaining(["2:258"]) }]);
  });
});
