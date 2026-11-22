import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { DecodedPhoneme, EngineEvent, PhonemeEvidence, PhraseAsset, PositionEvent, ReferenceAsset } from "../../src/engine/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8")) as PhraseAsset;

/** spliced-clean's decode of a help clip reciting 94:5 with its second word skipped (فإن مع العسر يسرا → فإن العسر يسرا). */
const SKIP_94_5: DecodedPhoneme[] = ([
  ["ف", 37, 37], ["َ", 39, 40], ["ء", 44, 45], ["ِ", 46, 47], ["ن", 51, 51], ["ن", 54, 54], ["ن", 72, 72], ["َ", 73, 74],
  ["ل", 79, 79], ["ع", 84, 84], ["ُ", 85, 86], ["ث", 91, 91], ["ر", 99, 99], ["ِ", 100, 101], ["ي", 105, 105], ["ُ", 106, 107],
  ["س", 111, 111], ["ر", 119, 120], ["َ", 121, 122], ["ا", 123, 124], ["ا", 127, 127],
] as const).map(([p, start, end]) => ({ p, start, end, conf: 0.9 }));

/** Evidence chunks of `ms` audio at 40 ms frames, cut the way a streaming listener cuts them: whole phonemes only. */
const pushedEvery = (ms: number, phonemes: DecodedPhoneme[], nFrames = 177): PhonemeEvidence[] => {
  const per = ms / 40;
  const out: PhonemeEvidence[] = [];
  for (let a = 0; a < nFrames; a += per) {
    out.push({ inventory: index.inventoryId, frameMs: 40, phonemes: phonemes.filter((p) => p.start >= a && p.start < a + per) });
  }
  return out;
};

const run = (chunks: PhonemeEvidence[]) => {
  const t = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  return [...chunks.flatMap((c) => t.push(c)), ...t.end()].filter((e: EngineEvent): e is PositionEvent => e.type === "position");
};
const locks = (events: PositionEvent[]) =>
  events.filter((e) => e.state === "locked").map((e) => `${e.observed[0].word.surah}:${e.observed[0].word.ayah}:${e.observed[0].word.word}`);

describe("tracker acquisition is independent of push cadence", () => {
  it("locks on the recited ayah whether evidence arrives every 320 ms, every second, or all at once", () => {
    const second = run(pushedEvery(1000, SKIP_94_5));
    const fine = run(pushedEvery(320, SKIP_94_5));
    const whole = run([{ inventory: index.inventoryId, frameMs: 40, phonemes: SKIP_94_5 }]);
    expect(new Set(locks(fine).map((k) => k.split(":").slice(0, 2).join(":")))).toEqual(new Set(["94:5"]));
    expect(fine).toEqual(second);
    expect(whole).toEqual(second);
  });

  it("acts on a second of evidence once the evidence says it is complete, without waiting for a later phoneme", () => {
    const t = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
    // Three frames earlier, the clip's last phoneme falls in the second that locks.
    const early = SKIP_94_5.map((p) => ({ ...p, start: p.start - 3, end: p.end - 3 }));
    const pushed = pushedEvery(320, early).flatMap((c, i) => t.push({ ...c, decodedTo: Math.min(177, (i + 1) * 8) }));
    expect(locks(pushed.filter((e): e is PositionEvent => e.type === "position"))[0]).toMatch(/^94:5:/);
  });
});
