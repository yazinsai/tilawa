import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { createChecker } from "../src/checker.js";
import { PosteriorFrames } from "../src/ctc.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PositionEvent } from "../src/events.js";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";
import { projectVowels, vowelPresenceMargin } from "../src/vowel-omission.js";

const inventory = [..."َُِاۦۥبتں"];
const id = (p: string) => inventory.indexOf(p) + 1;
function speech(text: string, startFrame = 7) {
  const ids: number[] = [], probs: number[] = [];
  for (const p of text) {
    ids.push(id(p), 0, id(p), 0);
    probs.push(0.98, 0.02, 0.98, 0.02);
    ids.push(0, id(p)); probs.push(0.98, 0.02);
  }
  const frames = new PosteriorFrames(inventory.length);
  frames.append({ startFrame, k: 2, nFrames: ids.length / 2, ids, probs });
  return frames;
}

describe("vowel-projected omission presence", () => {
  const words = [["بَ", "بِ", "بُاۦۥ", "تَ", "تِ"]];
  const margin = (frames: PosteriorFrames, text = words) =>
    vowelPresenceMargin(frames, inventory, text, { ayah: 0, word: 2 }, frames.startFrame, frames.endFrame);

  it("sums blank, consonants and ghunna while keeping all six vowels separate", () => {
    const frames = new PosteriorFrames(inventory.length);
    frames.append({ startFrame: 11, nFrames: 1, k: 4,
      ids: [0, id("ب"), id("ں"), id("ُ")], probs: [0.3, 0.2, 0.1, 0.25] });
    const before = Array.from(frames.row(11));
    const projected = projectVowels(frames, inventory, 11, 12)!;
    const mass = Array.from(projected.row(11), Math.exp);
    // The unlisted 0.15 is spread over the six remaining symbols: five vowels and ت.
    expect(mass[0]).toBeCloseTo(0.625);
    expect(mass[3]).toBeCloseTo(0.25);
    for (const vowel of [1, 2, 4, 5, 6]) expect(mass[vowel]).toBeCloseTo(0.025);
    expect(Array.from(frames.row(11))).toEqual(before);
  });

  it("keeps spoken vowels when all their consonants have changed", () => {
    expect(margin(speech("تَتِتُاۦۥبَبِ"))).toBeGreaterThanOrEqual(PHRASE_SWITCH);
    expect(margin(speech("بَںِںُاۦۥتَتِ"))).toBeCloseTo(margin(speech("تَتِتُاۦۥبَبِ"))!);
  });

  it("prefers omission when the word's vowel frames have been removed", () => {
    expect(margin(speech("بَبِتَتِ"))).toBeLessThan(0);
  });

  it("lets a word be repeated without changing the closed presence decision", () => {
    expect(margin(speech("بَبِبُاۦۥبُاۦۥتَتِ"))).toBeGreaterThanOrEqual(PHRASE_SWITCH);
  });

  it("retains distinct long and short vowel classes", () => {
    const frames = speech("َُِاۦۥ");
    const projected = projectVowels(frames, inventory, frames.startFrame, frames.endFrame)!;
    for (let v = 0; v < 6; v++) expect(projected.row(frames.startFrame + v * 3)[v + 1]).toBeGreaterThan(Math.log(0.9));
  });

  it("abstains on missing frames, a vowel-free target or an impossible comparison", () => {
    const frames = speech("ب");
    expect(projectVowels(frames, inventory, 0, 1)).toBeNull();
    expect(projectVowels(frames, inventory, 7, frames.endFrame + 1)).toBeNull();
    expect(projectVowels(frames, inventory, 7, 7)).toBeNull();
    expect(margin(frames, [["بَ", "بِ", "ں", "تَ", "تِ"]])).toBeNull();
    expect(margin(frames)).toBeNull();
  });
});

describe("checker's proposed single-word omission veto", () => {
  const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
  const from = { surah: 2, ayah: 255, word: 20 }, to = { surah: 2, ayah: 255, word: 27 };
  const target = { surah: 2, ayah: 255, word: 23 };
  const find = (word: number) => index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === word)!;
  const start = find(target.word).start - find(from.word).start;
  const end = find(target.word).end - find(from.word).start;
  const original = syntheticFixture(index, { id: "vowel-presence", from, to }).evidence[0];
  const position: PositionEvent = { type: "position", v: 1, timeMs: original.phonemes.at(-1)!.end * 20,
    referenceVersion: index.version, expected: to, state: "locked",
    observed: [{ word: to, span: { start: from, end: to }, confidence: 1, cost: 0, per: 0 }] };

  const run = (vowelsPresent: boolean, posteriors = true) => {
    const evidence = withPosteriors(index, original, j => {
      if (j < start || j >= end) return undefined;
      const p = original.phonemes[j].p;
      return vowelsPresent && "َُِاۦۥ".includes(p) ? [[p, 0.98], ["", 0.01]] : [["", 0.99]];
    });
    evidence.phonemes = evidence.phonemes.filter((_, j) => j < start || j >= end);
    if (!posteriors) delete evidence.posteriors;
    const before = JSON.stringify(evidence);
    const checker = createChecker(index, { passage: { from, to } });
    const events = [...checker.push(evidence, [position]), ...checker.end()];
    expect(JSON.stringify(evidence)).toBe(before);
    return events.filter(e => e.errorClass === "omitted_word" && e.location.word === target.word);
  };

  it("vetoes the proposal when the stored vowels fit despite missing greedy tokens", () => {
    expect(run(true)).toEqual([]);
    expect(run(true, false).some(e => e.state === "confirmed")).toBe(true);
  });

  it("keeps the omission when those vowels are absent", () => {
    expect(run(false).some(e => e.state === "confirmed")).toBe(true);
  });
});
