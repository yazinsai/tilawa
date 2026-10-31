import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[], posterior?: (j: number, p: string) => [string, number][] | undefined) {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const f = syntheticFixture(index, { id: "gap-ownership", from, to, edits: edits(at), chunkSize: 4 });
  if (posterior) {
    const all = f.evidence.flatMap(e => e.phonemes);
    const rows = withPosteriors(index, { ...f.evidence[0], phonemes: all }, j => posterior(j, all[j].p)).posteriors!;
    let startFrame = rows.startFrame;
    f.evidence = f.evidence.map((e, i) => {
      const end = f.evidence[i + 1]?.phonemes[0].start ?? rows.startFrame + rows.nFrames;
      const a = (startFrame - rows.startFrame) * rows.k, b = (end - rows.startFrame) * rows.k;
      const posteriors = { ...rows, startFrame, nFrames: end - startFrame, ids: Array.from(rows.ids).slice(a, b), probs: Array.from(rows.probs).slice(a, b) };
      startFrame = end;
      return { ...e, posteriors };
    });
  }
  const checker = createChecker(index, { passage: { from, to } });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

describe("ownership of sounds beside a missing word", () => {
  it("keeps a nasal ending of the following word with that word", { timeout: 30_000 }, () => {
    const from = w(52, 25, 1), to = w(52, 28, 9), omitted = w(52, 26, 2), after = w(52, 26, 3);
    expect(check(from, to, at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "sub", at: at(after), to: "ء" },
    ], (j, p) => j === find(after).start - find(from).start - text(omitted).length ? [["ء", 0.5], ["ك", 0.45], ["", 0.03]] : [[p, 0.98], ["", 0.01]])).toEqual([`omitted_word ${key(omitted)}`]);
  });

  it("leaves an ambiguous existing placement unchanged below the acoustic margin", { timeout: 30_000 }, () => {
    const from = w(52, 25, 1), to = w(52, 28, 9), omitted = w(52, 26, 2), after = w(52, 26, 3);
    expect(check(from, to, at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "sub", at: at(after), to: "ء" },
    ], j => j === find(after).start - find(from).start - text(omitted).length ? [["ء", 0.5], ["ك", 0.45], ["", 0.03]] : undefined)).toEqual([`omitted_word ${key(after)}`]);
  });

  it("keeps a noisy following word from filling the omitted word", () => {
    const from = w(3, 30, 1), to = w(3, 32, 10), omitted = w(3, 31, 2), after = w(3, 31, 3);
    expect(check(from, to, at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "sub", at: at(after), to: "ء" },
    ])).toEqual([`omitted_word ${key(omitted)}`]);
  });

  it("keeps clean neighboring words and harmless onset noise quiet", () => {
    expect(check(w(3, 30, 1), w(3, 32, 10), () => [])).toEqual([]);
    expect(check(w(3, 30, 1), w(3, 32, 10), at => [
      { kind: "sub", at: at(w(3, 31, 3)), to: "ء" },
    ])).toEqual([]);
  });

  it("does not invent an omission when the short word was actually said", () => {
    expect(check(w(52, 25, 1), w(52, 28, 9), at => [
      { kind: "sub", at: at(w(52, 26, 3)), to: "ء" },
    ], () => undefined)).toEqual([]);
  });

  it("keeps two genuinely omitted neighboring words in the gap", () => {
    const omitted = w(52, 26, 2), next = w(52, 26, 3);
    expect(check(w(52, 25, 1), w(52, 28, 9), at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "del", at: at(next), count: text(next).length },
    ], () => undefined)).toEqual([`omitted_word ${key(omitted)}`]);
  });

  it("preserves a real wrong word beside a real omission", () => {
    const from = w(52, 25, 1), to = w(52, 28, 9), omitted = w(52, 26, 2), next = w(52, 26, 3);
    const edits = (at: (r: WordRef) => number): Edit[] => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "del", at: at(next), count: text(next).length },
      { kind: "ins", at: at(next), text: "عَسِۦۦر" },
    ];
    expect(check(from, to, edits, () => undefined)).toEqual(check(from, to, edits));
    expect(check(from, to, edits, () => undefined).length).toBeGreaterThan(0);
  });
});
