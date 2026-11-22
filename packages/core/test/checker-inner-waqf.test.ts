import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { PhraseAsset } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../src/evidence.js";
import type { WordRef } from "../src/events.js";
import { procliticOnset, waqfMarked } from "../src/mushaf-marks.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const wi = (r: WordRef) => index.words.findIndex((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word);
const find = (r: WordRef) => index.words[wi(r)]!;
const phones = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const iso = (r: WordRef) => index.isolatedWordForms![wi(r)]!;
const from = w(38, 4, 1), to = w(38, 5, 1);
const prev = w(38, 4, 5), next = w(38, 4, 6);
const at = (r: WordRef) => find(r).start - find(from).start;
const neutral: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);

function paint(
  evidence: PhonemeEvidence,
  alter?: (set: (frame: number, token: string, p: number) => void, phones: DecodedPhoneme[]) => void,
  override?: (j: number) => [string, number][] | undefined,
): PhonemeEvidence {
  const painted = withPosteriors(index, evidence, (j) => override?.(j) ?? neutral, neutral.length);
  const post = painted.posteriors!;
  const ids = Array.from(post.ids as ArrayLike<number>);
  const probs = Array.from(post.probs as ArrayLike<number>);
  const idOf = (t: string) => (t ? [...index.tokens].indexOf(t) + 1 : 0);
  const set = (frame: number, token: string, p: number) => {
    const o = (frame - post.startFrame) * post.k;
    if (o < 0 || o + post.k > ids.length) return;
    ids[o] = idOf(token);
    probs[o] = p;
    ids[o + 1] = 0;
    probs[o + 1] = Math.max(0, 1 - p);
    for (let c = 2; c < post.k; c++) { ids[o + c] = 0; probs[o + c] = 0; }
  };
  alter?.(set, evidence.phonemes);
  return { ...painted, posteriors: { ...post, ids, probs } };
}

/** Pausal previous word, a real pause, then إن, then وقال with its و gone. */
function seamEdits(extra: Edit[] = [], dropWaw = true): Edit[] {
  return [
    { kind: "del", at: at(prev), count: phones(prev).length },
    { kind: "ins", at: at(prev), text: iso(prev) },
    { kind: "gap", at: at(next), frames: 20 },
    { kind: "ins", at: at(next), text: "ءِننن" },
    ...(dropWaw ? [{ kind: "del" as const, at: at(next), count: 1 }] : []),
    ...extra,
  ];
}

function run(edits: Edit[], frame?: (set: (frame: number, token: string, p: number) => void, phones: DecodedPhoneme[]) => void, override?: (j: number) => [string, number][] | undefined) {
  const fixture = syntheticFixture(index, { id: "inner-waqf", from, to, edits });
  const checker = createChecker(index, { passage: { from, to }, phrases });
  for (const evidence of fixture.evidence) checker.push(paint(evidence, frame, override), []);
  checker.end();
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

const added = [`added_word ${key(next)}`];

describe("inner waqf seam", () => {
  it("counts the waqf signs and the proclitic onsets", () => {
    expect(waqfMarked(wi(prev))).toBe(true);
    expect(waqfMarked(wi(next))).toBe(false);
    expect(procliticOnset(wi(next))).toBe(true);
    expect(procliticOnset(wi(w(9, 65, 5)))).toBe(false);
    let n = 0;
    for (let i = 0; i < index.words.length; i++) if (waqfMarked(i)) n++;
    expect(n).toBe(4257);
  });

  it("confirms a spelled word when the next word's only gap is its proclitic", () => {
    expect(run(seamEdits())).toEqual(added);
  });

  it("confirms the same word when the next visit is already complete", () => {
    expect(run(seamEdits([], false))).toEqual(added);
  });

  it("stays quiet when the pause is shorter than the detector's minimum", () => {
    const edits = seamEdits().map((e) => e.kind === "gap" ? { ...e, frames: 4 } : e);
    expect(run(edits)).toEqual([]);
  });

  it("still confirms when the frame after the previous token is that token's own class", () => {
    expect(run(seamEdits(), (set, ph) => {
      const i = ph.findIndex((p, k) => k > 0 && p.start - ph[k - 1]!.end - 1 >= 15);
      set(ph[i - 1]!.end + 1, ph[i - 1]!.p, 0.7);
    })).toEqual(added);
  });

  it("stays quiet when a later gap frame is another consonant", () => {
    expect(run(seamEdits(), (set, ph) => {
      const i = ph.findIndex((p, k) => k > 0 && p.start - ph[k - 1]!.end - 1 >= 15);
      set(ph[i - 1]!.end + 8, "ب", 0.8);
    })).toEqual([]);
  });

  it("stays quiet when the proclitic's frames hold another consonant", () => {
    const fixture = syntheticFixture(index, { id: "residue", from, to, edits: seamEdits() });
    const ph = fixture.evidence.flatMap((e) => e.phonemes);
    const fatha = ph.findIndex((p, i) => p.p === "َ" && ph[i - 1]?.p === "ن" && ph[i + 1]?.p === "ق");
    expect(fatha).toBeGreaterThan(0);
    expect(run(seamEdits(), undefined, (j) => j === fatha ? [["ب", 0.8], ["", 0.15]] : undefined)).toEqual([]);
  });

  it("does not treat a missing stem consonant as an erased proclitic", () => {
    const edits: Edit[] = [
      { kind: "del", at: at(prev), count: phones(prev).length },
      { kind: "ins", at: at(prev), text: iso(prev) },
      { kind: "gap", at: at(next), frames: 20 },
      { kind: "ins", at: at(next), text: "ءِننن" },
      { kind: "del", at: at(next) + 2, count: 1 },
    ];
    expect(run(edits)).toEqual([]);
  });

  it("stays quiet for a neighbour, a boundary with no sign, and a left word that is not the pausal form", () => {
    const neighbour: Edit[] = [
      { kind: "del", at: at(prev), count: phones(prev).length },
      { kind: "ins", at: at(prev), text: iso(prev) },
      { kind: "gap", at: at(next), frames: 20 },
      { kind: "ins", at: at(next), text: phones(w(38, 4, 8)) },
      { kind: "del", at: at(next), count: 1 },
    ];
    const unmarked = w(38, 4, 8);
    expect(waqfMarked(wi(w(38, 4, 7)))).toBe(false);
    const noSign: Edit[] = [
      { kind: "gap", at: at(unmarked), frames: 20 },
      { kind: "ins", at: at(unmarked), text: "ءِننن" },
    ];
    const notPausal: Edit[] = [
      { kind: "del", at: at(prev), count: phones(prev).length },
      { kind: "ins", at: at(prev), text: "غَظَضَقَ" },
      { kind: "gap", at: at(next), frames: 20 },
      { kind: "ins", at: at(next), text: "ءِننن" },
      { kind: "del", at: at(next), count: 1 },
    ];
    expect(run(neighbour)).toEqual([]);
    expect(run(noSign)).toEqual([]);
    expect(run(notPausal)).toEqual([]);
  });
});
