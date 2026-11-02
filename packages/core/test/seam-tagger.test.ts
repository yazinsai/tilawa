import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { withPosteriors, syntheticFixture } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";
import { loadReferenceIndex } from "../src/reference.js";
import { ayahSeamGeoms, OWN_LEFT, OWN_NONE, OWN_REPEAT, OWN_RIGHT, scoreSeam } from "../src/seam-tagger.js";
import type { SeamObs, SeamStep, SeamToken, SeamWeights } from "../src/seam-tagger.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;

function obs(phones: string): SeamObs[] {
  return [...phones].map((p, i) => ({ p, conf: 0.9, start: i * 3, end: i * 3 + 1 }));
}

describe("ayah seam geometry", () => {
  const phones = obs("ابقدهوز");
  const steps: SeamStep[] = [
    { op: "diag", ref: 0, obs: 0 },
    { op: "diag", ref: 1, obs: 1 },
    { op: "ins", obs: 2 },
    { op: "ins", obs: 3 },
    { op: "ins", obs: 4 },
    { op: "diag", ref: 10, obs: 5 },
    { op: "diag", ref: 11, obs: 6 },
  ];
  const base = {
    steps,
    obs: phones,
    wordAt: (r: number) => (r < 5 ? 4 : r >= 10 ? 9 : 20),
    wordLen: () => 2,
    prevWord: (wi: number) => wi - 5,
    isAyahStart: (wi: number) => wi === 9,
    matches: () => true,
    pauseBefore: (i: number) => i === 2,
  };

  it("spans the phones between two complete visits and marks ownership", () => {
    const [g] = ayahSeamGeoms(base);
    expect(g.rightWord).toBe(9);
    expect(g.obsFrom).toBe(2);
    expect(g.obsTo).toBe(5);
    expect(phones.slice(g.obsFrom, g.obsTo).map((p) => p.p).join("")).toBe("قده");
    const span = g.tokens.slice(g.span[0], g.span[1]);
    expect(span.map((t) => t.own)).toEqual([OWN_NONE, OWN_NONE, OWN_NONE]);
    expect(g.tokens[0].own).toBe(OWN_LEFT);
    expect(g.tokens.at(-1)!.own).toBe(OWN_RIGHT);
    expect(span[0].pause).toBe(1);
    expect(span[0].spanLen).toBe(3);
  });

  it("stays quiet when the right-hand visit does not cover its word", () => {
    const short = steps.filter((s) => s.obs !== 6);
    expect(ayahSeamGeoms({ ...base, steps: short })).toEqual([]);
  });

  it("calls a phone aligned to a third word a repeat", () => {
    const withRepeat: SeamStep[] = [
      ...steps.slice(0, 2),
      { op: "diag", ref: 7, obs: 2 },
      { op: "diag", ref: 8, obs: 3 },
      { op: "ins", obs: 4 },
      ...steps.slice(5),
    ];
    const [g] = ayahSeamGeoms({ ...base, steps: withRepeat });
    const span = g.tokens.filter((t) => t.obs >= 2 && t.obs < 5);
    expect(span.map((t) => t.own)).toEqual([OWN_REPEAT, OWN_REPEAT, OWN_NONE]);
  });
});

describe("seam score", () => {
  it("is the mean probability of the added class over the span", () => {
    const d = 4;
    const heads = 2;
    const ff = 8;
    const z = (n: number) => new Array<number>(n).fill(0);
    const eye = () => {
      const w = z(d * d);
      for (let i = 0; i < d; i++) w[i * d + i] = 1;
      return w;
    };
    const layer = {
      ln1g: new Array(d).fill(1), ln1b: z(d), q: eye(), k: eye(), v: eye(), wo: eye(),
      ln2g: new Array(d).fill(1), ln2b: z(d),
      w1: z(d * ff), b1: z(ff), w2: z(ff * d), b2: z(d),
    };
    const weights: SeamWeights = {
      tokens: "ءب", d, heads, ff, maxLen: 8, threshold: 0.5,
      embed: z(2 * d), pos: z(8 * d), featW: z(8 * d), featB: z(d),
      layers: [layer], headW: z(d * 4), headB: [8, 0, 0, 0],
    };
    const tok = (obs: number): SeamToken => ({ id: 0, conf: 1, pause: 0, own: OWN_NONE, step: 0, spanLen: 2, obs });
    const score = scoreSeam([tok(0), tok(1), tok(2)], [1, 3], weights);
    const p = Math.exp(8) / (Math.exp(8) + 3);
    expect(score).toBeCloseTo(p, 6);
  });

  it("does not fire without a model, and an empty span scores zero", () => {
    expect(scoreSeam([], [0, 0], null)).toBe(-1);
    expect(scoreSeam([], [0, 0])).toBe(0);
  });
});

describe("checker seam confirmation", () => {
  it("confirms an ayah-seam addition the support margin rejects when the tagger accepts it", () => {
    const from = w(37, 104, 1);
    const to = w(37, 111, 4);
    const added = w(37, 108, 1);
    const at = (r: WordRef) => index.words.find((x) => key(x) === key(r))!.start - index.words.find((x) => key(x) === key(from))!.start;
    const fixture = syntheticFixture(index, { id: "seam", from, to, edits: [{ kind: "ins", at: at(added), text: "قَلِۦۦلَاا" }] });
    const evidence = fixture.evidence.map((e) => withPosteriors(index, e));
    const run = (seamScore?: (tokens: SeamToken[], span: readonly [number, number]) => number) => {
      const checker = createChecker(index, {
        passage: { from, to },
        policy: { localWindow: { nats: 1e9 }, minConfidence: 0 },
        ...(seamScore ? { seamScore } : {}),
      });
      for (const e of evidence) checker.push(e, []);
      checker.end();
      return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
    };
    expect(run()).not.toContain(`added_word ${key(added)}`);
    expect(run(() => 0)).not.toContain(`added_word ${key(added)}`);
    expect(run(() => 1)).toContain(`added_word ${key(added)}`);
  });
});
