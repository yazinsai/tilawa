import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { CheckerOptions, WrongWordRecheck } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { WrongWordEventEvidence } from "../src/evidence.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const passage = { from: { surah: 2, ayah: 255, word: 1 }, to: { surah: 2, ayah: 256, word: 10 } };
const word = (surah: number, ayah: number, word: number) => index.words.find(w => w.surah === surah && w.ayah === ayah && w.word === word)!;
const first = word(2,255,1), target = word(2,255,23), alt = word(108,1,3);
const fixture = syntheticFixture(index, { id: "event-replay", ...passage, chunkSize: 8, edits: [
  { kind: "del", at: target.start - first.start, count: target.end - target.start },
  { kind: "ins", at: target.start - first.start, text: index.phonemes.slice(alt.start, alt.end) },
] });
const evidence = fixture.evidence.map(e => withPosteriors(index, e));
function run(recheck?: CheckerOptions["wrongWordRecheck"], metadata?: WrongWordEventEvidence) {
  const checker = createChecker(index, { passage, policy: { confirm: { after: "next-word" } }, wrongWordRecheck: recheck });
  const input = structuredClone(evidence);
  if (metadata) input[0].phonemes[0].wrongWordEvents = [metadata];
  const events = input.flatMap(e => checker.push(e, []));
  events.push(...checker.end());
  return { events, mistakes: checker.mistakes() };
}

describe("isolated word event replay", () => {
  it("restores pre-judgement state after multiple divergent replays and publishes only primary events", () => {
    let calls = 0;
    const baseline = run();
    const replayed = run((c, judge) => {
      calls++;
      const original = judge(c.phonemes);
      expect(original).toContain("wrong_word");
      judge([...c.phonemes].reverse());
      judge([]);
      expect(judge(c.phonemes)).toEqual(original);
      return false;
    });
    expect(calls).toBeGreaterThan(0);
    expect(replayed).toEqual(baseline);
    expect(baseline.mistakes.map(e => e.errorClass)).toEqual(["wrong_word"]);
  });

  it("binds serialized vetoes to the primary word, spelling and exact acoustic window", () => {
    let candidate: WrongWordRecheck | undefined;
    run(c => { candidate = c; return false; });
    expect(candidate).toBeDefined();
    const c = candidate!;
    const certificate: WrongWordEventEvidence = { word: c.word, observed: c.observed,
      fromFrame: c.fromFrame, toFrame: c.toFrame, quiet: .6, enlargedQuiet: .55 };
    expect(run(undefined, certificate).mistakes).toEqual([]);
    expect(run(undefined, { ...certificate, fromFrame: c.fromFrame + 1 })).mistakes).toEqual(run().mistakes);
    expect(run(undefined, { ...certificate, enlargedQuiet: .5 })).mistakes).toEqual(run().mistakes);
  });
});
