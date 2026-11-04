import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { trackerInput } from "../src/tracker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { PhonemeEvidence, SeamEvidence } from "../src/evidence.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 44, ayah: 9, word: 1 }, to = { surah: 44, ayah: 11, word: 5 }, seam = { surah: 44, ayah: 10, word: 1 };
const find = (r: WordRef) => index.words.find(w => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const said = text({ surah: 2, ayah: 118, word: 15 });
const at = find(seam).start - find(from).start;
const neutral: [string, number][] = ["", ...index.tokens].map(p => [p, 1 / (index.tokens.size + 1)]);

function evidence() {
  const teacher = withPosteriors(index, syntheticFixture(index, { id: "seam-ear", from, to,
    edits: [{ kind: "ins", at, text: said }] }).evidence[0], () => neutral, neutral.length);
  const insertStart = at * 3, insertEnd = (at + said.length) * 3;
  const main = { ...teacher, phonemes: teacher.phonemes.filter(p => p.start < insertStart || p.start >= insertEnd) };
  const span: SeamEvidence = { word: seam, fromFrame: Math.max(0, insertStart - 30), toFrame: insertEnd + text(seam).length * 3 + 30,
    phonemes: teacher.phonemes.filter(p => p.start >= Math.max(0, insertStart - 30) && p.start < insertEnd + text(seam).length * 3 + 30),
    posteriors: teacher.posteriors! };
  return { main, span };
}
function run(e: PhonemeEvidence) {
  const checker = createChecker(index, { passage: { from, to } });
  const events = [...checker.push(e, []), ...checker.end()];
  return { events, mistakes: checker.mistakes() };
}

describe("seam-only second-ear additions", () => {
  it("recovers a lexical addition without changing the main decode, tracker input or other warning classes", () => {
    const { main, span } = evidence();
    const base = run(main);
    const enriched = { ...main, phonemes: main.phonemes.map((p, i) => i ? p : { ...p, seams: [span] }) };
    expect(trackerInput(enriched)).toEqual(trackerInput(main));
    const ear = run(enriched);
    expect(base.mistakes).toEqual([]);
    expect(ear.mistakes.map(e => [e.errorClass, e.location])).toEqual([["added_word", seam]]);
    expect(ear.events.filter(e => e.errorClass !== "added_word")).toEqual(base.events.filter(e => e.errorClass !== "added_word"));
    expect(main.phonemes.some(p => p.seams)).toBe(false);
  });

  it("rejects second-ear additions belonging to a different seam and keeps the folded neighbour guard", () => {
    const { main, span } = evidence();
    const wrongSite = { ...span, word: { ...seam, word: 2 } };
    expect(run({ ...main, phonemes: main.phonemes.map((p, i) => i ? p : { ...p, seams: [wrongSite] }) }).mistakes).toEqual([]);
    const clean = withPosteriors(index, syntheticFixture(index, { id: "repeat-ear", from, to,
      edits: [{ kind: "ins", at, text: text(seam) }] }).evidence[0], () => neutral, neutral.length);
    const repeat = { ...span, phonemes: clean.phonemes.filter(p => p.start >= span.fromFrame && p.start < span.toFrame), posteriors: clean.posteriors! };
    const bare = { ...clean, phonemes: clean.phonemes.filter(p => p.start < at * 3 || p.start >= (at + text(seam).length) * 3) };
    expect(run({ ...bare, phonemes: bare.phonemes.map((p, i) => i ? p : { ...p, seams: [repeat] }) }).mistakes).toEqual([]);
  });

  it("observes overlapping crops once, with the same ownership for tokens and posterior rows", () => {
    const { main, span } = evidence();
    const enriched = (seams: SeamEvidence[]) => ({ ...main, phonemes: main.phonemes.map((p, i) => i ? p : { ...p, seams }) });
    expect(run(enriched([span, structuredClone(span)]))).toEqual(run(enriched([span])));
  });
});
