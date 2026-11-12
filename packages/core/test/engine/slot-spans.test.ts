import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replay, replayChecked, syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { PositionEvent, WordRef } from "../../src/engine/events.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { continuationDecision, longestSpanAt, slotSpans, wordSubstringCost } from "../../src/engine/slot-spans.js";
import { commitMargin, MISFIT_MAX_COST } from "../../src/engine/tracker.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const at = (r: WordRef) => index.words.findIndex((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word);
const phone = (r: WordRef) => index.phonemes.slice(index.words[at(r)].start, index.words[at(r)].end);
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const margin = (n: number) => commitMargin(n);

describe("slot-differing spans", () => {
  const spans = slotSpans(index);
  const slotOf = (span: (typeof spans)[number]) => span.starts.flatMap((st) => span.slots.map((off) => st + off));

  it("marks the Noah and Lut name slots and not an exact duplicate", () => {
    const noah = at(w(26, 105, 3));
    const lut = at(w(26, 160, 3));
    const span = spans.find((s) => slotOf(s).includes(noah) && slotOf(s).includes(lut));
    expect(span).toBeTruthy();
    const slots = slotOf(span!);
    expect(slots).toContain(at(w(26, 106, 5)));
    expect(slots).toContain(at(w(26, 161, 5)));
    expect(span!.length).toBeGreaterThanOrEqual(4);
    const allSlots = new Set(spans.flatMap(slotOf));
    expect(allSlots.has(at(w(26, 8, 1)))).toBe(false);
    expect(allSlots.has(at(w(26, 190, 1)))).toBe(false);
    expect(allSlots.has(at(w(26, 190, 4)))).toBe(false);
  });

  it("takes the longest span that covers a word", () => {
    const span = longestSpanAt(spans, at(w(26, 106, 5)));
    expect(span!.length).toBeGreaterThanOrEqual(30);
    expect(span!.starts.some((st) => index.words[st].ayah === 104)).toBe(true);
  });
});

describe("continuation word cost", () => {
  const fat = phone(w(26, 110, 1));
  const ata = phone(w(26, 165, 1));

  it("matches a word to itself at no cost", () => {
    expect(wordSubstringCost(ata, ata)).toMatchObject({ edits: 0, phones: ata.length });
  });

  it("gives the Lut word to Lut and the Noah word to Noah", () => {
    const candidates = [{ id: 110, phonemes: fat }, { id: 165, phonemes: ata }];
    expect(continuationDecision(ata, candidates, 110, margin, MISFIT_MAX_COST)?.id).toBe(165);
    expect(continuationDecision(fat, candidates, 110, margin, MISFIT_MAX_COST)?.id).toBe(110);
    expect(continuationDecision(ata, candidates, 165, margin, MISFIT_MAX_COST)?.id).toBe(165);
  });
});

describe("tracker continuation", () => {
  const positions = (events: { type: string }[]) => events.filter((e): e is PositionEvent => e.type === "position");

  it("stays on Noah when Noah's own continuation is what was heard", () => {
    const events = positions(replay(index, syntheticFixture(index, { id: "noah", from: w(26, 104, 1), to: w(26, 111, 1), chunkSize: 12 })));
    const moved = events.filter((e) => e.relocated && key(e.relocated.from) !== key(e.relocated.to));
    expect(moved).toEqual([]);
    expect(events.at(-1)!.observed[0].word.ayah).toBeGreaterThanOrEqual(110);
    expect(events.at(-1)!.observed[0].word.ayah).toBeLessThan(159);
  });

  it("moves a Noah lock onto Lut when the heard continuation is Lut's", () => {
    const from = w(26, 104, 1);
    const to = w(26, 110, 3);
    const at110 = at(w(26, 110, 1)) >= 0 ? index.words[at(w(26, 110, 1))].start - index.words[at(from)].start : 0;
    const len = index.words[at(w(26, 110, 3))].end - index.words[at(w(26, 110, 1))].start;
    const edits: Edit[] = [
      { kind: "del", at: at110, count: len },
      { kind: "ins", at: at110, text: phone(w(26, 165, 1)) },
    ];
    const events = positions(replay(index, syntheticFixture(index, { id: "lut-cont", from, to, edits, chunkSize: 12 })));
    const moved = events.filter((e) => e.relocated && key(e.relocated.from) !== key(e.relocated.to));
    expect(moved.length).toBeGreaterThan(0);
    expect(moved[0].relocated!.to).toMatchObject({ surah: 26, ayah: 159 });
    expect(moved[0].observed[0].word.ayah).toBeGreaterThanOrEqual(165);
  });

  it("does not relocate an exact duplicate refrain", () => {
    const events = positions(replay(index, syntheticFixture(index, { id: "dup", from: w(26, 8, 1), to: w(26, 9, 5), chunkSize: 12 })));
    expect(events.filter((e) => e.relocated && key(e.relocated.from) !== key(e.relocated.to))).toEqual([]);
  });

  it("does not resolve a refrain once the first lock has heard a word outside every slot span", () => {
    const spans = slotSpans(index);
    const noah = at(w(26, 105, 3));
    const span = longestSpanAt(spans, noah)!;
    const member = span.starts.find((st) => noah >= st && noah < st + span.length)!;
    let wi = member;
    let run = 0;
    while (wi > 0 && run < 10) {
      wi--;
      run = longestSpanAt(spans, wi) ? 0 : run + 1;
    }
    expect(longestSpanAt(spans, wi)).toBeNull();
    const origin = index.words[wi].start;
    const cont = at(w(26, 110, 1));
    const edits: Edit[] = [
      { kind: "del", at: index.words[cont].start - origin, count: index.words[at(w(26, 110, 3))].end - index.words[cont].start },
      { kind: "ins", at: index.words[cont].start - origin, text: phone(w(26, 165, 1)) },
    ];
    const from = index.words[wi];
    const events = positions(replay(index, syntheticFixture(index, {
      id: "after-unique", from: { surah: from.surah, ayah: from.ayah, word: from.word }, to: w(26, 110, 3), edits, chunkSize: 12,
    })));
    const first = events.find((e) => e.state === "locked" && e.observed[0]);
    expect(first).toBeTruthy();
    expect(longestSpanAt(spans, at(first!.observed[0].word))).toBeNull();
    expect(events.some((e) => e.spanHold)).toBe(false);
    expect(events.filter((e) => e.relocated && key(e.relocated.from) !== key(e.relocated.to))).toEqual([]);
  });
});

describe("word-1 surplus on the locked-step window", () => {
  it("confirms a spelled addition inside a span the first lock is still holding", () => {
    const spans = slotSpans(index);
    const noah = at(w(26, 105, 3));
    const span = longestSpanAt(spans, noah)!;
    const member = span.starts.find((st) => noah >= st && noah < st + span.length)!;
    let seam = -1;
    for (let wi = member + 1; wi < member + span.length; wi++) {
      const word = index.words[wi];
      if (word.word !== 1 || span.slots.includes(wi - member)) continue;
      const prev = index.words[wi - 1];
      if (prev?.surah === word.surah && prev.ayah === word.ayah - 1) { seam = wi; break; }
    }
    expect(seam).toBeGreaterThan(0);
    const fromW = index.words[member];
    const toW = index.words[Math.min(index.words.length - 1, member + span.length + 6)];
    const edits: Edit[] = [{ kind: "ins", at: index.words[seam].start - fromW.start, text: phone(w(2, 118, 15)) }];
    const positions = replay(index, syntheticFixture(index, {
      id: "span-surplus-track",
      from: { surah: fromW.surah, ayah: fromW.ayah, word: fromW.word },
      to: { surah: toW.surah, ayah: toW.ayah, word: toW.word },
      edits, chunkSize: 8,
    })).filter((e): e is PositionEvent => e.type === "position");
    expect(positions.some((e) => e.spanHold)).toBe(true);
    const mistakes = replayChecked(index, syntheticFixture(index, {
      id: "span-surplus",
      from: { surah: fromW.surah, ayah: fromW.ayah, word: fromW.word },
      to: { surah: toW.surah, ayah: toW.ayah, word: toW.word },
      edits, chunkSize: 8,
    })).filter((e) => e.state === "confirmed" && e.errorClass === "added_word");
    const loc = index.words[seam];
    expect(mistakes.map((e) => key(e.location))).toContain(key({ surah: loc.surah, ayah: loc.ayah, word: loc.word }));
  });
});
