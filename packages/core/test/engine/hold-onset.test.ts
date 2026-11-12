import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { CheckerEvent, PhraseAsset, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const at = (from: WordRef, word: WordRef) => find(word).start - find(from).start;

const positions = (from: WordRef, to: WordRef, edits: Edit[] = []) => {
  const f = syntheticFixture(index, { id: "h", from, to, edits, chunkSize: 8 });
  const t = createTracker(index, { phrases });
  return [...f.evidence.flatMap((e) => t.push(e)), ...t.end()].filter((e): e is PositionEvent => e.type === "position");
};

describe("misfit hold", () => {
  it("downgrades an issue that was already open and keeps delivering the continuation", () => {
    const from = w(2, 255, 1);
    const target = w(2, 255, 23);
    const alien = text(w(108, 1, 3));
    const f = syntheticFixture(index, {
      id: "h",
      from,
      to: w(2, 256, 10),
      chunkSize: 8,
      edits: [
        { kind: "del", at: at(from, target), count: text(target).length },
        { kind: "ins", at: at(from, target), text: alien },
        { kind: "noise", at: at(from, w(2, 255, 24)), length: 60, seed: 7 },
      ],
    });
    const tracker = createTracker(index, { phrases });
    const checker = createChecker(index, { phrases, policy: { confirm: { after: "delay", ms: 3000 } } });
    const pos: PositionEvent[] = [];
    const out: CheckerEvent[] = [];
    for (const e of f.evidence) {
      const p = tracker.push(e);
      pos.push(...p.filter((x): x is PositionEvent => x.type === "position"));
      out.push(...checker.push(e, p));
    }
    const end = tracker.end();
    pos.push(...end.filter((x): x is PositionEvent => x.type === "position"));
    out.push(...checker.end(end));
    expect(pos.some((e) => e.state === "lost")).toBe(false);
    expect(pos.at(-1)!.state).toBe("locked");
    expect(pos.at(-1)!.observed[0].word.surah).toBe(2);
    const wrong = out.filter((e) => key(e.location) === "2:255:23");
    expect(wrong.some((e) => e.state === "confirmed")).toBe(false);
    expect(wrong.some((e) => e.state === "uncertain")).toBe(true);
  });
});

describe("first-lock onset veto", () => {
  it("does not lock 26:126 then 26:127 onto 25:57", () => {
    const basmala = index.phonemes.slice(find(w(1, 1, 1)).start, find(w(1, 1, 4)).end);
    const lead = phrases.phrases.find((p) => p.name === "istiadha")!.p + basmala;
    for (const prefix of ["", lead]) {
      const edits: Edit[] = prefix ? [{ kind: "ins", at: 0, text: prefix }] : [];
      const events = positions(w(26, 126, 1), w(26, 129, 1), edits);
      const label = prefix ? "with lead-in" : "bare";
      const locked = events.filter((e) => e.state === "locked" && e.observed[0]);
      expect(locked.length, label).toBeGreaterThan(0);
      expect(locked.some((e) => e.observed[0].word.surah === 25), label).toBe(false);
      expect(locked.at(-1)!.observed[0].word.surah, label).toBe(26);
      expect(locked.at(-1)!.observed[0].word.ayah, label).toBeGreaterThanOrEqual(126);
    }
  });

  it("keeps a clean ayah-start opening on its own surah", () => {
    const events = positions(w(2, 255, 1), w(2, 255, 20));
    const first = events.find((e) => e.state === "locked");
    expect(first).toBeTruthy();
    expect(first!.observed[0].word).toMatchObject({ surah: 2, ayah: 255 });
    expect(first!.observed[0].span.start.surah).toBe(2);
    expect(events.at(-1)!.observed[0].word).toEqual(w(2, 255, 20));
  });
});
