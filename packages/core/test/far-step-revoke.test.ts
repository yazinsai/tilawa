import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../src/index.js";
import type { EngineEvent, PositionEvent, ReferenceAsset, WordRef } from "../src/index.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const at = (from: WordRef, word: WordRef) => find(word).start - find(from).start;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const positions = (events: EngineEvent[]) => events.filter((e): e is PositionEvent => e.type === "position");

const stream = (from: WordRef, to: WordRef, edits: Edit[] = []) => {
  const f = syntheticFixture(index, { id: "far", from, to, edits, chunkSize: 4 });
  const t = createTracker(index);
  return positions([...f.evidence.flatMap((e) => t.push(e)), ...t.end()]);
};

const lockedWords = (events: PositionEvent[]) => {
  const out: string[] = [];
  for (const e of events) if (e.state === "locked" && key(e.observed[0].word) !== out.at(-1)) out.push(key(e.observed[0].word));
  return out;
};

describe("an in-lock step that leaves words unread", () => {
  const from = w(54, 33, 1);
  const to = w(54, 38, 5);
  const drop = find(w(54, 36, 4));

  it("rebinds the landing to the next word when what follows stays in order", () => {
    const events = stream(from, to, [{ kind: "del", at: at(from, w(54, 36, 4)), count: drop.end - drop.start }]);
    const words = lockedWords(events);
    const near = words.indexOf("54:36:5");
    const cont = words.indexOf("54:37:1");
    expect(near).toBeGreaterThanOrEqual(0);
    expect(cont).toBeGreaterThan(near);
    const jumped = events.some((e) => e.state === "locked" && key(e.observed[0].span.start) === "54:37:9" && key(e.observed[0].word) === "54:37:9");
    expect(jumped).toBe(false);
  });

  it("leaves an in-order reading on its own run", () => {
    const words = lockedWords(stream(from, to));
    expect(words.indexOf("54:36:4")).toBeGreaterThanOrEqual(0);
    expect(words.indexOf("54:36:5")).toBe(words.indexOf("54:36:4") + 1);
    expect(words.indexOf("54:37:1")).toBe(words.indexOf("54:36:5") + 1);
  });
});
