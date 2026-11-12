import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { LocalPosition } from "../src/local.js";
import type { TokenEditModel } from "../src/heldout-token.js";
import type { ReferenceIndex } from "../src/reference.js";

const phonemes = "ببتتثثججححخخددذذررززسس";
const words = Array.from({ length: phonemes.length / 2 }, (_, i) => ({ surah: 1, ayah: 1, word: i + 1, start: i * 2, end: i * 2 + 2 }));
const index: ReferenceIndex = { version: "synthetic", inventoryId: "synthetic", tokens: new Set(phonemes), phonemes, words,
  gramSize: 4, grams: new Map(), wordAt: offset => Math.min(words.length - 1, Math.floor(Math.max(0, offset) / 2)) };
const phones = [...index.tokens];
const model: TokenEditModel = { phones, insertion: 0.1, inserted: phones.map(() => 1 / phones.length), block: 2,
  rows: phones.map(p => [...phones.map(q => p === q ? 0.8 : 0.1 / (phones.length - 1)), 0.1]) };
const start = { offset: 0, origin: 0, edits: 0 };

describe("local predictive validation", () => {
  it("leaves ordinary reading and audit mode on the original weighted path", () => {
    const plain = createLocalSearch(index, start), active = createLocalSearch(index, start, { model });
    for (const p of phonemes) {
      plain.step(p); active.step(p);
      expect(active.position()).toEqual(plain.position());
    }
    const original = createLocalSearch(index, start), audit = createLocalSearch(index, start, { model, audit: true });
    for (const p of "ببتتححخخدد") {
      original.step(p); audit.step(p);
      expect(audit.position()).toEqual(original.position());
    }
  });

  it("restores validation history along with the DP state", () => {
    const search = createLocalSearch(index, start, { model });
    for (const p of "ببتت") search.step(p);
    const saved = search.save();
    const follow = () => {
      const positions: LocalPosition[] = [];
      for (const p of "ححخخدد") { search.step(p); positions.push(search.position()); }
      return positions;
    };
    const first = follow();
    search.restore(saved);
    expect(follow()).toEqual(first);
  });

  it("continues the existing local path when a proposed restart cannot be validated", () => {
    const original = createLocalSearch(index, start);
    const guarded = createLocalSearch(index, start, { model: { ...model, block: 16 } });
    for (const p of "ببتتححخخدد") { original.step(p); guarded.step(p); }
    expect(original.position().origin).toBe(8);
    expect(guarded.position().origin).toBe(0);
  });

  it("owns model arrays and caps comparisons at the existing recent-cost horizon", () => {
    const supplied = { ...model, rows: model.rows.map(r => [...r]), inserted: [...model.inserted], phones: [...model.phones] };
    const counts: number[] = [];
    const a = createLocalSearch(index, start, { model: supplied, audit: true, onComparison: c => counts.push(c.heard.length) });
    const b = createLocalSearch(index, start, { model, audit: true });
    supplied.rows[0][0] = 0;
    supplied.inserted[0] = 0;
    for (const p of "ببتتثثججححخخددذذررححخخ") {
      a.step(p); b.step(p);
      expect(a.position()).toEqual(b.position());
    }
    expect(counts.length).toBeGreaterThan(0);
    expect(Math.max(...counts)).toBeLessThanOrEqual(16);
  });
});
