import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { DecodedPhoneme, PhraseAsset, PositionEvent, ReferenceAsset, TrackerOptions, WordRef } from "../../src/engine/index.js";
import { syntheticFixture } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;
const phrase = (name: string) => phrases.phrases.find((p) => p.name === name)!.p;
const basmala = asset.ayahs.find((x) => x.s === 1 && x.a === 1)!.p;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;

type Part = { recite: [WordRef, WordRef] } | { say: string };

/** Recite and say parts back to back, a short pause between them, streamed one second at a time. */
const run = (parts: Part[], options: TrackerOptions = {}) => {
  const phonemes: DecodedPhoneme[] = [];
  const starts: number[] = [];
  let frame = 0;
  const emit = (p: string) => {
    phonemes.push({ p, start: frame, end: frame + 1, conf: 0.99 });
    frame += 3;
  };
  for (const part of parts) {
    starts.push(phonemes.length);
    if ("say" in part) for (const p of part.say) emit(p);
    else {
      const f = syntheticFixture(index, { id: "p", from: part.recite[0], to: part.recite[1] });
      for (const ph of f.evidence[0].phonemes) emit(ph.p);
    }
    frame += 25;
  }
  const t = createTracker(index, { phrases, ...options });
  const events: PositionEvent[] = [];
  for (let i = 0; i < phonemes.length; i += 16) {
    for (const e of t.push({ inventory: index.inventoryId, frameMs: 20, phonemes: phonemes.slice(i, i + 16) })) if (e.type === "position") events.push(e);
  }
  for (const e of t.end()) if (e.type === "position") events.push(e);
  return { events, starts, phonemes };
};
const locked = (events: PositionEvent[]) => events.filter((e) => e.state === "locked");

describe("isti'adha and basmala", () => {
  it("never lock elsewhere, and al-Baqara locks once 2:2 distinguishes it", () => {
    const { events } = run([{ say: phrase("istiadha") }, { say: basmala }, { recite: [w(2, 1, 1), w(2, 2, lastWord(2, 2))] }]);
    for (const e of locked(events)) {
      const at = `${e.observed[0].word.surah}:${e.observed[0].word.ayah}`;
      expect(["2:67", "27:30", "27:31", "3:1", "3:2"]).not.toContain(at);
    }
    expect(events.at(-1)!.state).toBe("locked");
    expect(events.at(-1)!.observed[0].word).toEqual(w(2, 2, lastWord(2, 2)));
  });
});

describe("takbir", () => {
  it("before al-Fatiha doesn't delay the lock past 1:3", () => {
    const { events } = run([{ say: phrase("takbir") }, { recite: [w(1, 2, 1), w(1, 4, 3)] }]);
    const first = locked(events)[0];
    expect(first.observed[0].word.surah).toBe(1);
    // 1:3 has two words: the lock comes by its end.
    const order = (r: WordRef) => index.words.findIndex((x) => key(x) === key(r));
    expect(order(first.observed[0].word)).toBeLessThanOrEqual(order(w(1, 3, 2)));
  });
});

describe("al-Fatiha after a takbir", () => {
  it("doesn't override the audio when another al-hamd opening is recited", () => {
    for (const [s, words] of [[6, 1], [35, 1], [18, 1]] as const) {
      const end = w(s, words, lastWord(s, words));
      for (const parts of [[{ recite: [w(s, 1, 1), end] as [WordRef, WordRef] }], [{ say: phrase("takbir") }, { recite: [w(s, 1, 1), end] as [WordRef, WordRef] }]]) {
        const { events } = run(parts);
        for (const e of locked(events)) expect(e.observed[0].word.surah, `${s} after ${parts.length > 1 ? "takbir" : "nothing"}`).toBe(s);
        expect(events.at(-1)!.observed[0].word).toEqual(end);
      }
    }
  });
});

describe("Alif Lam Mim twice", () => {
  it("stays ambiguous and doesn't lock on 3:1–2", () => {
    const alm = asset.ayahs.find((x) => x.s === 2 && x.a === 1)!.p;
    const { events } = run([{ say: basmala }, { say: alm }, { say: alm }]);
    expect(locked(events)).toEqual([]);
    expect(events.at(-1)!.state).toBe("ambiguous");
  });
});

describe("amin", () => {
  it("after 1:7 doesn't move the bookmark", () => {
    const end = w(1, 7, lastWord(1, 7));
    const { events } = run([{ recite: [w(1, 2, 1), end] }, { say: phrase("amin") }]);
    expect(events.at(-1)!.observed[0].word).toEqual(end);
    expect(events.at(-1)!.expected).toEqual(end);
  });
});

describe("surah transitions", () => {
  it("follow a basmala into the next surah with no jump or flicker back", () => {
    const { events } = run([
      { recite: [w(112, 1, 1), w(112, 4, lastWord(112, 4))] },
      { say: basmala },
      { recite: [w(113, 1, 1), w(113, 5, lastWord(113, 5))] },
      { say: basmala },
      { recite: [w(114, 1, 1), w(114, 6, lastWord(114, 6))] },
    ]);
    const surahs = locked(events).map((e) => e.observed[0].word.surah);
    expect(surahs.every((s) => s >= 112)).toBe(true);
    // Surahs only ever go forward.
    expect(surahs).toEqual([...surahs].sort((a, b) => a - b));
    expect(surahs.at(-1)).toBe(114);
    expect(events.at(-1)!.observed[0].word).toEqual(w(114, 6, lastWord(114, 6)));
  });
});

describe("relabel", () => {
  it("a single-push clip gives the same events with or without phrases", () => {
    const f = syntheticFixture(index, { id: "r", from: w(2, 255, 1), to: w(2, 255, 12) });
    const once = (o: TrackerOptions) => {
      const t = createTracker(index, o);
      return [...t.push(f.evidence[0]), ...t.end()];
    };
    const withIstiadha = { ...f.evidence[0], phonemes: [...[...phrase("istiadha")].map((p, i) => ({ p, start: i * 3, end: i * 3 + 1, conf: 0.99 })), ...f.evidence[0].phonemes.map((ph) => ({ ...ph, start: ph.start + 200, end: ph.end + 200 }))] };
    const one = (o: TrackerOptions) => {
      const t = createTracker(index, o);
      return [...t.push(withIstiadha), ...t.end()];
    };
    expect(once({ phrases })).toEqual(once({}));
    expect(one({ phrases })).toEqual(one({}));
  });
});
