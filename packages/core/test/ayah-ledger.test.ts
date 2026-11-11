import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { absenceMargin } from "../src/ayah-ledger.js";
import { createChecker } from "../src/checker.js";
import { PosteriorFrames } from "../src/ctc.js";
import { withPosteriors } from "../src/fixtures/index.js";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { DiscontinuityIdentity, PositionEvent, WordRef } from "../src/events.js";
import { discontinuityIdentity } from "../src/transition-events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const words = (surah: number, ayah: number) => index.words.filter((x) => x.surah === surah && x.ayah === ayah);
const phones = (surah: number, ayah: number) => {
  const ws = words(surah, ayah);
  return index.phonemes.slice(ws[0].start, ws.at(-1)!.end);
};
const idOf = new Map([...index.tokens].map((t, i) => [t, i + 1]));

/** Each phone occupies `per` frames, peaked on itself. `blur` weakens one ayah's frames. */
function framesOf(parts: { phones: string; p: number }[], per = 3): PosteriorFrames {
  const frames = new PosteriorFrames(index.tokens.size);
  const k = 4;
  const ids: number[] = [];
  const probs: number[] = [];
  for (const part of parts) {
    for (const phone of part.phones) {
      const peak = idOf.get(phone) ?? 1;
      for (let f = 0; f < per; f++) {
        ids.push(peak, 0, peak === 1 ? 2 : 1, peak === 3 ? 2 : 3);
        probs.push(part.p, Math.max(0.02, 0.92 - part.p), 0.01, 0.01);
      }
    }
  }
  frames.append({ k, startFrame: 0, nFrames: ids.length / k, ids, probs });
  return frames;
}

const left = phones(55, 28);
const mid = phones(55, 29);
const right = phones(55, 30);

function ayahGroups(text: string): string[] {
  return [text];
}

describe("closed read-versus-skip fit", () => {
  it("matches a plain forward sum when no repeat arc is used", () => {
    const frames = framesOf([{ phones: left, p: 0.9 }]);
    const ids = [...left].map((p) => idOf.get(p) ?? 0);
    expect(frames.nll(ids, 0, frames.endFrame, [])).toBe(frames.nll(ids, 0, frames.endFrame));
  });

  it("prefers dropping an ayah the frames never say, by the phrase switch", () => {
    const frames = framesOf([{ phones: left + right, p: 0.9 }]);
    const margin = absenceMargin(frames, (p) => idOf.get(p) ?? 0, [ayahGroups(left), ayahGroups(mid), ayahGroups(right)], 1, 0, frames.endFrame);
    expect(margin).toBeGreaterThanOrEqual(PHRASE_SWITCH);
  });

  it("keeps an ayah the frames do say, including when the channel blurs it", () => {
    const clear = framesOf([{ phones: left + mid + right, p: 0.9 }]);
    const blurred = framesOf([
      { phones: left, p: 0.9 },
      { phones: mid, p: 0.42 },
      { phones: right, p: 0.9 },
    ]);
    const id = (p: string) => idOf.get(p) ?? 0;
    const ayahs = [ayahGroups(left), ayahGroups(mid), ayahGroups(right)];
    expect(absenceMargin(clear, id, ayahs, 1, 0, clear.endFrame)!).toBeLessThan(PHRASE_SWITCH);
    expect(absenceMargin(blurred, id, ayahs, 1, 0, blurred.endFrame)!).toBeLessThan(0);
  });
});

describe("ayah absence between committed locks", () => {
  /** Recite `ayahs` in order, locking each word, with posteriors on the phones actually said. */
  const recite = (ayahs: number[], provenance = false) => {
    const phonemes = [];
    const locks: WordRef[] = [];
    let frame = 0;
    for (const ayah of ayahs) {
      for (const word of words(55, ayah)) {
        const text = index.phonemes.slice(word.start, word.end);
        for (const p of text) {
          phonemes.push({ p, start: frame, end: frame + 2, conf: 0.99 });
          frame += 3;
        }
        locks.push({ surah: 55, ayah, word: word.word });
      }
    }
    const evidence = withPosteriors(index, { inventory: index.inventoryId, frameMs: 20, phonemes });
    const checker = createChecker(index, { passage: { from: w(55, ayahs[0], 1), to: w(55, ayahs.at(-1)!, 1) } });
    const events = [];
    let at = 0;
    let history: DiscontinuityIdentity | null = null;
    let previousAyah = ayahs[0];
    for (const lock of locks) {
      const word = words(55, lock.ayah).find((x) => x.word === lock.word)!;
      const n = word.end - word.start;
      const slice = evidence.phonemes.slice(at, at + n);
      at += n;
      const rows = evidence.posteriors!;
      const f0 = slice[0].start;
      const f1 = slice.at(-1)!.end + 1;
      if (lock.ayah > previousAyah + 1) {
        const source = evidence.phonemes[at - n - 1];
        history = discontinuityIdentity(1, lock.ayah, source, slice[0],
          words(55, previousAyah).at(-1)!.end, word.start, history);
      }
      previousAyah = lock.ayah;
      const pos: PositionEvent = {
        type: "position", v: 1, timeMs: f1 * 20, referenceVersion: index.version, expected: lock, state: "locked",
        observed: [{ word: lock, span: { start: w(55, lock.ayah, 1), end: lock }, confidence: 1, cost: 0, per: 0 }],
        ...(provenance ? { transitionHistory: { generation: 1, head: history } } : {}),
      };
      events.push(...checker.push({
        inventory: evidence.inventory, frameMs: 20, phonemes: slice,
        posteriors: { k: rows.k, startFrame: f0, nFrames: f1 - f0, ids: Array.from(rows.ids).slice(f0 * rows.k, f1 * rows.k), probs: Array.from(rows.probs).slice(f0 * rows.k, f1 * rows.k) },
      }, [pos]));
    }
    events.push(...checker.end());
    return events.filter((e) => e.state === "confirmed").map((e) => `${e.errorClass} ${key(e.location)}`);
  };

  it("warns when the lock jumps a unique ayah the frames do not contain", () => {
    expect(recite([28, 30, 31])).toContain("skipped_passage 55:29:1");
  });

  it("stays quiet when the lock reads that ayah", () => {
    expect(recite([28, 29, 30])).not.toContain("skipped_passage 55:29:1");
  });

  it("keeps a real ledger skip with exact transition provenance and no confirmed word alternative", () => {
    expect(recite([28, 30, 31], true)).toContain("skipped_passage 55:29:1");
    expect(recite([28, 29, 30], true)).not.toContain("skipped_passage 55:29:1");
  });
});

describe("a ledger skip and a seam skip on one unread run", () => {
  /**
   * Recite the listed words of surah 26, locking each one, with posteriors on the phones actually said.
   * `26:131` is a repeated ayah, so a bound word of it is refrain text.
   */
  const recite = (plan: { ayah: number; words?: number[] }[]) => {
    const phonemes = [];
    const locks: WordRef[] = [];
    let frame = 0;
    for (const part of plan) {
      for (const word of words(26, part.ayah)) {
        if (part.words && !part.words.includes(word.word)) continue;
        const text = index.phonemes.slice(word.start, word.end);
        if (!text) continue;
        for (const p of text) {
          phonemes.push({ p, start: frame, end: frame + 2, conf: 0.99 });
          frame += 3;
        }
        locks.push({ surah: 26, ayah: part.ayah, word: word.word });
      }
    }
    const evidence = withPosteriors(index, { inventory: index.inventoryId, frameMs: 20, phonemes });
    const checker = createChecker(index, { passage: { from: w(26, 129, 1), to: w(26, 140, 1) } });
    const events = [];
    let at = 0;
    const spanStart = locks[0];
    for (const lock of locks) {
      const word = words(26, lock.ayah).find((x) => x.word === lock.word)!;
      const n = word.end - word.start;
      const slice = evidence.phonemes.slice(at, at + n);
      at += n;
      const rows = evidence.posteriors!;
      const f0 = slice[0].start;
      const f1 = slice.at(-1)!.end + 1;
      const pos: PositionEvent = {
        type: "position", v: 1, timeMs: f1 * 20, referenceVersion: index.version, expected: lock, state: "locked",
        observed: [{ word: lock, span: { start: spanStart, end: lock }, confidence: 1, cost: 0, per: 0 }],
      };
      events.push(...checker.push({
        inventory: evidence.inventory, frameMs: 20, phonemes: slice,
        posteriors: { k: rows.k, startFrame: f0, nFrames: f1 - f0, ids: Array.from(rows.ids).slice(f0 * rows.k, f1 * rows.k), probs: Array.from(rows.probs).slice(f0 * rows.k, f1 * rows.k) },
      }, [pos]));
    }
    events.push(...checker.end());
    return events.filter((e) => e.state === "confirmed").map((e) => `${e.errorClass} ${key(e.location)}`);
  };

  const skips = (plan: { ayah: number; words?: number[] }[]) => recite(plan).filter((e) => e.startsWith("skipped_passage"));

  it("drops the seam skip that a bound refrain word separates from the ledger ayah", () => {
    // 26:130 unread, then only فاتقوا (26:131:1) is said, then 26:131:2–26:132:1 unread.
    const got = skips([
      { ayah: 129 },
      { ayah: 131, words: [1] },
      { ayah: 132, words: [2, 3, 4, 5] },
      { ayah: 133 },
      { ayah: 134 },
      { ayah: 135 },
    ]);
    expect(got).toContain("skipped_passage 26:130:1");
    expect(got).not.toContain("skipped_passage 26:131:2");
  });

  it("keeps both when a unique unread word separates them", () => {
    // 26:132:1 is unique and unread, between the ledger ayah and the seam at 26:135:2.
    const got = skips([
      { ayah: 129 },
      { ayah: 131 },
      { ayah: 132, words: [2, 3, 4, 5] },
      { ayah: 133 },
      { ayah: 134 },
      { ayah: 135, words: [1] },
      { ayah: 136 },
      { ayah: 137 },
    ]);
    expect(got).toContain("skipped_passage 26:130:1");
    expect(got).toContain("skipped_passage 26:135:2");
  });

  it("stays quiet when those ayahs are read", () => {
    expect(skips([{ ayah: 129 }, { ayah: 130 }, { ayah: 131 }, { ayah: 132 }, { ayah: 133 }])).toEqual([]);
  });
});
