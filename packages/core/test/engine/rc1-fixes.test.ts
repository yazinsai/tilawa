import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { createTracker } from "../../src/engine/tracker.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { syntheticFixture, withPosteriors } from "../fixtures/synthetic.js";
import type { CheckerEvent, PositionEvent, WordRef } from "../../src/engine/events.js";
import type { DecodedPhoneme } from "../../src/engine/evidence.js";
import type { PhraseAsset, ReferenceAsset } from "../../src/engine/reference.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (s: number, a: number) => asset.ayahs.find((x) => x.s === s && x.a === a)!.w.length;
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;

describe("rc.1 field-bug regression tests", () => {
  it("short-madd isti'adha and chunk-end isti'adha never lock onto 2:67 or 16:98", () => {
    const shortIstiadha = "ءَعُذُبِللَااهِمِنَششَيطَاانِررَجِۦۦم";
    const f = syntheticFixture(index, { id: "a87_18", from: w(87, 18, 1), to: w(87, 18, lastWord(87, 18)) });
    const phonemes: DecodedPhoneme[] = [];
    let frame = 20;
    for (const ch of shortIstiadha) {
      phonemes.push({ p: ch, start: frame, end: frame + 1, conf: 0.99 });
      frame += 3;
    }
    frame += 30;
    for (const ph of f.evidence[0].phonemes) {
      phonemes.push({ p: ph.p, start: frame, end: frame + 1, conf: 0.99 });
      frame += 3;
    }
    const t = createTracker(index, { phrases, acquireMs: 1000 });
    const events: PositionEvent[] = [];
    for (const ph of phonemes) {
      for (const e of t.push({ inventory: index.inventoryId, frameMs: 40, phonemes: [ph], decodedTo: ph.end + 1 })) {
        if (e.type === "position") events.push(e);
      }
    }
    for (const e of t.end()) if (e.type === "position") events.push(e);
    const locks = events.filter((e) => e.state === "locked");
    for (const e of locks) {
      expect(["2:67", "16:98"]).not.toContain(`${e.observed[0].word.surah}:${e.observed[0].word.ayah}`);
    }
    expect(locks[0]?.observed[0].word.surah).toBe(87);
  });

  it("fast written 22:1 with short madd/ghunna does not trigger parallel 2:21 false warning", () => {
    const fast = "يَاءييهَننَاسُتتَقُۥرَببَكُمءِننننَزَلزَلَتَ";
    const phonemes = [...fast].map((p, i) => ({ p, start: i * 2, end: i * 2, conf: 0.96 }));
    const ev = withPosteriors(index, { inventory: index.inventoryId, frameMs: 40, phonemes });
    const checker = createChecker(index, { passage: { from: w(22, 1, 1), to: w(22, 1, 6) }, policy: { confirm: { after: "next-word" } } });
    checker.push(ev, []);
    checker.end();
    expect(checker.mistakes()).toEqual([]);
  });

  it("mutashabihat slip 22:3->22:8->22:9 raises at most one confirmed skipped_passage", () => {
    const a = index.phonemes.slice(find(w(22, 1, 1)).start, find(w(22, 3, 8)).end);
    const b = index.phonemes.slice(find(w(22, 8, 9)).start, find(w(22, 9, 10)).end);
    const phones = [...(a + b)].map((ch, i) => ({ p: ch, start: i * 3, end: i * 3 + 1, conf: 0.96 }));
    const tracker = createTracker(index, { phrases });
    const checker = createChecker(index, { phrases });
    const out: CheckerEvent[] = [];
    for (let i = 0; i < phones.length; i += 8) {
      const ev = { inventory: index.inventoryId, frameMs: 40, phonemes: phones.slice(i, i + 8) };
      out.push(...checker.push(ev, tracker.push(ev)));
    }
    out.push(...checker.end(tracker.end()));
    const skips = out.filter((e) => e.state === "confirmed" && e.errorClass === "skipped_passage");
    expect(skips.length).toBeLessThanOrEqual(1);
  });
});
