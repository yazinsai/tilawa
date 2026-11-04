import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replayChecked, syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { CheckerEvent, Edit, ReferenceAsset, WordRef } from "../src/index.js";
import { loadReferenceIndex } from "../src/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

const from = w(2, 255, 1);
const at = (word: WordRef) => find(word).start - find(from).start;
const target = w(2, 255, 23);

/**
 * Recite 2:255–2:257 with `target` deleted. Phones beside the gap are at 0.4, so greedy clarity is under
 * the gate. `paint` puts the deleted word's own phones into the blank gap; the greedy string still skips it.
 */
const recite = (paint: boolean): CheckerEvent[] => {
  const edits: Edit[] = [{ kind: "del", at: at(target), count: text(target).length }];
  if (paint) edits.push({ kind: "gap", at: at(target), frames: text(target).length * 3 });
  const f = syntheticFixture(index, { id: "c", from, to: w(2, 257, 5), edits, chunkSize: 8 });
  const all = f.evidence.flatMap((x) => x.phonemes);
  const boundary = at(target);
  for (let i = Math.max(0, boundary - 3); i < Math.min(all.length, boundary + 3); i++) all[i]!.conf = 0.4;
  const whole = withPosteriors(index, { ...f.evidence[0]!, phonemes: all });
  if (paint) {
    const post = whole.posteriors!;
    const ids = [...index.tokens];
    const idOf = (t: string) => ids.indexOf(t) + 1;
    let frame = all[boundary - 1]!.end + 1;
    for (const p of text(target)) {
      for (let r = 0; r < 3; r++, frame++) {
        const base = (frame - post.startFrame) * post.k;
        post.ids[base] = idOf(p);
        post.probs[base] = 0.92;
        post.ids[base + 1] = 0;
        post.probs[base + 1] = 0.04;
        for (let c = 2; c < post.k; c++) {
          post.ids[base + c] = (c % ids.length) + 1;
          post.probs[base + c] = 0.01;
        }
      }
    }
  }
  const { k, startFrame, ids, probs } = whole.posteriors!;
  let a = startFrame;
  const evidence = f.evidence.map((x, i) => {
    const b = i + 1 < f.evidence.length ? f.evidence[i + 1]!.phonemes[0]!.start : startFrame + whole.posteriors!.nFrames;
    const rows = {
      k, startFrame: a, nFrames: b - a,
      ids: Array.from(ids).slice((a - startFrame) * k, (b - startFrame) * k),
      probs: Array.from(probs).slice((a - startFrame) * k, (b - startFrame) * k),
    };
    a = b;
    return { ...x, posteriors: rows };
  });
  return replayChecked(index, { ...f, evidence });
};

const atTarget = (events: CheckerEvent[]) => events.filter((e) => e.errorClass === "omitted_word" && key(e.location) === key(target));

describe("uncertain omission, read versus omit", () => {
  it("confirms when the frames never carry the word, even though greedy clarity is low", () => {
    const events = atTarget(recite(false));
    const confirmed = events.find((e) => e.state === "confirmed");
    expect(confirmed).toBeTruthy();
    expect(confirmed!.confidence).toBeLessThan(0.6);
  });

  it("stays uncertain when the gap's frames do carry the word", () => {
    const events = atTarget(recite(true));
    expect(events.some((e) => e.state === "suspected")).toBe(true);
    expect(events.some((e) => e.state === "confirmed")).toBe(false);
    expect(events.some((e) => e.state === "uncertain")).toBe(true);
    expect(events.find((e) => e.state === "uncertain")!.confidence).toBeLessThan(0.6);
  });
});
