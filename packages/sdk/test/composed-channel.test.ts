import { describe, expect, it } from "vitest";
import { PosteriorFrames, type Channel } from "@tilawa/core";
import { ComposedChannel } from "../src/composed-channel.js";

const tokens = [..."بت"];
const ch: Channel = { phones: tokens, index: new Map(tokens.map((p, i) => [p, i])),
  logSub: [[.65, .15], [.1, .75]].map(r => Float64Array.from(r, Math.log)),
  logDel: Float64Array.from([.2, .15], Math.log),
  logIns: Float64Array.from([.06, .04], Math.log), cut: 0 };
const scorer = new ComposedChannel(ch, tokens);
function frames(rows: number[][]) {
  const f = new PosteriorFrames(tokens.length);
  f.append({ startFrame: 0, nFrames: rows.length, k: 3, ids: rows.flatMap(() => [0, 1, 2]), probs: rows.flat() });
  return f;
}

/** Enumerate edit scripts directly, including their terminal loop-exit probability. */
function scripts(w: string, h: string, i = 0, j = 0): number {
  const exit = 1 / 1.1, wr = tokens.indexOf(w[i]), heard = tokens.indexOf(h[j]);
  let p = i === w.length && j === h.length ? exit : 0;
  if (j < h.length) p += exit * Math.exp(ch.logIns[heard]) * scripts(w, h, i, j + 1);
  if (i < w.length) {
    p += exit * Math.exp(ch.logDel[wr]) * scripts(w, h, i + 1, j);
    if (j < h.length) p += exit * Math.exp(ch.logSub[wr][heard]) * scripts(w, h, i + 1, j + 1);
  }
  return p;
}
function strings(max: number): string[] {
  let layer = [""]; const out = [...layer];
  for (let n = 1; n <= max; n++) { layer = layer.flatMap(s => tokens.map(p => s + p)); out.push(...layer); }
  return out;
}

describe("normalized channel composed with CTC", () => {
  it("matches exhaustive edit-script enumeration and assigns unit total string mass", () => {
    for (const w of ["", "ب", "بت", "بب"]) {
      let total = 0;
      for (const h of strings(10)) {
        const p = Math.exp(scorer.stringLogP(w, h)!);
        if (h.length < 4) expect(p).toBeCloseTo(scripts(w, h), 13);
        total += p;
      }
      expect(total).toBeCloseTo(1, 7);
    }
  });

  it("equals the sum over EVERY raw CTC path, including repeats, blanks and deletions", () => {
    const f = frames([[.3, .45, .25], [.4, .4, .2], [.65, .05, .3], [.1, .35, .55]]);
    for (const w of ["", "ب", "بت", "بب", "تبت"]) {
      let expected = 0;
      const enumerate = (t: number, last: number, h: string, p: number) => {
        if (t === 4) { expected += p * scripts(w, h); return; }
        const row = f.row(t);
        for (let id = 0; id < 3; id++) enumerate(t + 1, id, h + (id && id !== last ? tokens[id - 1] : ""), p * Math.exp(row[id]));
      };
      enumerate(0, 0, "", 1);
      expect(Math.exp(scorer.acousticLogP(w, f, 0, 4)!)).toBeCloseTo(expected, 13);
    }
  });

  it("can prefer the complete written pronunciation when the matched greedy channel prefers the rival", () => {
    const f = frames([[.143, .331, .526], [.384, .189, .427], [.601, .367, .032], [.443, .092, .465]]);
    const greedy = scorer.greedy(f, 0, 4);
    expect(greedy).toBe("تت");
    expect(scorer.stringLogP("بب", greedy)!).toBeGreaterThan(scorer.stringLogP("ب", greedy)!);
    expect(scorer.acousticLogP("ب", f, 0, 4)!).toBeGreaterThan(scorer.acousticLogP("بب", f, 0, 4)!);
  });

  it("reduces to ordinary complete CTC with an identity channel", () => {
    const identity = new ComposedChannel({ ...ch, logIns: new Float64Array(2).fill(-Infinity),
      logDel: new Float64Array(2).fill(-Infinity), logSub: [Float64Array.of(0, -Infinity), Float64Array.of(-Infinity, 0)] }, tokens);
    const f = frames([[.2, .7, .1], [.7, .15, .15], [.1, .4, .5]]);
    for (const w of ["", "ب", "بت", "بب", "ببت"])
      expect(identity.acousticLogP(w, f, 0, 3)).toBeCloseTo(-f.nll([...w].map(p => tokens.indexOf(p) + 1), 0, 3), 12);
  });

  it("does not recount epsilon deletions on blank frames or same-label continuations", () => {
    // Exact zero support isolates topology from the production row probability floor.
    const exact = (rows: number[][]) => ({ startFrame: 0, endFrame: rows.length,
      row: (f: number) => Float64Array.from(rows[f], Math.log) }) as unknown as PosteriorFrames;
    const blank = exact(Array.from({ length: 8 }, () => [1, 0, 0]));
    expect(scorer.acousticLogP("بت", blank, 0, 1)).toBeCloseTo(Math.log(scripts("بت", "")), 5);
    expect(scorer.acousticLogP("بت", blank, 0, 8)).toBeCloseTo(Math.log(scripts("بت", "")), 4);
    const single = exact([[0, 1, 0]]), held = exact([[0, 1, 0], [0, 1, 0]]);
    expect(scorer.acousticLogP("ب", single, 0, 1)).toBeCloseTo(scorer.acousticLogP("ب", held, 0, 2)!, 5);
    expect(scorer.greedy(held, 0, 2)).toBe("ب");
    expect(scorer.greedy(frames([[0, 1, 0], [1, 0, 0], [0, 1, 0]]), 0, 3)).toBe("بب");
  });

  it("abstains on missing rows or unknown phones and rejects inventory mismatch", () => {
    const f = frames([[.2, .7, .1]]);
    expect(scorer.acousticLogP("ث", f, 0, 1)).toBeNull();
    expect(scorer.acousticLogP("ب", f, -1, 1)).toBeNull();
    expect(scorer.acousticLogP("ب", f, 0, 2)).toBeNull();
    expect(scorer.stringLogP("ب", "ث")).toBeNull();
    expect(() => new ComposedChannel(ch, [..."بث"])).toThrow(/inventories/);
  });
});
