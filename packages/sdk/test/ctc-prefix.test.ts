import { describe, expect, it } from "vitest";
import { createPrefixDecoder } from "../src/ctc-prefix.js";
const decode = (rows: number[][], width = 8, lag = 25) => {
  const d = createPrefixDecoder("ab", width, lag), out = [];
  for (const [f, row] of rows.entries()) { d.push(row.map((p, id) => [id, p]), f); out.push(...d.take()); }
  return [...out, ...d.end()];
};
const exact = (rows: number[][]) => {
  const sums = new Map<string, number>();
  const visit = (at: number, prev: number, s: string, p: number) => {
    if (at === rows.length) { sums.set(s, (sums.get(s) ?? 0) + p); return; }
    rows[at].forEach((v, id) => visit(at + 1, id, s + (id && id !== prev ? "ab"[id - 1] : ""), p * v));
  };
  visit(0, 0, "", 1);
  return [...sums].sort((a, b) => b[1] - a[1])[0][0];
};
describe("audio-only prefix decoding", () => {
  it("sums paths instead of taking a transient winning frame", () => {
    const rows = [[0.6, 0.4, 0], [0.6, 0.4, 0]];
    expect(decode(rows).map(p => p.p).join("")).toBe("a");
  });
  it("keeps repetitions separated by blank and collapses sustained runs", () => {
    expect(decode([[0,1,0],[0,1,0],[1,0,0],[0,1,0]]).map(p => p.p).join("")).toBe("aa");
  });
  it("matches exhaustive CTC path sums on short inputs", () => {
    let seed = 17;
    for (let n = 0; n < 40; n++) {
      const rows = Array.from({length: 4}, () => {
        const r = Array.from({length: 3}, () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2**32 + 0.01; });
        const sum = r.reduce((a,b)=>a+b,0); return r.map(v=>v/sum);
      });
      expect(decode(rows, 128).map(p=>p.p).join("")).toBe(exact(rows));
    }
  });
  it("commits long streams with bounded lag and ordered timing", () => {
    const rows = Array.from({length: 100}, (_, i) => i % 2 ? [0.8,0.1,0.1] : [0.1,0.6,0.3]);
    const phones = decode(rows, 8, 5);
    expect(phones.length).toBeGreaterThan(0);
    for (let i=1;i<phones.length;i++) expect(phones[i].start).toBeGreaterThan(phones[i-1].end);
  });
});
