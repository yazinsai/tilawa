import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { EvidenceError, createTracker, loadReferenceIndex } from "../src/index.js";
import type { DecodedPhoneme, PhonemeEvidence, PositionEvent, ReferenceAsset } from "../src/index.js";
import { finalPosition, formatSpan, replay, withPosteriors } from "../src/fixtures/index.js";
import type { Fixture } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);

const tlogDir = new URL("./fixtures/tlog/", import.meta.url);
const tlog: Fixture[] = readdirSync(tlogDir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => JSON.parse(readFileSync(new URL(f, tlogDir), "utf8")));

const decoded = (text: string): PhonemeEvidence => ({
  inventory: index.inventoryId,
  frameMs: 20,
  phonemes: [...text].map((p, j): DecodedPhoneme => ({ p, start: 3 * j, end: 3 * j + 1, conf: 0.9 })),
});
const run = (evidence: PhonemeEvidence[]) => {
  const t = createTracker(index, { trace: true });
  const events = [...evidence.flatMap((e) => t.push(e)), ...t.end()];
  return finalPosition(events)!;
};

// 30:38 and 30:39 share a 31-phoneme window differing only at its centre: و in 30:38, ف in 30:39.
const occurrence = (surah: number, ayah: number) => {
  const words = index.words.filter((w) => w.surah === surah && w.ayah === ayah);
  return index.phonemes.slice(words[0].start, words.at(-1)!.end);
};
const window = (() => {
  const a = occurrence(30, 38);
  const b = occurrence(30, 39);
  for (let i = 0; i + 31 <= b.length; i++) {
    const w = b.slice(i, i + 31);
    if (w[15] === "ف" && a.includes(w.slice(0, 15) + "و" + w.slice(16))) return w;
  }
  throw new Error("shared window not found");
})();

describe("posterior reranking", () => {
  it("ranks the true span first when the decode is mangled but the posteriors favour it", () => {
    const mangled = decoded(window.slice(0, 15) + "و" + window.slice(16));
    const plain = run([mangled]);
    expect(plain.observed[0].span.start.ayah).toBe(38);

    const favoured = withPosteriors(index, mangled, (j) => (j === 15 ? [["ف", 0.7], ["و", 0.25], ["", 0.03]] : undefined));
    const reranked = run([favoured]);
    expect(reranked.observed[0].span.start.ayah).toBe(39);
  });

  it.each([
    ["shortened", -1],
    ["lengthened", 1],
  ])("doesn't charge a %s held sound", (_, delta) => {
    // 2:255 words 3..12, with its first madd run held one step shorter or longer.
    const words = index.words.filter((w) => w.surah === 2 && w.ayah === 255 && w.word >= 3 && w.word <= 12);
    const ref = index.phonemes.slice(words[0].start, words.at(-1)!.end);
    const at = ref.search(/ا{2,}/);
    expect(at).toBeGreaterThan(-1);
    const recited = delta < 0 ? ref.slice(0, at) + ref.slice(at + 1) : ref.slice(0, at) + "ا" + ref.slice(at);
    const e = run([withPosteriors(index, decoded(recited))]);
    expect(e.state).toBe("locked");
    expect(e.observed[0].cost).toBe(0);
  });

  it.each(tlog.map((f) => [f.id, f] as const))("%s: reranking only reorders candidates that retrieval produced", (_, f) => {
    const stripped = { ...f, evidence: f.evidence.map(({ posteriors: _, ...e }) => e) };
    const before = new Set(finalPosition(replay(index, stripped, { trace: true }))!.trace!.shortlist.map((c) => formatSpan(c.span)));
    const after = finalPosition(replay(index, f, { trace: true }))!.trace!.shortlist.map((c) => formatSpan(c.span));
    for (const s of after) expect(before.has(s)).toBe(true);
  });

  it("produces valid events on every real fixture", () => {
    for (const f of tlog) {
      const e = finalPosition(replay(index, f)) as PositionEvent;
      for (const c of e.observed) {
        expect(Number.isFinite(c.cost)).toBe(true);
        expect(c.confidence).toBeGreaterThanOrEqual(0);
      }
    }
  });
});

describe("posterior validation", () => {
  const good = () => withPosteriors(index, decoded(window));
  const push = (e: PhonemeEvidence) => () => createTracker(index).push(e);

  it("accepts consistent posteriors", () => {
    expect(push(good())).not.toThrow();
  });

  it("rejects rows that don't match nFrames x k", () => {
    const e = good();
    expect(push({ ...e, posteriors: { ...e.posteriors!, nFrames: e.posteriors!.nFrames + 1 } })).toThrow(EvidenceError);
  });

  it("rejects a mismatched frame stride, where phoneme frames fall outside the rows", () => {
    const e = good();
    const doubled = { ...e, phonemes: e.phonemes.map((p) => ({ ...p, start: p.start * 2, end: p.end * 2 })) };
    expect(push(doubled)).toThrow(/frame/);
  });

  it("rejects token ids outside the inventory", () => {
    const e = good();
    const ids = Array.from(e.posteriors!.ids);
    ids[0] = index.tokens.size + 1;
    expect(push({ ...e, posteriors: { ...e.posteriors!, ids } })).toThrow(/inventory/);
  });

  it("rejects posteriors in a different token order", () => {
    const e = good();
    const n = index.tokens.size;
    const shifted = Array.from(e.posteriors!.ids, (i) => (i === 0 ? 0 : (i % n) + 1));
    expect(push({ ...e, posteriors: { ...e.posteriors!, ids: shifted } })).toThrow(/inventory/);
  });

  it("rejects mixing chunks with and without posteriors", () => {
    const t = createTracker(index);
    const e = good();
    t.push(e);
    const next = { ...decoded("ب"), phonemes: [{ p: "ب", start: e.posteriors!.nFrames + 3, end: e.posteriors!.nFrames + 4, conf: 0.9 }] };
    expect(() => t.push(next)).toThrow(EvidenceError);
  });
});
