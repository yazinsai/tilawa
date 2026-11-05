import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { seamTextKey } from "../src/seam-replay.js";
import { lexicalWords } from "../src/checker-lexical.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const wi = (r: WordRef) => index.words.findIndex(x => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word);

function check(inner: boolean, extra: string, pause = true, endPause = true, replays = 2, veto = false, seen?: Set<number>) {
  const from = inner ? w(38, 4, 1) : w(44, 9, 1);
  const to = inner ? w(38, 5, 1) : w(44, 11, 5);
  const seam = inner ? w(38, 4, 6) : w(44, 10, 1);
  const start = index.words[wi(from)].start, right = index.words[wi(seam)];
  const prev = index.words[wi(seam) - 1];
  const isolated = index.isolatedWordForms![wi(seam) - 1];
  const edits: Edit[] = inner ? [
    { kind: "del", at: prev.start - start, count: prev.end - prev.start },
    { kind: "ins", at: prev.start - start, text: isolated },
  ] : [];
  if (pause) edits.push({ kind: "gap", at: right.start - start, frames: 20 });
  edits.push({ kind: "ins", at: right.start - start, text: extra });
  const fixture = syntheticFixture(index, { id: "excised-seam", from, to, edits, chunkSize: 10000 });
  const evidence = fixture.evidence[0];
  const at = right.start - start + extra.length + (inner ? isolated.length - (prev.end - prev.start) : 0);
  if (endPause) for (let i = at; i < evidence.phonemes.length; i++) {
    const p = evidence.phonemes[i];
    evidence.phonemes[i] = { ...p, start: p.start + 20, end: p.end + 20 };
  }
  const written = evidence.phonemes[at].p;
  evidence.phonemes[at] = { ...evidence.phonemes[at], p: "غ", conf: 0.5 };
  const painted = withPosteriors(index, evidence, j => j === at ? [["غ", 0.5], [written, 0.42], ["", 0.06]] : undefined);
  // Isolate lexical confirmation: ordinary added-word support must not make the
  // missing-pause control pass through an unrelated path.
  const spelling = lexicalWords(index).surplusWord(wi(seam), extra.replace(/ں+/gu, "ن"));
  const seamReplays = { reference: index.version, inventory: index.inventoryId,
    seams: { [seamTextKey(index, wi(seam))]: { replays, words: veto && spelling ? [spelling] : [] } } };
  const checker = createChecker(index, { seamReplays, onSeamSpelling: word => seen?.add(word), passage: { from, to }, phrases, policy: { localWindow: { nats: Infinity } } });
  checker.push(painted, []);
  checker.end();
  return checker.mistakes().filter(e => e.errorClass === "added_word" && wi(e.location) === wi(seam));
}

describe("spelled seam spans with posterior-certified neighbours", () => {
  it("finds a stopped nun word at an ayah seam despite a blurred next-word onset", () => {
    expect(check(false, "وَءِںںں")).toMatchObject([{ observed: "وَءِن" }]);
  });
  it("uses the same rule after a marked inner stop", () => {
    expect(check(true, "فَءِںںں")).toMatchObject([{ observed: "فَءِن" }]);
  });
  it("abstains with fewer than two clean replays", () => {
    expect(check(false, "وَءِںںں", true, true, 1)).toHaveLength(0);
  });
  it("vetoes a same-seam spelling found on a clean replay", () => {
    expect(check(false, "وَءِںںں", true, true, 2, true)).toHaveLength(0);
    expect(check(true, "فَءِںںں", true, true, 2, true)).toHaveLength(0);
  });
  it("records a through-reading as a clean replay even without an inner pause", () => {
    const seen = new Set<number>();
    expect(check(true, "", false, false, 2, false, seen)).toHaveLength(0);
    expect(seen.has(wi(w(38, 4, 6)))).toBe(true);
  });
  it("requires a pause at the inner boundary", () => {
    expect(check(true, "فَءِںںں", false)).toHaveLength(0);
  });
  it("does not turn an isolated hum into a word", () => {
    expect(check(false, "ںںں")).toHaveLength(0);
  });
  it("requires the whole addition to end before a pause", () => {
    expect(check(false, "وَءِںںں", true, false)).toHaveLength(0);
  });
  it("excludes an attested word embedded inside the neighbouring text", () => {
    const lex = lexicalWords(index), seam = wi(w(44, 10, 1));
    expect(lex.surplusWord(seam, "تَق")).not.toBeNull();
    expect(lex.fragmentOfSeam(seam, "تَق")).toBe(true);
    expect(lex.fragmentOfSeam(seam, "وَءِن")).toBe(false);
  });
});
