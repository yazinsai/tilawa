import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { CheckerEvent, EngineEvent, PhraseAsset, ReferenceAsset, WordRef } from "@tilawa/core";
import { syntheticFixture } from "../../core/src/fixtures/index.js";
import type { Edit } from "../../core/src/fixtures/index.js";
import { createEngine } from "../src/session.js";
import { loadNodeModel, readWav } from "../src/node.js";

const core = (p: string) => new URL(`../../core/assets/${p}`, import.meta.url);
const index = loadReferenceIndex(JSON.parse(readFileSync(core("ref-v2.json"), "utf8")) as ReferenceAsset);
const phrases = JSON.parse(readFileSync(core("phrases-v1.json"), "utf8")) as PhraseAsset;
const tiny = new URL("./fixtures/tiny", import.meta.url).pathname;

const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const from = w(2, 255, 1);
const target = w(2, 255, 23);
const at = (r: WordRef) => find(r).start - find(from).start;
/** يسمع for يشفع, then the right word: a self-correction. */
const selfCorrection: Edit[] = [{ kind: "ins", at: at(target), text: text(target).replace("ش", "س").replace("ف", "م") }];
const evidence = (edits: Edit[]) => syntheticFixture(index, { id: "s", from, to: w(2, 256, 10), edits, chunkSize: 2 }).evidence;
const summary = (es: EngineEvent[]) => es.filter((e): e is CheckerEvent => e.type === "checker").map((e) => `${e.state} ${e.errorClass} ${key(e.location)}`);

const last256 = index.words.filter((x) => x.surah === 2 && x.ayah === 256 && x.end > x.start).at(-1)!.word;

async function engine() {
  return createEngine({ reference: index, phrases, model: await loadNodeModel(tiny) });
}

describe("sessions", () => {
  it("start with the model's capabilities: every pronunciation class is unsupported", async () => {
    const seen: EngineEvent[] = [];
    const s = (await engine()).startMemorization({ surah: 2, fromAyah: 255, toAyah: 256 });
    s.subscribe((e) => seen.push(e));
    expect(seen[0]).toMatchObject({ type: "capabilities", v: 1, supported: ["wrong_word", "omitted_word", "added_word", "skipped_passage"] });
    expect(seen[0].type === "capabilities" && seen[0].unsupported).toContain("short_vowel");
    expect(seen[0].type === "capabilities" && seen[0].unsupported).toHaveLength(11);
  });

  it("declare the passage from the first word of fromAyah to the last of toAyah", async () => {
    const e = await engine();
    expect(e.startMemorization({ surah: 2, fromAyah: 255, toAyah: 256 }).passage).toEqual({ from: w(2, 255, 1), to: w(2, 256, last256) });
    expect(e.startFreeRecitation().passage).toBeNull();
    expect(() => e.startMemorization({ surah: 1, fromAyah: 5, toAyah: 3 })).toThrow();
    expect(() => e.startMemorization({ surah: 1, fromAyah: 1, toAyah: 8 })).toThrow();
  });

  it("report a self-correction as suspected then resolved, and count it only when the policy says so", async () => {
    const e = await engine();
    const run = (countSelfCorrections?: boolean) => {
      const s = e.startMemorization({ surah: 2, fromAyah: 255, toAyah: 256 }, { countSelfCorrections });
      const out: EngineEvent[] = [];
      for (const c of evidence(selfCorrection)) out.push(...s.pushEvidence(c));
      out.push(...s.endEvidence());
      return { out, mistakes: s.mistakes() };
    };
    const byDefault = run();
    expect(summary(byDefault.out)).toEqual(["suspected wrong_word 2:255:23", "resolved wrong_word 2:255:23"]);
    expect(byDefault.mistakes).toEqual([]);
    expect(summary(run(true).mistakes)).toEqual(["resolved wrong_word 2:255:23"]);
  });

  it("run audio through the listener into position events", async () => {
    const s = (await engine()).startFreeRecitation();
    const wav = await readWav(new URL("./fixtures/tlog-clean-00010-630.wav", import.meta.url).pathname);
    const out: EngineEvent[] = [];
    for (let a = 0; a < wav.samples.length; a += 8000) out.push(...(await s.pushAudio(wav.samples.subarray(a, a + 8000), wav.sampleRate)));
    out.push(...(await s.end()));
    expect(out.some((e) => e.type === "position")).toBe(true);
  });
});
