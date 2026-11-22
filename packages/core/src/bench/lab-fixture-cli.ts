/**
 * Writes a small synthetic lab test set (manifest, sessions, evidence) for smoke-running the scorer end to end:
 *
 *   pnpm --filter @tilawa/core exec tsx src/bench/lab-fixture-cli.ts --out $LAB_DATA/lab-scorer-fixture
 *
 * The audio paths are placeholders (no WAVs are written). The listener hears every planted wrong word as written on phone audio and decodes studio audio perfectly; the
 * teacher decodes both perfectly. Numbers from it test the scorer, not the engine.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import type { WordRef } from "../events.js";
import { loadReferenceIndex } from "../reference.js";
import { oracleEvidence } from "./lab.js";
import { buildSession, deafToWrongWords, donorWord, lastWordOf } from "./lab-fixture.js";
import type { FixturePart } from "./lab-fixture.js";
import { DEFAULT_CANDIDATE } from "./scoreboard.js";

const { values } = parseArgs({ options: { out: { type: "string", default: `${process.env.LAB_DATA ?? "."}/lab-scorer-fixture` },
  listener: { type: "string", default: DEFAULT_CANDIDATE } } });
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const W = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const last = (s: number, a: number) => lastWordOf(index, s, a);
const donor = (at: WordRef) => donorWord(index, at);
const plans: { id: string; split: "practice" | "exam"; tags: string[]; speaker: string; parts: FixturePart[] }[] = [
  { id: "fx-mulk", split: "practice", tags: [], speaker: "r1", parts: [
    { from: W(67, 1, 1), to: last(67, 11), wrong: [{ at: W(67, 3, 4), said: donor(W(67, 3, 4)), mutashabihat: true }], omit: [W(67, 5, 3)],
      add: [{ before: W(67, 7, 2), said: W(112, 1, 2) }], skip: [[W(67, 9, 1), last(67, 9)]] },
    { kind: "repeat", from: W(67, 11, 1), to: last(67, 11) }] },
  { id: "fx-rahman", split: "exam", tags: ["strict"], speaker: "r2", parts: [
    { from: W(55, 13, 2), to: last(55, 25), wrong: [{ at: W(55, 22, 2), said: donor(W(55, 22, 2)) }] }] },
  { id: "fx-yasin-clean", split: "practice", tags: [], speaker: "r1", parts: [
    { from: W(36, 1, 1), to: last(36, 9) }, { kind: "restart", from: W(36, 8, 1), to: last(36, 12) }] },
  { id: "fx-baqarah", split: "exam", tags: [], speaker: "r2", parts: [
    { from: W(2, 1, 1), to: last(2, 10), wrong: [{ at: W(2, 5, 3), said: donor(W(2, 5, 3)) }], omit: [W(2, 8, 4)] }] },
  { id: "fx-sharh", split: "practice", tags: [], speaker: "r3", parts: [
    { from: W(94, 1, 1), to: W(94, 8, 3), wrong: [{ at: W(94, 3, 2), said: donor(W(94, 3, 2)) }], add: [{ before: W(94, 6, 2), said: W(112, 1, 2) }] },
    { kind: "repeat", from: W(94, 8, 1), to: W(94, 8, 3) }] },
];
const out = resolve(values.out!);
const json = (path: string, x: unknown) => { mkdirSync(join(path, ".."), { recursive: true }); writeFileSync(path, JSON.stringify(x)); };
const sessions = plans.map(p => {
  const s = buildSession(index, p.id, p.parts, p.speaker);
  json(join(out, "sessions", `${p.id}.json`), s);
  const perfect = oracleEvidence(s, index), deaf = oracleEvidence(deafToWrongWords(s, index), index);
  json(join(out, "evidence", values.listener!, `${p.id}.phone.json`), deaf);
  json(join(out, "evidence", values.listener!, `${p.id}.studio.json`), perfect);
  for (const v of ["phone", "studio"]) json(join(out, "evidence", "teacher", `${p.id}.${v}.json`), perfect);
  console.log(`${p.id}: ${s.durationS} s, ${s.events.length} mistakes, ${s.nonErrors.length} non-error words`);
  return { id: p.id, split: p.split, tags: p.tags, session: `sessions/${p.id}.json`, audio: { phone: `audio/${p.id}.phone.wav`, studio: `audio/${p.id}.studio.wav` } };
});
writeFileSync(join(out, "manifest-v1.json"), JSON.stringify({ version: 1, testSet: "lab-scorer-fixture", note: "synthetic scorer fixture (lab-fixture-cli.ts)", sessions }, null, 1) + "\n");
console.log(`${sessions.length} sessions -> ${out}/manifest-v1.json`);
