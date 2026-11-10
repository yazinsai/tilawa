import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, type WrongWordAcousticCandidate } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { AcousticRecurrenceEvidence } from "../src/evidence.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 26, ayah: 142, word: 1 }, to = { surah: 26, ayah: 143, word: 3 };
const target = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 5)!;
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!.start;
const fixture = () => syntheticFixture(index, { id: "acoustic-certificate", from, to, chunkSize: 4, edits: [
  { kind: "del", at: target.start-start, count: target.end-target.start },
  { kind: "ins", at: target.start-start, text: "نُۥۥح" },
] }).evidence.map(e => withPosteriors(index,e));
const run = (metadata: AcousticRecurrenceEvidence[] = [], discovery?: (c: WrongWordAcousticCandidate) => boolean) => {
  const checker = createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},wrongWordAcoustic:discovery});
  const chunks = fixture();
  chunks[0].phonemes[0] = {...chunks[0].phonemes[0],acousticRecurrence:metadata};
  for (const e of chunks) checker.push(JSON.parse(JSON.stringify(e)),[]);
  checker.end();
  return checker.mistakes();
};

describe("exact-window listener recurrence certificates",()=> {
  it("discovers an otherwise confirmed complete substitution without changing it",()=> {
    const candidates: WrongWordAcousticCandidate[]=[];
    const baseline=run(), discovered=run([],c=>{candidates.push(c);return true;});
    expect(baseline.some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(discovered).toEqual(baseline);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c=>c.written.length && c.accused.length && c.toFrame>c.fromFrame)).toBe(true);
  });
  it("vetoes only a finite preference reversal with positive written relationship evidence",()=> {
    const candidates: WrongWordAcousticCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,localWritten:-2,localAccused:-1,relationalWritten:1,relationalAccused:0,positiveWritten:3}));
    expect(run(certificates).some(e=>e.errorClass==="wrong_word")).toBe(false);
    expect(run(certificates.map(c=>({...c,positiveWritten:0})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,relationalWritten:c.relationalAccused})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,localWritten:NaN})))).toEqual(run());
  });
  it("does not reuse a certificate on changed frames or pronunciations",()=> {
    const candidates: WrongWordAcousticCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,localWritten:-2,localAccused:-1,relationalWritten:1,relationalAccused:0,positiveWritten:3}));
    expect(run(certificates.map(c=>({...c,toFrame:c.toFrame+1})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,accused:c.accused+"ب"})))).toEqual(run());
  });
});
