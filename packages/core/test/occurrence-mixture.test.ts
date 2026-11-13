import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, type SequenceEvidenceCandidate } from "../src/checker.js";
import { parallelPaths } from "../src/parallel-phrases.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { OccurrenceMixtureEvidence } from "../src/evidence.js";
import type { CheckerEvent } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 26, ayah: 142, word: 1 }, to = { surah: 26, ayah: 143, word: 3 };
const target = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 5)!;
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!.start;
const fixture = () => syntheticFixture(index, { id: "occurrence-certificate", from, to, chunkSize: 4, edits: [
  { kind: "del", at: target.start-start, count: target.end-target.start },
  { kind: "ins", at: target.start-start, text: "نُۥۥح" },
] }).evidence.map(e => withPosteriors(index,e));
const run = (metadata: OccurrenceMixtureEvidence[] = [], discovery?: (c: SequenceEvidenceCandidate) => boolean, events?: CheckerEvent[]) => {
  const checker = createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},sequenceEvidence:discovery});
  const chunks = fixture();
  chunks[0].phonemes[0] = {...chunks[0].phonemes[0],occurrenceMixtureVetoes:metadata};
  for (const e of chunks) { const out=checker.push(JSON.parse(JSON.stringify(e)),[]); events?.push(...out); }
  const out=checker.end(); events?.push(...out);
  return checker.mistakes();
};

describe("exact-window listener occurrence-mixture certificates",()=> {
  it("discovers an otherwise confirmed complete substitution without changing it",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    const baseline=run(), discovered=run([],c=>{candidates.push(c);return true;});
    expect(baseline.some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(discovered).toEqual(baseline);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c=>c.written.length && c.rival.length && c.toFrame>c.fromFrame)).toBe(true);
  }, 30_000);
  it("vetoes only a finite preference reversal toward the complete written pronunciation",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,switchingWritten:-2,switchingRival:-1,tiedWritten:1,tiedRival:0}));
    expect(run(certificates).some(e=>e.errorClass==="wrong_word")).toBe(false);
    expect(run(certificates.map(c=>({...c,switchingWritten:c.switchingRival})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,tiedWritten:c.tiedRival})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,switchingWritten:NaN})))).toEqual(run());
  }, 30_000);
  it("does not reuse a certificate on changed frames or pronunciations",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,switchingWritten:-2,switchingRival:-1,tiedWritten:1,tiedRival:0}));
    expect(run(certificates.map(c=>({...c,toFrame:c.toFrame+1})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,rival:c.rival+"ب"})))).toEqual(run());
  }, 30_000);
  it("retains baseline issue creation and changes only the public final verdict",()=> {
    const baseline: CheckerEvent[]=[], vetoed: CheckerEvent[]=[];
    run([],undefined,baseline);
    run([],()=>false,vetoed);
    expect(vetoed).toEqual(baseline.map(e=>e.state==="confirmed" && e.errorClass==="wrong_word" ? {...e,state:"uncertain"} : e));
  }, 30_000);
  it("preserves graph confirmations when the side scorer rejects every string comparison",()=> {
    const paths=parallelPaths(index);
    const wi=index.words.findIndex(w=>w.surah===84 && w.ayah===25 && w.word===1);
    const path=paths.get(wi)!.find(p=>p.altWord==="ءِننننَ")!;
    const first=index.words[path.start], last=index.words[path.start+5], changed=index.words[path.word];
    const chunks=syntheticFixture(index,{id:"occurrence-graph-preservation",from:first,to:last,chunkSize:6,edits:[
      {kind:"del",at:changed.start-first.start,count:changed.end-changed.start},
      {kind:"ins",at:changed.start-first.start,text:path.altWord},
    ]}).evidence.map(e=>withPosteriors(index,e,j=>e.phonemes[j].p==="ن" ? [["ن",.6],["ل",.05],["",.3]] : undefined));
    const calls: SequenceEvidenceCandidate[]=[];
    const checker=createChecker(index,{passage:{from:first,to:last},policy:{confirm:{after:"delay",ms:60000}},
      sequenceEvidence(c) {calls.push(c);return false;}});
    for (const e of chunks) checker.push(e,[]);
    checker.end();
    expect(checker.mistakes().some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(calls.length).toBe(0);
  }, 30_000);

});
