import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, type SequenceEvidenceCandidate } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { NoisePowerEvidence } from "../src/evidence.js";
import type { CheckerEvent } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 26, ayah: 142, word: 1 }, to = { surah: 26, ayah: 143, word: 3 };
const target = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 5)!;
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!.start;
const fixture = () => syntheticFixture(index, { id: "noise power-certificate", from, to, chunkSize: 4, edits: [
  { kind: "del", at: target.start-start, count: target.end-target.start },
  { kind: "ins", at: target.start-start, text: "نُۥۥح" },
] }).evidence.map(e => withPosteriors(index,e));
const run = (metadata: NoisePowerEvidence[] = [], discovery?: (c: SequenceEvidenceCandidate) => boolean, events?: CheckerEvent[]) => {
  const checker = createChecker(index,{passage:{from,to},policy:{confirm:{after:"next-word"}},sequenceEvidence:discovery});
  const chunks = fixture();
  chunks[0].phonemes[0] = {...chunks[0].phonemes[0],noisePowerVetoes:metadata};
  for (const e of chunks) { const out=checker.push(JSON.parse(JSON.stringify(e)),[]); events?.push(...out); }
  const out=checker.end(); events?.push(...out);
  return checker.mistakes();
};

describe("exact-window listener noise power certificates",()=> {
  it("discovers an otherwise confirmed complete substitution without changing it",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    const baseline=run(), discovered=run([],c=>{candidates.push(c);return true;});
    expect(baseline.some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(discovered).toEqual(baseline);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c=>c.written.length && c.rival.length && c.toFrame>c.fromFrame)).toBe(true);
  });
  it("vetoes only a finite preference reversal toward the complete written pronunciation",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,controlWritten:-2,controlRival:-1,powerWritten:1,powerRival:0}));
    expect(run(certificates).some(e=>e.errorClass==="wrong_word")).toBe(false);
    expect(run(certificates.map(c=>({...c,controlWritten:c.controlRival})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,powerWritten:c.powerRival})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,controlWritten:NaN})))).toEqual(run());
  });
  it("does not reuse a certificate on changed frames or pronunciations",()=> {
    const candidates: SequenceEvidenceCandidate[]=[];
    run([],c=>{candidates.push(c);return true;});
    const certificates=candidates.map(c=>({...c,controlWritten:-2,controlRival:-1,powerWritten:1,powerRival:0}));
    expect(run(certificates.map(c=>({...c,toFrame:c.toFrame+1})))).toEqual(run());
    expect(run(certificates.map(c=>({...c,rival:c.rival+"ب"})))).toEqual(run());
  });
  it("preserves a parallel graph confirmation when every string comparison is vetoed",()=> {
    const a={surah:84,ayah:24,word:3}, b={surah:84,ayah:25,word:4};
    const target=index.words.find(w=>w.surah===84&&w.ayah===25&&w.word===1)!;
    const start=index.words.find(w=>w.surah===84&&w.ayah===24&&w.word===3)!.start;
    const chunks=syntheticFixture(index,{id:"noise-graph-preservation",from:a,to:b,chunkSize:6,edits:[
      {kind:"del",at:target.start-start,count:target.end-target.start},
      {kind:"ins",at:target.start-start,text:"ءِننننَ"},
    ]}).evidence.map(e=> {
      const heard={...e,phonemes:e.phonemes.map(p=>p.p==="ن"?{...p,conf:.95}:p)};
      return withPosteriors(index,heard,j=>heard.phonemes[j].p==="ن"?[["ن",.6],["ل",.05],["",.3]]:undefined);
    });
    const run=(veto:boolean)=> {
      const checker=createChecker(index,{passage:{from:a,to:b},policy:{confirm:{after:"next-word"}},sequenceEvidence:veto?()=>false:undefined});
      for (const e of chunks) checker.push(e,[]);checker.end();return checker.mistakes();
    };
    const baseline=run(false);expect(baseline.some(e=>e.errorClass==="wrong_word")).toBe(true);
    expect(run(true)).toEqual(baseline);
  });
  it("retains baseline issue creation and changes only the public final verdict",()=> {
    const baseline: CheckerEvent[]=[], vetoed: CheckerEvent[]=[];
    run([],undefined,baseline);
    run([],()=>false,vetoed);
    expect(vetoed).toEqual(baseline.map(e=>e.state==="confirmed" && e.errorClass==="wrong_word" ? {...e,state:"uncertain"} : e));
  });
});
