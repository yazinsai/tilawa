import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { SequenceEvidenceCandidate } from "../src/index.js";
import { replayChecked, syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json",import.meta.url),"utf8")));
const from = {surah:2,ayah:255,word:1}, target = {surah:2,ayah:255,word:23}, to = {surah:2,ayah:256,word:10};
const wi = (r: typeof from) => index.words.find(w=>w.surah===r.surah && w.ayah===r.ayah && w.word===r.word)!;
const text = (r: typeof from) => index.phonemes.slice(wi(r).start,wi(r).end);
const at = wi(target).start-wi(from).start;
const fixture = () => {
  const f = syntheticFixture(index,{id:"sequence-recheck",from,to,chunkSize:8,edits:[
  {kind:"del",at,count:text(target).length},{kind:"ins",at,text:text({surah:108,ayah:1,word:3})},
  ]});
  f.evidence = f.evidence.map(e=>withPosteriors(index,e));
  return f;
};
const wrong = (es: ReturnType<typeof replayChecked>) => es.filter(e=>e.state==="confirmed" && e.errorClass==="wrong_word");

describe("complete sequence listener veto transport", () => {
  it("discovers supported complete comparisons and lets a veto only remove confirmation", () => {
    const f = fixture(), captured: SequenceEvidenceCandidate[] = [];
    const baseline = replayChecked(index,f);
    expect(wrong(baseline)).toHaveLength(1);
    expect(replayChecked(index,f,{}, {sequenceEvidence(c) {captured.push(c);return true;}})).toEqual(baseline);
    expect(captured.length).toBeGreaterThan(0);
    expect(captured.every(c=>c.written.includes(text(target)) && c.rival.length && c.toFrame>c.fromFrame)).toBe(true);
    expect(wrong(replayChecked(index,f,{}, {sequenceEvidence:()=>false}))).toHaveLength(0);
    const clone = JSON.parse(JSON.stringify(f));
    clone.evidence[0].phonemes[0].sequenceVetoes = captured;
    expect(wrong(replayChecked(index,clone))).toHaveLength(0);
    expect(wrong(replayChecked(index,clone,{}, {sequenceEvidence:()=>true}))).toHaveLength(0);
    expect(f.evidence[0].phonemes[0].sequenceVetoes).toBeUndefined();
  });

  it("binds certificates to both complete pronunciations and exact frame boundaries", () => {
    const f = fixture(), captured: SequenceEvidenceCandidate[] = [];
    replayChecked(index,f,{}, {sequenceEvidence(c) {captured.push(c);return true;}});
    expect(captured.length).toBeGreaterThan(0);
    for (const change of [
      (c:SequenceEvidenceCandidate)=>({...c,fromFrame:c.fromFrame+1}),
      (c:SequenceEvidenceCandidate)=>({...c,rival:c.rival+"ب"}),
      (c:SequenceEvidenceCandidate)=>({...c,word:{...c.word,word:c.word.word+1}}),
    ]) {
      const clone = JSON.parse(JSON.stringify(f));
      clone.evidence[0].phonemes[0].sequenceVetoes = captured.map(change);
      expect(wrong(replayChecked(index,clone))).toHaveLength(1);
    }
  });
});
