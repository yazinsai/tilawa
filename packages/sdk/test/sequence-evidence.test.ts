import { describe, expect, it } from "vitest";
import type { PhonemeEvidence, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { sequenceEvidenceRecheck } from "../src/sequence-evidence.js";

const index = {tokens:new Set(["a","b","c"])} as ReferenceIndex;
const candidate: SequenceEvidenceCandidate = {word:{surah:1,ayah:1,word:1},written:"ab",rival:"ac",fromFrame:0,toFrame:4};
const probabilities = [
  [.13378794526775928,.1747308285347606,.4578575934391555,.23362363275832465],
  [.3731416754192928,.47951542205154785,.04588305451275701,.10145984801640236],
  [.1016383744397355,.22649346243426624,.5888832983642541,.08298486476174419],
  [.09081245231740456,.6592945080055808,.032458194729704015,.21743484494731066],
];
const chunk = (a:number,b:number):PhonemeEvidence => ({inventory:"test",frameMs:40,phonemes:[],
  posteriors:{startFrame:a,nFrames:b-a,k:4,ids:probabilities.slice(a,b).flatMap(()=>[0,1,2,3]),probs:probabilities.slice(a,b).flat()}});

describe("streaming sequence evidence recheck", () => {
  it("abstains on incomplete rows, then applies the same complete-window veto once available", () => {
    const recheck = sequenceEvidenceRecheck(index);
    expect(recheck.permits(candidate)).toBe(true);
    recheck.append(chunk(0,2));
    expect(recheck.permits(candidate)).toBe(true);
    recheck.append(chunk(2,4));
    expect(recheck.permits(candidate)).toBe(false);
    expect(recheck.permits(candidate)).toBe(false);
    expect(recheck.permits({...candidate,written:"ac",rival:"ab"})).toBe(true);
  });

  it("keeps primary evidence byte-identical and depends only on the complete comparison", () => {
    const e = chunk(0,4), original = JSON.stringify(e), recheck = sequenceEvidenceRecheck(index);
    recheck.append(e);
    expect(recheck.permits(candidate)).toBe(false);
    expect(recheck.permits({...candidate,word:{surah:2,ayah:2,word:2}})).toBe(false);
    expect(JSON.stringify(e)).toBe(original);
  });
});
