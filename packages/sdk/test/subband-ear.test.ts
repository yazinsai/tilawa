import { describe, expect, it } from "vitest";
import type { SubbandEvidence } from "@tilawa/core";
import type { ModelManifest } from "../src/model.js";
import { carrySubbandEvidence, subbandComparison, subbandCtcScore, subbandReversal, subbandVerdictRemovals, validateSubbandModel, type SubbandModel } from "../src/subband-ear.js";

const manifest={inventory:"test",tokens:"بت",sha256:"abc"} as ModelManifest;
const model={schema:1,inventory:"test",tokens:"بت",primarySha256:"abc",mels:80,stride:4,kernel:9,
  bands:[[0,32],[16,48],[32,64],[48,80]],learnedParameters:100,controlLearnedParameters:100} as SubbandModel;
const request={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:2};
const rows=(p:number)=>[[.1,p,.9-p],[.8,.1,.1]].map(r=>r.map(Math.log));

describe("reliability-fused subband complete-pronunciation evidence",()=> {
  it("requires positive written evidence plus a rival-preferring broadband control",()=> {
    const c=subbandComparison(model,rows(.8),rows(.1),request)!;
    expect(subbandReversal(c)).toBe(true);
    expect(subbandReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(subbandReversal({...c,fusedWritten:c.fusedRival})).toBe(false);
    expect(subbandReversal({...c,fusedWritten:Infinity})).toBe(false);
    expect(subbandComparison(model,rows(.8),rows(.8),request)).not.toBeNull();
    expect(subbandReversal(subbandComparison(model,rows(.8),rows(.8),request)!)).toBe(false);
    expect(subbandComparison(model,rows(.8),rows(.1),{...request,written:"x"})).toBeNull();
    expect(subbandComparison(model,rows(.8),rows(.1).slice(1),request)).toBeNull();
  });
  it("matches the exhaustive two-frame CTC sum and cannot ignore a requested phone",()=> {
    const p=[[.1,.8,.1],[.8,.1,.1]], x=p.map(r=>r.map(Math.log));
    expect(subbandCtcScore(x,[1])).toBeCloseTo(Math.log(.1*.1+.8*.1+.8*.8),12);
    expect(subbandCtcScore(x,[1,1])).toBe(-Infinity);
    expect(subbandCtcScore(x,[1,2])).toBeCloseTo(Math.log(.8*.1),12);
  });
  it("rejects wrong inventory, primary identity, band geometry or unequal capacity",()=> {
    expect(validateSubbandModel(model,manifest)).toEqual(model);
    for (const over of [{tokens:"b"},{primarySha256:"different"},{controlLearnedParameters:99},{stride:2},{bands:[[0,80]]}])
      expect(()=>validateSubbandModel({...model,...over},manifest)).toThrow();
  });
  it("carries certificates without changing the primary posterior or token values",()=> {
    const p={p:"ب",start:0,end:0,conf:.9};
    const chunks=[{inventory:"test",frameMs:40,phonemes:[p]}] as Parameters<typeof carrySubbandEvidence>[0];
    const c={...request,fusedWritten:1,fusedRival:0,controlWritten:0,controlRival:1} as SubbandEvidence;
    const enriched=carrySubbandEvidence(chunks,[c]);
    const {subbandVetoes,...plain}=enriched[0].phonemes[0];
    expect(plain).toEqual(p);expect(subbandVetoes).toEqual([c]);expect(chunks[0].phonemes[0]).toBe(p);
  });
  it("retracts only the exact live occurrence removed by a matching replay",()=> {
    const e={id:"a",errorClass:"wrong_word",timeMs:40,expected:"ب",observed:"ت",location:request.word} as Parameters<typeof subbandVerdictRemovals>[0][0];
    const repeated={...e,timeMs:80};
    expect(subbandVerdictRemovals([e,repeated],[e,repeated],[repeated])).toEqual([e]);
    expect(subbandVerdictRemovals([e],[{...e,observed:"x"}],[])).toEqual([]);
  });
});
