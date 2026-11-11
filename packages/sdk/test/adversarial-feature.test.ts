import { describe, expect, it } from "vitest";
import { carryAdversarialEvidence, featureBoundaryScore, featureComparison, featureReversal, validateAdversarialModel } from "../src/adversarial-feature.js";
import type { AdversarialModel, FeatureHead } from "../src/adversarial-feature.js";
import type { ModelManifest } from "../src/model.js";

const head=(bias:number[]):FeatureHead=>({hidden:Array.from({length:48},()=>Array(32).fill(0)),hiddenBias:Array(48).fill(0),
  output:bias.map(()=>Array(48).fill(0)),outputBias:bias});
const model=():AdversarialModel=>({schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:"a".repeat(64),
  featureOutput:"encoder",projection:Array.from({length:32},(_,i)=>Array.from({length:32},(_,j)=>Number(i===j))),
  center:Array(32).fill(0),scale:Array(32).fill(1),robust:head([-2,2,-2]),control:head([-2,-2,2])});
const candidate={word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:4,toFrame:8};
const features=Array.from({length:4},()=>Array(32).fill(0));

describe("matched acoustic pronunciation heads",()=>{
  it("scores both complete candidates symmetrically and requires the robust/control disagreement",()=>{
    const c=featureComparison(model(),features,candidate)!;
    expect(featureReversal(c)).toBe(true);
    const reverse=featureComparison(model(),features,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(reverse.robustWritten).toBe(c.robustRival);
    expect(reverse.robustRival).toBe(c.robustWritten);
    expect(reverse.controlWritten).toBe(c.controlRival);
    expect(featureReversal(reverse)).toBe(false);
    expect(featureReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(featureReversal({...c,robustWritten:NaN})).toBe(false);
  });
  it("marginalizes timing conditioned on the entire pronunciation being emitted",()=>{
    // With identity-neutral emissions, every feasible pronunciation has the same likelihood.
    const rows=Array.from({length:8},()=>[-1,-1,-1]);
    expect(featureBoundaryScore(rows,[1])).toBeCloseTo(-8,10);
    expect(featureBoundaryScore(rows,[1,1,2])).toBeCloseTo(-8,10);
    expect(featureBoundaryScore(rows.slice(0,1),[1,1])).toBe(-Infinity);
  });
  it("abstains on unknown phones, nonfinite features and impossible complete candidates",()=>{
    expect(featureComparison(model(),features,{...candidate,rival:"ث"})).toBeNull();
    expect(featureComparison(model(),[[NaN,...Array(31).fill(0)]],candidate)).toBeNull();
    expect(featureComparison(model(),features.slice(0,1),{...candidate,rival:"تت"})).toBeNull();
  });
  it("carries only metadata, retaining primary phones and posterior rows",()=>{
    const chunks=[{inventory:"test",frameMs:40,phonemes:[{p:"ت",start:4,end:7,conf:.99}],
      posteriors:{k:1,startFrame:4,nFrames:4,ids:[2,2,2,2],probs:[.99,.99,.99,.99]}}];
    const c=featureComparison(model(),features,candidate)!;
    const enriched=carryAdversarialEvidence(chunks,[c]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toEqual({...chunks[0].phonemes[0],adversarialFeatureVetoes:[c]});
    expect(chunks[0].phonemes[0]).not.toHaveProperty("adversarialFeatureVetoes");
  });
  it("rejects unmatched inventories, primary weights and head dimensions",()=>{
    const m=model(),manifest={inventory:m.inventory,tokens:m.tokens,sha256:m.primarySha256} as ModelManifest;
    expect(validateAdversarialModel(m,manifest)).toBe(m);
    expect(()=>validateAdversarialModel({...m,primarySha256:"b".repeat(64)},manifest)).toThrow();
    expect(()=>validateAdversarialModel({...m,robust:head([0,0])},manifest)).toThrow();
    expect(()=>validateAdversarialModel({...m,scale:Array(32).fill(0)},manifest)).toThrow();
  });
});
