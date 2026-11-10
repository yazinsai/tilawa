import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { decodeWav } from "../src/wav.js";
import { MaskingObservations, maskingComparison, maskingFeatures, maskingReversal, maskingVariance, scoreMaskedSequence, validateMaskingModel } from "../src/masking-evidence.js";
import type { MaskingModel } from "../src/masking-evidence.js";

const row = Array(24).fill(0) as number[];
row[1] = 8;
function model(): MaskingModel {
  const means = [row.slice(), row.slice(), row.slice()];
  means[1][0] = 4;
  means[2][5] = 1;
  return {schema:1,inventory:"test",tokens:"بت",frameMs:40,dimensions:24,center:Array(24).fill(0),scale:Array(24).fill(1),
    means,variances:means.map(()=>Array(24).fill(.1)),counts:[0,10,10],strength:1,maxFrames:512};
}
const candidate = {word:{surah:1,ayah:1,word:1},written:"ب",rival:"ت",fromFrame:0,toFrame:1};

describe("simultaneous masking observations", () => {
  it("discounts only weaker nearby spectral cues, with upward masking stronger", () => {
    const u=maskingVariance(row,1);
    expect(u[1]).toBe(0);
    expect(u[5]).toBe(0);
    expect(u[2]).toBeGreaterThan(u[0]);
    expect(maskingVariance(Array(24).fill(0),1)).toEqual(Array(24).fill(0));
    expect(maskingVariance(row,0)).toEqual(Array(24).fill(0));
  });
  it("observes the existing analysis without changing primary log mel", () => {
    const wav=decodeWav(readFileSync(new URL("./fixtures/tlog-clean-00010-630.wav",import.meta.url)));
    let observations=0;
    const observed=new Fbank().push(wav.samples,p=>{expect(p.length).toBe(256); observations++;});
    expect(observed).toEqual(new Fbank().push(wav.samples));
    const features=maskingFeatures(wav.samples);
    expect(features.length).toBe(Math.floor(observations/4));
    expect(features.every(r=>r.length===24 && r.every(Number.isFinite))).toBe(true);
    const stream=new MaskingObservations();
    for (let a=0;a<wav.samples.length;a+=1234) stream.push(wav.samples.subarray(a,a+1234));
    expect(stream.frames).toEqual(features);
  });
});

describe("integrated acoustic sequence evidence", () => {
  it("requires a matched control-to-written preference reversal", () => {
    const c=maskingComparison(model(),[row],candidate)!;
    expect(c.controlWritten).toBeLessThan(c.controlRival);
    expect(c.maskedWritten).toBeGreaterThan(c.maskedRival);
    expect(maskingReversal(c)).toBe(true);
    expect(maskingReversal({...c,controlWritten:c.controlRival})).toBe(false);
    expect(maskingReversal({...c,maskedWritten:c.maskedRival})).toBe(false);
    expect(maskingComparison({...model(),strength:0},[row],candidate)?.maskedWritten).toBe(c.controlWritten);
  });
  it("sums every complete phone placement with a normalized boundary prior", () => {
    const m=model(), raw=[row,row,row];
    const a=scoreMaskedSequence(m,[row],"ب",true)!, b=scoreMaskedSequence(m,[row],"ت",true)!;
    const expected=Math.log((Math.exp(2*a+b)+Math.exp(a+2*b))/2);
    expect(scoreMaskedSequence(m,raw,"بت",true)).toBeCloseTo(expected,9);
    expect(scoreMaskedSequence(m,[row],"بب",true)).toBeNull();
    expect(scoreMaskedSequence(m,[row,row],"بب",true)).toBeCloseTo(2*a,9);
  });
  it("abstains on unknown or untrained phones and invalid observations", () => {
    expect(scoreMaskedSequence(model(),[row],"ز",true)).toBeNull();
    expect(scoreMaskedSequence({...model(),counts:[0,0,10]},[row],"ب",true)).toBeNull();
    expect(scoreMaskedSequence(model(),[[NaN,...row.slice(1)]],"ب",true)).toBeNull();
    expect(()=>validateMaskingModel({...model(),variances:[[0]]},"test","بت")).toThrow();
    expect(()=>validateMaskingModel(model(),"other","بت")).toThrow();
    expect(validateMaskingModel(model(),"test","بت")).toEqual(model());
  });
});
