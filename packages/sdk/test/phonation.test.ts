import { describe, expect, it } from "vitest";
import { carryPhonationEvidence, phonationComparison, phonationFeatures, PhonationObservations, phonationReversal, scorePhonationSequence, validatePhonationModel } from "../src/phonation.js";
import type { PhonationModel } from "../src/phonation.js";
import type { PhonemeEvidence } from "@tilawa/core";
import { insertionCells } from "../src/seam-insertion.js";

export const toyModel=():PhonationModel => ({
  schema:1,inventory:"toy",tokens:"بت",frameMs:40,observationMs:10,dimensions:23,
  center:Array(23).fill(0),scale:Array(23).fill(1),
  means:[0,-1,1].map(v=>[v,...Array(19).fill(0)]),variances:Array.from({length:3},()=>Array(20).fill(.1)),
  offsets:[-2,0,2].map(v=>[v,...Array(19).fill(0)]),qualityMeans:[-2,0,2].map(v=>[v,0,0]),
  qualityVariances:Array.from({length:3},()=>Array(3).fill(.1)),counts:[20,20,20],stay:[.5,.5,.5],
  initial:[1/3,1/3,1/3],transition:[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]],maxFrames:4096,
});
const row=(mel:number,quality:number)=>[mel,...Array(19).fill(0),quality,0,0];
const candidate={word:{surah:1,ayah:1,word:1},written:"بت",rival:"تب",fromFrame:0,toFrame:1};

describe("shared within-word phonation evidence",()=> {
  it("reverses a fixed-state preference only when changing phonation explains the written articulation",()=> {
    const comparison=phonationComparison(toyModel(),[row(0,0),row(1,2),row(-1,-2),row(0,0)],candidate)!;
    expect(phonationReversal(comparison)).toBe(true);
    const realSwap=phonationComparison(toyModel(),[row(0,0),row(1,0),row(-1,0),row(0,0)],candidate)!;
    expect(phonationReversal(realSwap)).toBe(false);
    expect(realSwap.switchingRival>realSwap.switchingWritten).toBe(true);
  });
  it("matches exhaustive phone-boundary and phonation-state enumeration in both arms",()=> {
    const m=toyModel(), raw=[row(0,0),row(1,2),row(1,2),row(-1,-2),row(-1,-2),row(0,0)];
    for(const switching of [false,true]) {
      let mass=0,prior=0;
      const visit=(t:number,s:number,z:number,weight:number,pweight:number)=> {
        const p=[0,1,2,0][s], x=raw[t];
        for(let d=0;d<23;d++) {
          const mean=d<20?m.means[p][d]+m.offsets[z][d]:m.qualityMeans[z][d-20];
          const variance=d<20?m.variances[p][d]:m.qualityVariances[z][d-20];
          weight*=Math.exp(-.5*(Math.log(2*Math.PI*variance)+(x[d]-mean)**2/variance));
        }
        if(t===raw.length-1) { if(s===3){mass+=weight;prior+=pweight;}return; }
        for(const next of [s,s+1])if(next<4)for(let q=0;q<3;q++) {
          const zp=switching?m.transition[z][q]:(q===z?1:0);
          if(zp)visit(t+1,next,q,weight*.5*zp,pweight*.5*zp);
        }
      };
      for(let z=0;z<3;z++)visit(0,0,z,m.initial[z],m.initial[z]);
      expect(scorePhonationSequence(m,raw,"بت",switching)).toBeCloseTo(Math.log(mass/prior),10);
    }
  });
  it("uses the same prior when swapping candidate order and requires every repeated phone",()=> {
    const m=toyModel(),raw=[row(0,0),row(1,2),row(-1,-2),row(0,0)];
    const a=phonationComparison(m,raw,candidate)!,b=phonationComparison(m,raw,{...candidate,written:candidate.rival,rival:candidate.written})!;
    expect(a.fixedWritten).toBe(b.fixedRival);expect(a.switchingWritten).toBe(b.switchingRival);
    expect(scorePhonationSequence(m,raw,"ببب",true)).toBeNull();
    expect(scorePhonationSequence(m,raw,"خ",true)).toBeNull();
    m.counts[1]=0;expect(scorePhonationSequence(m,raw,"بت",true)).toBeNull();
  });
  it("is unchanged by audio push boundaries and produces finite silence observations",()=> {
    const audio=Float32Array.from({length:3200},(_,i)=>.1*Math.sin(2*Math.PI*130*i/16000));
    const obs=new PhonationObservations();
    for(let i=0;i<audio.length;i+=73)obs.push(audio.subarray(i,i+73));
    expect(obs.frames).toEqual(phonationFeatures(audio));
    const silence=phonationFeatures(new Float32Array(1600));
    expect(silence.every(r=>r.length===23&&r.every(Number.isFinite))).toBe(true);
  });
  it("rejects incompatible or corrupt shared-state parameters",()=> {
    const m=toyModel();expect(validatePhonationModel(m,"toy","بت")).toBe(m);
    expect(()=>validatePhonationModel({...m,initial:[1,1,1]},"toy","بت")).toThrow();
    expect(()=>validatePhonationModel({...m,transition:[[1,0,0]]},"toy","بت")).toThrow();
    expect(()=>validatePhonationModel(m,"other","بت")).toThrow();
    expect(()=>validatePhonationModel({...m,variances:[[0]]},"toy","بت")).toThrow();
  });
  it("transports exact reversal evidence without changing primary rows, tokens or insertion metadata",()=>{
    const certificate={...candidate,fixedWritten:-2,fixedRival:-1,switchingWritten:1,switchingRival:0};
    const chunks:PhonemeEvidence[]=[{inventory:"toy",frameMs:40,phonemes:[{p:"ب",start:1,end:2,conf:.9,
      seamInsertions:[{word:candidate.word,observed:"بت",fromFrame:1,toFrame:3,marginal:.8}]}],
      posteriors:{k:1,startFrame:0,nFrames:4,ids:[0,1,1,0],probs:[1,1,1,1]}}];
    const original=structuredClone(chunks),enriched=carryPhonationEvidence(chunks,[certificate]);
    expect(chunks).toEqual(original);expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes.map(({phonationVetoes:_,...p})=>p)).toEqual(chunks[0].phonemes);
    const cells=insertionCells(JSON.parse(JSON.stringify(enriched)));
    expect(cells.flatMap(e=>e.phonemes)[0].phonationVetoes).toEqual([certificate]);
    expect(cells.flatMap(e=>e.phonemes)[0].seamInsertions).toEqual(chunks[0].phonemes[0].seamInsertions);
    expect(cells.flatMap(e=>Array.from(e.posteriors!.ids))).toEqual(chunks[0].posteriors!.ids);
  });
});
