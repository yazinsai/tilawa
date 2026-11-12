import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import { BlankObservations, blankComparison, blankFeatures, blankObservation, blankReversal,
  scoreBlankSequence, validateBlankCauseModel } from "../src/blank-cause.js";
import type { BlankCauseModel } from "../src/blank-cause.js";

const model=():BlankCauseModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,dimensions:4,
  center:[0,0,0,0],scale:[1,1,1,1],means:[[-1,0,0,0],[1,0,1,0]],variances:[[1,1,1,1],[1,1,1,1]],
  initial:[.5,.5],transition:[[.9,.1],[.1,.9]],maxFrames:2048});
const rows=[[.1,.8,.1],[.8,.1,.1],[.8,.1,.1],[.1,.1,.8]].map(r=>r.map(Math.log));
const features=[[1,.2,.9],[-1,.1,.1],[-1,.1,.1],[1,.2,.9]];

/** Independent exhaustive enumeration of emitted tokens and acoustic states, including complete CTC collapse. */
function enumerate(m:BlankCauseModel,pronunciation:string,dynamic:boolean):number {
  let total=0;
  const a=m.transition[0][1],b=m.transition[1][0],prior=[b/(a+b),a/(a+b)];
  for (let output=0;output<3**rows.length;output++) {
    let encoded=output;const ids=[];
    for (let t=0;t<rows.length;t++) {ids.push(encoded%3);encoded=Math.floor(encoded/3);}
    const collapsed=ids.filter((id,t)=>id!==0&&(t===0||id!==ids[t-1])).map(id=>m.tokens[id-1]).join("");
    if (collapsed!==pronunciation) continue;
    for (let latent=0;latent<2**rows.length;latent++) {
      const z=Array.from({length:rows.length},(_,t)=>(latent>>t)&1);
      let p=prior[z[0]];
      for (let t=0;t<rows.length;t++) {
        if (t) p*=dynamic?m.transition[z[t-1]][z[t]]:prior[z[t]];
        const x=blankObservation(features[t],rows[t]);
        for (let d=0;d<4;d++) {
          const y=(x[d]-m.center[d])/m.scale[d],variance=m.variances[z[t]][d];
          p*=Math.exp(-.5*(y-m.means[z[t]][d])**2/variance)/Math.sqrt(2*Math.PI*variance);
        }
        p*=z[t]===0?(ids[t]===0?1:0):Math.exp(rows[t][ids[t]]);
      }
      total+=p;
    }
  }
  return Math.log(total);
}

describe("normalized two-cause blank sequence evidence",()=> {
  it("sums every complete token/state path, including repeated labels and edge blanks",()=> {
    for (const p of ["","a","ab","aa","aba"]) for (const dynamic of [false,true])
      expect(scoreBlankSequence(model(),rows,features,p,dynamic)).toBeCloseTo(enumerate(model(),p,dynamic),11);
  });
  it("does not accept partial pronunciations or collapse repeated consonants without a blank",()=> {
    expect(scoreBlankSequence(model(),rows.slice(0,1),features.slice(0,1),"aa",true)).toBe(-Infinity);
    expect(scoreBlankSequence(model(),rows.slice(0,2),features.slice(0,2),"aba",true)).toBe(-Infinity);
  });
  it("equals the collapsed control when state dynamics are independent",()=> {
    const m=model();m.transition=[[.3,.7],[.3,.7]];
    for (const p of ["","a","ab","aa"]) expect(scoreBlankSequence(m,rows,features,p,true))
      .toBeCloseTo(scoreBlankSequence(m,rows,features,p,false)!,11);
  });
  it("makes blank cause persistence affect complete comparisons symmetrically",()=> {
    const dynamic=scoreBlankSequence(model(),rows,features,"ab",true)!-scoreBlankSequence(model(),rows,features,"aa",true)!;
    const control=scoreBlankSequence(model(),rows,features,"ab",false)!-scoreBlankSequence(model(),rows,features,"aa",false)!;
    expect(Math.abs(dynamic-control)).toBeGreaterThan(.001);
    const swapped=scoreBlankSequence(model(),rows,features,"aa",true)!-scoreBlankSequence(model(),rows,features,"ab",true)!;
    expect(swapped).toBe(-dynamic);
  });
  it("requires finite strict reversals, compatible rows, and exact complete windows",()=> {
    const frames=new PosteriorFrames(2);
    frames.append({k:3,startFrame:0,nFrames:4,ids:[0,1,2,0,1,2,0,1,2,0,1,2],probs:rows.flatMap(r=>r.map(Math.exp))});
    const candidate={word:{surah:1,ayah:1,word:1},written:"ab",rival:"aa",fromFrame:0,toFrame:4};
    const c=blankComparison(model(),features,frames,candidate)!;
    expect(c.written).toBe("ab");expect(c.toFrame).toBe(4);
    expect(blankComparison(model(),features,frames,{...candidate,toFrame:5})).toBeNull();
    expect(scoreBlankSequence(model(),rows,features,"c",true)).toBeNull();
    expect(scoreBlankSequence(model(),rows,features.slice(1),"ab",true)).toBeNull();
    expect(blankReversal({...c,controlWritten:-2,controlRival:-1,stateWritten:-1,stateRival:-2})).toBe(true);
    expect(blankReversal({...c,controlWritten:-1,controlRival:-1,stateWritten:-1,stateRival:-2})).toBe(false);
    expect(blankReversal({...c,controlWritten:-2,controlRival:-1,stateWritten:Infinity,stateRival:-2})).toBe(false);
  });
  it("rejects unnormalized state transitions and incompatible assets",()=> {
    expect(validateBlankCauseModel(model(),"test","ab")).toEqual(model());
    expect(()=>validateBlankCauseModel({...model(),transition:[[.9,.2],[.1,.9]]},"test","ab")).toThrow();
    expect(()=>validateBlankCauseModel(model(),"other","ab")).toThrow();
    expect(()=>validateBlankCauseModel({...model(),variances:[[0,1,1,1],[1,1,1,1]]},"test","ab")).toThrow();
  });
  it("extracts causal observations independent of audio chunking on the native CTC clock",()=> {
    const audio=Float32Array.from({length:8200},(_,i)=>.1*Math.sin(2*Math.PI*150*i/16000));
    const whole=blankFeatures(audio),stream=new BlankObservations();
    for (let from=0;from<audio.length;from+=137) stream.push(audio.subarray(from,from+137));
    expect(stream.frames).toEqual(whole);
    const fbankFrames=1+Math.floor((audio.length-400)/160);
    expect(whole.length).toBe(Math.floor((Math.floor((fbankFrames-7)/2)+1)/2));
    expect(whole.every(r=>r.length===3&&r.every(Number.isFinite))).toBe(true);
    expect(blankFeatures(new Float32Array(4000)).every(r=>r[1]===0&&r[2]===0)).toBe(true);
  });
});
