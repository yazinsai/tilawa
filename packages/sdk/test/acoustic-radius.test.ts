import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { estimateRadius, pauseBasis, searchRadius, RADIUS_SEARCH } from "../src/acoustic-radius.js";
import { createListener } from "../src/listener.js";
import { PosteriorFrames } from "@tilawa/core";
import type { LoadedModel, Tensor } from "../src/model.js";
import { insertionCells } from "../src/seam-insertion.js";
import type { PhonemeEvidence } from "@tilawa/core";

describe("background decision radius", () => {
  it("finds the minimum-energy direction rather than counting perturbed agreements", async () => {
    const result=await searchRadius(async x=>30-10*x[0]+3*x[1],20,{...RADIUS_SEARCH,dimensions:2});
    expect(result.energy).not.toBeNull();
    expect(result.energy!).toBeCloseTo(100/109,2);
    expect(result.boundary!).toBeLessThan(20);
    const again=await searchRadius(async x=>30-10*x[0]+3*x[1],20,{...RADIUS_SEARCH,dimensions:2});
    expect(again).toEqual(result);
  });
  it("abstains on a failed search and on an already absent confirmation",async()=> {
    expect((await searchRadius(async()=>30,20,RADIUS_SEARCH)).energy).toBeNull();
    expect((await searchRadius(async()=>19,20,RADIUS_SEARCH)).energy).toBeNull();
    expect((await searchRadius(async()=>NaN,20,RADIUS_SEARCH)).energy).toBeNull();
  });
  it("uses only preceding pause samples and measures additive waveform energy",()=> {
    let state=1;
    const audio=Float32Array.from({length:64000},(_,i)=> {
      state=(Math.imul(state,1664525)+1013904223)>>>0;
      return (state/2**32-.5)*(i<16000?.002:.2);
    });
    const chunks:PhonemeEvidence[]=[{inventory:"test",frameMs:40,phonemes:[
      {p:"ب",start:25,end:35,conf:.9},{p:"ت",start:36,end:45,conf:.9},
    ]}];
    const basis=pauseBasis(audio,chunks,50,8000,4)!;
    expect(basis).not.toBeNull();
    const coefficients=[.2,-.3,.4,.1];
    const energy=basis.columns[0].reduce((s,_,i)=>s+basis.columns.reduce((v,b,j)=>v+b[i]*coefficients[j],0)**2,0)/8000;
    expect(energy).toBeCloseTo(coefficients.reduce((s,x)=>s+x*x,0),10);
    const changed=audio.slice();changed.fill(1,50*640);
    expect(pauseBasis(changed,chunks,50,8000,4)).toEqual(basis);
    expect(pauseBasis(audio,chunks,20,8000,4)).toBeNull();
    expect(pauseBasis(new Float32Array(64000),chunks,50,8000,4)).toBeNull();
  });
  it("recomputing sample-aligned local analysis frames equals full waveform analysis",()=> {
    const audio=Float32Array.from({length:32000},(_,i)=>.1*Math.sin(i*.07));
    const from=12000,to=18000,changed=audio.slice();
    for(let i=from;i<to;i++) changed[i]+=.001*Math.cos(i*.38);
    const base=new Fbank().push(audio), full=new Fbank().push(changed);
    const first=Math.max(0,Math.ceil((from-399)/160)), last=Math.min(base.length,Math.ceil(to/160));
    const local=new Fbank().push(changed.subarray(first*160,Math.min(audio.length,(last-1)*160+400)));
    for(let i=0;i<local.length;i++) base[first+i]=local[i];
    expect(base).toEqual(full);
  });
  it("retains a veto certificate through ordinary JSON and native evidence cells",()=> {
    const evidence:PhonemeEvidence[]=[{inventory:"test",frameMs:40,phonemes:[{p:"ب",start:1,end:1,conf:.9,
      acousticRadii:[{word:{surah:1,ayah:2,word:1},written:"ب",accused:"ت",fromFrame:1,toFrame:3,
        need:10,baselineMargin:20,boundaryMargin:9,energy:.1,rejectionEnergy:.2}]}],
      posteriors:{k:1,startFrame:0,nFrames:3,ids:[0,1,0],probs:[1,1,1]}}];
    const cells=insertionCells(JSON.parse(JSON.stringify(evidence)));
    expect(cells.flatMap(e=>e.phonemes)).toEqual(evidence[0].phonemes);
    expect(cells.flatMap(e=>Array.from(e.posteriors!.ids))).toEqual([0,1,0]);
  });
  it("reproduces the native stateful stream before accepting any perturbation result",async()=> {
    let rng=1;
    const audio=Float32Array.from({length:64000},(_,i)=> {
      rng=(Math.imul(rng,1664525)+1013904223)>>>0;
      return (rng/2**32-.5)*(i<16000?.002:.2);
    });
    const model:LoadedModel={supported:["wrong_word"],manifest:{schema:1,id:"stub",format:"float32",arch:"zipformer2-streaming-ctc",
      inventory:"test",tokens:"بت",blank:0,sampleRate:16000,frameMs:40,fbank:{mels:80,padValue:0},window:17,shift:8,
      states:[{name:"history",shape:[1],dtype:"float32"}],file:"",bytes:0,sha256:""},backend:{name:"stateful-stub",
      async release(){},async run(feeds){
        const history=(feeds.history.data as Float32Array)[0], x=feeds.x.data as Float32Array;
        const change=Math.tanh(history+x[0])*.001;
        const lp=Float32Array.from([.03,.00001,.96999+change,.03,.00001,.96999-change],Math.log);
        return {log_probs:{type:"float32",data:lp,dims:[1,2,3]},
          new_history:{type:"float32",data:Float32Array.of(history*.5+x[0]),dims:[1]}} as Record<string,Tensor>;
      }}};
    const listener=createListener(model,{posteriors:true,topK:3}), raw:PhonemeEvidence[]=[];
    const body=await listener.push(audio,16000),tail=await listener.end();if(body)raw.push(body);if(tail)raw.push(tail);
    const frames=new PosteriorFrames(2);for(const e of raw)frames.append(e.posteriors!);
    // Independent pause bounds are supplied by the primary stream; original rows stay intact.
    const chunks=raw.map((e,i)=>({...e,phonemes:i?[]:[{p:"ب",start:25,end:35,conf:.9},{p:"ت",start:36,end:45,conf:.9}]}));
    const margin=frames.nll([1],50,60)-frames.nll([2],50,60);
    const c={word:{surah:1,ayah:2,word:1},written:"ب",accused:"ت",fromFrame:50,toFrame:60,need:1,margin};
    const search={...RADIUS_SEARCH,dimensions:2,iterations:1};
    const result=await estimateRadius(model,audio,chunks,c,search);
    expect(result.status).toBe("no-crossing");
    expect(result.baselineMargin).toBeCloseTo(margin,10);
    expect((await estimateRadius(model,audio,chunks,{...c,margin:margin+1},search)).status).toBe("baseline-mismatch");
  });
});
