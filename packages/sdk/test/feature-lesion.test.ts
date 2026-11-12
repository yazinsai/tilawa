import { describe, expect, it } from "vitest";
import type { InferenceBackend, ModelManifest } from "../src/model.js";
import type { CheckerEvent } from "@tilawa/core";
import { createListener } from "../src/listener.js";
import { featureLesionComparison, featureLesionReversal, featureLesionSequenceScore,
  featureLesionObservations, featureLesionVerdictRemovals, meanLesionedLogProbs, validateFeatureLesionModel, type FeatureLesionModel } from "../src/feature-lesion.js";

const model: FeatureLesionModel = { schema: 1, inventory: "test", tokens: "بت", primarySha256: "a".repeat(64),
  width: 1, featureOutput: "features", logitOutput: "logits", effectiveWeight: [[0,-2,2]], bias: [0,0,0], mean: [-1], mask: [0] };
const manifest = { inventory: model.inventory, tokens: model.tokens, sha256: model.primarySha256 } as ModelManifest;
const request = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", rival: "ت", fromFrame: 10, toFrame: 11 };
const softmax = (z: number[]) => { const n=Math.log(z.reduce((s,v)=>s+Math.exp(v),0)); return z.map(v=>v-n); };

describe("frozen causal feature lesion", () => {
  it("sums only paths spelling the complete sequence, including repeated phones", () => {
    const probabilities = [[.2,.5,.3],[.4,.2,.4],[.3,.6,.1]];
    for (const sequence of [[],[1],[1,1],[1,2],[1,2,1]]) {
      let mass=0;
      for (let a=0;a<3;a++) for (let b=0;b<3;b++) for (let c=0;c<3;c++) {
        const path=[a,b,c], collapsed=path.filter((p,i)=>p!==0 && (i===0 || p!==path[i-1]));
        if (JSON.stringify(collapsed)===JSON.stringify(sequence)) mass+=probabilities[0][a]*probabilities[1][b]*probabilities[2][c];
      }
      expect(Math.exp(featureLesionSequenceScore(probabilities.map(r=>r.map(Math.log)),sequence))).toBeCloseTo(mass,12);
    }
    expect(featureLesionSequenceScore([[Math.log(.1),Math.log(.9),-Infinity]],[1,1])).toBe(-Infinity);
  });
  it("replaces the selected channel with its mean under unchanged head weights", () => {
    const logits=[[0,-2,2]], features=[[1]];
    meanLesionedLogProbs(model,features,logits)[0].forEach((v,i)=>expect(v).toBeCloseTo(softmax([0,2,-2])[i],12));
    meanLesionedLogProbs({...model,mask:[]},features,logits)[0].forEach((v,i)=>expect(v).toBeCloseTo(softmax(logits[0])[i],12));
    expect(features).toEqual([[1]]);
    expect(logits).toEqual([[0,-2,2]]);
  });
  it("certifies only a finite strict reversal on the complete exact window", () => {
    const lp=[softmax([0,-2,2])];
    const result=featureLesionComparison(model,[[1]],[[0,-2,2]],lp,request)!;
    expect(featureLesionReversal(result)).toBe(true);
    expect(featureLesionReversal({...result,originalWritten:result.originalRival})).toBe(false);
    expect(featureLesionReversal({...result,lesionedWritten:Infinity})).toBe(false);
    expect(featureLesionComparison(model,[[1]],[[0,-2,2]],lp,{...request,toFrame:12})).toBeNull();
    expect(featureLesionComparison(model,[[1]],[[0,-2,2]],lp,{...request,written:"بب"})).toBeNull();
    expect(featureLesionComparison(model,[[1]],[[0,-2,2]],lp,{...request,rival:"x"})).toBeNull();
  });
  it("binds a nonempty sparse mask to the exact shipped graph and alphabet", () => {
    expect(validateFeatureLesionModel(model,manifest)).toBe(model);
    for (const altered of [{...model,mask:[]},{...model,mask:[0,0]},{...model,mask:[1]},
      {...model,mean:[NaN]},{...model,primarySha256:"b".repeat(64)},{...model,tokens:"تب"}]) {
      expect(()=>validateFeatureLesionModel(altered,manifest)).toThrow("Invalid frozen feature lesion");
    }
  });
  it("retracts only an identical live wrong-word occurrence removed by certified replay",()=> {
    const wrong={id:"wrong",timeMs:1000,location:request.word,expected:"ب",observed:"ت",errorClass:"wrong_word"} as CheckerEvent;
    const added={...wrong,id:"added",errorClass:"added_word"} as CheckerEvent;
    const other={...wrong,id:"other",timeMs:1040};
    expect(featureLesionVerdictRemovals([wrong,added,other],[wrong,added],[])).toEqual([wrong]);
    expect(featureLesionVerdictRemovals([wrong],[wrong],[wrong])).toEqual([]);
    expect(featureLesionVerdictRemovals([other],[wrong],[])).toEqual([]);
  });
  it("replays the listener's full stateful stream, including its padded final window", async () => {
    const clock = { ...manifest, sampleRate:16000, frameMs:40, window:45, shift:32,
      fbank:{mels:80,padValue:-23.0259}, states:[{name:"state",shape:[1],dtype:"float32"}] } as ModelManifest;
    const run = () => {
      const calls: number[][]=[];
      const backend: InferenceBackend={name:"test",release:async()=>{},run:async feeds=> {
        const state=Number(feeds.state.data[0]);
        calls.push([...Array.from(feeds.x.data as Float32Array),state]);
        return {new_state:{type:"float32",dims:[1],data:new Float32Array([state+1])},
          log_probs:{type:"float32",dims:[1,8,3],data:Float32Array.from(Array.from({length:8},()=>softmax([0,-2,2])).flat())},
          features:{type:"float32",dims:[1,8,1],data:new Float32Array(8).fill(state)},
          logits:{type:"float32",dims:[1,8,3],data:Float32Array.from(Array.from({length:8},()=>[0,-2,2]).flat())}};
      }};
      return {backend,calls};
    };
    const primary=run(), side=run(), audio=Float32Array.from({length:32000},(_,i)=>.1*Math.sin(i*.1));
    const listener=createListener({manifest:clock,backend:primary.backend,supported:[]},{posteriors:true});
    for(let a=0;a<audio.length;a+=5120) await listener.push(audio.subarray(a,a+5120),16000);
    await listener.end();
    const observations=await featureLesionObservations(side.backend,clock,model,audio);
    expect(side.calls).toEqual(primary.calls);
    expect(observations.features).toHaveLength(listener.decoded);
    expect(observations.logProbs).toHaveLength(listener.decoded);
  });
});
