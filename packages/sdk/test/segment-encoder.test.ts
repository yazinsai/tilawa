import {describe,expect,it,vi} from "vitest";
import {createSegmentFeatureStream} from "../src/segment-encoder.js";
import type {ModelManifest} from "../src/model.js";

describe("frozen feature stream",()=>{
  it("pools to the primary clock, preserves state per session and truncates final padding",async()=>{
    const run=vi.fn(async feeds=>({
      log_probs:{type:"float32" as const,data:new Float32Array(6),dims:[1,2,3]},
      mid:{type:"float32" as const,data:Float32Array.from([1,2,3,4,5,6,7,8]),dims:[4,1,2]},
      new_cache:{type:"float32" as const,data:Float32Array.from([Number(feeds.cache.data[0])+1]),dims:[1]},
    }));
    const backend={name:"test",run,release:vi.fn()};
    const manifest={states:[{name:"cache",shape:[1],dtype:"float32"}]} as ModelManifest;
    const a=createSegmentFeatureStream(async()=>backend,manifest,"mid",2);
    const b=createSegmentFeatureStream(async()=>backend,manifest,"mid",2);
    const x={type:"float32" as const,data:new Float32Array(80),dims:[1,1,80]};
    await a.step(x,1);await a.step(x,2);await b.step(x,2);
    expect(a.rows.map(r=>[...r])).toEqual([[2,3],[2,3],[6,7]]);
    expect(b.rows.map(r=>[...r])).toEqual([[2,3],[6,7]]);
    expect(run.mock.calls.map(([f])=>Number(f.cache.data[0]))).toEqual([0,1,0]);
    expect(run.mock.calls.every(([f])=>f.x===x)).toBe(true);
  });
  it("fails on an incompatible auxiliary clock",async()=>{
    const backend={name:"test",run:async()=>({log_probs:{type:"float32" as const,data:new Float32Array(6),dims:[1,2,3]},
      mid:{type:"float32" as const,data:new Float32Array(6),dims:[3,1,2]}}),release:async()=>{}};
    const s=createSegmentFeatureStream(async()=>backend,{states:[]} as unknown as ModelManifest,"mid",2);
    await expect(s.step({type:"float32",data:new Float32Array(1),dims:[1]},2)).rejects.toThrow("clock or shape");
  });
});
