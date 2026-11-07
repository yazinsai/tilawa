import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { createListener } from "../src/listener.js";
import { pauseFloor } from "../src/pause-mask.js";
import type { LoadedModel, ModelManifest, Tensor } from "../src/model.js";
describe("complete pause-trained streaming input",()=>{
  it("keeps the original native mask clock and leaves frames outside its span unmasked",()=>{
    expect(Array.from(pauseFloor([{fromFrame:2,toFrame:4}],0,10))).toEqual([-1n,-1n,-1n,-1n,4n,4n,4n,4n,-1n,-1n]);
  });
  it("supplies the mask input on every streaming call, including unmasked full-session frames",async()=>{
    const m=JSON.parse(readFileSync(new URL("./fixtures/tiny/manifest.json",import.meta.url),"utf8")) as ModelManifest;
    m.pauseMask={input:"pause_floor",clockMs:20};
    const run=vi.fn(async(feeds:Record<string,Tensor>)=>({
      log_probs:{type:"float32" as const,dims:[1,m.shift/4,m.tokens.length+1],data:new Float32Array(m.shift/4*(m.tokens.length+1))},
      ...Object.fromEntries(m.states.map(s=>[`new_${s.name}`,feeds[s.name]])),
    }));
    const model:LoadedModel={manifest:m,supported:[],backend:{name:"test",run,release:async()=>{}}};
    const listener=createListener(model);
    await listener.push(new Float32Array(16000),16000);await listener.end();
    expect(run.mock.calls.length).toBeGreaterThan(1);
    for(const [feeds] of run.mock.calls){
      expect(feeds.pause_floor.dims).toEqual([1,m.shift/2]);
      expect(Array.from(feeds.pause_floor.data)).toEqual(new Array(m.shift/2).fill(-1n));
    }
  });
});
