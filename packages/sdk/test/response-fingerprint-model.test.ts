import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadNodeModel } from "../src/node.js";
import { createListener } from "../src/listener.js";
import { fingerprintMetadata, fingerprintObservations } from "../src/response-fingerprint.js";

describe("declared response-fingerprint bundle",()=>{
  it("executes symmetric interventions on a real streaming graph and preserves every baseline posterior",async()=>{
    const model=await loadNodeModel(new URL("./fixtures/tiny/",import.meta.url).pathname);
    try {
      const audio=Float32Array.from({length:16000},(_,i)=>.02*Math.sin(i*.17));
      const rows=await fingerprintObservations(model.backend,model.manifest,fingerprintMetadata(model.manifest),audio,f=>f===0);
      const vocab=model.manifest.tokens.length+1;
      expect(rows.response[0]).toHaveLength(2*vocab);
      expect(rows.response[0].every(Number.isFinite)).toBe(true);
      const listener=createListener(model,{posteriors:true,topK:vocab});
      const chunks=[await listener.push(audio,16000),await listener.end()].filter(e=>e!==null);
      expect(chunks.reduce((n,e)=>n+e.posteriors!.nFrames,0)).toBe(rows.logProbs.length);
      for(const chunk of chunks) {
        const p=chunk.posteriors!;
        for(let t=0;t<p.nFrames;t++) for(let k=0;k<vocab;k++) {
          const id=p.ids[t*vocab+k],expected=Math.exp(rows.logProbs[p.startFrame+t][id]);
          expect(p.probs[t*vocab+k]).toBeCloseTo(expected,3);
        }
      }
    } finally {await model.backend.release();}
  });
  it("rejects missing, corrupt and orphan fitted assets instead of using the primary",async()=>{
    const dir=await mkdtemp(join(process.cwd(),".fingerprint-bundle-"));
    const fixture=new URL("./fixtures/tiny/",import.meta.url);
    try {
      for(const name of ["manifest.json","model.onnx"])await writeFile(join(dir,name),await readFile(new URL(name,fixture)));
      const descriptor={schema:1,files:[
        {file:"response-fingerprint-emissions.json",bytes:2,sha256:createHash("sha256").update("{}").digest("hex")},
      ]};
      await writeFile(join(dir,"response-fingerprint-asset.json"),JSON.stringify(descriptor));
      await expect(loadNodeModel(dir)).rejects.toThrow();
      await writeFile(join(dir,"response-fingerprint-emissions.json"),"[]");
      await expect(loadNodeModel(dir)).rejects.toThrow("checksum mismatch");
      await rm(join(dir,"response-fingerprint-asset.json"));
      await expect(loadNodeModel(dir)).rejects.toThrow("lack their asset declaration");
    } finally {await rm(dir,{recursive:true,force:true});}
  });
});
