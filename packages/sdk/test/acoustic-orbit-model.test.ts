import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadNodeModel } from "../src/node.js";

// Loader rejection must happen before inference: a missing declared scorer cannot silently use the primary.
describe("declared preactivation orbit bundle",()=>{
  it("fails closed for missing and corrupt declared assets",async()=>{
    const dir=await mkdtemp(join(process.cwd(),".orbit-bundle-"));
    const fixture=new URL("./fixtures/tiny/",import.meta.url);
    try {
      for(const name of ["manifest.json","model.onnx"])await writeFile(join(dir,name),await readFile(new URL(name,fixture)));
      const descriptor={schema:1,files:[
        {file:"acoustic-orbit-emissions.json",bytes:2,sha256:createHash("sha256").update("{}").digest("hex")},
        {file:"acoustic-orbit.onnx",bytes:1,sha256:"a".repeat(64)},
      ]};
      await writeFile(join(dir,"acoustic-orbit-asset.json"),JSON.stringify(descriptor));
      await expect(loadNodeModel(dir)).rejects.toThrow();
      await writeFile(join(dir,"acoustic-orbit-emissions.json"),"[]");
      await expect(loadNodeModel(dir)).rejects.toThrow("checksum mismatch");
    } finally {await rm(dir,{recursive:true,force:true});}
  });
});
