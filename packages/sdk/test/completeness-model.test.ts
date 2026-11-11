import { createHash } from "node:crypto";
import { mkdtemp,readFile,rm,writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe,expect,it } from "vitest";
import { loadNodeModel } from "../src/node.js";

describe("declared pronunciation completeness bundle",()=>{
  it("rejects missing and corrupt declared files before decoding",async()=>{
    const dir=await mkdtemp(join(process.cwd(),".completeness-bundle-"));
    const fixture=new URL("./fixtures/tiny/",import.meta.url);
    try {
      for(const name of ["manifest.json","model.onnx"])await writeFile(join(dir,name),await readFile(new URL(name,fixture)));
      const descriptor=Object.fromEntries(["pronunciation-completeness.json","completeness-features.onnx","completeness-head.onnx"].map(file=>
        [file,{bytes:2,sha256:createHash("sha256").update("{}").digest("hex")}]));
      await writeFile(join(dir,"completeness-assets.json"),JSON.stringify(descriptor));
      await expect(loadNodeModel(dir)).rejects.toThrow();
      await writeFile(join(dir,"pronunciation-completeness.json"),"[]");
      await expect(loadNodeModel(dir)).rejects.toThrow("checksum mismatch");
    } finally {await rm(dir,{recursive:true,force:true});}
  });
});
