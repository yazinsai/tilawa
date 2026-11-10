import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
const manifest={file:"primary.onnx",inventory:"toy",tokens:"بت",sampleRate:16000,frameMs:40};
const primary={name:"primary",run:vi.fn(),release:vi.fn()};
const asset={schema:1,inventory:"toy",tokens:"بت",frameMs:40,observationMs:10,dimensions:23,
  center:Array(23).fill(0),scale:Array(23).fill(1),means:Array.from({length:3},()=>Array(20).fill(0)),
  variances:Array.from({length:3},()=>Array(20).fill(1)),offsets:Array.from({length:3},()=>Array(20).fill(0)),
  qualityMeans:Array.from({length:3},()=>Array(3).fill(0)),qualityVariances:Array.from({length:3},()=>Array(3).fill(1)),
  counts:[1,1,1],stay:[.5,.5,.5],initial:[1/3,1/3,1/3],transition:Array.from({length:3},()=>[1/3,1/3,1/3]),maxFrames:4096};
const bytes=Buffer.from(JSON.stringify(asset));
const descriptor={schema:1,file:"phonation-emissions.json",bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")};
beforeEach(()=>{
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  mocks.ortBackend.mockResolvedValue(primary);
  mocks.readFile.mockImplementation(async (path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
    path.endsWith("phonation-asset.json")?JSON.stringify(descriptor):bytes);
});
describe("verified phonation bundle loading",()=>{
  it("loads the side model while retaining the original primary backend and manifest",async()=>{
    const model=await loadNodeModel("/models/caller");
    expect(model.phonation).toEqual(asset);expect(model.backend).toBe(primary);expect(model.manifest).toEqual(manifest);
    expect(primary.run).not.toHaveBeenCalled();expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it("allows an undeclared primary-only bundle",async()=>{
    mocks.readFile.mockImplementation(async(path:string)=>{
      if(path.endsWith("manifest.json"))return JSON.stringify(manifest);
      throw Object.assign(new Error("absent"),{code:"ENOENT"});
    });
    expect((await loadNodeModel("/models/caller")).phonation).toBeUndefined();
  });
  it("rejects a declared missing file and releases the primary backend",async()=>{
    mocks.readFile.mockImplementation(async(path:string)=>{
      if(path.endsWith("manifest.json"))return JSON.stringify(manifest);
      if(path.endsWith("phonation-asset.json"))return JSON.stringify(descriptor);
      throw Object.assign(new Error("absent"),{code:"ENOENT"});
    });
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("absent");
    expect(primary.release).toHaveBeenCalledOnce();
  });
  it("rejects corruption and path traversal before inference",async()=>{
    mocks.readFile.mockImplementation(async(path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
      path.endsWith("phonation-asset.json")?JSON.stringify({...descriptor,file:"../phonation-emissions.json"}):bytes);
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("descriptor");
    mocks.readFile.mockImplementation(async(path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
      path.endsWith("phonation-asset.json")?JSON.stringify(descriptor):Buffer.from("bad"));
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("checksum");
    expect(primary.run).not.toHaveBeenCalled();
  });
});
