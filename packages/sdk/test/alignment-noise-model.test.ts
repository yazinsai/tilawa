import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
const manifest={file:"model.onnx",tokens:"ab",inventory:"test",sampleRate:16000,frameMs:40};
const emission={mean:Array(8).fill(0),variance:Array(8).fill(1),count:10};
const asset={schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:8,
  center:Array(8).fill(0),scale:Array(8).fill(1),control:Array(3).fill(emission),noise:Array(3).fill(emission),
  contamination:[0,.1,.1],stay:.8,maxFrames:2048};
const bytes=Buffer.from(JSON.stringify(asset));
const descriptor={schema:1,file:"alignment-noise-emissions.json",bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")};
beforeEach(()=>{
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  mocks.readFile.mockImplementation(async(file:string)=>file.endsWith("manifest.json")?JSON.stringify(manifest):
    file.endsWith("alignment-noise-asset.json")?JSON.stringify(descriptor):bytes);
  mocks.ortBackend.mockResolvedValue({name:"primary",release:vi.fn()});
});
describe("declared alignment noise bundle",()=>{
  it("loads the two checked emission heads without another primary backend",async()=>{
    const model=await loadNodeModel("/model");
    expect(model.alignmentNoise).toEqual(asset);
    expect(model.manifest).toEqual(manifest);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it("fails when a declared file is missing and releases the primary runtime",async()=>{
    const denied=Object.assign(new Error("missing emissions"),{code:"ENOENT"});
    mocks.readFile.mockImplementation(async(file:string)=>{
      if(file.endsWith("manifest.json"))return JSON.stringify(manifest);
      if(file.endsWith("alignment-noise-asset.json"))return JSON.stringify(descriptor);
      throw denied;
    });
    await expect(loadNodeModel("/model")).rejects.toBe(denied);
    expect((await mocks.ortBackend.mock.results[0].value).release).toHaveBeenCalledOnce();
  });
  it("rejects corrupt bytes instead of falling back to the shipped decoder alone",async()=>{
    mocks.readFile.mockImplementation(async(file:string)=>file.endsWith("manifest.json")?JSON.stringify(manifest):
      file.endsWith("alignment-noise-asset.json")?JSON.stringify({...descriptor,sha256:"0".repeat(64)}):bytes);
    await expect(loadNodeModel("/model")).rejects.toThrow("checksum");
  });
});
