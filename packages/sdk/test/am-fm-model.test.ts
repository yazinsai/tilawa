import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
const manifest={file:"model.onnx",tokens:"ab",inventory:"test",sampleRate:16000,frameMs:40};
const side={schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,bands:[0],center:[[0,0]],scale:[[1,1]],
  means:[[0,0],[0,0],[0,0]],variances:[[1,1],[1,1],[1,1]],covariance:[[0],[.2],[-.2]],counts:[0,10,10],stay:.7,maxFrames:4096};
const bytes=Buffer.from(JSON.stringify(side)),descriptor={schema:1,file:"am-fm-emissions.json",bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")};
let primary:{name:string;release:ReturnType<typeof vi.fn>};
beforeEach(()=>{
  vi.resetAllMocks();primary={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  mocks.readFile.mockImplementation(async (path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
    path.endsWith("am-fm-asset.json")?JSON.stringify(descriptor):bytes);
});
describe("verified AM-FM side asset loading",()=>{
  it("preserves the primary model and loads a verified scorer",async()=>{
    const loaded=await loadNodeModel("/bundle");expect(loaded.backend).toBe(primary);expect(loaded.manifest).toEqual(manifest);
    expect(loaded.amFm).toEqual(side);expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
  });
  it("accepts an ordinary bundle with no side declaration",async()=>{
    mocks.readFile.mockImplementation(async (path:string)=>{if(path.endsWith("manifest.json"))return JSON.stringify(manifest);
      throw Object.assign(new Error("absent"),{code:"ENOENT"});});
    expect((await loadNodeModel("/bundle")).amFm).toBeUndefined();
  });
  it("fails on a declared missing scorer and releases the backend",async()=>{
    mocks.readFile.mockImplementation(async (path:string)=>{if(path.endsWith("manifest.json"))return JSON.stringify(manifest);
      if(path.endsWith("am-fm-asset.json"))return JSON.stringify(descriptor);throw new Error("missing declared asset");});
    await expect(loadNodeModel("/bundle")).rejects.toThrow("missing declared asset");expect(primary.release).toHaveBeenCalledOnce();
  });
  it("fails on corrupt bytes and releases the backend",async()=>{
    mocks.readFile.mockImplementation(async (path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
      path.endsWith("am-fm-asset.json")?JSON.stringify(descriptor):Buffer.from("corrupt"));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("checksum");expect(primary.release).toHaveBeenCalledOnce();
  });
});
