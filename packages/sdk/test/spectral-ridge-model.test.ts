import { beforeEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";

const mocks=vi.hoisted(()=>({access:vi.fn(),readFile:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async importOriginal=>({
  ...await importOriginal<typeof import("node:fs/promises")>(),access:mocks.access,readFile:mocks.readFile,
}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";

const manifest={file:"primary.onnx",tokens:"بت",inventory:"test",sampleRate:16000,frameMs:40,supported:["added_word"]};
const scorer={schema:1,inventory:manifest.inventory,tokens:manifest.tokens,frameMs:40,observationMs:10,dimensions:56,controlDimensions:32,
  radius:1,center:Array(56).fill(0),scale:Array(56).fill(1),counts:[0,1,1],stay:.8,maxFrames:2048,
  means:Array.from({length:3},()=>Array(56).fill(0)),variances:Array.from({length:3},()=>Array(56).fill(1))};
const missing=()=>Object.assign(new Error("absent"),{code:"ENOENT"});
beforeEach(()=>{vi.resetAllMocks();mocks.access.mockRejectedValue(missing());});
function install(asset:Buffer|null,descriptorBytes?:Buffer,descriptorPresent=true) {
  const bytes=Buffer.from(JSON.stringify(scorer));
  const descriptor={schema:1,file:"ridge-emissions.json",bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")};
  mocks.readFile.mockImplementation(async path=>{
    if(String(path).endsWith("/manifest.json"))return JSON.stringify(manifest);
    if(String(path).endsWith("/ridge-asset.json")&&descriptorPresent)return (descriptorBytes??Buffer.from(JSON.stringify(descriptor))).toString("utf8");
    if(String(path).endsWith("/ridge-emissions.json")&&asset)return asset;
    throw missing();
  });
}
describe("Node spectral ridge bundle loading",()=>{
  it("loads a verified side scorer without replacing or executing the primary model",async()=>{
    install(Buffer.from(JSON.stringify(scorer)));
    const primary={name:"primary",run:vi.fn(),release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
    const loaded=await loadNodeModel("/models/caller");
    expect(loaded.backend).toBe(primary);expect(loaded.manifest).toEqual(manifest);expect(loaded.spectralRidge).toEqual(scorer);
    expect(primary.run).not.toHaveBeenCalled();
  });
  it("accepts an ordinary primary bundle with no scorer descriptor",async()=>{
    install(null,undefined,false);mocks.ortBackend.mockResolvedValue({name:"primary"});
    expect((await loadNodeModel("/models/caller")).spectralRidge).toBeUndefined();
  });
  it("fails and releases the backend when the declared scorer is missing",async()=>{
    install(null);const primary={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
    await expect(loadNodeModel("/models/caller")).rejects.toMatchObject({code:"ENOENT"});expect(primary.release).toHaveBeenCalledOnce();
  });
  it("rejects checksum corruption and invalid descriptors",async()=>{
    const primary={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
    install(Buffer.from("{}"));await expect(loadNodeModel("/models/caller")).rejects.toThrow("checksum mismatch");
    install(Buffer.from(JSON.stringify(scorer)),Buffer.from("{}"));
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("Invalid spectral ridge asset descriptor");
    expect(primary.release).toHaveBeenCalledTimes(2);
  });
});
