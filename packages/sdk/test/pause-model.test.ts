import { beforeEach,describe,expect,it,vi } from "vitest";
const mocks=vi.hoisted(()=>({ access:vi.fn(),readFile:vi.fn(),ortBackend:vi.fn(),digest:vi.fn() }));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),access:mocks.access,readFile:mocks.readFile}));
vi.mock("node:fs",()=>({createReadStream:()=>[new Uint8Array(4)]}));
vi.mock("node:crypto",()=>({createHash:()=>({update:vi.fn(),digest:mocks.digest})}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
vi.mock("../src/registry.js",()=>({MODEL_REGISTRY:{schema:1,tiers:[],pauseMask:{
  id:"side",precision:"int8",inventory:"test",file:"pause-mask/model.onnx",manifestFile:"pause-mask/manifest.json",bytes:4,sha256:"hash",
}}}));
import {loadNodeModel} from "../src/node.js";
const primary={id:"primary",file:"primary.onnx",tokens:"بت",inventory:"test",sampleRate:16000,frameMs:40,window:45,shift:32,
  states:[{name:"processed_lens",dtype:"int64",shape:[1]}]};
const side={...primary,id:"side",file:"model.onnx",format:"onnx-int8",sha256:"hash",bytes:4,pauseMask:{input:"pause_floor",clockMs:20}};
beforeEach(()=>{
  vi.resetAllMocks();mocks.access.mockResolvedValue(undefined);mocks.digest.mockReturnValue("hash");
  mocks.readFile.mockImplementation(async (path:string)=>JSON.stringify(path.includes("pause-mask")?side:primary));
});
describe("registry-declared pause ear",()=>{
  it("keeps the primary manifest/backend and releases both sessions",async()=>{
    const parent={name:"primary",run:vi.fn(),release:vi.fn()},child={name:"side",run:vi.fn(),release:vi.fn()};
    const releasePrimary=parent.release;
    mocks.ortBackend.mockResolvedValueOnce(parent).mockResolvedValueOnce(child);
    const model=await loadNodeModel("/models",{threads:2});
    expect(model.manifest).toEqual(primary);expect(model.backend).toBe(parent);expect(model.pauseModel?.backend).toBe(child);
    expect(mocks.ortBackend.mock.calls.map(c=>c[1])).toEqual(["/models/primary.onnx","/models/pause-mask/model.onnx"]);
    expect(parent.run).not.toHaveBeenCalled();await model.backend.release();
    expect(releasePrimary).toHaveBeenCalledOnce();expect(child.release).toHaveBeenCalledOnce();
  });
  it("leaves a bundle with no installed ear unchanged",async()=>{
    const parent={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(parent);
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
    const model=await loadNodeModel("/models");expect(model.pauseModel).toBeUndefined();expect(model.manifest).toEqual(primary);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it.each(["clock","weights"])("rejects a mismatched %s instead of replacing the primary",async mismatch=>{
    const parent={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(parent);
    if(mismatch==="clock") mocks.readFile.mockImplementation(async(path:string)=>JSON.stringify(path.includes("pause-mask")?{...side,frameMs:20}:primary));
    else mocks.digest.mockReturnValue("corrupt");
    await expect(loadNodeModel("/models")).rejects.toThrow(/Pause ear/);
    expect(parent.release).toHaveBeenCalledOnce();expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
});
