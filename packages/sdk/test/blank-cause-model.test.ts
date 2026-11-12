import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { BlankCauseModel } from "../src/blank-cause.js";
const mocks=vi.hoisted(()=>({access:vi.fn(),readFile:vi.fn(),ortBackend:vi.fn(),release:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),access:mocks.access,readFile:mocks.readFile}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
const parameters:BlankCauseModel={schema:1,inventory:"test",tokens:"ab",frameMs:40,dimensions:4,center:[0,0,0,0],scale:[1,1,1,1],
  means:[[-1,0,0,0],[1,0,1,0]],variances:[[1,1,1,1],[1,1,1,1]],initial:[.5,.5],transition:[[.9,.1],[.1,.9]],maxFrames:2048};
const bytes=Buffer.from(JSON.stringify(parameters)),manifest={file:"primary.onnx",inventory:"test",tokens:"ab",sampleRate:16000,frameMs:40};
const descriptor={schema:1,file:"blank-cause-model.json",bytes:bytes.length,sha256:createHash("sha256").update(bytes).digest("hex")};
const absent=()=>Object.assign(new Error("absent"),{code:"ENOENT"});
beforeEach(()=> {
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(absent());
  mocks.ortBackend.mockResolvedValue({name:"primary",release:mocks.release});
  mocks.readFile.mockImplementation((path:string)=>Promise.resolve(path.endsWith("manifest.json")?JSON.stringify(manifest):
    path.endsWith("blank-cause-asset.json")?JSON.stringify(descriptor):bytes));
});
describe("blank-cause bundle loading",()=> {
  it("loads verified side parameters through the normal primary model loader",async()=> {
    const model=await loadNodeModel("/models/test");
    expect(model.blankCause).toEqual(parameters);expect(model.manifest).toEqual(manifest);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();expect(mocks.release).not.toHaveBeenCalled();
  });
  it("retains baseline behavior when no side scorer is declared",async()=> {
    mocks.readFile.mockImplementation((path:string)=>path.endsWith("manifest.json")?Promise.resolve(JSON.stringify(manifest)):Promise.reject(absent()));
    expect((await loadNodeModel("/models/test")).blankCause).toBeUndefined();
  });
  it("fails and releases the backend on a missing or corrupt declared asset",async()=> {
    for (const missing of [false,true]) {
      mocks.readFile.mockImplementation((path:string)=>path.endsWith("manifest.json")?Promise.resolve(JSON.stringify(manifest)):
        path.endsWith("blank-cause-asset.json")?Promise.resolve(JSON.stringify(descriptor)):
        missing?Promise.reject(absent()):Promise.resolve(Buffer.from("corrupt")));
      await expect(loadNodeModel("/models/test")).rejects.toThrow();
    }
    expect(mocks.release).toHaveBeenCalledTimes(2);
  });
});
