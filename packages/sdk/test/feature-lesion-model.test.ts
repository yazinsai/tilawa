import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),
  readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";

const primary=Buffer.from("primary graph"), extractor=Buffer.from("feature graph");
const sha=(b:Buffer)=>createHash("sha256").update(b).digest("hex");
const manifest={file:"model.onnx",tokens:"بت",inventory:"test",sampleRate:16000,frameMs:40,sha256:sha(primary)};
const mask=Buffer.from(JSON.stringify({schema:1,inventory:"test",tokens:"بت",primarySha256:sha(primary),width:1,
  featureOutput:"features",logitOutput:"logits",effectiveWeight:[[0,-2,2]],bias:[0,0,0],mean:[-1],mask:[0]}));
const descriptor={schema:1,files:[{file:"feature-lesion-mask.json",bytes:mask.length,sha256:sha(mask)},
  {file:"feature-lesion.onnx",bytes:extractor.length,sha256:sha(extractor)}]};
let files:Record<string,string|Buffer>;
beforeEach(()=> {
  vi.resetAllMocks();
  files={"manifest.json":JSON.stringify(manifest),"model.onnx":primary,"feature-lesion-mask.json":mask,
    "feature-lesion.onnx":extractor,"feature-lesion-asset.json":JSON.stringify(descriptor)};
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  mocks.readFile.mockImplementation(async(path:string)=> {
    const name=path.slice(path.lastIndexOf("/")+1);
    if(!(name in files)) throw Object.assign(new Error("absent"),{code:"ENOENT"});
    return files[name];
  });
});
describe("complete frozen feature lesion bundle",()=> {
  it("loads a bound side graph and releases it with the unchanged primary",async()=> {
    const primaryRelease=vi.fn(), main={release:primaryRelease}, side={release:vi.fn()};
    mocks.ortBackend.mockResolvedValueOnce(main).mockResolvedValueOnce(side);
    const loaded=await loadNodeModel("/bundle");
    expect(loaded.backend).toBe(main);
    expect(loaded.featureLesion?.backend).toBe(side);
    expect(loaded.manifest).toEqual(manifest);
    await loaded.backend.release();
    expect(primaryRelease).toHaveBeenCalledOnce();
    expect(side.release).toHaveBeenCalledOnce();
  });
  it("rejects missing and corrupted declared files instead of evaluating the primary alone",async()=> {
    for(const invalid of [undefined,Buffer.from("corrupt")]) {
      const main={release:vi.fn()};mocks.ortBackend.mockResolvedValueOnce(main);
      if(invalid)files["feature-lesion.onnx"]=invalid;else delete files["feature-lesion.onnx"];
      await expect(loadNodeModel("/bundle")).rejects.toThrow();
      expect(main.release).toHaveBeenCalledOnce();
    }
    expect(mocks.ortBackend).toHaveBeenCalledTimes(2);
  });
  it("permits the ordinary primary when no lesion was declared",async()=> {
    delete files["feature-lesion-asset.json"];
    const main={release:vi.fn()};mocks.ortBackend.mockResolvedValueOnce(main);
    expect((await loadNodeModel("/bundle")).featureLesion).toBeUndefined();
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
});
