import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn(),primaryRelease:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
import { MODEL_REGISTRY } from "../src/registry.js";

const manifest={file:"model.onnx",sha256:"a".repeat(64),inventory:"test",tokens:"بت",sampleRate:16000,frameMs:40};
const metadata={schema:1,inventory:"test",tokens:"بت",dimensions:32,primarySha256:manifest.sha256,featureOutput:"encoder",
  projection:Array.from({length:32},()=>[1]),center:new Array(32).fill(0),scale:new Array(32).fill(1)};
const head={embedding:Array.from({length:3},()=>new Array(16).fill(0)),weight:Array.from({length:32},()=>new Array(48).fill(0)),
  bias:new Array(32).fill(0),output:new Array(32).fill(0),intercept:0};
let files:Record<string,Buffer>;
let primary: {name:string;run:ReturnType<typeof vi.fn>;release:ReturnType<typeof vi.fn>};
const features={name:"features",run:vi.fn(),release:vi.fn()};
beforeEach(()=>{
  vi.resetAllMocks();
  primary={name:"primary",run:vi.fn(),release:mocks.primaryRelease};
  files={"manifest.json":Buffer.from(JSON.stringify(manifest)),"duration-negatives.json":Buffer.from(JSON.stringify(metadata)),
    "duration-negatives-features.onnx":Buffer.from("graph"),"duration-negatives-matched.json":Buffer.from(JSON.stringify(head)),
    "duration-negatives-random.json":Buffer.from(JSON.stringify(head))};
  const names=Object.keys(files).filter(n=>n!=="manifest.json");
  files["duration-negatives-assets.json"]=Buffer.from(JSON.stringify(Object.fromEntries(names.map(n=>[n,
    {bytes:files[n].length,sha256:createHash("sha256").update(files[n]).digest("hex")}]))));
  mocks.readFile.mockImplementation(async(path:string,encoding?:string)=>{
    const name=path.split("/").at(-1)!;
    if(!files[name])throw Object.assign(new Error("absent"),{code:"ENOENT"});
    return encoding?files[name].toString():files[name];
  });
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(features);
});
describe("complete matched-negative asset bundle",()=>{
  it("validates both frozen heads and releases the separate feature backend",async()=>{
    const model=await loadNodeModel("/model");
    expect(model.backend).toBe(primary);
    expect(model.durationNegatives?.matched).toEqual(head);
    expect(model.durationNegatives?.random).toEqual(head);
    expect(features.run).not.toHaveBeenCalled();
    await model.backend.release();
    expect(features.release).toHaveBeenCalledOnce();
    expect(mocks.primaryRelease).toHaveBeenCalledOnce();
  });
  it("rejects a missing control instead of evaluating a fallback",async()=>{
    delete files["duration-negatives-random.json"];
    await expect(loadNodeModel("/model")).rejects.toThrow("absent");
    expect(mocks.primaryRelease).toHaveBeenCalledOnce();
  });
  it("rejects a corrupt declared feature graph",async()=>{
    files["duration-negatives-features.onnx"]=Buffer.from("different");
    await expect(loadNodeModel("/model")).rejects.toThrow("checksum");
  });
  it("rejects heads fitted on a different primary even with valid asset checksums",async()=>{
    files["duration-negatives.json"]=Buffer.from(JSON.stringify({...metadata,primarySha256:"b".repeat(64)}));
    const descriptor=JSON.parse(files["duration-negatives-assets.json"].toString());
    descriptor["duration-negatives.json"]={bytes:files["duration-negatives.json"].length,
      sha256:createHash("sha256").update(files["duration-negatives.json"]).digest("hex")};
    files["duration-negatives-assets.json"]=Buffer.from(JSON.stringify(descriptor));
    await expect(loadNodeModel("/model")).rejects.toThrow("Invalid duration-negative model");
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
    expect(mocks.primaryRelease).toHaveBeenCalledOnce();
  });
  it("retains the inherited insertion teacher alongside the added head",async()=>{
    const asset=MODEL_REGISTRY.auxiliary!.find(a=>a.purpose==="seam-insertion")!;
    const inheritedManifest={...manifest,inventory:asset.inventory};
    files["manifest.json"]=Buffer.from(JSON.stringify(inheritedManifest));
    files["duration-negatives.json"]=Buffer.from(JSON.stringify({...metadata,inventory:asset.inventory}));
    const descriptor=JSON.parse(files["duration-negatives-assets.json"].toString());
    descriptor["duration-negatives.json"]={bytes:files["duration-negatives.json"].length,
      sha256:createHash("sha256").update(files["duration-negatives.json"]).digest("hex")};
    files["duration-negatives-assets.json"]=Buffer.from(JSON.stringify(descriptor));
    mocks.access.mockResolvedValue(undefined);
    const teacher={name:"teacher",run:vi.fn().mockResolvedValue({log_probs:{type:"float32",
      data:Float32Array.from([.9,.05,.05],Math.log),dims:[1,1,3]}}),release:vi.fn()};
    mocks.ortBackend.mockResolvedValueOnce(teacher);
    const model=await loadNodeModel("/inherited");
    expect(model.backend).toBe(primary);
    expect(model.manifest).toEqual(inheritedManifest);
    expect(model.durationNegatives).toBeDefined();
    expect(model.insertionTeacher).toBeDefined();
    await model.insertionTeacher!.decode(Float32Array.from({length:1600},(_,i)=>Math.sin(i*.1)),3);
    expect(teacher.run).toHaveBeenCalledOnce();
    expect(primary.run).not.toHaveBeenCalled();
    expect(mocks.ortBackend.mock.calls.map(c=>c[1])).toEqual([
      "/inherited/model.onnx","/inherited/duration-negatives-features.onnx",`/inherited/${asset.file}`]);
    await model.backend.release();
    expect(teacher.release).toHaveBeenCalledOnce();
    expect(features.release).toHaveBeenCalledOnce();
    expect(mocks.primaryRelease).toHaveBeenCalledOnce();
  });
  it("keeps ordinary models working without an auxiliary declaration",async()=>{
    delete files["duration-negatives-assets.json"];
    const model=await loadNodeModel("/model");
    expect(model.durationNegatives).toBeUndefined();
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
});
