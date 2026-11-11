import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({access:vi.fn(),readFile:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),access:mocks.access,readFile:mocks.readFile}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";

const manifest={file:"primary.onnx",tokens:"a",inventory:"test",supported:["wrong_word"],sampleRate:16000,frameMs:40};
const arm={intercept:Array.from({length:2},()=>Array(80).fill(0)),coefficient:Array.from({length:2},()=>Array(80).fill(.5)),
  variance:Array.from({length:2},()=>Array(80).fill(1)),counts:[10,10]};
const asset=Buffer.from(JSON.stringify({schema:1,inventory:"test",tokens:"a",frameMs:40,observationMs:10,dimensions:80,
  center:Array(80).fill(0),scale:Array(80).fill(1),forward:arm,reverse:arm,maxFrames:8192}));
const descriptor={schema:1,file:"reverse-emissions.json",bytes:asset.length,sha256:createHash("sha256").update(asset).digest("hex")};
let files: Map<string,string|Buffer>;
let primary: {release: ReturnType<typeof vi.fn>};
beforeEach(()=> {
  vi.resetAllMocks();
  primary={release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
  files=new Map<string,string|Buffer>([["manifest.json",JSON.stringify(manifest)],["reverse-asset.json",JSON.stringify(descriptor)],["reverse-emissions.json",asset]]);
  mocks.readFile.mockImplementation(async (path: string)=> {
    const value=files.get(path.split("/").at(-1)!);
    if (value===undefined) throw Object.assign(new Error("absent"),{code:"ENOENT"});
    return value;
  });
});
describe("declared reverse predictor bundle",()=> {
  it("loads both predictors without changing the primary manifest or backend",async()=> {
    const model=await loadNodeModel("/bundle");
    expect(model.backend).toBe(primary);expect(model.manifest).toEqual(manifest);
    expect(model.reversePredictive?.forward).toEqual(arm);
    expect(model.reversePredictive?.reverse).toEqual(arm);
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
  });
  it("permits primary-only installations",async()=> {
    files.delete("reverse-asset.json");
    expect((await loadNodeModel("/bundle")).reversePredictive).toBeUndefined();
  });
  it("fails on declared missing files and corruption",async()=> {
    files.delete("reverse-emissions.json");
    await expect(loadNodeModel("/bundle")).rejects.toThrow("absent");
    expect(primary.release).toHaveBeenCalledOnce();
    files.set("reverse-emissions.json",Buffer.from("corrupt"));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("checksum");
  });
});
