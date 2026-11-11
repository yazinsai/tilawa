import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),
  readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";

const hash=(bytes:Buffer)=>createHash("sha256").update(bytes).digest("hex");
const primaryBytes=Buffer.from("primary graph"),extractorBytes=Buffer.from("unchanged graph with extra output");
const manifest={file:"primary.onnx",tokens:"بت",inventory:"test",sha256:hash(primaryBytes),sampleRate:16000,frameMs:40};
const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>Number(i===j)));
const arm={center:Array(16).fill(0),scale:Array(16).fill(1),phones:[0,1,2].map(()=>({
  mean:Array(16).fill(0),precision:eye,logDet:0,count:10}))};
const conv=(output:number,input:number)=>Array.from({length:output},()=>Array.from({length:input},()=>Array(8).fill(0)));
const fitted={schema:1,inventory:manifest.inventory,tokens:manifest.tokens,primarySha256:manifest.sha256,
  dimensions:16,patch:400,hop:160,output:"encoder",controlProjection:Array.from({length:16},()=>[1,0]),
  codec:arm,control:arm,weights:{"encoder.0.weight":conv(8,1),"encoder.0.bias":Array(8).fill(0),
    "encoder.2.weight":conv(16,8),"encoder.2.bias":Array(16).fill(0),codebook:Array.from({length:32},()=>Array(16).fill(0))}};
let files:Record<string,Buffer>;
beforeEach(()=> {
  vi.resetAllMocks();
  const emissions=Buffer.from(JSON.stringify(fitted));
  files={"manifest.json":Buffer.from(JSON.stringify(manifest)),"primary.onnx":primaryBytes,
    "wavecodec-emissions.json":emissions,"wavecodec-control.onnx":extractorBytes,
    "wavecodec-asset.json":Buffer.from(JSON.stringify({schema:1,files:[
      {file:"wavecodec-emissions.json",bytes:emissions.length,sha256:hash(emissions)},
      {file:"wavecodec-control.onnx",bytes:extractorBytes.length,sha256:hash(extractorBytes)}]}))};
  mocks.access.mockRejectedValue(Object.assign(new Error("absent teacher"),{code:"ENOENT"}));
  mocks.readFile.mockImplementation(async (path:string,encoding?:string)=> {
    const bytes=files[path.split("/").at(-1)!];
    if (!bytes) throw Object.assign(new Error("missing asset"),{code:"ENOENT"});
    return encoding?bytes.toString("utf8"):bytes;
  });
});
describe("declared waveform codec bundle",()=> {
  it("loads both frozen arms and releases the primary and extractor",async()=> {
    const primary={name:"primary",release:vi.fn()},extractor={name:"extractor",release:vi.fn()};
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(extractor);
    const model=await loadNodeModel("/models/bundle");
    expect(model.wavecodec?.model).toEqual(fitted);
    expect(model.backend).toBe(primary);
    expect(model.wavecodec?.backend).toBe(extractor);
    await model.backend.release();
    expect(primary.release).toHaveBeenCalledOnce();
    expect(extractor.release).toHaveBeenCalledOnce();
  });
  it.each(["wavecodec-control.onnx","wavecodec-emissions.json"])("rejects missing declared %s",async file=> {
    const release=vi.fn();mocks.ortBackend.mockResolvedValue({name:"primary",release});
    delete files[file];
    await expect(loadNodeModel("/models/bundle")).rejects.toThrow("missing asset");
    expect(release).toHaveBeenCalledOnce();
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it("rejects corruption instead of silently running the primary model",async()=> {
    const release=vi.fn();mocks.ortBackend.mockResolvedValue({name:"primary",release});
    files["wavecodec-control.onnx"]=Buffer.from("corrupted");
    await expect(loadNodeModel("/models/bundle")).rejects.toThrow("checksum mismatch");
    expect(release).toHaveBeenCalledOnce();
  });
  it("keeps the ordinary loader when no codec is declared",async()=> {
    delete files["wavecodec-asset.json"];
    const primary={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
    const model=await loadNodeModel("/models/bundle");
    expect(model.wavecodec).toBeUndefined();expect(model.backend).toBe(primary);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
});
