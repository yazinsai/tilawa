import {createHash} from "node:crypto";
import {beforeEach,describe,expect,it,vi} from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import {loadNodeModel} from "../src/node.js";
const sha=(b:Buffer)=>createHash("sha256").update(b).digest("hex");
const encoder=Buffer.from("frozen weights");
const scorer=Buffer.from(JSON.stringify({schema:1,hidden:1,tokens:"ab",inventory:"test",featureNode:"mid",encoderSha256:sha(encoder),
  mean:[0],std:[1],weights:{"acoustic.weight":[[0]],"acoustic.bias":[0],"text.weight":[Array(39).fill(0)],"text.bias":[0],duration:[0,0],skip_logit:0}}));
const manifest={file:"primary.onnx",tokens:"ab",inventory:"test",states:[],segmentPresence:{file:"segment-scorer.json",sha256:sha(scorer),encoder:{file:"segment-encoder.onnx",sha256:sha(encoder)}}};
beforeEach(()=>{
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("missing optional teacher"),{code:"ENOENT"}));
  mocks.readFile.mockImplementation(async(path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):path.endsWith("segment-scorer.json")?scorer:encoder);
});
describe("required segment assets",()=>{
  it("loads the declared scorer and feature graph lazily and releases both backends",async()=>{
    const primaryRelease=vi.fn();
    const primary={name:"primary",release:primaryRelease}, auxiliary={name:"frozen",release:vi.fn(),run:vi.fn().mockResolvedValue({
      log_probs:{type:"float32",data:new Float32Array(3),dims:[1,1,3]},mid:{type:"float32",data:Float32Array.from([1]),dims:[1,1,1]}})};
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(auxiliary);
    const model=await loadNodeModel("/bundle");
    expect(model.backend).toBe(primary);expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
    const s=model.segmentPresence!.stream!();await s.step({type:"float32",data:new Float32Array(1),dims:[1]},1);
    expect(s.rows[0][0]).toBe(1);expect(mocks.ortBackend).toHaveBeenCalledTimes(2);
    await model.backend.release();expect(auxiliary.release).toHaveBeenCalledOnce();expect(primaryRelease).toHaveBeenCalledOnce();
  });
  it("fails on missing declared files and releases the primary session",async()=>{
    const primary={name:"primary",release:vi.fn()};mocks.ortBackend.mockResolvedValue(primary);
    mocks.readFile.mockImplementation(async(path:string)=>{if(path.endsWith("manifest.json"))return JSON.stringify(manifest);throw new Error("missing declared scorer");});
    await expect(loadNodeModel("/bundle")).rejects.toThrow("missing declared scorer");expect(primary.release).toHaveBeenCalledOnce();
  });
  it("fails on corrupt feature weights",async()=>{
    mocks.ortBackend.mockResolvedValue({name:"primary",release:vi.fn()});
    mocks.readFile.mockImplementation(async(path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):path.endsWith("segment-scorer.json")?scorer:Buffer.from("corrupt"));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("hash mismatch");
  });
});
