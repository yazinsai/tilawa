import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks=vi.hoisted(()=>({readFile:vi.fn(),access:vi.fn(),ortBackend:vi.fn()}));
vi.mock("node:fs/promises",async original=>({...await original<typeof import("node:fs/promises")>(),readFile:mocks.readFile,access:mocks.access}));
vi.mock("../src/ort.js",()=>({ortBackend:mocks.ortBackend}));
import { loadNodeModel } from "../src/node.js";
const sha=(b:Buffer)=>createHash("sha256").update(b).digest("hex");
const primaryBytes=Buffer.from("primary"),fusedBytes=Buffer.from("fused"),controlBytes=Buffer.from("control");
const manifest={file:"primary.onnx",inventory:"test",tokens:"بت",sampleRate:16000,frameMs:40,sha256:sha(primaryBytes)};
const cfg={schema:1,inventory:"test",tokens:"بت",primarySha256:manifest.sha256,mels:80,stride:4,kernel:9,
  bands:[[0,32],[16,48],[32,64],[48,80]],learnedParameters:100,controlLearnedParameters:100};
const configBytes=Buffer.from(JSON.stringify(cfg));
const asset={schema:1,files:[['subband-ear-config.json',configBytes],['subband-fused.onnx',fusedBytes],['subband-control.onnx',controlBytes]]
  .map(([file,bytes])=>({file,bytes:(bytes as Buffer).length,sha256:sha(bytes as Buffer)}))};
let files:Record<string,Buffer>;
beforeEach(()=> {
  vi.resetAllMocks();
  files={'manifest.json':Buffer.from(JSON.stringify(manifest)),'primary.onnx':primaryBytes,
    'subband-ear-asset.json':Buffer.from(JSON.stringify(asset)),'subband-ear-config.json':configBytes,
    'subband-fused.onnx':fusedBytes,'subband-control.onnx':controlBytes};
  mocks.access.mockRejectedValue(Object.assign(new Error('absent'),{code:'ENOENT'}));
  mocks.readFile.mockImplementation(async (path:string,encoding?:string)=> {
    const b=files[path.split('/').at(-1)!];
    if (!b) throw Object.assign(new Error('absent'),{code:'ENOENT'});
    return encoding ? b.toString(encoding as BufferEncoding) : b;
  });
});
const backend=()=>({name:'fake',run:vi.fn(),release:vi.fn().mockResolvedValue(undefined)});
describe('declared subband paired asset loading',()=> {
  it('loads both ears and releases all graph instances',async()=> {
    const primary=backend(),fused=backend(),control=backend(),primaryRelease=primary.release;
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(fused).mockResolvedValueOnce(control);
    const loaded=await loadNodeModel('/bundle',{threads:2});
    expect(loaded.subbandEar).toEqual({model:cfg,fused,control});
    await loaded.backend.release();
    expect(primaryRelease).toHaveBeenCalledOnce();
    for (const b of [fused,control]) expect(b.release).toHaveBeenCalledOnce();
    expect(mocks.ortBackend.mock.calls[2][3]).toMatchObject({intraOpNumThreads:2});
  });
  it('keeps the ordinary model when no side asset is declared',async()=> {
    delete files['subband-ear-asset.json'];mocks.ortBackend.mockResolvedValue(backend());
    expect((await loadNodeModel('/bundle')).subbandEar).toBeUndefined();
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it.each(['subband-fused.onnx','subband-control.onnx','subband-ear-config.json'])('rejects a declared missing file: %s',async file=> {
    delete files[file];const primary=backend();mocks.ortBackend.mockResolvedValue(primary);
    await expect(loadNodeModel('/bundle')).rejects.toThrow('absent');
    expect(primary.release).toHaveBeenCalledOnce();
  });
  it('rejects a corrupt graph before running the pair',async()=> {
    files['subband-control.onnx']=Buffer.from('corrupt');const primary=backend();mocks.ortBackend.mockResolvedValue(primary);
    await expect(loadNodeModel('/bundle')).rejects.toThrow('checksum');expect(primary.release).toHaveBeenCalledOnce();
  });
  it('releases the fused graph and primary when control creation fails',async()=> {
    const primary=backend(),fused=backend();
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(fused).mockRejectedValueOnce(new Error('control failed'));
    await expect(loadNodeModel('/bundle')).rejects.toThrow('control failed');
    expect(primary.release).toHaveBeenCalledOnce();expect(fused.release).toHaveBeenCalledOnce();
  });
});
