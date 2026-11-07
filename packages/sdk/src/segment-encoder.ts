import { Fbank } from "./fbank.js";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";

export interface SegmentFeatureStream {
  readonly rows: Float32Array[];
  /** Same fbank window and output-frame truncation as the primary listener. */
  step(x: Tensor, keep: number): Promise<void>;
}

export function createSegmentFeatureStream(backend: () => Promise<InferenceBackend>, m: ModelManifest,
  node: string, dim: number): SegmentFeatureStream {
  let states: Record<string, Tensor> = Object.fromEntries(m.states.map(s => {
    const size=s.shape.reduce((a,b)=>a*b,1);
    return [s.name,{type:s.dtype,dims:s.shape,data:s.dtype==="int64"?new BigInt64Array(size):new Float32Array(size)}];
  }));
  const rows: Float32Array[]=[];
  return { rows, async step(x, keep) {
    const out=await (await backend()).run({x,...states});
    states=Object.fromEntries(m.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const z=out[node], count=out.log_probs.dims[1], ratio=z?.dims[0]/count;
    if (!z || z.type!=="float32" || z.dims.length!==3 || z.dims[1]!==1 || z.dims[2]!==dim || !Number.isInteger(ratio) || ratio<1)
      throw new Error("Segment encoder feature clock or shape differs from primary evidence");
    const values=z.data as Float32Array;
    for (let t=0;t<Math.min(keep,count);t++) {
      const row=new Float32Array(dim);
      for (let j=0;j<dim;j++) {
        let sum=0;
        for (let r=0;r<ratio;r++) sum+=values[(t*ratio+r)*dim+j];
        row[j]=sum/ratio;
      }
      rows.push(row);
    }
  } };
}

/** Offline discovery uses exactly the SDK frontend and streaming windows. */
export async function segmentFeatures(samples: Float32Array, m: ModelManifest, stream: SegmentFeatureStream): Promise<Float32Array[]> {
  if (m.sampleRate!==16000) throw new Error("Segment encoder requires 16 kHz audio");
  const fbank=new Fbank(), feats=fbank.push(samples);
  const total=Math.max(0,Math.floor((Math.floor((feats.length-7)/2)+1)/2));
  const pad=new Float32Array(m.fbank.mels).fill(m.fbank.padValue);
  for (let at=0;stream.rows.length<total;at+=m.shift) {
    const x=new Float32Array(m.window*m.fbank.mels);
    for (let i=0;i<m.window;i++) x.set(feats[at+i]??pad,i*m.fbank.mels);
    await stream.step({type:"float32",data:x,dims:[1,m.window,m.fbank.mels]},total-stream.rows.length);
  }
  return stream.rows;
}
