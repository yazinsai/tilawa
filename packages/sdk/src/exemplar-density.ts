import { createChecker, createTracker, trackerInput, PosteriorFrames } from "@tilawa/core";
import type { CheckerOptions, ExemplarEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export interface ExemplarModel {
  schema: 1; inventory: string; tokens: string; primarySha256: string; output: string;
  center: number[]; projection: number[][]; bank: number[][]; soft: number[][];
  recording: number[]; weights: number[]; frequencies: number[]; recordingCount: number; k: number; alpha: number;
}
const logadd = (a: number,b: number): number => {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
/** Ordinary CTC: complete pronunciations and identical blank/timing topology in both arms. */
export function exemplarSequenceScore(rows: number[][], ids: number[]): number {
  if (!rows.length||!ids.length||ids.some(id=>id<=0||id>=rows[0].length)) return -Infinity;
  const labels=[0,...ids.flatMap(id=>[id,0])];
  let a=labels.map(()=>-Infinity);a[0]=rows[0][0];a[1]=rows[0][labels[1]];
  for (let t=1;t<rows.length;t++) a=a.map((v,s)=>logadd(logadd(v,s?a[s-1]:-Infinity),
    s>1&&labels[s]!==0&&labels[s]!==labels[s-2]?a[s-2]:-Infinity)+rows[t][labels[s]]);
  return logadd(a.at(-1)!,a.at(-2)!);
}
/** The bank has unit weight per recording and fractional, transcript-marginal phone labels. */
export function exemplarRows(m: ExemplarModel,raw: number[][],lp: number[][]): number[][] {
  const grouped=Array.from({length:m.recordingCount},()=>[] as number[]);
  m.recording.forEach((r,j)=>grouped[r].push(j));
  return raw.map((r,t)=> {
    const x=m.projection.map(p=>p.reduce((s,v,d)=>s+v*(r[d]-m.center[d]),0));
    const distance=m.bank.map(b=>b.reduce((s,v,d)=>s+(v-x[d])**2,0));
    const neighbours=m.frequencies.map((_,p)=>grouped.flatMap(group=> {
      let best=-1,cost=Infinity;
      for (const j of group) if (m.soft[j][p]>0) {
        const value=distance[j]-2*Math.log(m.soft[j][p]);
        if (value<cost) {best=j;cost=value;}
      }
      return best<0?[]:[{j:best,cost}];
    }).sort((a,b)=>a.cost-b.cost||a.j-b.j).slice(0,m.k).map(v=>v.j));
    const distances=[...new Set(neighbours.flat())].map(j=>distance[j]).sort((a,b)=>a-b);
    const middle=Math.floor(distances.length/2);
    const bandwidth=Math.max(distances.length?(distances.length%2?distances[middle]:(distances[middle-1]+distances[middle])/2):1,1e-8);
    const density=neighbours.map((ns,p)=>m.frequencies[p]>0?ns.reduce((s,j)=>
      s+Math.exp(-distance[j]/(2*bandwidth))*m.soft[j][p]*m.weights[j],0)/m.frequencies[p]:0);
    const smoothing=m.alpha*Math.max(density.reduce((s,v)=>s+v,0)/density.length,1e-12);
    const total=density.reduce((s,v)=>s+v+smoothing,0),speech=1-Math.exp(lp[t][0]);
    return [lp[t][0],...density.map(v=>Math.log(Math.max(speech*(v+smoothing)/total,1e-12)))];
  });
}
export function exemplarComparison(m:ExemplarModel,raw:number[][],lp:number[][],c:SequenceEvidenceCandidate):ExemplarEvidence|null {
  const ids=[c.written,c.rival].map(s=>[...s].map(p=>m.tokens.indexOf(p)+1));
  if (!raw.length||raw.length!==lp.length||raw.some(r=>r.length!==m.center.length||r.some(v=>!Number.isFinite(v)))||
    lp.some(r=>r.length!==m.tokens.length+1||r.some(v=>!Number.isFinite(v)))||
    ids.some(s=>!s.length||s.some(p=>p<1||m.frequencies[p-1]<=0))) return null;
  const rows=exemplarRows(m,raw,lp),head=ids.map(s=>exemplarSequenceScore(lp,s)),exemplar=ids.map(s=>exemplarSequenceScore(rows,s));
  return [...head,...exemplar].every(Number.isFinite)?{...c,headWritten:head[0],headRival:head[1],exemplarWritten:exemplar[0],exemplarRival:exemplar[1]}:null;
}
export const exemplarReversal=(c:ExemplarEvidence):boolean=>
  [c.headWritten,c.headRival,c.exemplarWritten,c.exemplarRival].every(Number.isFinite)&&c.headWritten<c.headRival&&c.exemplarWritten>c.exemplarRival;

/** Separate frozen extractor state with the primary window, lookahead, padding and native clock. */
export async function exemplarObservations(backend:InferenceBackend,manifest:ModelManifest,output:string,audio:Float32Array) {
  const fb=new Fbank().push(audio),features:number[][]=[],logProbs:number[][]=[];
  const total=Math.floor((Math.floor((fb.length-7)/2)+1)/2),vocab=manifest.tokens.length+1;
  let states:Record<string,Tensor>=Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64"?new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)):new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;features.length<total;f+=manifest.shift) {
    const data=new Float32Array(manifest.window*manifest.fbank.mels);
    for (let i=0;i<manifest.window;i++) data.set(fb[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const tensor=out[output],n=out.log_probs.data.length/vocab;
    if (!tensor||tensor.type!=="float32"||tensor.dims.length!==3||tensor.data.length/tensor.dims[2]!==n)
      throw new Error("Pre-head extractor does not match the primary frame clock");
    const width=tensor.dims[2];
    for (let t=0;t<n&&features.length<total;t++) {
      features.push(Array.from(tensor.data.slice(t*width,(t+1)*width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {features,logProbs};
}
export async function addExemplarEvidence(scorer:{model:ExemplarModel;backend:InferenceBackend}|undefined,manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:ExemplarEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const primaryRows=new PosteriorFrames(manifest.tokens.length);
  for(const chunk of chunks)if(chunk.posteriors)primaryRows.append(chunk.posteriors);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if (!candidates.size)return chunks;
  const observations=await exemplarObservations(scorer.backend,manifest,scorer.model.output,audio),certificates:ExemplarEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=primaryRows.startFrame&&c.toFrame<=primaryRows.endFrame&&c.toFrame<=observations.features.length?exemplarComparison(scorer.model,
      observations.features.slice(c.fromFrame,c.toFrame),Array.from({length:c.toFrame-c.fromFrame},(_,t)=>Array.from(primaryRows.row(c.fromFrame+t))),c):null;
    inspect?.(result,c);if(result&&exemplarReversal(result))certificates.push(result);
  }
  if(!certificates.length)return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,exemplarVetoes:certificates})});
}
export function validateExemplarModel(value:unknown,manifest:ModelManifest):ExemplarModel {
  const m=value as ExemplarModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    typeof m.output!=="string"||!m.output||!Array.isArray(m.center)||!vector(m.center,m.center.length)||!m.center.length||
    !Array.isArray(m.projection)||!m.projection.length||!m.projection.every(r=>vector(r,m.center.length))||
    !Number.isInteger(m.k)||m.k<=0||!Number.isFinite(m.alpha)||m.alpha<=0||
    !Number.isInteger(m.recordingCount)||m.recordingCount<3||!m.bank?.length||
    !m.bank.every(r=>vector(r,m.projection.length))||m.soft?.length!==m.bank.length||
    !m.soft.every(r=>vector(r,m.tokens.length)&&r.every(v=>v>=0&&v<=1)&&r.reduce((s,v)=>s+v,0)<=1+1e-6)||
    !vector(m.weights,m.bank.length)||m.weights.some(v=>v<=0)||!vector(m.frequencies,m.tokens.length)||m.frequencies.some(v=>v<0)||
    !vector(m.recording,m.bank.length)||m.recording.some(r=>!Number.isInteger(r)||r<0||r>=m.recordingCount))throw new Error("Invalid exemplar datastore");
  const totals=Array(m.recordingCount).fill(0);m.recording.forEach((r,i)=>totals[r]+=m.weights[i]);
  if(totals.some(v=>Math.abs(v-1)>1e-6))throw new Error("Each exemplar recording must have unit weight");
  for(let p=0;p<m.tokens.length;p++)if(Math.abs(m.soft.reduce((s,r,i)=>s+r[p]*m.weights[i],0)-m.frequencies[p])>1e-6)
    throw new Error("Datastore phone frequencies differ from soft labels");
  return m;
}
