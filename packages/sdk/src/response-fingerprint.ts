import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, ResponseFingerprintEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleFingerprintTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

export interface FingerprintEmission { mean: number[]; precision: number[][]; logDet: number; count: number }
export interface FingerprintArm { center: number[]; scale: number[]; phones: FingerprintEmission[] }
export interface FingerprintMetadata {
  schema: 1; inventory: string; tokens: string; dimensions: number; primarySha256: string;
  basis: number[][]; epsilon: number; radius: number;
}
export interface FingerprintModel extends FingerprintMetadata { response: FingerprintArm; control: FingerprintArm }
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function fingerprintBoundaryScore(values: number[][], ids: number[]): number {
  if (!values.length || !ids.length) return -Infinity;
  const labels = [0, ...ids.flatMap(p => [p,0])];
  const edges = labels.map((p,s) => [s, ...(s+1<labels.length?[s+1]:[]),
    ...(s+2<labels.length && labels[s+2]!==0 && labels[s+2]!==p ? [s+2]:[])]);
  let a = labels.map(()=>-Infinity), prior = a.slice();
  a[0]=values[0][0]; a[1]=values[0][ids[0]]; prior[0]=prior[1]=0;
  for (let t=1;t<values.length;t++) {
    const b=labels.map(()=>-Infinity), q=b.slice();
    for (let s=0;s<labels.length;s++) for (const next of edges[s]) {
      const cost=Math.log(edges[s].length);
      b[next]=logadd(b[next],a[s]-cost); q[next]=logadd(q[next],prior[s]-cost);
    }
    a=b.map((v,s)=>v+values[t][labels[s]]); prior=q;
  }
  const n=labels.length;
  const likelihood=logadd(a[n-1],a[n-2]), timing=logadd(prior[n-1],prior[n-2]);
  return Number.isFinite(likelihood) && Number.isFinite(timing) ? likelihood-timing : -Infinity;
}
function scores(arm: FingerprintArm, raw: number[][], sequences: number[][]): number[] {
  const x=raw.map(r=>r.map((v,d)=>(v-arm.center[d])/arm.scale[d]));
  const used=[...new Set([0,...sequences.flat()])];
  const rows=x.map(r=> {
    const values=arm.phones.map(()=>-Infinity);
    for (const p of used) {
      const e=arm.phones[p]; let q=0;
      for (const row of e.precision) { let z=0; for (let d=0;d<r.length;d++) z+=row[d]*(r[d]-e.mean[d]); q+=z*z; }
      values[p]=-.5*(r.length*Math.log(2*Math.PI)+e.logDet+q);
    }
    return values;
  });
  return sequences.map(s=>fingerprintBoundaryScore(rows,s));
}
export function fingerprintReversal(c: ResponseFingerprintEvidence): boolean {
  return [c.responseWritten,c.responseRival,c.controlWritten,c.controlRival].every(Number.isFinite) &&
    c.responseWritten>c.responseRival && c.controlWritten<c.controlRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export const fingerprintEventKey = (e: CheckerEvent): string => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
export function fingerprintVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[],
  eligible?: ReadonlySet<string>): CheckerEvent[] {
  const identity = fingerprintEventKey;
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)) && (!eligible || eligible.has(identity(e))));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryFingerprintEvidence(chunks: PhonemeEvidence[], certificates: ResponseFingerprintEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, responseFingerprintVetoes:certificates,
  })});
}
export function fingerprintComparison(model: FingerprintModel, fingerprint: number[][], control: number[][],
  candidate: SequenceEvidenceCandidate): ResponseFingerprintEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  const valid=(x:number[][])=>x.length>0&&x.every(r=>r.length===model.dimensions&&r.every(Number.isFinite));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 ||
    !Number.isInteger(candidate.toFrame) || candidate.toFrame <= candidate.fromFrame ||
    candidate.toFrame-candidate.fromFrame !== fingerprint.length || fingerprint.length!==control.length || !valid(fingerprint)||!valid(control) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.response.phones[p].count<=0||model.control.phones[p].count<=0))) return null;
  const a=scores(model.response,fingerprint,sequences), b=scores(model.control,control,sequences);
  const result={...candidate,responseWritten:a[0],responseRival:a[1],controlWritten:b[0],controlRival:b[1]};
  return [a[0],a[1],b[0],b[1]].every(Number.isFinite)?result:null;
}

/** Fixed, disjoint smooth spectral patches. No candidate pronunciation selects a perturbation. */
export function fingerprintMetadata(m: ModelManifest): FingerprintMetadata {
  const basis = [0.28, 0.72].map(center => Array.from({length:m.fbank.mels}, (_,i) => {
    const distance = Math.abs(i/(m.fbank.mels-1)-center)/0.2;
    return distance < 1 ? (1+Math.cos(Math.PI*distance))/2 : 0;
  }));
  return {schema:1, inventory:m.inventory, tokens:m.tokens, dimensions:2*(m.tokens.length+1),
    primarySha256:m.sha256, basis, epsilon:0.05, radius:4};
}

export function validateFingerprintModel(value: unknown, manifest: ModelManifest): FingerprintModel {
  const m = value as FingerprintModel, fixed = fingerprintMetadata(manifest);
  const vector = (r: unknown, n: number): r is number[] => Array.isArray(r) && r.length === n && r.every(Number.isFinite);
  if (!m || Object.keys(fixed).some(k => JSON.stringify(m[k as keyof FingerprintModel]) !== JSON.stringify(fixed[k as keyof FingerprintMetadata])))
    throw new Error("Fingerprint geometry or primary checksum differs from the frozen extractor");
  for (const arm of [m.response,m.control]) {
    if (!arm || !vector(arm.center,m.dimensions) || !vector(arm.scale,m.dimensions) || arm.scale.some(v=>v<=0) ||
        !Array.isArray(arm.phones) || arm.phones.length !== m.tokens.length+1) throw new Error("Invalid fingerprint emission arm");
    for (const e of arm.phones) if (!e || !vector(e.mean,m.dimensions) ||
        !Array.isArray(e.precision) || e.precision.length!==m.dimensions ||
        e.precision.some((r,i)=>!vector(r,m.dimensions)||r[i]<=0||r.some((v,j)=>j>i&&v!==0)) ||
        !Number.isFinite(e.logDet) || !Number.isFinite(e.count) || e.count<0) throw new Error("Invalid fingerprint emission");
  }
  return m;
}

const centered = (row: ArrayLike<number>): number[] => {
  const x=Array.from(row), mean=x.reduce((a,b)=>a+b,0)/x.length;
  return x.map(v=>v-mean);
};
const cloneStates = (states: Record<string,Tensor>): Record<string,Tensor> => Object.fromEntries(
  Object.entries(states).map(([k,v])=>[k,{...v,data:v.data.slice()}]));
const logProbabilityRows = (out: Record<string,Tensor>): Float32Array => {
  const tensor=out.log_probs;
  if (!tensor || tensor.type!=="float32" || !(tensor.data instanceof Float32Array))
    throw new Error("Primary log probabilities must be float32");
  return tensor.data;
};

/** Replay the exact primary graph with independent state. Each symmetric intervention starts
 * from the same unmodified incoming cache and perturbs only a 90 ms local spectral patch.
 * Centered log probabilities equal centered pre-softmax logits exactly; no graph surgery is needed.
 * `wanted` avoids expensive responses outside existing requested windows. */
export async function fingerprintObservations(backend: InferenceBackend, m: ModelManifest, meta: FingerprintMetadata,
  audio: Float32Array, wanted?: (frame:number)=>boolean, deadline=Infinity) {
  if (m.sampleRate!==16000 || m.frameMs!==40 || m.fbank.mels!==80 || m.shift<=0 || m.shift%4!==0 ||
    JSON.stringify(meta)!==JSON.stringify(fingerprintMetadata(m)))
    throw new Error("Unsupported or unfrozen fingerprint geometry");
  const features=new Fbank().push(audio), total=Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  const response:number[][]=[],control:number[][]=[],logProbs:number[][]=[];
  let states:Record<string,Tensor>=Object.fromEntries(m.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad=new Float32Array(m.fbank.mels).fill(m.fbank.padValue),vocab=m.tokens.length+1;
  let previous:number[]|undefined;
  for(let f=0;logProbs.length<total;f+=m.shift) {
    if(Date.now()>deadline) throw new Error("Fingerprint extraction exceeded its fixed compute budget");
    const x=new Float32Array(m.window*m.fbank.mels);
    for(let i=0;i<m.window;i++)x.set(features[f+i]??pad,i*m.fbank.mels);
    const run=(data:Float32Array)=> {
      if(Date.now()>deadline) throw new Error("Fingerprint extraction exceeded its fixed compute budget");
      return backend.run({x:{type:"float32",dims:[1,m.window,m.fbank.mels],data},...cloneStates(states)});
    };
    const baseline=await run(x.slice()), baselineRows=logProbabilityRows(baseline).slice(), n=baselineRows.length/vocab;
    if(n!==m.shift/4) throw new Error("Invalid primary output clock");
    // Some backends reuse output buffers. Capture the baseline before any recheck.
    const nextStates=cloneStates(Object.fromEntries(m.states.map(s=>[s.name,baseline[`new_${s.name}`]])));
    for(let t=0;t<n&&logProbs.length<total;t++) {
      const frame=logProbs.length;
      const raw=Array.from(baselineRows.slice(t*vocab,(t+1)*vocab),Number),staticRow=centered(raw);
      if(!raw.every(Number.isFinite)) throw new Error("Nonfinite primary observations");
      control.push([...staticRow,...(previous??staticRow)]);previous=staticRow;logProbs.push(raw);
      if(wanted&&!wanted(frame)) {response.push([]);continue;}
      const result:number[]=[];
      for(const spectral of meta.basis) {
        const arms:number[][]=[];
        for(const sign of [-1,1]) {
          const perturbed=x.slice(),center=t*4+3;
          for(let i=Math.max(0,center-meta.radius);i<=Math.min(m.window-1,center+meta.radius);i++) {
            if(f+i>=features.length) continue;
            const envelope=(1+Math.cos(Math.PI*(i-center)/(meta.radius+1)))/2;
            for(let b=0;b<m.fbank.mels;b++)perturbed[i*m.fbank.mels+b]+=sign*meta.epsilon*envelope*spectral[b];
          }
          const out=await run(perturbed);
          const logProbs=logProbabilityRows(out);
          if(logProbs.length!==baselineRows.length) throw new Error("Perturbation changed output geometry");
          arms.push(centered(logProbs.slice(t*vocab,(t+1)*vocab)));
        }
        result.push(...arms[1].map((v,p)=>(v-arms[0][p])/(2*meta.epsilon)));
      }
      if(!result.every(Number.isFinite)||!raw.every(Number.isFinite)) throw new Error("Nonfinite fingerprint observations");
      response.push(result);
    }
    states=nextStates;
  }
  return {response,control,logProbs};
}

export async function addFingerprintEvidence(scorer: {model:FingerprintModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:ResponseFingerprintEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if(!scorer||!audio.length)return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of insertionCells(chunks))checker.push(e,tracker.push(trackerInput(e)));checker.end(tracker.end());
  if(!candidates.size)return chunks;
  const intervals=[...candidates.values()];
  const rows=await fingerprintObservations(scorer.backend,manifest,fingerprintMetadata(manifest),audio,
    f=>intervals.some(c=>c.fromFrame<=f&&f<c.toFrame));
  const certificates:ResponseFingerprintEvidence[]=[];
  for(const c of intervals) {
    const result=c.toFrame<=rows.response.length ? fingerprintComparison(scorer.model,
      rows.response.slice(c.fromFrame,c.toFrame),rows.control.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c);if(result&&fingerprintReversal(result))certificates.push(result);
  }
  return carryFingerprintEvidence(chunks,certificates);
}
