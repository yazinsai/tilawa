import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerEvent, CheckerOptions, AcousticOrbitEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import type { InferenceBackend, ModelManifest, Tensor } from "./model.js";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

/** None labels are canonical supervision only when the recording is usable and has no edits/repeats. */
export function eligibleAcousticOrbitTake(r: {kind?:string;split?:string;speaker?:string;exclude?:boolean|string|null;excluded?:boolean|string|null;scored?:boolean;events?:unknown[];non_errors?:unknown[]}, excluded:ReadonlySet<string>):boolean {
  return r.kind === "none" && r.split === "train" && !!r.speaker && !excluded.has(r.speaker) &&
    !r.exclude && !r.excluded && r.scored !== false && !r.events?.length && !r.non_errors?.length;
}

/** A query-local SPD metric field. Anchors are acoustic coordinates, never phone labels. */
export interface AcousticOrbitRegion { center: number[]; precision: number[][] }
export interface AcousticOrbitMetadata {
  schema: 1; inventory: string; tokens: string; dimensions: number; width: number;
  featureOutput: string; logitOutput: string; primarySha256: string;
  featureCenter: number[]; projection: number[][]; trainingRecordings: number;
}
export interface AcousticOrbitModel extends AcousticOrbitMetadata {
  means: number[][]; counts: number[];
  globalPrecision: number[][]; regions: AcousticOrbitRegion[];
  neighborhoodScale: number; temperature: number;
}
const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : -Infinity;
};
/** Complete CTC pronunciation, with all boundary paths marginalized. Legal outgoing arcs have
 * equal probability; conditioning the timing prior on completion avoids a length/path-count reward. */
export function orbitBoundaryScore(values: number[][], ids: number[]): number {
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
function quadratic(delta: number[], precision: number[][]): number {
  return precision.reduce((sum,row)=>sum+row.reduce((s,v,d)=>s+v*delta[d],0)**2,0);
}
/** Identical means and timing in all three arms; only acoustic geometry differs. */
export function orbitEmissions(model: AcousticOrbitModel, observations: number[][], arm: "local"|"euclidean"|"global"): number[][] {
  return observations.map(x=> {
    const nearest=model.regions.map((r,i)=>({i,d:r.center.reduce((s,v,d)=>s+(v-x[d])**2,0)}))
      .sort((a,b)=>a.d-b.d||a.i-b.i).slice(0,3);
    const weights=nearest.map(n=>Math.exp(-(n.d-nearest[0].d)/(2*model.neighborhoodScale)));
    const total=weights.reduce((s,v)=>s+v,0);
    const values=model.means.map((mean,p)=> {
      if(model.counts[p]<=0)return -Infinity;
      const delta=x.map((v,d)=>v-mean[d]);
      const distance=arm==="euclidean" ? delta.reduce((s,v)=>s+v*v,0) : arm==="global" ? quadratic(delta,model.globalPrecision) :
        nearest.reduce((s,n,j)=>s+weights[j]*quadratic(delta,model.regions[n.i].precision),0)/total;
      return -.5*distance/model.temperature;
    });
    const normalizer=values.reduce(logadd,-Infinity);
    return values.map(v=>v-normalizer);
  });
}
export function orbitReversal(c: AcousticOrbitEvidence): boolean {
  return [c.localWritten,c.localRival,c.euclideanWritten,c.euclideanRival,c.globalWritten,c.globalRival].every(Number.isFinite) &&
    c.localWritten>c.localRival && c.euclideanWritten<c.euclideanRival && c.globalWritten<c.globalRival;
}
/** Only retract the exact live occurrence removed by an otherwise identical evidence replay.
 * Repeated words and later graph confirmations may share a spelling or issue id. */
export function orbitVerdictRemovals(live: CheckerEvent[], baseline: CheckerEvent[], reviewed: CheckerEvent[]): CheckerEvent[] {
  const identity = (e: CheckerEvent) => JSON.stringify([e.id, e.timeMs, e.location, e.expected, e.observed]);
  const kept = new Set(reviewed.map(identity));
  const vetoed = new Set(baseline.filter(e => e.errorClass === "wrong_word" && !kept.has(identity(e))).map(identity));
  return live.filter(e => e.errorClass === "wrong_word" && vetoed.has(identity(e)));
}
/** Carry auxiliary judgments without changing any primary token or posterior. */
export function carryAcousticOrbitEvidence(chunks: PhonemeEvidence[], certificates: AcousticOrbitEvidence[]): PhonemeEvidence[] {
  if (!certificates.length) return chunks;
  const first = chunks.findIndex(e => e.phonemes.length);
  return chunks.map((e,i) => i !== first ? e : {...e, phonemes:e.phonemes.map((p,j) => j ? p : {
    ...p, acousticOrbitVetoes:certificates,
  })});
}
/** All arms see identical unaugmented frames and complete candidate pronunciations. */
export function orbitComparison(model: AcousticOrbitModel, observations: number[][],
  candidate: SequenceEvidenceCandidate): AcousticOrbitEvidence | null {
  const sequences=[candidate.written,candidate.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if (!Number.isInteger(candidate.fromFrame) || candidate.fromFrame < 0 ||
    !Number.isInteger(candidate.toFrame) || candidate.toFrame-candidate.fromFrame !== observations.length ||
    !observations.length || observations.some(r=>r.length!==model.dimensions||!r.every(Number.isFinite)) ||
    sequences.some(s=>!s.length||s.some(p=>p<1||model.counts[p]<=0))) return null;
  const scores=(arm: "local"|"euclidean"|"global")=> {
    const rows=orbitEmissions(model,observations,arm);
    return sequences.map(s=>orbitBoundaryScore(rows,s));
  };
  const a=scores("local"),b=scores("euclidean"),g=scores("global");
  const result={...candidate,localWritten:a[0],localRival:a[1],euclideanWritten:b[0],euclideanRival:b[1],globalWritten:g[0],globalRival:g[1]};
  return [a[0],a[1],b[0],b[1],g[0],g[1]].every(Number.isFinite)?result:null;
}
export function validateAcousticOrbitModel(value: unknown, manifest: ModelManifest): AcousticOrbitModel {
  const m=value as AcousticOrbitModel;
  const vector=(r:unknown,n:number):r is number[]=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite);
  const matrix=(v:unknown,rows:number,cols:number):v is number[][]=>Array.isArray(v)&&v.length===rows&&v.every(r=>vector(r,cols));
  const precision=(v:unknown,n:number)=>matrix(v,n,n)&&v.every((r,i)=>r[i]>0&&r.every((x,j)=>j<=i||x===0));
  if(!m||m.schema!==1||m.inventory!==manifest.inventory||m.tokens!==manifest.tokens||m.primarySha256!==manifest.sha256||
    !Number.isSafeInteger(m.dimensions)||m.dimensions<=0||m.dimensions>32||!Number.isSafeInteger(m.width)||m.width<m.dimensions||
    !m.featureOutput||!m.logitOutput||!vector(m.featureCenter,m.width)||!matrix(m.projection,m.dimensions,m.width)||
    !Number.isSafeInteger(m.trainingRecordings)||m.trainingRecordings<3||
    !matrix(m.means,m.tokens.length+1,m.dimensions)||!vector(m.counts,m.tokens.length+1)||m.counts.some(v=>v<0)||
    !precision(m.globalPrecision,m.dimensions)||!Array.isArray(m.regions)||m.regions.length<2||m.regions.length>64||
    m.regions.some(r=>!r||!vector(r.center,m.dimensions)||!precision(r.precision,m.dimensions))||
    !Number.isFinite(m.neighborhoodScale)||m.neighborhoodScale<=0||!Number.isFinite(m.temperature)||m.temperature<=0)
    throw new Error("Invalid acoustic-orbit model");
  return m;
}
function project(x: Float32Array, matrix: number[][], center: number[]): number[] {
  return matrix.map(row => row.reduce((s,v,d) => s+v*(x[d]-center[d]),0));
}
/** Full-session separate streaming state. The final linear input includes activation quantization.
 * Both arms retain the same frame clock; primary rows and caches are untouched. */
export async function encoderObservations(backend: InferenceBackend, manifest: ModelManifest, meta: Pick<AcousticOrbitMetadata,"featureOutput"|"logitOutput"|"width">, audio: Float32Array) {
  const features = new Fbank().push(audio), logProbs: number[][] = [], encoder: number[][] = [];
  const total = Math.max(0,Math.floor((Math.floor((features.length-7)/2)+1)/2));
  let states: Record<string,Tensor> = Object.fromEntries(manifest.states.map(s=>[s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype === "int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const pad = new Float32Array(manifest.fbank.mels).fill(manifest.fbank.padValue);
  for (let f=0;logProbs.length<total;f+=manifest.shift) {
    const input = new Float32Array(manifest.window*manifest.fbank.mels);
    for(let i=0;i<manifest.window;i++)input.set(features[f+i]??pad,i*manifest.fbank.mels);
    const out=await backend.run({x:{type:"float32",dims:[1,manifest.window,manifest.fbank.mels],data:input},...states});
    states=Object.fromEntries(manifest.states.map(s=>[s.name,out[`new_${s.name}`]]));
    const vocab=manifest.tokens.length+1, n=out.log_probs.data.length/vocab;
    const h=out[meta.featureOutput],z=out[meta.logitOutput];
    if(!Number.isSafeInteger(n)||n<=0||!h||!z||h.type!=="float32"||z.type!=="float32"||
       h.data.length!==n*meta.width||z.data.length!==n*vocab) throw new Error("Invalid final projection geometry");
    for(let t=0;t<n && logProbs.length<total;t++) {
      encoder.push(Array.from(h.data.slice(t*meta.width,(t+1)*meta.width),Number));
      logProbs.push(Array.from(out.log_probs.data.slice(t*vocab,(t+1)*vocab),Number));
    }
  }
  return {encoder,logProbs};
}
export async function orbitObservations(backend: InferenceBackend, manifest: ModelManifest, meta: AcousticOrbitMetadata, audio: Float32Array) {
  const {encoder,logProbs}=await encoderObservations(backend,manifest,meta,audio);
  return {observations:encoder.map(r=>project(Float32Array.from(r),meta.projection,meta.featureCenter)),logProbs};
}
export async function addAcousticOrbitEvidence(scorer: {model:AcousticOrbitModel;backend:InferenceBackend}|undefined, manifest:ModelManifest,
  audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},
  mode:"free"|"memorization"="free",inspect?:(c:AcousticOrbitEvidence|null,request:SequenceEvidenceCandidate)=>void):Promise<PhonemeEvidence[]> {
  if (!scorer||!audio.length) return chunks;
  const candidates=new Map<string,SequenceEvidenceCandidate>(),cells=insertionCells(chunks);
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  if (!candidates.size) return chunks;
  const features=await orbitObservations(scorer.backend,manifest,scorer.model,audio),certificates:AcousticOrbitEvidence[]=[];
  for (const c of candidates.values()) {
    const result=c.fromFrame>=0&&c.toFrame<=features.observations.length ? orbitComparison(scorer.model,
      features.observations.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(result,c); if (result&&orbitReversal(result)) certificates.push(result);
  }
  return carryAcousticOrbitEvidence(chunks,certificates);
}
