import { createChecker, createTracker, trackerInput, minGapFrames } from "@tilawa/core";
import type { CheckerOptions, DecodedPhoneme, NoisePowerEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

// Individual positive-frequency FFT bins; neither DC nor Nyquist has complex circular noise.
export const POWER_BINS = [4, 8, 12, 18, 26, 36, 48, 62, 78, 96, 116, 138, 162, 188, 216, 240];
export interface Quadrature { nodes: number[]; weights: number[] }
export interface NoisePowerModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; bins: number[];
  means: number[][]; deviations: number[][]; counts: number[];
  quadrature: Quadrature; refined: Quadrature; maxFrames: number;
}
export const logadd = (a: number, b: number): number => {
  const m = Math.max(a,b); return Number.isFinite(m) ? m + Math.log(Math.exp(a-m)+Math.exp(b-m)) : m;
};
/** log(I0(z))-z: series at small arguments and the positive asymptotic expansion at large ones. */
export function logI0Scaled(z: number): number {
  if (z < 50) {
    let term=1, sum=1;
    for (let k=1;k<200;k++) { term *= (z*z/4)/(k*k); sum += term; if (term < sum*1e-15) break; }
    return Math.log(sum)-z;
  }
  let term=1, sum=1;
  for (let k=1;k<=12;k++) { term *= (2*k-1)**2/(8*k*z); sum += term; }
  return -.5*Math.log(2*Math.PI*z)+Math.log(sum);
}
/** If E|N|²=n, 2|S+N|²/n is noncentral chi-square(2, 2|S|²/n).
 * Evaluate its density in power units, avoiding cancellation at high SNR. */
export function noisyPowerLogP(y: number, s: number, n: number): number {
  const rootY=Math.sqrt(y),rootS=Math.sqrt(s),z=2*rootY*rootS/n;
  return -Math.log(n)-(rootY-rootS)**2/n+logI0Scaled(z);
}
/** Matched delta-method additive error in log power: zero mean, variance 2n/s.
 * Both arms integrate the same clean-power uncertainty and use the same power-density Jacobian. */
export function logPowerControl(y: number, s: number, n: number): number {
  const variance=2*n/s;
  return -.5*(Math.log(2*Math.PI*variance)+(Math.log(y)-Math.log(s))**2/variance)-Math.log(y);
}
export function marginalEmission(y: number[], noise: number[], mean: number[], deviation: number[], q: Quadrature, explicit: boolean): number {
  let result=0;
  for (let d=0;d<y.length;d++) {
    let sum=-Infinity;
    for (let j=0;j<q.nodes.length;j++) {
      const clean=Math.exp(mean[d]+deviation[d]*q.nodes[j]);
      sum=logadd(sum,Math.log(q.weights[j])+(explicit ? noisyPowerLogP(y[d],clean,noise[d]) : logPowerControl(y[d],clean,noise[d])));
    }
    result+=sum;
  }
  return result;
}
/** Uniform prior over every complete monotonic phone segmentation, normalized separately for each sequence. */
export function boundaryLogP(emissions: number[][]): number {
  if (!emissions.length || !emissions[0].length || emissions.length<emissions[0].length) return -Infinity;
  let scores=emissions[0].map((v,i)=>i ? -Infinity : v), prior=scores.map((_,i)=>i ? -Infinity : 0);
  for (let t=1;t<emissions.length;t++) {
    scores=scores.map((v,s)=>logadd(v,s?scores[s-1]:-Infinity)+emissions[t][s]);
    prior=prior.map((v,s)=>logadd(v,s?prior[s-1]:-Infinity));
  }
  return scores.at(-1)!-prior.at(-1)!;
}
/** Optional noise-only intervals between mandatory phone states. Repeated articulations
 * remain separate compulsory states. The same normalized segmentation prior serves both arms. */
export function boundaryWithNoise(emissions: number[][], background: number[]): number {
  if (!emissions.length || emissions.length<emissions[0].length) return -Infinity;
  const n=2*emissions[0].length+1;
  let score=Array(n).fill(-Infinity),prior=score.slice();
  score[0]=background[0];score[1]=emissions[0][0];prior[0]=prior[1]=0;
  for (let t=1;t<emissions.length;t++) {
    score=score.map((v,s)=>logadd(logadd(v,s?score[s-1]:-Infinity),s>1 && s%2 ? score[s-2] : -Infinity)+
      (s%2 ? emissions[t][(s-1)/2] : background[t]));
    prior=prior.map((v,s)=>logadd(logadd(v,s?prior[s-1]:-Infinity),s>1 && s%2 ? prior[s-2] : -Infinity));
  }
  return logadd(score[n-1],score[n-2])-logadd(prior[n-1],prior[n-2]);
}
export class PowerObservations {
  readonly frames: number[][]=[];
  private readonly fbank=new Fbank();
  push(audio: Float32Array): void { this.fbank.push(audio,p=>this.frames.push(POWER_BINS.map(k=>p[k]))); }
  /** Only completed preceding token gaps satisfying the engine's existing pause length qualify.
   * Remove windows that could intersect either bordering speech token. Freeze the resulting mean. */
  noiseBefore(fromFrame: number, phones: DecodedPhoneme[]): number[] | null {
    const sum=POWER_BINS.map(()=>0); let count=0;
    for (let i=1;i<phones.length;i++) {
      const left=phones[i-1],right=phones[i];
      if (right.start>fromFrame || right.start-left.end-1<minGapFrames(40)) continue;
      const a=(left.end+1)*4+3, b=Math.min(right.start*4-3,fromFrame*4,this.frames.length);
      for (let f=a;f<b;f++) { for (let d=0;d<sum.length;d++) sum[d]+=this.frames[f][d]; count++; }
    }
    const mean=sum.map(v=>v/count);
    return count && mean.every(v=>Number.isFinite(v)&&v>0) ? mean : null;
  }
}
export function powerComparison(model: NoisePowerModel, observations: PowerObservations, phones: DecodedPhoneme[], c: SequenceEvidenceCandidate): NoisePowerEvidence | null {
  if (!Number.isInteger(c.fromFrame)||!Number.isInteger(c.toFrame)||c.fromFrame<0||c.toFrame*4>observations.frames.length) return null;
  const rows=observations.frames.slice(c.fromFrame*4,c.toFrame*4), noise=observations.noiseBefore(c.fromFrame,phones);
  const sequences=[c.written,c.rival].map(s=>[...s].map(p=>model.tokens.indexOf(p)+1));
  if (!noise || !rows.length || rows.length>model.maxFrames || rows.some(r=>r.some(v=>!Number.isFinite(v)||v<=0)) ||
    sequences.some(s=>!s.length||s.length>rows.length||s.some(p=>p<1||model.counts[p]<=0))) return null;
  const used=[...new Set(sequences.flat())];
  const background=rows.map(r=>r.reduce((s,y,d)=>s-Math.log(noise[d])-y/noise[d],0));
  const scores=(q: Quadrature,explicit: boolean)=> {
    const values=new Map(used.map(p=>[p,rows.map(r=>marginalEmission(r,noise,model.means[p],model.deviations[p],q,explicit))]));
    return sequences.map(s=>boundaryWithNoise(rows.map((_,t)=>s.map(p=>values.get(p)![t])),background));
  };
  const control=scores(model.quadrature,false), power=scores(model.quadrature,true);
  const refinedControl=scores(model.refined,false), refinedPower=scores(model.refined,true);
  if (![...control,...power,...refinedControl,...refinedPower].every(Number.isFinite)) return null;
  // Refined integration must preserve both strict signs; instability abstains.
  if ((control[0]<control[1]) !== (refinedControl[0]<refinedControl[1]) ||
      (power[0]>power[1]) !== (refinedPower[0]>refinedPower[1])) return null;
  return {...c,controlWritten:refinedControl[0],controlRival:refinedControl[1],powerWritten:refinedPower[0],powerRival:refinedPower[1]};
}
export const powerReversal=(c: NoisePowerEvidence): boolean =>
  [c.controlWritten,c.controlRival,c.powerWritten,c.powerRival].every(Number.isFinite) && c.controlWritten<c.controlRival && c.powerWritten>c.powerRival;
export function validateNoisePowerModel(value: unknown, inventory: string, tokens: string): NoisePowerModel {
  const m=value as NoisePowerModel,n=tokens.length+1;
  const vector=(v: number[],length: number,positive=false)=>Array.isArray(v)&&v.length===length&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v: number[][],positive=false)=>Array.isArray(v)&&v.length===n&&v.every(r=>vector(r,16,positive));
  const q=(v: Quadrature,count: number)=>v&&vector(v.nodes,count)&&vector(v.weights,count,true)&&Math.abs(v.weights.reduce((a,b)=>a+b,0)-1)<1e-10;
  if (!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||JSON.stringify(m.bins)!==JSON.stringify(POWER_BINS)||
    !matrix(m.means)||!matrix(m.deviations,true)||!vector(m.counts,n)||m.counts.some(c=>c<0)||!q(m.quadrature,17)||!q(m.refined,33)||
    !Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>2048) throw new Error("Invalid noise power acoustic model or inventory");
  return m;
}
export function addPowerEvidence(model: NoisePowerModel | undefined,audio: Float32Array,chunks: PhonemeEvidence[],index: ReferenceIndex,
  phrases?: PhraseAsset,options: CheckerOptions={},mode: "free" | "memorization"="free",
  inspect?: (comparison: NoisePowerEvidence | null,candidate: SequenceEvidenceCandidate)=>void): PhonemeEvidence[] {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks); if (!cells.length) return chunks;
  if (cells[0].frameMs!==40) throw new Error("Noise power scorer requires a 40 ms evidence clock");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c) {candidates.set(JSON.stringify(c),c);return true;}});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e))); checker.end(tracker.end());
  const observations=new PowerObservations(); observations.push(audio);
  const phones=chunks.flatMap(e=>e.phonemes),certificates: NoisePowerEvidence[]=[];
  for (const c of candidates.values()) {
    const comparison=powerComparison(model,observations,phones,c); inspect?.(comparison,c);
    if (comparison && powerReversal(comparison)) certificates.push(comparison);
  }
  if (!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,noisePowerVetoes:certificates})});
}
