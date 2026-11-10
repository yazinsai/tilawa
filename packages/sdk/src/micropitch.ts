import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, MicropitchEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";

export interface PitchFrame { logPitch: number; reliability: number }
export interface MicropitchModel {
  schema: 1; inventory: string; tokens: string;
  radius: number; reliability: number; shrinkage: number;
  scale: number; variance: number;
  stay: number[];
  means: Record<string, number>; control: Record<string, number>; counts: Record<string, number>;
}
const VOWELS = "َُِاۦۥ";
const CONSONANTS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي";

/** Normalized lag correlation on 40 ms windows, on a 10 ms clock. The
 * estimator sees waveform only; neither candidate can choose the melody.
 */
export function pitchFrames(audio: Float32Array): PitchFrame[] {
  const out: PitchFrame[] = [];
  for (let start=0; start+640<=audio.length; start+=160) {
    const x=Array.from(audio.subarray(start,start+640));
    const mean=x.reduce((a,b)=>a+b,0)/x.length;
    for(let j=0;j<x.length;j++) x[j]-=mean;
    const cor=Array<number>(321).fill(0);
    for(let lag=32;lag<=320;lag++) {
      let xy=0,xx=0,yy=0;
      for(let j=0;j<x.length-lag;j++) {xy+=x[j]*x[j+lag];xx+=x[j]*x[j];yy+=x[j+lag]*x[j+lag];}
      cor[lag]=xx*yy>1e-16 ? xy/Math.sqrt(xx*yy) : 0;
    }
    // First strong local maximum avoids choosing integer multiples of the period.
    const peaks=Array.from({length:287},(_,i)=>i+33).filter(i=>cor[i]>=cor[i-1]&&cor[i]>cor[i+1]);
    const maximum=Math.max(0,...peaks.map(i=>cor[i]));
    const lag=peaks.find(i=>cor[i]>=maximum*.95);
    if(lag===undefined) {out.push({logPitch:0,reliability:0});continue;}
    const denom=cor[lag-1]-2*cor[lag]+cor[lag+1];
    const offset=denom ? Math.max(-.5,Math.min(.5,.5*(cor[lag-1]-cor[lag+1])/denom)) : 0;
    out.push({logPitch:Math.log(16000/(lag+offset)),reliability:Math.min(1,Math.max(0,cor[lag]))});
  }
  return out;
}

/** Robust leave-neighbourhood-out linear contour. Excluding the central
 * 60 ms keeps the brief disturbance out of its own melodic baseline.
 * Only completed observations in the supplied audio participate.
 */
export function pitchResiduals(frames: PitchFrame[], radius: number, reliable: number): number[][] {
  const half=Math.round(radius*100);
  return frames.map((f,t)=> {
    if(f.reliability<reliable) return [0,0];
    const neighbours=[];
    for(let j=Math.max(0,t-half);j<Math.min(frames.length,t+half+1);j++)
      if(Math.abs(j-t)>6 && frames[j].reliability>=reliable) neighbours.push(j);
    if(neighbours.length<8 || !neighbours.some(j=>j<t) || !neighbours.some(j=>j>t)) return [0,0];
    const median=(a:number[])=>a.slice().sort((a,b)=>a-b)[Math.floor(a.length/2)];
    const center=median(neighbours.map(j=>frames[j].logPitch));
    const spread=Math.max(.02,median(neighbours.map(j=>Math.abs(frames[j].logPitch-center)))*3);
    const keep=neighbours.filter(j=>Math.abs(frames[j].logPitch-center)<=spread);
    let w=0,sx=0,sy=0,sxx=0,sxy=0;
    for(const j of keep) {const a=frames[j].reliability, x=(j-t)/100,y=frames[j].logPitch;w+=a;sx+=a*x;sy+=a*y;sxx+=a*x*x;sxy+=a*x*y;}
    const determinant=w*sxx-sx*sx;
    if(determinant<=1e-8) return [0,0];
    const baseline=(sy*sxx-sxy*sx)/determinant;
    // Gross period jumps cannot be evidence for a consonant.
    const delta=f.logPitch-baseline;
    return Math.abs(delta)>.25 ? [0,0] : [delta,1];
  });
}

export interface PitchState { key: string; control: string; phase: number; adjacent: boolean }
/** All pronunciation symbols remain obligatory. Only a vowel edge immediately
 * adjoining a consonant is conditioned on it; expanded vowel interiors do not
 * manufacture another consonant disturbance.
 */
export function pitchStates(pronunciation: string): PitchState[] {
  const p=[...pronunciation], out:PitchState[]=[];
  for(let i=0;i<p.length;i++) {
    if(!VOWELS.includes(p[i])) {out.push({key:"other",control:"other",phase:1,adjacent:false});continue;}
    const left=i>0&&CONSONANTS.includes(p[i-1])?p[i-1]:"_";
    const right=i+1<p.length&&CONSONANTS.includes(p[i+1])?p[i+1]:"_";
    for(let phase=0;phase<3;phase++) {
      const control=p[i]+":"+phase, c=phase===0?left:phase===2?right:"_";
      out.push({key:control+":"+c,control,phase,adjacent:phase!==1&&c!=="_"});
    }
  }
  return out;
}
const logadd=(a:number,b:number)=> {const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;};

/** Forward sum over every phone/phase placement with a normalized shared
 * geometric duration law. Unreliable pitch has a marginalized unit likelihood.
 * Subtract the identity-free timing partition so path count alone cannot win.
 */
export function scoreMicropitch(m:MicropitchModel, raw:number[][], pronunciation:string, conditioned:boolean): {score:number; evidence:number}|null {
  const states=pitchStates(pronunciation), n=states.length;
  if(!n || raw.length<n || raw.length>6000 || [...pronunciation].some(p=>!m.tokens.includes(p)) ||
    raw.some(r=>r.length!==2||r.some(v=>!Number.isFinite(v))||(r[1]!==0&&r[1]!==1)) ||
    states.some(s=>!(s.control in m.control) || !(s.key in m.means) || (s.adjacent && !(m.counts[s.key]>0)))) return null;
  const stay=states.map(s=>Math.log(m.stay[s.phase])), step=states.map(s=>Math.log1p(-m.stay[s.phase]));
  let a=Array<number>(n).fill(-Infinity), prior=a.slice(), evidence=Array<number>(n).fill(0);
  for(let t=0;t<raw.length;t++) {
    const [y,r]=raw[t];
    const next=a.slice(), np=prior.slice(), ne=evidence.slice();
    for(let s=0;s<n;s++) {
      const self=t?a[s]+stay[s]:-Infinity, advance=t&&s?a[s-1]+step[s-1]:t===0&&s===0?0:-Infinity;
      const z=logadd(self,advance);
      const mu=conditioned?m.means[states[s].key]:m.control[states[s].control];
      const emission=r ? -.5*(Math.log(2*Math.PI*m.variance)+(y/m.scale-mu)**2/m.variance)-Math.log(m.scale) : 0;
      next[s]=z+emission;
      np[s]=logadd(t?prior[s]+stay[s]:-Infinity,t&&s?prior[s-1]+step[s-1]:t===0&&s===0?0:-Infinity);
      ne[s]=Number.isFinite(z)?(Number.isFinite(self)?Math.exp(self-z)*evidence[s]:0)+(Number.isFinite(advance)&&s?Math.exp(advance-z)*evidence[s-1]:0)+(r&&states[s].adjacent?1:0):0;
    }
    a=next;prior=np;evidence=ne;
  }
  const score=a[n-1]-prior[n-1];
  return Number.isFinite(score)?{score,evidence:evidence[n-1]}:null;
}
export const micropitchReversal=(c:MicropitchEvidence)=>
  [c.controlWritten,c.controlRival,c.micropitchWritten,c.micropitchRival].every(Number.isFinite) &&
  c.controlWritten<c.controlRival && c.micropitchWritten>c.micropitchRival;
export function micropitchComparison(m:MicropitchModel, raw:number[][], c:SequenceEvidenceCandidate):MicropitchEvidence|null {
  const w=scoreMicropitch(m,raw,c.written,true),r=scoreMicropitch(m,raw,c.rival,true);
  const cw=scoreMicropitch(m,raw,c.written,false),cr=scoreMicropitch(m,raw,c.rival,false);
  if(!w||!r||!cw||!cr||Math.min(w.evidence,r.evidence)<3) return null;
  return {...c,controlWritten:cw.score,controlRival:cr.score,micropitchWritten:w.score,micropitchRival:r.score};
}

export function validateMicropitchModel(value:unknown, inventory:string, tokens:string):MicropitchModel {
  const m=value as MicropitchModel;
  if(!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||
    !Number.isFinite(m.radius)||m.radius<.15||m.radius>.5||!Number.isFinite(m.reliability)||m.reliability<.7||m.reliability>=1||
    !Number.isFinite(m.shrinkage)||m.shrinkage<=0||!Number.isFinite(m.scale)||m.scale<=0||!Number.isFinite(m.variance)||m.variance<=0||
    !Array.isArray(m.stay)||m.stay.length!==3||m.stay.some(x=>!Number.isFinite(x)||x<=0||x>=1)||
    !m.means||!m.control||!m.counts||Object.values(m.means).some(x=>!Number.isFinite(x))||
    Object.values(m.control).some(x=>!Number.isFinite(x))||Object.values(m.counts).some(x=>!Number.isFinite(x)||x<0)||
    !('other' in m.means)||!('other' in m.control)||Object.keys(m.means).some(k=>!(k in m.counts)))
    throw new Error("Invalid consonant micropitch model");
  return m;
}

/** Append-only audio; contour observations are shared by both candidates. */
export class MicropitchObservations {
  readonly frames:PitchFrame[]=[];
  private tail=new Float32Array();
  push(audio:Float32Array):void {
    const joined=new Float32Array(this.tail.length+audio.length);joined.set(this.tail);joined.set(audio,this.tail.length);
    const frames=pitchFrames(joined);this.frames.push(...frames);
    this.tail=joined.slice(frames.length*160);
  }
}

/** Generic discovery on unchanged tracker/checker evidence. Certificates are
 * keyed by complete pronunciation and exact frame interval, never by a test case.
 */
export function addMicropitchEvidence(model:MicropitchModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],
  index:ReferenceIndex,phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(comparison:MicropitchEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if(!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);
  if(!cells.length) return chunks;
  if(cells[0].frameMs!==40) throw new Error("Micropitch scorer requires a 40 ms evidence clock");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const raw=pitchResiduals(pitchFrames(audio),model.radius,model.reliability),certificates:MicropitchEvidence[]=[];
  for(const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame*4<=raw.length?micropitchComparison(model,raw.slice(c.fromFrame*4,c.toFrame*4),c):null;
    inspect?.(comparison,c);
    if(comparison&&micropitchReversal(comparison)) certificates.push(comparison);
  }
  if(!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,micropitchVetoes:[...(p.micropitchVetoes??[]),...certificates]})});
}
