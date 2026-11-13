import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { CheckerOptions, MovingZeroEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, SequenceEvidenceCandidate } from "@tilawa/core";
import { Fbank, fft512 } from "./fbank.js";
import { insertionCells } from "./seam-insertion.js";

export interface RootFrame { poles: number[]; zeros: number[][]; valid: boolean }
interface Gaussian { mean: number[][]; variance: number[][] }
interface ZeroArm { mean: number[][][]; variance: number[][][] }
interface MovingArm extends ZeroArm { alpha: number[][]; noise: number[][] }
export interface MovingZeroModel {
  schema: 1; inventory: string; tokens: string; frameMs: 40; stay: number; maxFrames: number;
  phoneCounts: number[]; center: number[]; scale: number[]; stationaryVariance: number[];
  poles: Gaussian; control: ZeroArm; moving: MovingArm;
}
const PERMS = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
const add = (a: number,b: number): number => {
  const m=Math.max(a,b);return Number.isFinite(m)?m+Math.log(Math.exp(a-m)+Math.exp(b-m)):-Infinity;
};
const sum = (xs:number[]):number=>xs.reduce(add,-Infinity);
const normal = (x:number[],m:number[],v:number[]):number=>x.reduce((s,y,d)=>s-.5*(Math.log(2*Math.PI*v[d])+(y-m[d])**2/v[d]),0);
type Complex = [number,number];
const mul = (a:Complex,b:Complex):Complex=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
const div = (a:Complex,b:Complex):Complex=> { const d=b[0]**2+b[1]**2;return [(a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d]; };

/** Pivoted normal-equation solve; ridge keeps the fixed rational fit bounded. */
function solve(matrix:number[][],target:number[]):number[] {
  const a=matrix.map((r,i)=>[...r,target[i]]),n=target.length;
  for(let j=0;j<n;j++) {
    let pivot=j;for(let k=j+1;k<n;k++) if(Math.abs(a[k][j])>Math.abs(a[pivot][j])) pivot=k;
    [a[j],a[pivot]]=[a[pivot],a[j]];
    if(Math.abs(a[j][j])<1e-14) throw new Error("Singular rational fit");
    for(let k=j+1;k<n;k++) {const factor=a[k][j]/a[j][j];for(let d=j;d<=n;d++) a[k][d]-=factor*a[j][d];}
  }
  const x=Array<number>(n).fill(0);
  for(let j=n-1;j>=0;j--) {let v=a[j][n];for(let k=j+1;k<n;k++) v-=a[j][k]*x[k];x[j]=v/a[j][j];}
  return x;
}

/** Simultaneous polynomial roots, with no root-index identity assumption. */
export function rootPairs(coefficients:number[]):number[][] {
  if(Math.abs(coefficients[0])<1e-12) throw new Error("Degenerate numerator");
  const c=coefficients.map(x=>x/coefficients[0]),n=c.length-1;
  let roots:Complex[]=Array.from({length:n},(_,j)=>[.7*Math.cos(2*Math.PI*(j+.25)/n),.7*Math.sin(2*Math.PI*(j+.25)/n)]);
  let converged=false;
  for(let iteration=0;iteration<160;iteration++) {
    let change=0;
    const next=roots.map((z,j)=> {
      let value:Complex=[c[0],0],den:Complex=[1,0];
      for(let k=1;k<c.length;k++) {value=mul(value,z);value[0]+=c[k];}
      for(let k=0;k<n;k++) if(k!==j) den=mul(den,[z[0]-roots[k][0],z[1]-roots[k][1]]);
      const delta=div(value,den);change=Math.max(change,Math.hypot(...delta));
      return [z[0]-delta[0],z[1]-delta[1]] as Complex;
    });
    roots=next;
    if(!roots.flat().every(Number.isFinite)) throw new Error("Nonfinite rational roots");
    if(change<1e-10) {converged=true;break;}
  }
  if(!converged) throw new Error("Unresolved rational roots");
  const pairs=roots.filter(z=>z[1]>1e-6),real=roots.filter(z=>Math.abs(z[1])<=1e-6).sort((a,b)=>a[0]-b[0]);
  for(let j=0;j+1<real.length;j+=2) pairs.push([(real[j][0]+real[j+1][0])/2,0]);
  if(pairs.length!==n/2) throw new Error("Unresolved conjugate roots");
  return pairs.map(z=> {
    const radius=Math.hypot(...z),r=Math.max(.01,Math.min(.999,radius,1/Math.max(radius,1e-8)));
    return [Math.abs(Math.atan2(z[1],z[0]))/Math.PI,-Math.log(r)/Math.PI];
  }).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
}

/** Fixed 8-pole/6-zero minimum-phase rational fit to the smoothed spectrum. */
export function spectralRoots(power:Float64Array):RootFrame {
  if(power.length!==512||!Array.from(power).every(x=>Number.isFinite(x)&&x>=0)) throw new Error("Invalid spectrum");
  const energy=Math.log(Math.max(power.reduce((s,x)=>s+x,0)/512,1e-12));
  try {
    const level=Math.max(power.reduce((s,x)=>s+x,0)/512,1e-30);
    const re=Float64Array.from(power,x=>.5*Math.log(Math.max(x/level,1e-12))),im=new Float64Array(512);
    const mean=re.reduce((s,x)=>s+x,0)/512;for(let j=0;j<512;j++) re[j]-=mean;
    // Inverse transform of the real log magnitude via conjugation.
    fft512(re,im);for(let j=0;j<512;j++) {re[j]/=512;im[j]=-im[j]/512;}
    for(let j=1;j<512;j++) re[j]=j<=16?2*re[j]:0;
    im.fill(0);fft512(re,im);
    const h:Complex[]=[],z:Complex[]=[],aa:Complex[][]=[],bb:Complex[][]=[];
    for(let k=1;k<256;k+=4) {
      h.push([Math.exp(re[k])*Math.cos(im[k]),Math.exp(re[k])*Math.sin(im[k])]);
      z.push([Math.cos(-2*Math.PI*k/512),Math.sin(-2*Math.PI*k/512)]);
      const a:Complex[]=[],b:Complex[]=[[1,0]];let v:Complex=[1,0];
      for(let j=1;j<=8;j++) {v=mul(v,z[z.length-1]);a.push(v);if(j<=6) b.push(v);}
      aa.push(a);bb.push(b);
    }
    let a=Array<number>(8).fill(0),c:number[]=[];
    for(let iteration=0;iteration<3;iteration++) {
      const xx=Array.from({length:15},(_,j)=>Array.from({length:15},(_,k)=>j===k?.01:0));
      const xy=Array<number>(15).fill(0);
      for(let k=0;k<h.length;k++) {
        const denominator:Complex=[1,0];for(let j=0;j<8;j++) {denominator[0]+=aa[k][j][0]*a[j];denominator[1]+=aa[k][j][1]*a[j];}
        const weight=1/Math.max(Math.hypot(...h[k]),.1)/Math.max(Math.hypot(...denominator),.1);
        const design=[...aa[k].map(v=>mul(h[k],v)),...bb[k].map(v=>[-v[0],-v[1]] as Complex)];
        for(let component=0;component<2;component++) {
          const row=design.map(v=>v[component]*weight),y=-h[k][component]*weight;
          for(let j=0;j<15;j++) {xy[j]+=row[j]*y;for(let l=0;l<15;l++) xx[j][l]+=row[j]*row[l];}
        }
      }
      c=solve(xx,xy);a=c.slice(0,8);
    }
    const poles=rootPairs([1,...a]),zeros=rootPairs(c.slice(8));
    const moments=[1,2,3,4].map(j=>poles.reduce((s,p)=>s+p[0]**j,0)/4);
    moments.push(poles.reduce((s,p)=>s+p[1],0)/4,poles.reduce((s,p)=>s+p[1]**2,0)/4,poles.reduce((s,p)=>s+p[0]*p[1],0)/4,energy);
    return {poles:moments,zeros,valid:power.reduce((s,x)=>s+x,0)>1e-9};
  } catch {return {poles:[0,0,0,0,0,0,0,energy],zeros:[[0,0],[0,0],[0,0]],valid:false};}
}

/** The first native 25 ms analysis in each 40 ms evidence cell. */
export class MovingZeroObservations {
  readonly frames:RootFrame[]=[];
  private native=0;
  private readonly frontend=new Fbank({onSpectrum:power=> {if(this.native++%4===0) this.frames.push(spectralRoots(power));}});
  push(audio:Float32Array):void {this.frontend.push(audio);}
}
export function movingZeroFeatures(audio:Float32Array):RootFrame[] {const o=new MovingZeroObservations();o.push(audio);return o.frames;}

/** Forward sum over complete phone paths, all six root correspondences, and
 * a single nuisance center per window. The control resets correspondence every
 * frame; the moving density conditions on the preceding articulation roots.
 */
export function scoreMovingZeroSequence(m:MovingZeroModel,raw:RootFrame[],pronunciation:string,moving:boolean):number|null {
  const labels=[...pronunciation].map(p=>m.tokens.indexOf(p)+1);
  if(!labels.length||!raw.length||raw.length>m.maxFrames||labels.some(p=>p<=0||m.phoneCounts[p-1]<=0)||
    raw.some(r=>!validFrame(r))||!raw.some(r=>r.valid)) return null;
  const ranked=raw.filter(r=>r.valid).map(r=>r.zeros.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]));
  const centers=Array.from({length:3},(_,j)=>[0,1].map(d=>ranked.reduce((s,r)=>s+r[j][d],0)/ranked.length));
  const xs=raw.map(r=>PERMS.map(p=>[...r.zeros[p[1]],...r.zeros[p[2]]]));
  const nuisance=raw.map(r=>centers.map(center=>PERMS.map(p=>r.valid?normal(r.zeros[p[0]],center,m.stationaryVariance):0)));
  const pole=raw.map(r=>m.poles.mean.map((mean,p)=>normal(r.poles.map((v,d)=>(v-m.center[d])/m.scale[d]),mean,m.poles.variance[p])));
  const states=[0];for(const p of labels) states.push(p,0);
  const n=states.length,skip=(s:number)=>s>=2&&states[s]!==0&&states[s]!==states[s-2];
  const count=states.map((_,s)=>1+(s+1<n?1:0)+(s+2<n&&skip(s+2)?1:0));
  const self=count.map(k=>k===1?0:Math.log(m.stay)),step=count.map(k=>k>1?Math.log((1-m.stay)/(k-1)):-Infinity);
  const arm=moving?m.moving:m.control;
  const initial=(t:number,p:number,j:number):number=>!raw[t].valid?0:moving?normal(xs[t][j],arm.mean[p][0],arm.variance[p][0]):
    sum([0,1].map(k=>normal(xs[t][j],arm.mean[p][k],arm.variance[p][k])))-Math.log(2);
  let a=Array.from({length:3},()=>Array.from({length:n},()=>Array<number>(6).fill(-Infinity)));
  for(let h=0;h<3;h++) for(let s=0;s<2;s++) for(let j=0;j<6;j++) a[h][s][j]=initial(0,states[s],j)+pole[0][states[s]]+nuisance[0][h][j]-Math.log(36);
  let prior=Array<number>(n).fill(-Infinity);prior[0]=prior[1]=-Math.log(2);
  for(let t=1;t<raw.length;t++) {
    const transitions=states.map(p=>Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=> {
      if(!raw[t].valid) return 0;
      if(!moving||!raw[t-1].valid) return initial(t,p,j);
      const predicted=m.moving.mean[p][0].map((v,d)=>v+m.moving.alpha[p][d]*(xs[t-1][i][d]-v));
      return normal(xs[t][j],predicted,m.moving.noise[p]);
    })));
    const next=Array.from({length:3},()=>Array.from({length:n},()=>Array<number>(6).fill(-Infinity)));
    for(let h=0;h<3;h++) for(let s=0;s<n;s++) {
      const incoming=Array.from({length:6},(_,i)=> {
        let v=a[h][s][i]+self[s];if(s>0) v=add(v,a[h][s-1][i]+step[s-1]);
        if(skip(s)) v=add(v,a[h][s-2][i]+step[s-2]);return v;
      });
      for(let j=0;j<6;j++) next[h][s][j]=sum(incoming.map((v,i)=>v+transitions[s][i][j]))-Math.log(6)+pole[t][states[s]]+nuisance[t][h][j];
    }
    const np=prior.map((p,s)=> {
      let v=p+self[s];if(s>0) v=add(v,prior[s-1]+step[s-1]);if(skip(s)) v=add(v,prior[s-2]+step[s-2]);return v;
    });
    a=next;prior=np;
  }
  const value=sum(a.flatMap(h=>h.slice(-2).flat()))-add(prior[n-1],prior[n-2]);
  return Number.isFinite(value)?value:null;
}
export function movingZeroComparison(m:MovingZeroModel,raw:RootFrame[],c:SequenceEvidenceCandidate):MovingZeroEvidence|null {
  if(!Number.isInteger(c.fromFrame)||c.fromFrame<0||!Number.isInteger(c.toFrame)||c.toFrame-c.fromFrame!==raw.length) return null;
  const controlWritten=scoreMovingZeroSequence(m,raw,c.written,false),controlRival=scoreMovingZeroSequence(m,raw,c.rival,false);
  const movingWritten=scoreMovingZeroSequence(m,raw,c.written,true),movingRival=scoreMovingZeroSequence(m,raw,c.rival,true);
  if(controlWritten===null||controlRival===null||movingWritten===null||movingRival===null) return null;
  return {...c,controlWritten,controlRival,movingWritten,movingRival};
}
export const movingZeroReversal=(c:MovingZeroEvidence):boolean=>[c.controlWritten,c.controlRival,c.movingWritten,c.movingRival].every(Number.isFinite)&&
  c.controlWritten<c.controlRival&&c.movingWritten>c.movingRival;
function vector(v:unknown,n:number,positive=false):v is number[] {return Array.isArray(v)&&v.length===n&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));}
function validFrame(r:RootFrame):boolean {return !!r&&vector(r.poles,8)&&Array.isArray(r.zeros)&&r.zeros.length===3&&r.zeros.every(z=>vector(z,2))&&typeof r.valid==="boolean";}
export function validateMovingZeroModel(value:unknown,inventory:string,tokens:string):MovingZeroModel {
  const m=value as MovingZeroModel,n=tokens.length+1;
  const matrix=(x:number[][],rows:number,cols:number,positive=false)=>Array.isArray(x)&&x.length===rows&&x.every(r=>vector(r,cols,positive));
  if(!m||m.schema!==1||m.inventory!==inventory||m.tokens!==tokens||m.frameMs!==40||!(m.stay>0&&m.stay<1)||
    !Number.isSafeInteger(m.maxFrames)||m.maxFrames<2||m.maxFrames>2048||!vector(m.phoneCounts,tokens.length)||m.phoneCounts.some(c=>c<0)||
    !vector(m.center,8)||!vector(m.scale,8,true)||!vector(m.stationaryVariance,2,true)||
    !m.poles||!matrix(m.poles.mean,n,8)||!matrix(m.poles.variance,n,8,true)) throw new Error("Invalid moving spectral zero model");
  for(const [arm,components] of [[m.control,2],[m.moving,1]] as const) {
    if(!arm||!Array.isArray(arm.mean)||arm.mean.length!==n||!Array.isArray(arm.variance)||arm.variance.length!==n||
      arm.mean.some(x=>!matrix(x,components,4))||arm.variance.some(x=>!matrix(x,components,4,true))) throw new Error("Invalid pole-zero emissions");
  }
  if(!matrix(m.moving.alpha,n,4)||m.moving.alpha.flat().some(a=>Math.abs(a)>=1)||!matrix(m.moving.noise,n,4,true)) throw new Error("Invalid zero dynamics");
  return m;
}

/** Read-only discovery; certificates preserve primary tokens and issue ledgers. */
export function addMovingZeroEvidence(model:MovingZeroModel|undefined,audio:Float32Array,chunks:PhonemeEvidence[],index:ReferenceIndex,
  phrases?:PhraseAsset,options:CheckerOptions={},mode:"free"|"memorization"="free",
  inspect?:(comparison:MovingZeroEvidence|null,candidate:SequenceEvidenceCandidate)=>void):PhonemeEvidence[] {
  if(!model||!audio.length) return chunks;
  const cells=insertionCells(chunks);if(!cells.length) return chunks;
  if(cells[0].frameMs!==40) throw new Error("Pole-zero scorer clock differs from primary evidence");
  const candidates=new Map<string,SequenceEvidenceCandidate>();
  const tracker=createTracker(index,{mode,phrases,acquireMs:1000});
  const checker=createChecker(index,{...options,phrases,sequenceEvidence(c){candidates.set(JSON.stringify(c),c);return true;}});
  for(const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=movingZeroFeatures(audio),certificates:MovingZeroEvidence[]=[];
  for(const c of candidates.values()) {
    const comparison=c.fromFrame>=0&&c.toFrame<=features.length?movingZeroComparison(model,features.slice(c.fromFrame,c.toFrame),c):null;
    inspect?.(comparison,c);if(comparison&&movingZeroReversal(comparison)) certificates.push(comparison);
  }
  if(!certificates.length) return chunks;
  const first=chunks.findIndex(e=>e.phonemes.length);
  return chunks.map((e,i)=>i!==first?e:{...e,phonemes:e.phonemes.map((p,j)=>j?p:{...p,movingZeroVetoes:[...(p.movingZeroVetoes??[]),...certificates]})});
}
