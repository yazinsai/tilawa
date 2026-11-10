import { createChecker, createTracker, trackerInput, PosteriorFrames, minGapFrames } from "@tilawa/core";
import type { AcousticRadiusCandidate, AcousticRadiusEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex } from "@tilawa/core";
import { Fbank } from "./fbank.js";
import { topK } from "./listener.js";
import type { LoadedModel, Tensor } from "./model.js";
import { insertionCells } from "./seam-insertion.js";

export interface RadiusCalibration {
  schema: 1;
  inventory: string;
  primarySha256: string;
  rejectionEnergy: number;
  /** Search parameters selected without practice audio. */
  search: { dimensions: number; iterations: number; difference: number; maxEnergy: number };
}

export interface RadiusResult {
  energy: number | null;
  baselineMargin: number;
  boundaryMargin: number | null;
  evaluations: number;
  status: "crossed" | "no-crossing" | "baseline-mismatch" | "no-pause";
}

export const RADIUS_SEARCH = { dimensions: 4, iterations: 8, difference: 0.05, maxEnergy: 4 };

export function radiusKey(c: AcousticRadiusCandidate): string {
  return JSON.stringify([c.word, c.written, c.accused, c.fromFrame, c.toFrame, c.need]);
}

/** Observe existing final confirmations; the returned tracker events are never changed by the side scorer. */
export function radiusCandidates(chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset) {
  const candidates = new Map<string, AcousticRadiusCandidate>();
  const tracker = createTracker(index, { mode: "free", phrases, acquireMs: 1000 });
  const checker = createChecker(index, { phrases, acousticRadius(c) { candidates.set(radiusKey(c), c); return true; } });
  for (const e of insertionCells(chunks)) checker.push(e, tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const confirmed = checker.mistakes().filter(e => e.errorClass === "wrong_word");
  return [...candidates.values()].filter(c => confirmed.some(e =>
    e.location.surah === c.word.surah && e.location.ayah === c.word.ayah && e.location.word === c.word.word));
}

/** An orthonormal additive waveform basis built only from completed, token-free preceding pauses.
 * No speech samples are replaced. Each column has unit mean-square, making ||coefficients||² an energy ratio. */
export function pauseBasis(audio: Float32Array, chunks: PhonemeEvidence[], beforeFrame: number, n: number,
  dimensions: number): { columns: Float64Array[]; power: number } | null {
  const frameMs = chunks[0]?.frameMs;
  if (!frameMs || !n) return null;
  const samplesPerFrame = frameMs * 16;
  const phones = chunks.flatMap(e => e.phonemes).filter(p => p.end < beforeFrame);
  const gaps: [number, number][] = [];
  // A clip-start pause is eligible only when the first emission independently bounds its end.
  let previous = 0;
  for (const p of phones) {
    if (p.start - previous >= minGapFrames(frameMs)) gaps.push([previous, p.start]);
    previous = Math.max(previous, p.end + 1);
  }
  const patches: Float32Array[] = [];
  const width = Math.round(samplesPerFrame);
  for (const [a, b] of gaps.reverse()) {
    // Keep whole analysis frames away from both speech edges.
    for (let start = Math.round((a + 1) * samplesPerFrame); start + width <= (b - 1) * samplesPerFrame; start += width)
      patches.push(audio.subarray(start, start + width));
    if (patches.length >= dimensions * 2) break;
  }
  if (patches.length < dimensions) return null;
  const power = patches.reduce((s, p) => s + p.reduce((v, x) => v + x*x, 0), 0) / (patches.length * width);
  if (!(power > 0) || !Number.isFinite(power)) return null;
  // Reject a supposed pause louder than the preceding speech's median waveform power.
  const speech = phones.map(p => {
    const x = audio.subarray(Math.round(p.start*samplesPerFrame), Math.round((p.end+1)*samplesPerFrame));
    return x.length ? x.reduce((s,v) => s+v*v,0)/x.length : 0;
  }).sort((a,b) => a-b);
  if (!speech.length || power >= speech[Math.floor(speech.length/2)]) return null;
  const columns: Float64Array[] = [];
  for (const patch of patches) {
    const mean = patch.reduce((s,x) => s+x,0)/patch.length;
    const b = Float64Array.from({length:n}, (_,i) => patch[i%patch.length]-mean);
    for (const q of columns) {
      const projection = b.reduce((s,x,i) => s+x*q[i],0)/n;
      for (let i=0;i<n;i++) b[i] -= projection*q[i];
    }
    const norm = Math.sqrt(b.reduce((s,x) => s+x*x,0)/n);
    if (norm <= Math.sqrt(power)*1e-8) continue;
    for (let i=0;i<n;i++) b[i] /= norm;
    columns.push(b);
    if (columns.length === dimensions) break;
  }
  return columns.length === dimensions ? { columns, power } : null;
}

/** Deterministic finite differences seek a low-energy crossing, then bisect the witnessed direction.
 * This is an upper estimate of the nearest radius, never a proof of absence of a closer boundary. */
export async function searchRadius(evaluate: (coefficients: number[]) => Promise<number>, need: number,
  search: RadiusCalibration["search"]) {
  const d = search.dimensions, zero = Array(d).fill(0) as number[];
  let evaluations = 0;
  const value = async (x: number[]) => { evaluations++; return evaluate(x); };
  const baseline = await value(zero);
  if (!Number.isFinite(baseline) || baseline < need) return { energy:null, baseline, boundary:null, evaluations };
  let best: {x:number[]; margin:number; energy:number} | null = null;
  const norm2 = (x:number[]) => x.reduce((s,v) => s+v*v,0);
  const consider = async (x:number[], margin:number) => {
    if (!(Number.isFinite(margin) && margin < need) || norm2(x)>search.maxEnergy) return;
    let lo=0, hi=1, boundary=margin;
    for (let j=0;j<12;j++) {
      const mid=(lo+hi)/2, m=await value(x.map(v => v*mid));
      if (Number.isFinite(m) && m < need) { hi=mid; boundary=m; } else lo=mid;
    }
    const point=x.map(v => v*hi), energy=norm2(point);
    if (!best || energy < best.energy) best={x:point,margin:boundary,energy};
  };
  let x=zero, current=baseline;
  for (let iteration=0;iteration<search.iterations;iteration++) {
    const gradient:number[]=[];
    for (let axis=0;axis<d;axis++) {
      const plus=[...x], minus=[...x]; plus[axis]+=search.difference; minus[axis]-=search.difference;
      const p=await value(plus), m=await value(minus);
      gradient.push(Number.isFinite(p)&&Number.isFinite(m) ? (p-m)/(2*search.difference) : 0);
      await consider(plus,p); await consider(minus,m);
    }
    const g=Math.sqrt(norm2(gradient));
    if (!(g>0)) break;
    // Projected descent; energy shells grow from the observed floor, independent of target labels.
    const step=Math.sqrt(search.maxEnergy)*(iteration+1)/search.iterations;
    const next=x.map((v,i) => v-step*gradient[i]/g), n=Math.sqrt(norm2(next));
    if (n>Math.sqrt(search.maxEnergy)) for (let i=0;i<d;i++) next[i]*=Math.sqrt(search.maxEnergy)/n;
    const m=await value(next); await consider(next,m);
    if (m<current) { x=next; current=m; }
    else x=zero;
  }
  const winner = best as {x:number[];margin:number;energy:number} | null;
  return {energy:winner?.energy??null,baseline,boundary:winner?.margin??null,evaluations};
}

/** Resume the exact native-grid inference state before the first affected input window. */
export async function estimateRadius(model: LoadedModel, audio: Float32Array, chunks: PhonemeEvidence[],
  candidate: AcousticRadiusCandidate, search=RADIUS_SEARCH): Promise<RadiusResult> {
  const m=model.manifest;
  if (m.sampleRate!==16000 || m.frameMs!==chunks[0]?.frameMs || m.shift%4) throw new Error("Radius replay requires native 16 kHz / fourfold subsampling");
  const sampleFrom=Math.round(candidate.fromFrame*m.frameMs*16), sampleTo=Math.min(audio.length,Math.round(candidate.toFrame*m.frameMs*16));
  const basis=pauseBasis(audio,chunks,candidate.fromFrame,sampleTo-sampleFrom,search.dimensions);
  if (!basis) return {energy:null,baselineMargin:candidate.margin,boundaryMargin:null,evaluations:0,status:"no-pause"};
  const baseFeatures=new Fbank().push(audio);
  const pad=new Float32Array(m.fbank.mels).fill(m.fbank.padValue);
  const firstFeature=Math.max(0,Math.ceil((sampleFrom-399)/160));
  const startWindow=Math.max(0,Math.floor((firstFeature-m.window)/m.shift)+1);
  const startOutput=startWindow*m.shift/4;
  let states:Record<string,Tensor>=Object.fromEntries(m.states.map(s => [s.name,{type:s.dtype,dims:s.shape,
    data:s.dtype==="int64" ? new BigInt64Array(s.shape.reduce((a,b)=>a*b,1)) : new Float32Array(s.shape.reduce((a,b)=>a*b,1))}]));
  const run = async (features:Float32Array[], window:number, state:Record<string,Tensor>) => {
    const x=new Float32Array(m.window*m.fbank.mels);
    for (let i=0;i<m.window;i++) x.set(features[window*m.shift+i]??pad,i*m.fbank.mels);
    const out=await model.backend.run({x:{type:"float32",data:x,dims:[1,m.window,m.fbank.mels]},...state});
    return {lp:out.log_probs.data as Float32Array,states:Object.fromEntries(m.states.map(s => [s.name,out[`new_${s.name}`]]))};
  };
  for (let w=0;w<startWindow;w++) states=(await run(baseFeatures,w,states)).states;
  const saved:Record<string,Tensor>=Object.fromEntries(Object.entries(states).map(([name,s]) => [name,{...s,data:s.data.slice()}]));
  const ids=(s:string) => [...s].map(p => m.tokens.indexOf(p)+1);
  const evaluate=async (coefficients:number[]) => {
    const changed=Float32Array.from(audio);
    const scale=Math.sqrt(basis.power);
    for (let i=0;i<sampleTo-sampleFrom;i++) {
      let noise=0; for (let j=0;j<basis.columns.length;j++) noise+=coefficients[j]*basis.columns[j][i];
      changed[sampleFrom+i]=audio[sampleFrom+i]+scale*noise;
    }
    const features=baseFeatures.slice();
    if (!coefficients.every(v=>v===0)) {
      // Recompute only analysis frames whose actual sample windows intersect the added noise.
      const lastFeature=Math.min(baseFeatures.length,Math.ceil(sampleTo/160));
      const local=new Fbank().push(changed.subarray(firstFeature*160,Math.min(audio.length,(lastFeature-1)*160+400)));
      for (let t=0;t<local.length;t++) features[firstFeature+t]=local[t];
    }
    let state:Record<string,Tensor>=Object.fromEntries(Object.entries(saved).map(([name,s]) => [name,{...s,data:s.data.slice()}]));
    const frames=new PosteriorFrames(m.tokens.length), k=chunks[0].posteriors!.k;
    let output=startOutput;
    for (let w=startWindow;output<candidate.toFrame;w++) {
      const out=await run(features,w,state); state=out.states;
      const rows=out.lp.length/(m.tokens.length+1), rowIds:number[]=[], rowProbs:number[]=[];
      for (let t=0;t<rows;t++) for (const [id,p] of topK(out.lp.subarray(t*(m.tokens.length+1),(t+1)*(m.tokens.length+1)),k)) {
        rowIds.push(id); rowProbs.push(Math.round(p*1e4)/1e4);
      }
      frames.append({k,startFrame:output,nFrames:rows,ids:rowIds,probs:rowProbs}); output+=rows;
      if (!rows) throw new Error("Radius replay produced no frames");
    }
    return frames.nll(ids(candidate.written),candidate.fromFrame,candidate.toFrame)-
      frames.nll(ids(candidate.accused),candidate.fromFrame,candidate.toFrame);
  };
  const baseline=await evaluate(Array(search.dimensions).fill(0));
  if (!Number.isFinite(baseline) || Math.abs(baseline-candidate.margin)>1e-6) return {
    energy:null,baselineMargin:baseline,boundaryMargin:null,evaluations:1,status:"baseline-mismatch"};
  const result=await searchRadius(evaluate,candidate.need,search);
  return {energy:result.energy,baselineMargin:result.baseline,boundaryMargin:result.boundary,
    evaluations:result.evaluations+1,status:result.energy===null ? "no-crossing" : "crossed"};
}

export async function addAcousticRadiusEvidence(model:LoadedModel,audio:Float32Array,chunks:PhonemeEvidence[],
  index:ReferenceIndex,phrases?:PhraseAsset):Promise<PhonemeEvidence[]> {
  const calibration=model.acousticRadius;
  if (!calibration || !audio.length) return chunks;
  if (calibration.inventory!==mInventory(model) || calibration.primarySha256!==model.manifest.sha256)
    throw new Error("Radius calibration does not match the primary listener");
  const metadata:AcousticRadiusEvidence[]=[];
  for (const c of radiusCandidates(chunks,index,phrases)) {
    const r=await estimateRadius(model,audio,chunks,c,calibration.search);
    if (r.energy!==null && r.energy<calibration.rejectionEnergy && r.boundaryMargin!==null)
      metadata.push({...c,baselineMargin:r.baselineMargin,boundaryMargin:r.boundaryMargin,energy:r.energy,
        rejectionEnergy:calibration.rejectionEnergy});
  }
  if (!metadata.length) return chunks;
  const first=chunks.findIndex(c=>c.phonemes.length);
  return chunks.map((c,i)=>i!==first?c:{...c,phonemes:c.phonemes.map((p,j)=>j?p:{...p,acousticRadii:metadata})});
}
const mInventory=(model:LoadedModel)=>model.manifest.inventory;
