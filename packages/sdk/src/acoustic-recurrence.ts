import { createChecker, createTracker, trackerInput } from "@tilawa/core";
import type { AcousticRecurrenceEvidence, PhonemeEvidence, PhraseAsset, ReferenceIndex, WrongWordAcousticCandidate } from "@tilawa/core";
import { insertionCells } from "./seam-insertion.js";
import { Fbank } from "./fbank.js";

/** Fitted only on correctly transcribed training speakers. No words or passage identities. */
export interface RecurrenceModel {
  schema: 1;
  inventory: string;
  tokens: string;
  frameMs: 40;
  dimensions: number;
  center: number[];
  scale: number[];
  means: number[][];
  variances: number[][];
  pairMeans: number[][];
  pairVariances: number[][];
  nullMean: number;
  nullVariance: number;
  weight: number;
  beam: number;
  maxFrames: number;
}

/** Same Kaldi features as the listener; pool four native feature frames on its 40 ms clock. */
export function recurrenceFeatures(audio: Float32Array, dimensions = 16): number[][] {
  const mel = new Fbank().push(audio), out: number[][] = [];
  for (let t = 0; t + 3 < mel.length; t += 4) {
    const row = Array(dimensions).fill(0) as number[];
    for (let d = 0; d < dimensions; d++) {
      const a = Math.floor(d * 80 / dimensions), b = Math.floor((d + 1) * 80 / dimensions);
      for (let f = t; f < t + 4; f++) for (let m = a; m < b; m++) row[d] += mel[f][m] / (4 * (b-a));
    }
    out.push(row);
  }
  return out;
}
const gaussian = (x: number, mean: number, variance: number) => -.5 * (Math.log(2*Math.PI*variance) + (x-mean)**2/variance);
const logsum = (xs: number[]) => {
  const max = Math.max(...xs);
  return Number.isFinite(max) ? max + Math.log(xs.reduce((s,x) => s+Math.exp(x-max),0)) : -Infinity;
};
export interface RecurrenceScore { local: number; relational: number; positive: number; paths: number }

/** Bounded monotonic segmentation pool, shared by the local and relational calculations.
 * Every pronunciation phone is consumed, including repeated articulations; no phone can be erased.
 * Relationships connect frames separated by at least one intervening frame, within this occurrence only.
 */
export function scoreRecurrence(model: RecurrenceModel, raw: number[][], pronunciation: string,
  relationshipRange: [number, number] = [0, raw.length]): RecurrenceScore | null {
  const ids = [...pronunciation].map(p => model.tokens.indexOf(p)+1);
  if (!ids.length || ids.some(id => id < 1) || raw.length < ids.length || raw.length > model.maxFrames || raw.some(row => row.length !== model.dimensions || row.some(v => !Number.isFinite(v)))) return null;
  const [begin, end] = relationshipRange;
  if (!Number.isInteger(begin) || !Number.isInteger(end) || begin < 0 || end > raw.length || end <= begin) return null;
  const x = raw.map(row => row.map((v,d) => (v-model.center[d])/model.scale[d]));
  const similarity = x.map((a,t) => x.slice(0,t).map(b => {
    let dot=0, aa=0, bb=0;
    for (let d=0;d<model.dimensions;d++) { dot+=a[d]*b[d]; aa+=a[d]**2; bb+=b[d]**2; }
    return dot / Math.sqrt(Math.max(aa*bb,1e-12));
  }));
  const emissions = x.map(row => ids.map(id => row.reduce((s,v,d) => s+gaussian(v,model.means[id][d],model.variances[id][d]),0)));
  type Path = { state: number; local: number; labels: number[] };
  let paths: Path[] = [{state:0,local:emissions[0][0],labels:[ids[0]]}];
  for (let t=1;t<x.length;t++) {
    const next: Path[] = [];
    for (const path of paths) for (const advance of [0,1]) {
      const state=path.state+advance;
      if (state>=ids.length || ids.length-1-state>x.length-1-t) continue;
      next.push({state,local:path.local-Math.log(2)+emissions[t][state],labels:[...path.labels,ids[state]]});
    }
    // Preserve competing phone placements at every frontier, rather than letting one duration dominate the pool.
    const groups = new Map<number,Path[]>();
    for (const p of next) { const group=groups.get(p.state)??[]; group.push(p); groups.set(p.state,group); }
    paths=[...groups.values()].flatMap(g=>g.sort((a,b)=>b.local-a.local).slice(0,model.beam));
  }
  paths=paths.filter(p=>p.state===ids.length-1);
  if (!paths.length) return null;
  const relations=paths.map(path=> {
    let score=0;
    for (let t=begin+2;t<end;t++) {
      let sum=0;
      for (let s=begin;s<t-1;s++) {
        const a=path.labels[s],b=path.labels[t],sim=similarity[t][s];
        sum+=gaussian(sim,model.pairMeans[a][b],model.pairVariances[a][b])-gaussian(sim,model.nullMean,model.nullVariance);
      }
      // Degree normalization prevents long held phones from quadratically accumulating the same evidence.
      score+=sum/(t-begin-1);
    }
    return score;
  });
  const local=logsum(paths.map(p=>p.local));
  const relational=logsum(paths.map((p,i)=>p.local+model.weight*relations[i]));
  // Positive relationship evidence against the phone-independent similarity null, marginalized under the local pool.
  return {local,relational,positive:relational-local,paths:paths.length};
}

export function recurrenceVeto(written: RecurrenceScore | null, accused: RecurrenceScore | null): boolean {
  return !!written && !!accused && written.local < accused.local && written.relational > accused.relational && written.positive > 0;
}

export function validateRecurrenceModel(value: unknown, inventory: string, tokens: string): RecurrenceModel {
  const m=value as RecurrenceModel, n=tokens.length+1;
  const vector=(v: number[],size: number,positive=false)=>Array.isArray(v)&&v.length===size&&v.every(x=>Number.isFinite(x)&&(!positive||x>0));
  const matrix=(v: number[][],size: number,positive=false)=>Array.isArray(v)&&v.length===n&&v.every(row=>vector(row,size,positive));
  if (!m || m.schema!==1 || m.inventory!==inventory || m.tokens!==tokens || m.frameMs!==40 ||
    !Number.isInteger(m.dimensions) || m.dimensions<1 || m.dimensions>80 || !vector(m.center,m.dimensions)||!vector(m.scale,m.dimensions,true)||
    !matrix(m.means,m.dimensions)||!matrix(m.variances,m.dimensions,true)||!matrix(m.pairMeans,n)||!matrix(m.pairVariances,n,true)||
    !Number.isFinite(m.nullMean)||!(m.nullVariance>0)||!Number.isFinite(m.nullVariance)||!(m.weight>0)||!Number.isFinite(m.weight)||
    !Number.isInteger(m.beam)||m.beam<1||m.beam>64||!Number.isInteger(m.maxFrames)||m.maxFrames<1||m.maxFrames>256)
    throw new Error("Invalid acoustic recurrence model or inventory");
  return m;
}

/** Discovery observes the original checker windows. It never changes candidates, pronunciations or boundaries. */
export async function addRecurrenceEvidence(model: RecurrenceModel | undefined, audio: Float32Array,
  chunks: PhonemeEvidence[], index: ReferenceIndex, phrases?: PhraseAsset): Promise<PhonemeEvidence[]> {
  if (!model || !audio.length) return chunks;
  const cells=insertionCells(chunks);
  if (!cells.length || cells[0].frameMs!==model.frameMs) return chunks;
  const candidates=new Map<string,WrongWordAcousticCandidate>();
  const tracker=createTracker(index,{mode:"free",phrases});
  const checker=createChecker(index,{phrases,wrongWordAcoustic(c) { candidates.set(JSON.stringify(c),c); return true; }});
  for (const e of cells) checker.push(e,tracker.push(trackerInput(e)));
  checker.end(tracker.end());
  const features=recurrenceFeatures(audio,model.dimensions), metadata: AcousticRecurrenceEvidence[]=[];
  for (const c of candidates.values()) {
    if (c.fromFrame<0 || c.toFrame>features.length) continue;
    const window=features.slice(c.fromFrame,c.toFrame);
    const interval: [number, number] = [c.relationshipFromFrame-c.fromFrame,c.relationshipToFrame-c.fromFrame];
    const written=scoreRecurrence(model,window,c.written,interval), accused=scoreRecurrence(model,window,c.accused,interval);
    if (recurrenceVeto(written,accused)) metadata.push({...c,localWritten:written!.local,localAccused:accused!.local,
      relationalWritten:written!.relational,relationalAccused:accused!.relational,positiveWritten:written!.positive});
  }
  if (!metadata.length) return chunks;
  const first=chunks.findIndex(c=>c.phonemes.length);
  return chunks.map((c,i)=>i!==first?c:{...c,phonemes:c.phonemes.map((p,j)=>j?p:{...p,acousticRecurrence:metadata})});
}
