/** Small weak-label segment model. All boundaries are marginalized; no visits
 * or forced word cuts are inputs to this acoustic/text scorer. */
export interface SegmentPresenceModel {
  schema: 1;
  tokens: string;
  inventory: string;
  featureNode: string;
  encoderSha256: string;
  mean: number[];
  std: number[];
  hidden: number;
  weights: {
    "acoustic.weight": number[][];
    "acoustic.bias": number[];
    "text.weight": number[][];
    "text.bias": number[];
    duration: number[];
    skip_logit: number;
  };
}

export function segmentTextFeatures(words: string[], tokens: string): number[][] {
  const ids = new Map([...tokens].map((p,i) => [p,i])), k = tokens.length;
  return words.map(word => {
    const phones = [...word].filter(p => ids.has(p));
    if (!phones.length) throw new Error("Empty word pronunciation");
    const q = Array(3*k+33).fill(0);
    for (const p of phones) q[ids.get(p)!] += 1/phones.length;
    q[k+ids.get(phones[0])!] = 1;
    q[2*k+ids.get(phones.at(-1)!)!] = 1;
    for (let i=1;i<phones.length;i++) q[3*k+(ids.get(phones[i-1])!*k+ids.get(phones[i])!)%32] += 1/Math.max(1,phones.length-1);
    q[q.length-1] = Math.log(phones.length);
    return q;
  });
}

const logadd = (a: number,b: number) => a === -Infinity ? b : b === -Infinity ? a : Math.max(a,b)+Math.log1p(Math.exp(-Math.abs(a-b)));

/** Segment score = sum(frame compatibility) + log exit + (duration-1) log stay. */
export function segmentPartition(emissions: ArrayLike<number>[], stay: number[], exit: number[],
  skipLogit: number, omitted?: number): number {
  const n = emissions.length, t = emissions[0]?.length ?? 0;
  if (!n || !t) return -Infinity;
  let alpha = new Float64Array(t+1).fill(-Infinity); alpha[0]=0;
  for (let w=0;w<n;w++) {
    if (w===omitted) { alpha = alpha.map(v => v+skipLogit-Math.log(n)); continue; }
    const next = new Float64Array(t+1).fill(-Infinity);
    let prefix = 0, sum = -Infinity;
    for (let b=1;b<=t;b++) {
      const a=b-1;
      sum=logadd(sum,alpha[a]-prefix-a*stay[w]);
      prefix+=emissions[w][a];
      next[b]=prefix+(b-1)*stay[w]+exit[w]+sum;
    }
    alpha=next;
  }
  return alpha[t];
}

function project(x: ArrayLike<number>, weight: number[][], bias: number[]): number[] {
  return weight.map((row,i) => Math.tanh(row.reduce((s,v,j) => s+v*x[j],bias[i])));
}

export function segmentReadMargin(model: SegmentPresenceModel, features: ArrayLike<number>[],
  words: string[], omitted: number): { read: number; omit: number; margin: number } | null {
  if (!Number.isInteger(omitted) || omitted<0 || omitted>=words.length || !features.length) return null;
  const d=model.mean.length, w=model.weights;
  if (features.some(f => f.length!==d || Array.from(f).some(v => !Number.isFinite(v)))) throw new Error("Invalid frozen encoder features");
  const acoustic=features.map(f => project(Array.from(f,(v,i) => (v-model.mean[i])/model.std[i]),w["acoustic.weight"],w["acoustic.bias"]));
  const text=segmentTextFeatures(words,model.tokens).map(q => project(q,w["text.weight"],w["text.bias"]));
  const emissions=text.map(q => acoustic.map(x => q.reduce((s,v,i) => s+v*x[i],0)/Math.sqrt(model.hidden)));
  const rate=features.length/words.reduce((s,v) => s+v.length,0);
  const stay=words.map(v => {
    const mean=1+v.length*rate*Math.exp(w.duration[0]);
    const exit=Math.max(1e-5,Math.min(1-1e-5,1/mean));
    return Math.log1p(-exit)*Math.exp(w.duration[1]);
  });
  const exit=stay.map(s => Math.log(-Math.expm1(s)));
  const read=segmentPartition(emissions,stay,exit,w.skip_logit);
  const omit=segmentPartition(emissions,stay,exit,w.skip_logit,omitted);
  return Number.isFinite(read) && Number.isFinite(omit) ? { read,omit,margin:read-omit } : null;
}
