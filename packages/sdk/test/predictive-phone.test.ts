import { describe, expect, it } from "vitest";
import { boundaryLogP, cholesky, integratedEmission, logadd, PosteriorRandom, predictiveComparison, predictiveFeatures, PredictiveObservations,
  predictiveReversal, predictiveScores, samplePosterior, validatePredictiveModel } from "../src/predictive-phone.js";
import type { PredictiveModel, PhonePosterior } from "../src/predictive-phone.js";

const diagonal=(v:number)=>Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>i===j?v:0));
const posterior=(mean:number,count=10000):PhonePosterior=>({mean:Array(8).fill(mean),kappa:count+1,nu:count+10,psi:diagonal(count+1),count});
const model=():PredictiveModel=>({schema:1,inventory:"test",tokens:"ab",frameMs:40,observationMs:10,dimensions:8,
  center:Array(8).fill(0),scale:Array(8).fill(1),phones:[posterior(0,0),posterior(0),posterior(1)],stay:.8,draws:128,maxFrames:2048});
const row=(v:number)=>Array(8).fill(v);

describe("Bayesian complete-pronunciation evidence",()=> {
  it("marginalizes every complete phone boundary with the same normalized duration prior",()=> {
    const e=[[.1,.8],[.2,.3],[.9,.4],[.1,.7]],stay=.8;
    let mass=-Infinity,prior=-Infinity;
    for (let boundary=1;boundary<e.length;boundary++) {
      let p=Math.log1p(-stay)+(e.length-2)*Math.log(stay),value=0;
      for (let t=0;t<e.length;t++) value+=e[t][t<boundary?0:1];
      prior=logadd(prior,p); mass=logadd(mass,p+value);
    }
    expect(boundaryLogP(e,stay)).toBeCloseTo(mass-prior,12);
    expect(boundaryLogP([[0,0]],stay)).toBe(-Infinity);
  });
  it("integrates parameter uncertainty once for an entire pronunciation and is candidate-order independent",()=> {
    const m=model(),x=[row(0),row(.1),row(.8),row(1)];
    const a=predictiveScores(m,x,["ab","ba","aa"])!,b=predictiveScores(m,x,["aa","ba","ab"])!;
    expect(a.predictive).toEqual(b.predictive.slice().reverse());
    expect(a.point).toEqual(b.point.slice().reverse());
    // Repeated labels retain two required articulations; they share one sampled phone distribution.
    expect(predictiveScores(m,[row(0)],["aa"])).toBeNull();
    expect(predictiveScores(m,x,["aa"])).not.toBeNull();
    expect(a.predictive[0]).toBeGreaterThan(a.predictive[1]);
  });
  it("reverses an overconfident sparse emission without changing the point-estimate control",()=> {
    const m=model();
    m.phones[1]={mean:row(0),kappa:10,nu:12,psi:diagonal(3),count:1};
    m.phones[2]={mean:[.1,0,0,0,0,0,0,0],kappa:10000,nu:10010,psi:diagonal(10001),count:10000};
    const c={word:{surah:1,ayah:1,word:1},written:"a",rival:"b",fromFrame:0,toFrame:1};
    const result=predictiveComparison(m,[Array.from({length:8},(_,d)=>d===0?8:0)],c)!;
    expect(result.pointRival).toBeGreaterThan(result.pointWritten);
    expect(predictiveReversal(result)).toBe(true);
    expect(predictiveReversal({...result,halfWritten:result.halfRival-1})).toBe(false);
    expect(predictiveReversal({...result,predictiveWritten:NaN})).toBe(false);
  });
  it("matches an independent multivariate Student-t calculation for one observed frame",()=> {
    const p:PhonePosterior={mean:row(0),kappa:10,nu:12,psi:diagonal(3),count:1};
    // scipy.stats.multivariate_t.logpdf: df=5, scale=.66*I, eight coordinates.
    expect(integratedEmission(p,[[8,0,0,0,0,0,0,0]])).toBeCloseTo(-23.575775721061724,10);
  });
  it("samples full covariances with the NIW mean rather than independent scalar variance guesses",()=> {
    const p=posterior(0,20); p.psi[0][1]=p.psi[1][0]=10.5;
    const rng=new PosteriorRandom(); let cross=0,variance=0;
    for (let n=0;n<2000;n++) {
      const e=samplePosterior(p,rng);
      variance+=e.mean[0]**2; cross+=e.mean[0]*e.mean[1];
    }
    // E[sigma]=psi/(nu-d-1), and Var(mu)=E[sigma]/kappa.
    expect(variance/2000).toBeCloseTo(.05,2);
    expect(cross/2000).toBeCloseTo(.025,2);
    expect(()=>cholesky([[1,2],[2,1]])).toThrow();
  });
  it("abstains on unknown phones, unsupported training classes and nonfinite observations",()=> {
    const m=model();
    expect(predictiveScores(m,[row(0)],["c"])).toBeNull();
    expect(predictiveScores(m,[row(NaN)],["a"])).toBeNull();
    m.phones[1].count=0;
    expect(predictiveScores(m,[row(0)],["a"])).toBeNull();
  });
  it("retains the native feature clock under different push sizes",()=> {
    const wave=Float32Array.from({length:3000},(_,i)=>.2*Math.sin(i*.1));
    const streamed=new PredictiveObservations();
    for (let i=0;i<wave.length;i+=137) streamed.push(wave.slice(i,i+137));
    expect(streamed.frames).toEqual(predictiveFeatures(wave));
  });
  it("validates inventory, covariance and posterior degrees of freedom",()=> {
    const m=model(); expect(validatePredictiveModel(m,"test","ab")).toBe(m);
    expect(()=>validatePredictiveModel(m,"wrong","ab")).toThrow();
    m.phones[1].nu=9;
    expect(()=>validatePredictiveModel(m,"test","ab")).toThrow();
  });
});
