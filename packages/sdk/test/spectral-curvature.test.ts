import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { CurvatureObservations, curvatureFeatures, curvatureView, curvatureChain, curvatureComparison, curvatureReversal, scoreCurvatureSequence, validateCurvatureModel, type CurvatureModel } from "../src/spectral-curvature.js";

const model = (): CurvatureModel => {
  const arm = () => ({ center: Array(14).fill(0), scale: Array(14).fill(1),
    emissions: [0, 1, 2].map(p => ({ mean: Array(14).fill(p), variance: Array(14).fill(1) })) });
  return { schema: 1, inventory: "test", tokens: "ab", frameMs: 40, resolution: 16,
    phoneCounts: [3, 3], stay: .8, maxFrames: 2048, control: arm(), curvature: arm() };
};

// Independently enumerate legal paths for a small complete pronunciation.
function enumerate(rows: number[][], labels: number[], stay: number): number | null {
  const states = [0]; for (const p of labels) states.push(p, 0);
  const skip = (s: number) => s >= 2 && states[s] !== 0 && states[s] !== states[s-2];
  let mass = 0, timing = 0;
  const walk = (t: number, s: number, probability: number, density: number) => {
    const observed = density * Math.exp(rows[t][states[s]]);
    if (t === rows.length-1) {
      if (s >= states.length-2) { mass += probability*observed; timing += probability; }
      return;
    }
    const destinations = [s, ...(s+1 < states.length ? [s+1] : []), ...(skip(s+2) && s+2 < states.length ? [s+2] : [])];
    for (const d of destinations) walk(t+1, d, probability * (destinations.length === 1 ? 1 : d === s ? stay : (1-stay)/(destinations.length-1)), observed);
  };
  walk(0, 0, .5, 1); walk(0, 1, .5, 1);
  return timing ? Math.log(mass/timing) : null;
}

describe("spectral curvature pronunciation evidence", () => {
  it("removes additive gain and linear frequency tilt while retaining a notch", () => {
    const envelope = Array.from({length:16}, (_, j) => Math.sin(j*.7));
    const colored = envelope.map((x,j) => x + 2.4 - .13*j);
    const original = curvatureView(envelope, true), transformed = curvatureView(colored, true);
    for (let j=0; j<original.length; j++) expect(transformed[j]).toBeCloseTo(original[j], 12);
    expect(curvatureView(envelope, false)).not.toEqual(curvatureView(colored, false));
    const notch = Array(16).fill(0); notch[8] = -1;
    expect(curvatureView(notch, true).slice(6,9)).toEqual([-1,2,-1]);
  });

  it("has finite chunk-independent observations without altering primary fbank", () => {
    const audio = Float32Array.from({length:16000}, (_, i) => .2*Math.sin(i*.2)+.03*Math.sin(i*.67));
    const all = curvatureFeatures(audio,16), streamed = new CurvatureObservations(16);
    for (let i=0; i<audio.length; i+=137) streamed.push(audio.subarray(i,i+137));
    expect(streamed.frames).toEqual(all);
    expect(all.length).toBe(Math.floor(new Fbank().push(audio).length/4));
    expect(all.flat().every(Number.isFinite)).toBe(true);
    const plain = new Fbank().push(audio);
    const observed = new Fbank({onSpectrum: p => expect(p.length).toBe(256)}).push(audio);
    expect(observed).toEqual(plain);
  });

  it("marginalizes every complete path with shared timing, including repeated phones", () => {
    const rows = Array.from({length:6}, (_,t) => [0,1,2].map(p => -.2*(t-p)**2));
    for (const labels of [[1], [1,2], [1,1], [2,1,2]]) {
      expect(curvatureChain(rows,labels,.8)).toBeCloseTo(enumerate(rows,labels,.8)!, 12);
      expect(curvatureChain(rows.map(r => r.map(() => 0)),labels,.8)).toBeCloseTo(0, 12);
    }
    expect(curvatureChain(rows.slice(0,2),[1,1],.8)).toBeNull();
    expect(curvatureChain(rows.slice(0,1),[1,2],.8)).toBeNull();
  });

  it("abstains on missing phones, partial pronunciations, invalid frames and unknown inventory", () => {
    const m = model(), rows = Array.from({length:6}, () => Array(16).fill(0));
    expect(scoreCurvatureSequence(m,rows,"ab",true)).not.toBeNull();
    expect(scoreCurvatureSequence(m,rows,"z",true)).toBeNull();
    expect(scoreCurvatureSequence(m,rows.slice(0,1),"ab",true)).toBeNull();
    m.phoneCounts[1] = 0;
    expect(scoreCurvatureSequence(m,rows,"ab",false)).toBeNull();
    rows[0][0] = NaN;
    expect(scoreCurvatureSequence(m,rows,"a",true)).toBeNull();
    expect(() => validateCurvatureModel(model(),"other","ab")).toThrow();
    const invalid = model(); invalid.curvature.emissions[0].variance[0] = 0;
    expect(() => validateCurvatureModel(invalid,"test","ab")).toThrow();
  });

  it("requires a matched control reversal rather than written preference alone", () => {
    const m = model(), rows = Array.from({length:4}, () => Array(16).fill(0));
    m.control.emissions[2].mean.fill(0);
    m.curvature.emissions[1].mean.fill(0);
    const candidate = { word: {surah:1, ayah:1, word:1}, written:"a", rival:"b", fromFrame:4, toFrame:8 };
    const certificate = curvatureComparison(m,rows,candidate)!;
    expect(curvatureReversal(certificate)).toBe(true);
    expect(JSON.parse(JSON.stringify(certificate))).toEqual(certificate);
    m.control.emissions[1].mean.fill(0);
    expect(curvatureReversal(curvatureComparison(m,rows,candidate)!)).toBe(false);
  });
});
