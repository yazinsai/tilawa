import { describe, expect, it } from "vitest";
import { boundaryLogP, supportComparison, supportFeatures, SupportObservations, supportReversal, supportScores,
  validateTrainingSupportModel } from "../src/training-support.js";
import type { EmissionFit, TrainingSupportModel } from "../src/training-support.js";

const fit = (means: number[]): EmissionFit => ({ center: Array(8).fill(0), scale: Array(8).fill(1),
  phones: [0, ...means].map(mean => ({ mean: Array(8).fill(mean), variance: Array(8).fill(1), count: 10 })) });
const model = (): TrainingSupportModel => ({ schema: 1, inventory: "toy", tokens: "بت", frameMs: 40, observationMs: 10,
  dimensions: 8, stay: .8, maxFrames: 2048, cutoff: 0, full: fit([4, 1]), deletions: [fit([0, 1]), fit([4, 2])] });
const candidate = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", rival: "ت", fromFrame: 0, toFrame: 1 };
const x = [Array(8).fill(0), Array(8).fill(0), Array(8).fill(0), Array(8).fill(0)];

describe("complete recording-jackknife pronunciation evidence", () => {
  it("matches exhaustive complete boundary summation with a normalized timing prior", () => {
    const e = [[-.2,-2],[-.7,-.3],[-2,-.4]], stay = .8;
    const masses = [Math.exp(e[0][0]+e[1][0]+e[2][1])*stay*(1-stay),
      Math.exp(e[0][0]+e[1][1]+e[2][1])*(1-stay)*stay];
    expect(boundaryLogP(e, stay)).toBeCloseTo(Math.log((masses[0]+masses[1])/(2*stay*(1-stay))), 12);
    expect(boundaryLogP([[0,0,0]], stay)).toBe(-Infinity);
    expect(boundaryLogP([[0,0],[0,0],[0,0]], stay)).toBeCloseTo(0, 12);
  });
  it("finds a deletion reversal and measures influence over the complete sequence", () => {
    const c = supportComparison(model(), x, candidate)!;
    expect(c.recording).toBe(0);
    expect(c.fullWritten).toBeLessThan(c.fullRival);
    expect(c.removedWritten).toBeGreaterThan(c.removedRival);
    expect(c.influence).toBeCloseTo((c.removedWritten-c.removedRival-c.fullWritten+c.fullRival)/x.length, 12);
    expect(supportReversal(c)).toBe(true);
    expect(supportReversal({ ...c, cutoff: c.influence })).toBe(false);
    expect(supportReversal({ ...c, removedWritten: c.removedRival })).toBe(false);
  });
  it("does not ignore a larger influential recording that supports the written word", () => {
    const m=model();
    m.deletions.push(fit([20,1]));
    const c=supportComparison(m,x,candidate)!;
    expect(c.recording).toBe(2);
    expect(c.influence).toBeLessThan(0);
    expect(supportReversal(c)).toBe(false);
  });
  it("is invariant to candidate order and retains repeated articulation requirements", () => {
    const m = model();
    const a = supportScores(m,m.full,x,["بت","تب"]), b = supportScores(m,m.full,x,["تب","بت"]);
    expect(a).toEqual(b?.slice().reverse());
    expect(supportScores(m,m.full,[x[0]],["بب"])).toBeNull();
    expect(supportScores(m,m.full,x,["بب"])?.[0]).toBeCloseTo(supportScores(m,m.full,x,["ب"])![0], 12);
  });
  it("abstains on unsupported phones, missing frames and non-finite observations", () => {
    const m = model();
    expect(supportComparison(m,[],candidate)).toBeNull();
    expect(supportComparison(m,[Array(8).fill(NaN)],candidate)).toBeNull();
    expect(supportComparison(m,x,{...candidate,rival:"ن"})).toBeNull();
    m.full.phones[1].count = 0;
    expect(supportComparison(m,x,candidate)).toBeNull();
  });
  it("uses only fixed frozen observations and preserves frontend rechunking", () => {
    const audio = Float32Array.from({length:4000},(_,i)=>Math.sin(i*.1));
    const o = new SupportObservations(); o.push(audio.slice(0,751)); o.push(audio.slice(751));
    expect(o.frames).toEqual(supportFeatures(audio));
  });
  it("rejects mismatched clocks, missing fits and corrupt emissions", () => {
    const m=model();
    expect(validateTrainingSupportModel(m,"toy","بت")).toBe(m);
    expect(()=>validateTrainingSupportModel(m,"other","بت")).toThrow();
    expect(()=>validateTrainingSupportModel({...m,deletions:[]},"toy","بت")).toThrow();
    m.deletions[0].phones[1].variance[0]=0;
    expect(()=>validateTrainingSupportModel(m,"toy","بت")).toThrow();
  });
});
