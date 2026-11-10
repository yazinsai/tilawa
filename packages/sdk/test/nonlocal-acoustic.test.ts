import { describe, expect, it } from "vitest";
import { NonlocalAcousticDenoiser } from "../src/nonlocal-acoustic.js";

const signal = () => Float32Array.from({length:16000},(_,i) =>
  0.15*Math.sin(2*Math.PI*400*i/16000)+0.08*Math.sin(2*Math.PI*1200*i/16000));

describe("recording-local acoustic reconstruction", () => {
  it("requires three mutually sample-disjoint donors entirely before the disputed word", () => {
    const denoiser = new NonlocalAcousticDenoiser(signal());
    const from = 12000, to = 13200, ds = denoiser.retrieve(77,from,to);
    expect(ds).toHaveLength(3);
    for (const d of ds) {
      expect(d.toSample).toBeLessThanOrEqual(from);
      expect(d.toSample).toBeLessThanOrEqual((77-5)*160);
      expect(d.gain).toBeGreaterThan(0);
    }
    for (let i = 0; i < ds.length; i++) for (let j = i+1; j < ds.length; j++)
      expect(ds[i].toSample <= ds[j].fromSample || ds[j].toSample <= ds[i].fromSample).toBe(true);
  });

  it("abstains without enough earlier support and leaves PCM exactly unchanged", () => {
    const audio = signal(), d = new NonlocalAcousticDenoiser(audio);
    const result = d.repair(800,1200);
    expect(result.patches).toEqual([]);
    expect(result.samples).toEqual(audio);
    expect(new NonlocalAcousticDenoiser(new Float32Array(16000)).repair(12000,13200).patches).toEqual([]);
  });

  it("preserves length, phase of a stationary rendition, and all samples outside the disputed interval", () => {
    const audio = signal(), result = new NonlocalAcousticDenoiser(audio).repair(12000,13200);
    expect(result.patches.length).toBeGreaterThan(0);
    expect(result.samples.length).toBe(audio.length);
    expect(result.samples.subarray(0,12000)).toEqual(audio.subarray(0,12000));
    expect(result.samples.subarray(13200)).toEqual(audio.subarray(13200));
    expect(Math.max(...result.samples.map((x,i) => Math.abs(x-audio[i])))).toBeLessThan(1e-5);
  });

  it("does not retrieve a patch's center to explain itself or a later rendition", () => {
    const d = new NonlocalAcousticDenoiser(signal());
    expect(d.retrieve(15,2400,3200)).toEqual([]);
    const result = d.repair(12000,13200);
    for (const p of result.patches) for (const donor of p.donors)
      expect(donor.toSample).toBeLessThanOrEqual((p.center-5)*160);
  });

  it("rejects invalid geometries and repair intervals", () => {
    expect(() => new NonlocalAcousticDenoiser(signal(),{radius:2,contextLimit:1,consistencyLimit:1,donors:3})).toThrow();
    expect(() => new NonlocalAcousticDenoiser(signal()).repair(-1,100)).toThrow();
  });

  it("ranks donors without looking at the query's central observation", () => {
    const audio = signal(), changed = audio.slice();
    for (let i = 77*160; i < 77*160+400; i++) changed[i] += 0.2*Math.sin(i*2.1);
    const original = new NonlocalAcousticDenoiser(audio).retrieve(77,12000,13200);
    const obscured = new NonlocalAcousticDenoiser(changed).retrieve(77,12000,13200);
    expect(obscured).toEqual(original);
  });

  it("reduces independent central contamination using actual earlier observations", () => {
    const clean = signal(), noisy = clean.slice();
    for (let i = 77*160; i < 77*160+400; i++) noisy[i] += 0.2*Math.sin(i*2.1);
    const result = new NonlocalAcousticDenoiser(noisy).repair(12000,13200);
    expect(result.patches.length).toBeGreaterThan(0);
    let rawError = 0, repairedError = 0;
    for (let i = 77*160; i < 77*160+400; i++) {
      rawError += (noisy[i]-clean[i])**2;
      repairedError += (result.samples[i]-clean[i])**2;
    }
    expect(repairedError).toBeLessThan(rawError);
  });

  it("abstains when all center-blind matches have mutually inconsistent central spectra", () => {
    const d = new NonlocalAcousticDenoiser(signal());
    // Keep the flank descriptors fixed while independently corrupting the withheld central observations.
    for (let t = 0; t < 65; t++) d.power[t] = Float32Array.from(d.power[t],x => x*Math.exp(t/2));
    expect(d.retrieve(77,12000,13200)).toEqual([]);
  });
});
