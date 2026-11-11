import { describe, expect, it } from "vitest";
import { backgroundSamples, createAcceptanceVolume, fitPhoneBackground, logAcceptanceVolume, minimumAcousticCost } from "../src/acceptance-volume.js";
import { deletionCost, insertionCost, subCost, UNIT } from "../src/fuzzy.js";
import { anchorFitCost } from "../src/tracker.js";

describe("reference acceptance volume", () => {
  it("uses the existing directed acoustic prices, including duration charges", () => {
    expect(minimumAcousticCost("ننننب", "ننب")).toBe((2*deletionCost("ننننب",0))/UNIT);
    expect(minimumAcousticCost("ق", "ك")).toBe(subCost("ق","ك")/UNIT);
    expect(minimumAcousticCost("", "ُ")).toBe(insertionCost("",0,"ُ")/UNIT);
    expect(minimumAcousticCost("ب", "ب")).toBe(0);
  });
  it("retains the existing fixed-entry and open-suffix semantics", () => {
    for (const ref of ["بت", "ننننب", "قَت", "تَاا"]) for (const heard of ["ب", "نن", "كب", "تَاب", "ُ"]) {
      expect(minimumAcousticCost(ref,heard)).toBe(anchorFitCost(ref,heard)! / UNIT);
    }
    expect(minimumAcousticCost("بت", "ب")).toBe(0);
  });
  it("counts strings once even when many minimum alignments tie", () => {
    const samples = ["ب","بب","ت"];
    const want = Math.log(samples.reduce((s,x)=>s+Math.exp(-minimumAcousticCost("بب",x)),0)/samples.length);
    expect(logAcceptanceVolume("بب",samples)).toBeCloseTo(want,14);
  });
  it("penalizes a generically easy reference when acoustic costs tie", () => {
    const samples = ["ب","ب","ب","ت"];
    const broad = logAcceptanceVolume("ب",samples);
    const rare = logAcceptanceVolume("ت",samples);
    expect(broad).toBeGreaterThan(rare);
    expect(1+broad).toBeGreaterThan(1+rare);
  });
  it("shares deterministic samples independently of the requested reference and length", () => {
    const model = fitPhoneBackground("بت",[{speaker:"training",split:"train",kind:"none",phones:"ببتت"}]);
    const long = backgroundSamples(model,8,16);
    expect(backgroundSamples(model,3,16)).toEqual(long.map(s=>s.slice(0,3)));
    const a = createAcceptanceVolume(model,16), b = createAcceptanceVolume(model,16);
    a("بت",5); b("تب",5);
    expect(a("تب",5)).toBe(b("تب",5));
  });
  it("excludes scripted edits and held-out speakers from fitting", () => {
    const train = {speaker:"training",split:"train",kind:"none",phones:"ببت"};
    const fitted = fitPhoneBackground("بت",[train]);
    expect(fitPhoneBackground("بت",[train,{speaker:"evaluation",split:"eval",kind:"none",phones:"تتت"},
      {speaker:"training",split:"train",kind:"omission",phones:"تتت"}])).toEqual(fitted);
    expect(()=>fitPhoneBackground("بت",[train,{...train,split:"eval"}])).toThrow("disjoint");
  });
});
