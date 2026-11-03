import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { BLANK_CUT, NATS_PER_EDIT, PHRASE_SWITCH, SOFT, SUPPORT, TODAY, UNSURE, anchorFrom, calibrationNoise, derive, pairCost, quantileCut } from "../src/calibration.js";
import { subCost } from "../src/fuzzy.js";
import type { CalibrationStats } from "../src/calibration.js";

const decoded = (name: string): CalibrationStats =>
  JSON.parse(readFileSync(new URL(`./fixtures/listener-calibration/${name}.json`, import.meta.url), "utf8"));

/** A sample whose rates sit under today's floors, so the estimator's floor is what reproduces the constants. */
function shippedSample(): CalibrationStats {
  return {
    wordConf: [...Array(200).fill(0.5), ...Array(800).fill(0.8)],
    phoneNll: Array(100).fill(0.4),
    sub: { "ه>ح": 80, "ح>ه": 80, "َ>ُ": 200 },
    refCount: { ه: 400, ح: 400, "َ": 500, "ُ": 500, "ِ": 500, ا: 500 },
    phraseMargins: [...Array(99).fill(-8), 1],
    blankCosts: [2, 9.5],
  };
}

describe("listener calibration", () => {
  it("reproduces today's constants on the shipped student's gold-excluded decodes", () => {
    const stats = decoded("shipped");
    expect(calibrationNoise(derive(stats, anchorFrom(stats)))).toEqual([]);
  });

  it("wires the wide student's derived constants", () => {
    const got = derive(decoded("wide"), anchorFrom(decoded("shipped")));
    expect(got.unsure).toBeCloseTo(UNSURE, 10);
    expect(got.support).toBeCloseTo(SUPPORT, 10);
    expect(got.natsPerEdit).toBeCloseTo(NATS_PER_EDIT, 10);
    expect(got.phraseSwitch).toBeCloseTo(PHRASE_SWITCH, 10);
    expect(got.soft).toBe(SOFT);
    expect(got.blankCut).toBeCloseTo(BLANK_CUT, 10);
    expect(got.costs["ثس"]).toBe(pairCost("ث", "س"));
    expect(pairCost("ث", "س")).toBe(6);
  });

  it("reproduces today's constants on a shipped-shaped sample", () => {
    const stats = shippedSample();
    const got = derive(stats, anchorFrom(stats));
    expect(calibrationNoise(got)).toEqual([]);
    expect(got.unsure).toBeCloseTo(TODAY.unsure, 5);
    expect(got.support).toBeCloseTo(TODAY.support, 5);
    expect(got.phraseSwitch).toBeCloseTo(TODAY.phraseSwitch, 5);
    expect(got.blankCut).toBeCloseTo(20, 5);
  });

  it("moves the cut and the margin when the other student is sharper, and the switch only once its tail crosses today's penalty", () => {
    const anchor = anchorFrom(shippedSample());
    const wide: CalibrationStats = {
      ...shippedSample(),
      wordConf: [...Array(200).fill(0.7), ...Array(800).fill(0.95)],
      phoneNll: Array(100).fill(0.5),
      phraseMargins: [...Array(99).fill(-4), 3],
      blankCosts: [4, 14],
    };
    const got = derive(wide, anchor);
    expect(got.unsure).toBeCloseTo(0.95, 5);
    expect(got.support).toBeCloseTo(12.5, 5);
    expect(got.phraseSwitch).toBeCloseTo(TODAY.phraseSwitch, 5);
    expect(got.blankCut).toBeCloseTo(24.5, 5);
    expect(quantileCut(wide.wordConf, anchor.passRate)).toBeCloseTo(0.95, 5);
    const crossed: CalibrationStats = { ...wide, phraseMargins: Array(100).fill(5.5) };
    expect(derive(crossed, anchor).phraseSwitch).toBeCloseTo(5.5, 5);
  });

  it("floors a more-common confusion and raises one the new student confuses less", () => {
    const shipped = shippedSample();
    shipped.sub = { ...shipped.sub, "ص>س": 400, "س>ص": 400 };
    shipped.refCount = { ...shipped.refCount, ص: 800, س: 800 };
    const wide = shippedSample();
    wide.sub = { ...wide.sub, "ه>ح": 300, "ح>ه": 300, "ص>س": 2, "س>ص": 2 };
    wide.refCount = { ...wide.refCount, ه: 400, ح: 400, ص: 800, س: 800 };
    const got = derive(wide, anchorFrom(shipped));
    expect(got.costs["هح"]).toBe(5);
    expect(got.costs["صس"]).toBeGreaterThan(5);
    expect(pairCost("ه", "ح")).toBe(5);
    expect(subCost("ه", "ح")).toBe(5);
  });
});
