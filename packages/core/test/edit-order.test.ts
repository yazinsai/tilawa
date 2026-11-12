import { describe, expect, it } from "vitest";
import { appendEdit, editOrderLogRatio, editSequence, emptyEditScript, marginalOrderLogRatio, orderedAlignments, orderFeatures, shuffledOrderFeatures, validateEditOrderModel } from "../src/edit-order.js";
import type { EditOperation, EditOrderModel } from "../src/edit-order.js";

const script = (ops: EditOperation[]) => ops.reduce(appendEdit, emptyEditScript());
const weights = Array(25).fill(0); weights[3 * 5] = 2;
const model: EditOrderModel = { version: 1, ordered: weights, shuffled: weights, scale: 1 };

describe("ordered edit evidence", () => {
  it("distinguishes ordering with the same edit totals and reference length, while the control ties", () => {
    const a = script(["del", "match", "match", "sub", "boundary"]);
    const b = script(["match", "del", "sub", "match", "boundary"]);
    expect(a.counts).toEqual(b.counts);
    expect(editOrderLogRatio(a, model, "ordered")).toBeGreaterThan(editOrderLogRatio(b, model, "ordered"));
    expect(editOrderLogRatio(a, model, "shuffled")).toBe(editOrderLogRatio(b, model, "shuffled"));
  });

  it("centers transitions at the exact uniform count-preserving permutation mean", () => {
    const sequences: EditOperation[][] = [["match", "match", "del"], ["match", "del", "match"], ["del", "match", "match"]];
    const features = sequences.map(ops => orderFeatures(script(ops)));
    for (let i = 0; i < 25; i++) expect(features.reduce((s, f) => s + f[i], 0)).toBeCloseTo(0);
    expect(shuffledOrderFeatures([2, 0, 0, 1, 0])).toEqual(shuffledOrderFeatures([2, 0, 0, 1, 0]));
  });

  it("marginalizes ambiguous scripts with normalized acoustic weights rather than awarding more paths free credit", () => {
    const a = script(["del", "match"]), b = script(["match", "del"]);
    const paths = [{ script: a, cost: 10 }, { script: b, cost: 10 }];
    const want = Math.log((Math.exp(editOrderLogRatio(a, model, "ordered")) + Math.exp(editOrderLogRatio(b, model, "ordered"))) / 2);
    expect(marginalOrderLogRatio(paths, model, "ordered")).toBeCloseTo(want);
    expect(marginalOrderLogRatio([...paths, ...paths], model, "ordered")).toBeCloseTo(want);
    expect(marginalOrderLogRatio(paths, { ...model, scale: 0 }, "ordered")).toBeCloseTo(0);
  });

  it("retains directed operations and boundaries without inventing terminal phone consumption", () => {
    const paths = orderedAlignments("ببت", "بت", new Set([0, 2]));
    const best = paths.filter(p => p.cost === paths[0].cost).map(p => editSequence(p.script));
    expect(best).toContainEqual(["boundary", "del", "match", "boundary", "match"]);
    expect(best).toContainEqual(["boundary", "match", "del", "boundary", "match"]);
    for (const p of paths) {
      expect(p.script.counts[0] + p.script.counts[1] + p.script.counts[3]).toBe(3);
      expect(p.script.counts[0] + p.script.counts[1] + p.script.counts[2]).toBe(2);
    }
    const a = emptyEditScript(), b = appendEdit(a, "ins");
    expect(editSequence(a)).toEqual([]);
    expect(editSequence(b)).toEqual(["ins"]);
  });

  it("rejects malformed or non-finite calibration rather than changing candidate ranking silently", () => {
    expect(() => validateEditOrderModel({ ...model, scale: NaN })).toThrow();
    expect(() => validateEditOrderModel({ ...model, scale: -1 })).toThrow();
    expect(() => validateEditOrderModel({ ...model, ordered: [0] })).toThrow();
  });
});
