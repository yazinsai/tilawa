import { describe, expect, it } from "vitest"
import { ctcForward as coreForward } from "@tilawa/core"
import { ctcForward } from "../src/ctc.js"

describe("sdk ctc forward", () => {
  it("scores a label sequence on raw model log-probs the same way the checker does", () => {
    const vocab = 3
    const rows = [
      [Math.log(0.7), Math.log(0.2), Math.log(0.1)],
      [Math.log(0.1), Math.log(0.8), Math.log(0.1)],
      [Math.log(0.6), Math.log(0.1), Math.log(0.3)],
      [Math.log(0.2), Math.log(0.2), Math.log(0.6)],
    ]
    const flat = rows.flat()
    const labels = [1, 2]
    const got = ctcForward(flat, vocab, labels, 0, 4)
    const core = coreForward((f) => rows[f], labels, 0, 4)
    expect(got).toBeCloseTo(core, 10)
    // The heard sequence over a window shifted by two frames is a different hypothesis.
    expect(ctcForward(flat, vocab, labels, 1, 4)).not.toBeCloseTo(got, 5)
    expect(ctcForward(new Float64Array(flat), vocab, [2], 2, 4)).toBeCloseTo(coreForward((f) => rows[f], [2], 2, 4), 10)
  })

  it("is finite on an empty label over blank frames and infinite on an empty span with labels", () => {
    const blank = [0, -20, -20, 0, -20, -20]
    expect(ctcForward(blank, 3, [], 0, 2)).toBeCloseTo(0, 8)
    expect(ctcForward(blank, 3, [1], 1, 1)).toBe(Infinity)
  })
})
