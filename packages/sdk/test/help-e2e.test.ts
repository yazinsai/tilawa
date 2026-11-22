/**
 * The shipped runtime end to end: the help eval clips through the session API on onnxruntime-node reproduce the
 * scoreboard's spliced-clean numbers (scored there from the PyTorch decode). Needs the exported model and the private
 * help clips, so it runs only with TILAWA_MODEL_DIR and HELP_DIR set; about 7 minutes on an M-series Mac.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import type { PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { runHelpBench } from "../src/bench/help.js";
import { loadNodeModel } from "../src/node.js";

const { TILAWA_MODEL_DIR: modelDir, HELP_DIR: helpDir } = process.env;
const core = (p: string) => JSON.parse(readFileSync(new URL(`../../core/assets/${p}`, import.meta.url), "utf8"));

describe.runIf(modelDir && helpDir)("help clips through the session API on onnxruntime-node", () => {
  it("reproduce the scoreboard's spliced-clean declared and free arms", { timeout: 30 * 60_000 }, async () => {
    const res = await runHelpBench({
      helpDir: helpDir!,
      reference: loadReferenceIndex(core("ref-v2.json") as ReferenceAsset),
      phrases: core("phrases-v1.json") as PhraseAsset,
      model: await loadNodeModel(modelDir!),
    });
    // Scoreboard row 2d94ea0 / spliced-clean: declared 56 / 11 / 8, free 27 / 6 / 37 (tp / fp / fn).
    expect(res.clips.length).toBe(388);
    expect(Math.abs(res.declared.precision - 56 / 67)).toBeLessThan(0.03);
    expect(Math.abs(res.declared.recall - 56 / 64)).toBeLessThan(0.03);
    expect(Math.abs(res.free.precision - 27 / 33)).toBeLessThan(0.05);
    expect(Math.abs(res.free.recall - 27 / 64)).toBeLessThan(0.03);
  });
});
