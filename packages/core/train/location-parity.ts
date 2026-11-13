/** Small cross-language recurrence oracle emitted by the actual engine, not by a second test implementation. */
import { writeFileSync } from "node:fs";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { TRAJECTORY_FEATURES } from "../src/trajectory.js";
const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "ابتثجحنَس" },
  ayahs: Array.from({ length: 8 }, (_, i) => ({ s: 1, a: i + 1, w: ["x", "y"], p: "ابتننننَثجح", o: [0, 3], iso: ["ابت", "ننننَثجح"] })) });
const cases = [];
for (const weights of [TRAJECTORY_FEATURES.map(() => 0), TRAJECTORY_FEATURES.map((_, i) => (i % 3 - 1) / 100)]) {
  for (const phones of ["ابتثجحابتننَث", "ابتننننَثجح".repeat(5), "ثجحابتسَننث", "ابتثجح".repeat(3)]) {
    const local = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 }, { model: { version: 1, features: TRAJECTORY_FEATURES, weights } });
    const states = [];
    for (const p of phones) {
      local.step(p);
      const state = local.save() as { lo: number; cost: Float64Array; origin: Int32Array; best: number };
      states.push({ lo: state.lo, best: state.best, cost: Array.from(state.cost, c => Number.isFinite(c) ? c / 10 : null), origin: [...state.origin] });
    }
    cases.push({ weights, phones, states });
  }
}
if (!process.argv[2]) throw new Error("Output path required");
writeFileSync(process.argv[2], JSON.stringify({ reference: { phonemes: index.phonemes, words: index.words }, cases }));
