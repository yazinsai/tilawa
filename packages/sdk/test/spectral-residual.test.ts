import { describe, expect, it } from "vitest";
import { Fbank } from "../src/fbank.js";
import { createSpectralResidualScorer, residualLikelihood, spectralObservations, logadd, type SpectralResidualParameters } from "../src/spectral-residual.js";
import { carrySpectralResidualEvidence, compareSpectralWindow } from "../src/spectral-residual-evidence.js";

function parameters(): SpectralResidualParameters {
  return { schema: 1, tokens: "بت", projection: [Array.from({ length: 256 }, (_, k) => k === 20 ? 1 : 0)],
    predictor: Array.from({ length: 81 }, () => [0]), scale: [1], means: [[0], [1], [-1]], variance: [1], covered: [true, true, true] };
}
const request = { word: 0, written: "ب", accused: "ت", fromFrame: 0, toFrame: 2, slotFromFrame: 0, slotToFrame: 2 };

describe("within-band spectral evidence", () => {
  it("marginalizes exactly over CTC paths with a normalized placement prior", () => {
    const rows = [[-.2, -1, -2], [-.1, -.2, -3], [-1, -.4, -.7], [-.3, -1.2, -.2]];
    for (const labels of [[1], [1, 2], [1, 1]]) {
      let mass = -Infinity, count = 0;
      function enumerate(path: number[]) {
        if (path.length === rows.length) {
          const collapsed = path.filter((p, t) => p !== 0 && (!t || path[t - 1] !== p));
          if (JSON.stringify(collapsed) !== JSON.stringify(labels)) return;
          mass = logadd(mass, path.reduce((s, p, t) => s + rows[t][p], 0)); count++; return;
        }
        for (let p = 0; p < 3; p++) enumerate([...path, p]);
      }
      enumerate([]);
      expect(residualLikelihood(rows, labels)).toBeCloseTo(mass - Math.log(count), 12);
    }
    expect(residualLikelihood([[0, 0], [0, 0]], [1, 1])).toBe(-Infinity);
    expect(residualLikelihood(Array.from({ length: 8 }, () => [2, 2, 2]), [1, 2])).toBeCloseTo(16, 12);
  });

  it("compares complete hypotheses symmetrically and abstains on missing phones or windows", () => {
    const p = parameters(), scorer = createSpectralResidualScorer(p), features = Array.from({ length: 8 }, () => new Float32Array([1]));
    const score = scorer.compare(features, request)!;
    expect(score.writtenLogLikelihood).toBeGreaterThan(score.accusedLogLikelihood);
    expect(scorer.compare(features, { ...request, written: "ت", accused: "ب" })).toEqual({
      writtenLogLikelihood: score.accusedLogLikelihood, accusedLogLikelihood: score.writtenLogLikelihood });
    expect(scorer.compare(features, { ...request, accused: "ب" })!.writtenLogLikelihood).toBe(score.writtenLogLikelihood);
    expect(scorer.compare(features, { ...request, written: "ث" })).toBeNull();
    expect(scorer.compare(features.slice(0, 7), request)).toBeNull();
    p.covered[2] = false;
    expect(scorer.compare(features, request)).toBeNull();
    expect(() => createSpectralResidualScorer({ ...parameters(), variance: [0] })).toThrow();
  });

  it("keeps primary mel frames identical even if a spectrum consumer mutates its copies", () => {
    const audio = Float32Array.from({ length: 2400 }, (_, i) => Math.sin(i * .12) * .1);
    const primary = new Fbank().push(audio);
    const powers: number[][] = [];
    const f = new Fbank({ onSpectrum(power, mel) { powers.push(Array.from(power)); power.fill(1e9); mel.fill(1e9); } });
    const chunked = [...f.push(audio.subarray(0, 633)), ...f.push(audio.subarray(633))];
    expect(chunked).toEqual(primary);
    expect(powers).toHaveLength(primary.length);
    const observed = spectralObservations(audio, parameters());
    expect(observed).toHaveLength(primary.length);
    expect(observed.every(x => Number.isFinite(x[0]))).toBe(true);
  });

  it("extracts byte-identical original windows and scores without reprocessing the full session", () => {
    const audio = Float32Array.from({ length: 4000 }, (_, i) => Math.sin(i * .131) * .1);
    const p = parameters(), scorer = createSpectralResidualScorer(p), full = spectralObservations(audio, p);
    const r = { ...request, fromFrame: 1, toFrame: 3, slotFromFrame: 1, slotToFrame: 3 };
    expect(spectralObservations(audio.subarray(640, 3 * 640 + 240), p)).toEqual(full.slice(4, 12));
    expect(compareSpectralWindow(scorer, audio, r)).toEqual(scorer.compare(full, r));
    expect(compareSpectralWindow(scorer, audio.subarray(0, 1400), r)).toBeNull();
  });

  it("carries certificates through JSON without changing primary phones or insertion certificates", () => {
    const chunks = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 2, conf: .9, seamInsertions: [] }] }];
    const certificates = [{ ...request, word: { surah: 1, ayah: 1, word: 1 }, writtenLogLikelihood: -1, accusedLogLikelihood: -2 }];
    const enriched = JSON.parse(JSON.stringify(carrySpectralResidualEvidence(chunks, certificates)));
    expect(enriched[0].phonemes[0].spectralResiduals).toEqual(certificates);
    delete enriched[0].phonemes[0].spectralResiduals;
    expect(enriched).toEqual(chunks);
    expect(chunks[0].phonemes[0]).not.toHaveProperty("spectralResiduals");
  });
});
