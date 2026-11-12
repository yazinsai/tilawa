import { describe, expect, it } from "vitest";
import type { DecodedPhoneme } from "../src/evidence.js";
import { COPY_HISTORY, copyInsertionCost, copyObservationProbability, copyPhoneClass, copyRecordingLogP, copySources, fitCopyChannel } from "../src/copy-channel.js";
import type { CopyChannelModel } from "../src/copy-channel.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { insertionCost } from "../src/fuzzy.js";

const observation = (p: string, start: number): DecodedPhoneme => ({ p, start, end: start, conf: 0.9 });
const phones = (s: string) => [...s].map((p, i) => observation(p, i * 2));
function channel(inventory = [..."بت"], linked = true): CopyChannelModel {
  const counts = Array.from({ length: 12 }, (_, c) => inventory.filter(p => copyPhoneClass(p) === c).length);
  const background = counts.map(n => n / inventory.length);
  return { version: 1, linked, inventory, insertion: 0.2, copy: 0.8, same: 0.95,
    ordinary: background, sourcePrior: inventory.map(() => 1 / inventory.length),
    copyClasses: Array.from({ length: 12 }, () => [...background]),
    delay: Array.from({ length: COPY_HISTORY }, () => 1 / COPY_HISTORY), sourceClasses: counts.map(n => n ? 1 : 0) };
}
const index = loadReferenceIndex({ version: "synthetic-copy", text: "synthetic", profile: {}, inventory: { id: "copy", tokens: "بتدنۥ" },
  ayahs: [{ s: 1, a: 1, w: ["first", "second", "third"], p: "بتدبتدننننۥۥ", o: [0, 3, 6], iso: ["بتد", "بتد", "ننننۥۥ"] }] });

describe("delayed recognition copies", () => {
  it("normalizes every observation row, including empty history and the independent control", () => {
    for (const linked of [false, true]) for (const history of [[], phones("بتب")]) {
      const model = channel([..."بتدَ"], linked);
      const sum = model.inventory.reduce((s, p) => s + copyObservationProbability(model, history, observation(p, 12)), 0);
      expect(sum).toBeCloseTo(1, 12);
    }
  });

  it("requires a distinct earlier occurrence and normalizes source selection", () => {
    const model = channel(), p = observation("ب", 10);
    expect(copySources(model, [observation("ب", 10)], p)).toHaveLength(0);
    const source = observation("ب", 0), history = [source, source, observation("ت", 2)];
    const sources = copySources(model, history, p);
    expect(sources).toHaveLength(2);
    expect(sources.reduce((s, p) => s + p.weight, 0)).toBeCloseTo(1, 12);
    const old = Array.from({ length: COPY_HISTORY + 1 }, (_, i) => observation(i === 0 ? "ب" : "ت", i * 2));
    expect(copySources(model, old, observation("ب", 100)).every(s => s.phone === "ت")).toBe(true);
  });

  it("prices a source-linked copy below its source-independent alternative while keeping insertions charged", () => {
    const model = channel(), history = phones("بب"), p = observation("ب", 8);
    const linked = copyInsertionCost(model, "ت", 0, history, p);
    const control = copyInsertionCost({ ...model, linked: false }, "ت", 0, history, p);
    expect(linked).toBeLessThan(control);
    expect(linked).toBeGreaterThan(0);
    expect(copyInsertionCost(model, "ت", 0, history, observation("ت", 8))).toBeGreaterThan(linked);
    expect(copyObservationProbability(model, [], p)).toBe(copyObservationProbability({ ...model, copy: 0 }, [], p));
  });

  it("keeps held-sound duration charges and unknown-phone fallback", () => {
    const model = channel([..."بتننۥ"].filter((p, i, a) => a.indexOf(p) === i));
    for (const ref of ["نننن", "ۥۥ"]) {
      const p = observation(ref[0], 8);
      expect(copyInsertionCost(model, ref, 1, phones("بت"), p)).toBe(insertionCost(ref, 1, p.p));
    }
    expect(copyInsertionCost(model, "ت", 0, [], observation("ق", 4))).toBe(insertionCost("ت", 0, "ق"));
  });

  it("normalizes complete latent edit/copy paths rather than awarding free alignment multiplicity", () => {
    const model = { ...channel(), insertion: 0.02 };
    let mass = 0;
    const visit = (text: string) => {
      mass += Math.exp(copyRecordingLogP(model, { written: "بت", observations: phones(text) }));
      if (text.length < 7) for (const p of model.inventory) visit(text + p);
    };
    visit("");
    expect(mass).toBeCloseTo(1, 7);
  });

  it("fits latent edits with normalized phonetic-class sharing, without unknown-symbol supervision", () => {
    const records = ["بتب", "تببت", "بت", "تب"].map(text => ({ written: "بت", observations: phones(text) }));
    const model = fitCopyChannel(records, [..."بتد"], 8, true);
    expect(model.insertion).toBeGreaterThan(0);
    expect(model.insertion).toBeLessThan(1);
    expect(model.copy).toBeGreaterThan(0);
    expect(model.copy).toBeLessThan(1);
    expect(model.delay.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (const row of [model.ordinary, model.sourcePrior, model.sourceClasses, ...model.copyClasses]) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(Number.isFinite(copyRecordingLogP(model, records[0]))).toBe(true);
    expect(() => fitCopyChannel([{ written: "ق", observations: [] }], [..."بت"], 8, true)).toThrow("Unknown training phone");
  });

  it("learns copy dependence that a pooled insertion model cannot predict on a new order", () => {
    const texts = ["بتد", "بدت", "تبد", "تدب", "دتب"];
    const records = Array.from({ length: 8 }, () => texts.map(written => ({ written, observations: phones(written[0] + written) }))).flat();
    const linked = fitCopyChannel(records, [..."بتد"], 8, true);
    const control = fitCopyChannel(records, [..."بتد"], 8, false);
    const query = { written: "دبت", observations: phones("ددبت") };
    expect(copyRecordingLogP(linked, query)).toBeGreaterThan(copyRecordingLogP(control, query));
  });

  it("restores source history exactly across rewind, including separate timestamps", () => {
    const model = channel([...index.tokens]), start = { offset: 0, origin: 0, edits: 0 };
    const a = createLocalSearch(index, start, model), b = createLocalSearch(index, start, model);
    for (const p of phones("بتد")) { a.step(p.p, p); b.step(p.p, p); }
    const saved = a.save();
    a.step("ت", observation("ت", 8)); a.restore(saved);
    for (const p of [observation("ب", 10), observation("ت", 12), observation("د", 14)]) { a.step(p.p, p); b.step(p.p, p); }
    expect(a.position()).toEqual(b.position());
  });

  it("retains ordinary local paths byte for byte when no trained model is supplied", () => {
    const start = { offset: 0, origin: 0, edits: 0 }, a = createLocalSearch(index, start), b = createLocalSearch(index, start, null);
    for (const p of phones("بتبتدنننۥۥ")) { a.step(p.p); b.step(p.p, p); expect(a.position()).toEqual(b.position()); }
  });

  it("leaves real word omissions represented by a charged movement", () => {
    const local = createLocalSearch(index, { offset: 3, origin: 0, edits: 0 }, channel([...index.tokens]));
    let cost = 0;
    for (const p of phones("ننننۥۥ")) { local.step(p.p, p); cost += local.position().stepCost; }
    expect(local.position().origin).toBe(6);
    expect(cost).toBeGreaterThanOrEqual(2);
    expect(local.position().offset).toBe(12);
  });
});
