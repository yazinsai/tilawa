import { describe, expect, it } from "vitest";
import { composePhoneChannel, fitSpokenPhoneModel, localPhoneCosts, phoneLogP } from "../src/spoken-phone-channel.js";
import type { PhoneChannel, PhoneRecording } from "../src/spoken-phone-channel.js";
import { createLocalSearch } from "../src/local.js";
import { deletionCost, insertionCost, subCost } from "../src/fuzzy.js";
import { loadReferenceIndex } from "../src/reference.js";

const recognition: PhoneChannel = { phones: ["َ", "ُ"], rows: [[0.75, 0.15, 0.1], [0.1, 0.85, 0.05]], insertion: [0.4, 0.6], insertionRate: 0.1 };
const pronunciation = [[0.8, 0.2], [0.1, 0.9]];
const record = (written: string, heard: string, variable = Array(written.length).fill(false)): PhoneRecording => ({ written, heard, variable });
const toyIndex = () => loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "َُ" },
  ayahs: [{ s: 1, a: 1, w: ["first", "next", "rhyme"], p: "َََُُُ", o: [0, 2, 4], iso: ["ََ", "ُُ", "َُ"] }] });

describe("latent spoken-phone composition", () => {
  it("sums alternative latent pronunciations, including their recognition deletions", () => {
    const channel = composePhoneChannel(recognition, pronunciation);
    expect(channel.rows[0][0]).toBeCloseTo(0.8 * 0.75 + 0.2 * 0.1, 12);
    expect(channel.rows[0][2]).toBeCloseTo(0.8 * 0.1 + 0.2 * 0.05, 12);
    for (const row of channel.rows) expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(channel.insertionRate).toBe(recognition.insertionRate);
    expect(composePhoneChannel(recognition, [[1, 0], [0, 1]])).toEqual(recognition);
  });

  it("normalizes complete observed strings and marginalizes insertion placement", () => {
    const channel = composePhoneChannel(recognition, pronunciation), stop = 1 - channel.insertionRate;
    expect(Math.exp(phoneLogP(channel, record("َ", "")))).toBeCloseTo(stop ** 2 * channel.rows[0][2], 12);
    expect(Math.exp(phoneLogP(channel, record("َ", "ُ")))).toBeCloseTo(stop ** 2 * (channel.rows[0][1] + 2 * channel.insertionRate * channel.insertion[1] * channel.rows[0][2]), 12);
    let total = 0, strings = [""];
    for (let n = 0; n <= 8; n++) {
      for (const heard of strings) total += Math.exp(phoneLogP(channel, record("َ", heard)));
      strings = strings.flatMap(s => channel.phones.map(p => s + p));
    }
    expect(total).toBeLessThanOrEqual(1 + 1e-12);
    expect(total).toBeGreaterThan(0.99999);
  });

  it("applies the script mask to latent pronunciation, without changing the clean anchor", () => {
    const composed = composePhoneChannel(recognition, pronunciation);
    expect(phoneLogP(recognition, record("َ", "ُ", [false]), pronunciation)).toBe(phoneLogP(recognition, record("َ", "ُ")));
    expect(phoneLogP(recognition, record("َ", "ُ", [true]), pronunciation)).toBeCloseTo(phoneLogP(composed, record("َ", "ُ")), 12);
    expect(() => composePhoneChannel(recognition, [[1, -1], [0, 1]])).toThrow();
  });

  it("keeps recognition anchored when pronunciation supervision changes", () => {
    const clean = Array.from({ length: 8 }, () => record("َُ", "َُ"));
    const first = fitSpokenPhoneModel([...clean, ...Array.from({ length: 12 }, () => record("َ", "ُ", [true]))], recognition.phones, 8, 6);
    const other = fitSpokenPhoneModel([...clean, ...Array.from({ length: 12 }, () => record("ُ", "َ", [true]))], recognition.phones, 8, 6);
    expect(first.recognition).toEqual(other.recognition);
    expect(first.pronunciation[0][1]).toBeGreaterThan(other.pronunciation[0][1]);
    expect(first.control).not.toEqual(other.control);
    for (const channel of [first.recognition, first.control, composePhoneChannel(first.recognition, first.pronunciation)]) {
      for (const row of channel.rows) expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 10);
      expect(Number.isFinite(phoneLogP(channel, record("َُ", "َُُ")))).toBe(true);
    }
  });

  it("retains explicit duration charges and falls back on unknown phones", () => {
    const costs = localPhoneCosts(composePhoneChannel(recognition, pronunciation));
    expect(costs.del("َََ", 1)).toBe(deletionCost("َََ", 1));
    expect(costs.sub("َََ", 1, "ُ")).toBe(subCost("َ", "ُ"));
    expect(costs.ins("ااا", 1, "ا")).toBe(insertionCost("ااا", 1, "ا"));
    expect(costs.sub("ب", 0, "ُ")).toBe(subCost("ب", "ُ"));
    expect(costs.del("ب", 0)).toBe(deletionCost("ب", 0));
    expect(costs.sub("َ", 0, "َ")).toBe(0);
  });

  it("uses the same model for continuation and movement, and rewinds exactly", () => {
    const index = toyIndex(), start = { offset: 0, origin: 0, edits: 0 };
    const local = createLocalSearch(index, start, composePhoneChannel(recognition, pronunciation));
    local.step("َ");
    const saved = local.save();
    local.step("ُ");
    const position = local.position(), candidates = local.candidates();
    candidates[0].cost = 999; // exported diagnostics cannot mutate a retained DP row
    local.restore(saved);
    local.step("ُ");
    expect(local.position()).toEqual(position);
    expect(local.candidates()[0].cost).not.toBe(999);
    const ordinary = createLocalSearch(index, start), absent = createLocalSearch(index, start, undefined);
    for (const p of "ََُُ") { ordinary.step(p); absent.step(p); }
    expect(absent.position()).toEqual(ordinary.position());
    expect(absent.candidates()).toEqual(ordinary.candidates());
  });

  it("changes pronunciation likelihood without erasing the observed phone edit", () => {
    const index = toyIndex(), start = { offset: 0, origin: 0, edits: 0 };
    const channel = composePhoneChannel(recognition, [[0, 1], [0, 1]]);
    const modelled = createLocalSearch(index, start, channel), ordinary = createLocalSearch(index, start);
    modelled.step("ُ"); ordinary.step("ُ");
    expect(modelled.position().offset).toBe(1);
    expect(modelled.position().origin).toBe(0);
    expect(modelled.position().edits).toBe(1);
    expect(modelled.position().stepCost).toBe(0);
    expect(ordinary.position().stepCost).toBeGreaterThan(modelled.position().stepCost);
  });
});
