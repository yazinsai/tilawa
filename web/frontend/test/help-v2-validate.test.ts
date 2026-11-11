import { describe, expect, test } from "vitest";
import { validateReport } from "../server/help/report-validate";
import { makeReport, makeWav, sha256 } from "./help-v2-fixture";

const wav = makeWav(2000);
const check = (patch: (r: any) => void, w = wav, opts = {}) => {
  const r = makeReport(w, patch);
  const res = validateReport(JSON.stringify(r), w, opts);
  return res.ok ? [] : res.errors;
};

describe("validateReport (contract §6)", () => {
  test("a valid artifact passes", () => {
    expect(check(() => {})).toEqual([]);
    expect(check(() => {}, wav, { consentVersions: ["2026-11-report-1"] })).toEqual([]);
  });
  test("schema, uuids and consent", () => {
    expect(check((r) => (r.schema = "tilawa.report/2"))).toContain("schema");
    expect(check((r) => (r.reportId = "not-a-uuid"))).toContain("uuid.reportId");
    expect(check((r) => (r.reportId = r.reportId.toUpperCase()))).toContain("uuid.reportId");
    expect(check((r) => (r.incidents[0].id = "x"))).toContain("uuid.incidents[0].id");
    expect(check((r) => (r.consent.send = false))).toContain("consent.send");
    expect(check(() => {}, wav, { consentVersions: [] })).toContain("consent.version");
    expect(check((r) => (r.contributor = "has space"))).toContain("fields.contributor");
    expect(check((r) => (r.incidents[0].kind = "bogus"))).toContain("fields.incidents[0]");
  });
  test("extra fields are allowed", () => {
    expect(check((r) => ((r.future = { x: 1 }), (r.audio.extra = true)))).toEqual([]);
  });
  test("canonical WAV header only", () => {
    expect(check(() => {}, makeWav(2000, { rate: 44100 }))).toContain("wav.header");
    expect(check(() => {}, makeWav(2000, { channels: 2 }))).toContain("wav.header");
    const truncated = wav.slice(0, wav.length - 2);
    const r = makeReport(wav);
    expect((validateReport(JSON.stringify(r), truncated) as any).errors).toContain("wav.header");
    const notWav = new Uint8Array(wav); notWav[0] = 0x58;
    expect(check(() => {}, notWav)).toContain("wav.header");
  });
  test("duration tolerance is ±1 ms", () => {
    expect(check((r) => (r.audio.durationMs += 1))).toEqual([]);
    expect(check((r) => (r.audio.durationMs += 2))).toContain("audio.duration");
  });
  test("sha256 of the WAV bytes", () => {
    expect(check((r) => (r.audio.sha256 = sha256("other")))).toContain("audio.sha256");
  });
  test("incident spans stay inside the audio unless truncated", () => {
    const after = (r: any) => (r.incidents[0].endMs = r.audio.startMs + r.audio.durationMs + 5);
    expect(check(after)).toContain("incident.span[0]");
    expect(check((r) => (after(r), (r.audio.truncatedEnd = true)))).toEqual([]);
    const before = (r: any) => (r.incidents[0].startMs = r.audio.startMs - 5);
    expect(check(before)).toContain("incident.span[0]");
    expect(check((r) => (before(r), (r.audio.truncatedStart = true)))).toEqual([]);
    expect(check((r) => ((r.incidents[0].startMs = r.incidents[0].endMs + 1), (r.audio.truncatedEnd = true)))).toContain("incident.span[0]");
  });
  test("limits", () => {
    const inc = (r: any) => r.incidents[0];
    expect(check((r) => (r.incidents = Array.from({ length: 101 }, () => ({ ...inc(r), id: crypto.randomUUID() }))))).toContain("incidents.max");
    expect(check((r) => (r.incidents = Array.from({ length: 100 }, () => ({ ...inc(r), id: crypto.randomUUID() }))))).toEqual([]);
    expect(check((r) => (inc(r).claim.note = "x".repeat(2001)))).toContain("claim.note[0]");
    expect(check((r) => (inc(r).claim.note = "x".repeat(2000)))).toEqual([]);
    expect(check((r) => (r.observations.pad = "x".repeat(4 * 1024 * 1024)))).toEqual(["size.json"]);
    expect(check((r) => (r.audio.durationMs = 300_001))).toContain("audio.max");
    expect((validateReport("{nope", wav) as any).errors).toEqual(["json"]);
  });
});
