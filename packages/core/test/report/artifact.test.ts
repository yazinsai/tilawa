import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { createRecorder, decodeReport, encodeReport, encodeWav, ReportValidationError, validateReport } from "../../src/report/index.js";
import type { TilawaReport } from "../../src/report/index.js";
import { sha256Hex } from "../../src/report/crypto.js";
import { CONSENT, FakeSession } from "./fake-session.js";

const VALIDATOR = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../tools/report_validate.py");

let base: { json: string; wav: Uint8Array };

beforeAll(async () => {
  const s = new FakeSession();
  const rec = createRecorder(s, { source: "test", contributor: "anon-1", consent: CONSENT });
  s.pushMs(4000);
  s.checker("c1", 4000);
  s.pushMs(6000);
  base = encodeReport(await rec.report({ kind: "false_warning", eventId: "c1", claim: { question: "Was this a mistake?", answer: "no", note: "I read it right" } }));
});

/** Applies `edit` to a copy of the report and, unless `keepSha`, re-signs the audio sha so only the edit is wrong. */
async function variant(edit: (r: TilawaReport, wav: Uint8Array) => Uint8Array | void, keepSha = false) {
  const r = JSON.parse(base.json) as TilawaReport;
  const wav = (edit(r, base.wav.slice()) as Uint8Array | undefined) ?? base.wav;
  if (!keepSha) r.audio.sha256 = await sha256Hex(wav);
  return { json: JSON.stringify(r), wav };
}

async function errorsOf(v: { json: string; wav: Uint8Array }) {
  return (await validateReport(v.json, v.wav)).errors;
}

describe("encodeReport", () => {
  it("writes a canonical 44-byte PCM16 mono 16 kHz WAV", () => {
    const wav = encodeWav(new Int16Array([0, 1, -1, 32767, -32768]));
    const view = new DataView(wav.buffer);
    expect(wav.length).toBe(44 + 10);
    expect(new TextDecoder().decode(wav.subarray(0, 4))).toBe("RIFF");
    expect(view.getUint32(4, true)).toBe(46);
    expect([view.getUint32(16, true), view.getUint16(20, true), view.getUint16(22, true), view.getUint32(24, true), view.getUint32(28, true), view.getUint16(32, true), view.getUint16(34, true)]).toEqual([16, 1, 1, 16000, 32000, 2, 16]);
    expect(view.getUint32(40, true)).toBe(10);
    expect(view.getInt16(44 + 8, true)).toBe(-32768);
  });

  it("throws for a report object without attached audio", () => {
    expect(() => encodeReport(JSON.parse(base.json))).toThrow(/no audio attached/);
  });
});

describe("round trip", () => {
  it("encode -> decode -> validate -> encode is stable", async () => {
    expect((await validateReport(base.json, base.wav)).ok).toBe(true);
    const decoded = await decodeReport(base.json, base.wav);
    expect(decoded.incidents[0].claim.note).toBe("I read it right");
    const again = encodeReport(decoded);
    expect(again.json).toBe(base.json);
    expect(again.wav).toBe(base.wav);
    expect((await validateReport(new TextEncoder().encode(again.json), again.wav)).ok).toBe(true);
  });

  it("decodeReport throws ReportValidationError for a bad artifact", async () => {
    await expect(decodeReport("{", base.wav)).rejects.toBeInstanceOf(ReportValidationError);
  });
});

describe("validateReport rejects", () => {
  it("a bad fmt tuple", async () => {
    const v = await variant((_r, wav) => {
      new DataView(wav.buffer).setUint32(24, 44_100, true);
      return wav;
    });
    expect(await errorsOf(v)).toEqual([expect.stringMatching(/fmt is \(16,1,1,44100,32000,2,16\)/)]);
  });

  it("a non-canonical header", async () => {
    const v = await variant((_r, wav) => {
      wav.set([0x4c, 0x49, 0x53, 0x54], 36); // "LIST"
      return wav;
    });
    expect(await errorsOf(v)).toEqual([expect.stringMatching(/canonical 44-byte header/)]);
  });

  it("a sha256 mismatch", async () => {
    const v = await variant((_r, wav) => {
      wav[100] ^= 0xff;
      return wav;
    }, true);
    expect(await errorsOf(v)).toEqual(["audio.sha256 does not match audio.wav"]);
  });

  it("a duration that does not match the samples", async () => {
    expect(await errorsOf(await variant((r) => void (r.audio.durationMs += 2)))).toEqual([expect.stringMatching(/durationMs/)]);
    expect(await errorsOf(await variant((r) => void (r.audio.durationMs += 1)))).toEqual([]);
  });

  it("an incident outside the span unless that side is truncated", async () => {
    const before = await variant((r) => void (r.incidents[0].startMs = r.audio.startMs - 1));
    expect(await errorsOf(before)).toEqual(["incidents[0] starts before the audio"]);
    const after = await variant((r) => void (r.incidents[0].endMs = r.audio.startMs + r.audio.durationMs + 1));
    expect(await errorsOf(after)).toEqual(["incidents[0] ends after the audio"]);
    const truncated = await variant((r) => {
      r.incidents[0].endMs = r.audio.startMs + r.audio.durationMs + 1;
      r.audio.truncatedEnd = true;
    });
    expect(await errorsOf(truncated)).toEqual([]);
  });

  it("more than 100 incidents", async () => {
    const v = await variant((r) => void (r.incidents = Array.from({ length: 101 }, () => r.incidents[0])));
    expect(await errorsOf(v)).toEqual(["at most 100 incidents"]);
  });

  it("a note over 2000 chars", async () => {
    expect(await errorsOf(await variant((r) => void (r.incidents[0].claim.note = "x".repeat(2001))))).toEqual(["incidents[0].claim.note must be at most 2000 chars"]);
    expect(await errorsOf(await variant((r) => void (r.incidents[0].claim.note = "x".repeat(2000))))).toEqual([]);
  });

  it("an unknown schema major, bad uuids, missing consent and an oversize json", async () => {
    expect(await errorsOf(await variant((r) => void ((r as { schema: string }).schema = "tilawa.report/2")))).toEqual(["unsupported report schema major 2"]);
    expect(await errorsOf(await variant((r) => void ((r as { schema: string }).schema = "tilawa.report/1.1")))).toEqual([]);
    expect(await errorsOf(await variant((r) => void (r.reportId = "nope")))).toEqual(["reportId is not a uuid"]);
    expect(await errorsOf(await variant((r) => void (r.consent.send = false as unknown as true)))).toEqual(["consent.send must be true"]);
    expect(await errorsOf(await variant((r) => void (r.contributor = "has space")))).toEqual(["contributor must be 1-64 chars of [A-Za-z0-9_-]"]);
    const big = await variant((r) => void ((r as unknown as { pad: string }).pad = "x".repeat(4 * 1024 * 1024)));
    expect(await errorsOf(big)).toEqual(["report.json is larger than 4 MiB"]);
  });

  it("audio over 300 s", async () => {
    const wav = encodeWav(new Int16Array(300 * 16000 + 16));
    const v = await variant((r) => {
      r.audio.durationMs = 300_001;
      return wav;
    });
    expect(await errorsOf(v)).toEqual(expect.arrayContaining(["audio.wav is longer than 300 s", "audio.durationMs must be in (0, 300000]"]));
  });
});

const hasPython = spawnSync("python3", ["--version"]).status === 0;

describe.skipIf(!hasPython)("tools/report_validate.py", () => {
  it("accepts a TS-encoded fixture and rejects corrupted ones", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tilawa-report-"));
    const write = (name: string, v: { json: string; wav: Uint8Array }) => {
      writeFileSync(join(dir, `${name}.json`), v.json);
      writeFileSync(join(dir, `${name}.wav`), v.wav);
      return spawnSync("python3", [VALIDATOR, join(dir, `${name}.json`), join(dir, `${name}.wav`)], { encoding: "utf8" });
    };
    const ok = write("ok", base);
    expect(ok.stderr).toBe("");
    expect(ok.status).toBe(0);
    expect(execFileSync("python3", [VALIDATOR, join(dir, "ok.json"), join(dir, "ok.wav")], { encoding: "utf8" }).trim()).toBe("ok");

    const cases: Record<string, { json: string; wav: Uint8Array }> = {
      sha: await variant((_r, wav) => {
        wav[60] ^= 1;
        return wav;
      }, true),
      fmt: await variant((_r, wav) => {
        new DataView(wav.buffer).setUint16(22, 2, true);
        return wav;
      }),
      span: await variant((r) => void (r.incidents[0].startMs = r.audio.startMs - 5)),
      many: await variant((r) => void (r.incidents = Array.from({ length: 101 }, () => r.incidents[0]))),
      note: await variant((r) => void (r.incidents[0].claim.note = "ن".repeat(2001))),
      schema: await variant((r) => void ((r as { schema: string }).schema = "tilawa.report/2")),
      bools: await variant((r) => void ((r.audio as unknown as { channels: boolean }).channels = true)),
    };
    for (const [name, v] of Object.entries(cases)) {
      const ts = await validateReport(v.json, v.wav);
      const py = write(name, v);
      expect({ name, status: py.status }).toEqual({ name, status: 1 });
      expect(ts.ok).toBe(false);
      expect(py.stderr.trim().split("\n")).toHaveLength(ts.errors.length);
    }
  });
});
