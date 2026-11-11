import { describe, expect, it } from "vitest";
import type { EngineEvent, TilawaSession } from "../../src/index.js";
import { createRecorder, encodeReport, validateReport } from "../../src/report/index.js";
import { tinyTilawa, wav } from "../helpers.js";

const tilawa = await tinyTilawa({ recordEvidence: true });
const clip = await wav();
const consent = { policy: "tilawa-report", version: "test", send: true as const, training: false, publicDataset: false };

async function run(withRecorder: boolean) {
  const session: TilawaSession = tilawa.startFreeRecitation();
  const events: EngineEvent[] = [];
  session.subscribe((e) => events.push(e));
  const recorder = withRecorder
    ? createRecorder(session, { retain: { take: true, maxMs: 300_000 }, source: "test", evidence: true })
    : null;
  const chunk = 1234;
  const calls: Promise<unknown>[] = [];
  for (let a = 0; a < clip.samples.length; a += chunk) calls.push(session.pushAudio(clip.samples.subarray(a, a + chunk), clip.sampleRate));
  const mid = withRecorder ? recorder!.mark({ kind: "stumble" }) : null;
  calls.push(session.end());
  await Promise.all(calls);
  const report = recorder ? await recorder.contribution({ consent, incidents: mid ? [mid] : [] }) : null;
  recorder?.dispose();
  return { events, evidence: session.evidence(), report, session };
}

describe("recorder on a real session", () => {
  it("does not change events or evidence", async () => {
    const off = await run(false);
    const on = await run(true);
    expect(on.events).toEqual(off.events);
    expect(on.evidence).toEqual(off.evidence);
  });

  it("produces a take that validates and covers the whole session clock", async () => {
    const { report, session } = await run(true);
    const { json, wav: audio } = encodeReport(report!);
    const result = await validateReport(json, audio);
    expect(result.errors).toEqual([]);
    expect(report!.sessionId).toBe(session.id);
    expect(report!.audio.startMs).toBe(0);
    expect(Math.abs(report!.audio.durationMs - session.now())).toBeLessThanOrEqual(1);
    expect(report!.engine.eventSchema).toBe(1);
    expect(report!.engine.sdkVersion).not.toBe("unknown");
    expect(report!.observations.events[0]?.type).toBe("capabilities");
  });
});
