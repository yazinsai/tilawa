// Builds valid tilawa.report/1 artifacts for server tests, independent of the SDK.
import { createHash, randomUUID } from "node:crypto";

export function makeWav(durationMs: number, opts: { rate?: number; channels?: number } = {}): Uint8Array {
  const rate = opts.rate ?? 16000;
  const channels = opts.channels ?? 1;
  const samples = Math.round((durationMs * rate) / 1000) * channels;
  const buf = new Uint8Array(44 + samples * 2);
  const v = new DataView(buf.buffer);
  const tag = (o: number, s: string) => [...s].forEach((ch, i) => (buf[o + i] = ch.charCodeAt(0)));
  tag(0, "RIFF"); v.setUint32(4, buf.length - 8, true); tag(8, "WAVE");
  tag(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, channels, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * channels * 2, true); v.setUint16(32, channels * 2, true); v.setUint16(34, 16, true);
  tag(36, "data"); v.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i++) v.setInt16(44 + i * 2, Math.round(8000 * Math.sin(i / 7)), true);
  return buf;
}

export const sha256 = (b: Uint8Array | string) => createHash("sha256").update(b).digest("hex");

export function makeReport(wav: Uint8Array, patch: (r: any) => void = () => {}) {
  const durationMs = (wav.length - 44) / 2 / 16;
  const startMs = 12_000;
  const r: any = {
    schema: "tilawa.report/1",
    reportId: randomUUID(),
    sessionId: randomUUID(),
    createdAt: "2026-11-11T10:00:00.000Z",
    source: "tilawa.dev/help",
    contributor: "spk_abc123",
    engine: { sdkVersion: "1.0.0", modelId: "zipformer-a0w", modelVersion: "ep1-a0.5", modelSha256: null,
              referenceVersion: "ref-v1", eventSchema: 1 },
    session: { mode: "free", passage: { surah: 1, fromAyah: 1, toAyah: 7 }, policy: {}, startedAt: "2026-11-11T09:59:00.000Z", vad: false },
    audio: { format: "pcm_s16le", sampleRate: 16000, channels: 1, startMs, durationMs, sha256: sha256(wav),
             requested: { startMs, endMs: startMs + durationMs }, truncatedStart: false, truncatedEnd: false },
    incidents: [{ id: randomUUID(), kind: "false_warning", startMs: startMs + Math.floor(durationMs * 0.1), endMs: startMs + Math.floor(durationMs * 0.6), eventIds: ["e1"],
                  markedAt: { sessionMs: startMs + Math.floor(durationMs * 0.7), wallMs: 1_762_855_200_000 }, claim: { answer: "no", note: "I said it right" } }],
    observations: { events: [] },
    consent: { policy: "tilawa-report", version: "2026-11-report-1", send: true, training: false, publicDataset: false },
  };
  patch(r);
  return r;
}

export function reportForm(report: unknown, wav: Uint8Array, reportText = JSON.stringify(report)) {
  const form = new FormData();
  form.set("report", new Blob([reportText], { type: "application/json" }), "report.json");
  form.set("audio", new Blob([wav], { type: "audio/wav" }), "audio.wav");
  return form;
}
