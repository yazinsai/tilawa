import { beforeAll, describe, expect, it, vi } from "vitest";
import { ConsentError, createRecorder, ReportSendError, sendReport } from "../../src/report/index.js";
import type { TilawaReport } from "../../src/report/index.js";
import { CONSENT, FakeSession } from "./fake-session.js";

let report: TilawaReport;
const ENDPOINT = "https://example.test/help/api/v2/reports";

beforeAll(async () => {
  const s = new FakeSession();
  const rec = createRecorder(s, { source: "test", consent: CONSENT });
  s.pushMs(8000);
  report = await rec.report({ kind: "other", atMs: 2000 });
});

const reply = (status: number, body?: unknown) => new Response(body === undefined ? null : JSON.stringify(body), { status });

describe("sendReport", () => {
  it("posts multipart report + audio and returns the receipt", async () => {
    const fetch = vi.fn(async (_url: string, init: RequestInit) => {
      const form = init.body as FormData;
      const json = form.get("report") as File;
      const audio = form.get("audio") as File;
      expect(json.name).toBe("report.json");
      expect(JSON.parse(await json.text()).reportId).toBe(report.reportId);
      expect(audio.type).toBe("audio/wav");
      expect(audio.size).toBe(44 + report.audio.durationMs * 32);
      expect(init.headers).toEqual({ "x-ingest-key": "k" });
      return reply(201, { reportId: report.reportId, receivedAt: "2026-11-11T12:00:00Z", deleteToken: "tok" });
    });
    const receipt = await sendReport(report, { endpoint: ENDPOINT, fetch: fetch as never, headers: { "x-ingest-key": "k" } });
    expect(receipt).toEqual({ reportId: report.reportId, receivedAt: "2026-11-11T12:00:00Z", deleteToken: "tok", duplicate: false, status: 201 });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe(ENDPOINT);
  });

  it("retries network errors and 5xx with backoff", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const fetch = vi
        .fn()
        .mockRejectedValueOnce(new TypeError("offline"))
        .mockResolvedValueOnce(reply(503))
        .mockResolvedValueOnce(reply(200, { reportId: report.reportId, receivedAt: "t", deleteToken: "d" }));
      const p = sendReport(report, { endpoint: ENDPOINT, fetch, backoffMs: 100 });
      while (fetch.mock.calls.length === 0) await new Promise((r) => setImmediate(r)); // validation hashes first
      await vi.advanceTimersByTimeAsync(99);
      expect(fetch).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(fetch).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(200);
      expect((await p).deleteToken).toBe("d");
      expect(fetch).toHaveBeenCalledTimes(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("treats 409 (different payload or deleted id) as an error, without retrying", async () => {
    const fetch = vi.fn().mockResolvedValue(reply(409));
    await expect(sendReport(report, { endpoint: ENDPOINT, fetch })).rejects.toMatchObject({ status: 409 });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("treats a 200 retry reply as a duplicate success", async () => {
    const receipt = await sendReport(report, { endpoint: ENDPOINT, fetch: vi.fn().mockResolvedValue(reply(200)) });
    expect(receipt).toMatchObject({ reportId: report.reportId, duplicate: true, status: 200 });
  });

  it("fails fast on other 4xx and gives up after the retries", async () => {
    const bad = vi.fn().mockResolvedValue(new Response("nope", { status: 400 }));
    await expect(sendReport(report, { endpoint: ENDPOINT, fetch: bad })).rejects.toMatchObject({ name: "ReportSendError", status: 400, body: "nope" });
    expect(bad).toHaveBeenCalledTimes(1);
    const down = vi.fn().mockImplementation(async () => reply(500));
    await expect(sendReport(report, { endpoint: ENDPOINT, fetch: down, retries: 2, backoffMs: 0 })).rejects.toBeInstanceOf(ReportSendError);
    expect(down).toHaveBeenCalledTimes(3);
  });

  it("stops on abort and refuses reports without consent or audio", async () => {
    const ctrl = new AbortController();
    ctrl.abort();
    const fetch = vi.fn();
    await expect(sendReport(report, { endpoint: ENDPOINT, fetch, signal: ctrl.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(fetch).not.toHaveBeenCalled();
    const noConsent = { ...report, consent: { ...report.consent, send: false } } as unknown as TilawaReport;
    await expect(sendReport(noConsent, { endpoint: ENDPOINT, fetch })).rejects.toBeInstanceOf(ConsentError);
    await expect(sendReport({ ...report }, { endpoint: ENDPOINT, fetch })).rejects.toThrow(/no audio attached/);
    await expect(sendReport(report, {} as never)).rejects.toThrow(/endpoint/);
  });
});
