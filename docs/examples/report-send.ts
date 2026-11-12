// docs/reporting.md: "sendReport" and "Consent"
import { createRecorder, decodeReport, encodeReport, sendReport } from "@tilawa/core/report";
import type { ReportConsent } from "@tilawa/core/report";
import { createTilawa } from "@tilawa/core/web";

const consent: ReportConsent = {
  policy: "my-app-report",   // which policy text the reader saw
  version: "2026-11-01",     // which version of it
  send: true,                // required: they agreed to send this report
  training: false,           // optional: they agreed to model training and testing
  publicDataset: false,      // optional: they agreed to publication in a public dataset
};

const tilawa = await createTilawa();
const session = tilawa.startFreeRecitation();
const recorder = createRecorder(session, {
  source: "my-app",
  consent,
  // Known issue in 1.0.0-rc.0: record the model version explicitly.
  engine: { modelVersion: tilawa.model.version ?? undefined },
});
const report = await recorder.report({ kind: "missed_mistake", claim: { answer: "yes", note: "I skipped a word" } });

const signal = AbortSignal.timeout(60_000);
const receipt = await sendReport(report, {
  endpoint: "https://reports.example.com/tilawa", // required; nothing is built in
  headers: {},                                    // optional, e.g. an ingestion key from your server
  retries: 3,                                     // default 3: on network errors, 429 and 5xx
  backoffMs: 500,                                 // default 500, doubled per retry
  signal,                                         // optional AbortSignal
});
// receipt: { reportId, receivedAt, deleteToken, duplicate, status }
console.log(receipt.reportId, receipt.deleteToken);

// Save an unsent report, and send it after a reload.
const { json, wav } = encodeReport(report);
const again = await decodeReport(json, wav);
await sendReport(again, { endpoint: "https://reports.example.com/tilawa" });
