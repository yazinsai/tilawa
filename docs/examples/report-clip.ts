// docs/reporting.md: "Report a moment"
import type { CheckerEvent } from "@tilawa/core";
import { createRecorder, encodeReport, sendReport } from "@tilawa/core/report";
import type { ReportConsent } from "@tilawa/core/report";
import { createTilawa } from "@tilawa/core/web";

// What the reader agreed to, in your app's own consent screen.
const consent: ReportConsent = {
  policy: "my-app-report",
  version: "2026-11-01",
  send: true, // required: the reader agreed to send this
  training: false,
  publicDataset: false,
};

const tilawa = await createTilawa();
const session = tilawa.startFreeRecitation();
// Off by default. With a recorder, the last ~45 s of audio is kept in memory and nothing leaves the device.
const recorder = createRecorder(session, { source: "my-app", consent });

let lastWarning: CheckerEvent | null = null;
session.subscribe((e) => {
  if (e.type === "checker" && e.state === "confirmed") lastWarning = e;
});
const mic = await tilawa.listen(session);

// The reader taps "That's not a mistake" under a warning:
async function disagree(warning: CheckerEvent) {
  const report = await recorder.report({ kind: "false_warning", eventId: warning.id, claim: { answer: "no" } });
  const { json, wav } = encodeReport(report); // report.json + audio.wav, e.g. to save or attach
  console.log(json.length, wav.length);
  const receipt = await sendReport(report, { endpoint: "https://reports.example.com/tilawa" });
  console.log("stored", receipt.reportId);
}

stopButton.onclick = async () => {
  // report() waits for 5 s of audio after the moment, or for the session to end (then truncatedEnd is true).
  const pending = lastWarning ? disagree(lastWarning) : null;
  await mic.stop();
  await pending;
  recorder.dispose(); // drops every retained sample
  await tilawa.dispose();
};
