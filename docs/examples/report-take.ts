// docs/reporting.md: "Send a whole take"
import { createRecorder, sendReport } from "@tilawa/core/report";
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa();
const session = tilawa.startFreeRecitation();
const recorder = createRecorder(session, {
  retain: { take: true, maxMs: 300_000 }, // keep the whole take, up to 5 minutes
  source: "my-app",
  contributor: "c_7f3a91", // optional pseudonymous id; never a name or an email
});
const mic = await tilawa.listen(session);

// During the take: the reader taps "I stumbled here". Synchronous; uses the session clock.
startButton.onclick = () => recorder.mark({ kind: "stumble" });

stopButton.onclick = async () => {
  await mic.stop();
  const report = await recorder.contribution({
    incidents: recorder.incidents(),
    consent: { policy: "my-app-report", version: "2026-11-01", send: true, training: true, publicDataset: false },
  });
  await sendReport(report, { endpoint: "https://reports.example.com/tilawa" });
  recorder.dispose();
};
