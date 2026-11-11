# Reporting

`@tilawa/core/report` lets your readers tell you when the engine got something wrong: a warning that wasn't a mistake, a mistake it missed, or the wrong ayah on screen. A report is a short audio clip plus what the engine showed, in one portable artifact. You decide where it goes.

**Off by default.** Without a recorder, no audio is kept. Nothing is ever sent unless your app calls `sendReport`. The SDK makes no network request of its own for reporting.

## Report a moment

```ts
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
```

1. `createRecorder(session, options)` starts keeping recent audio, in memory, from the moment it is created. Create it right after starting the session.
2. `recorder.report(...)` cuts a clip around the moment and resolves with a `TilawaReport`.
3. `encodeReport(report)` gives you the artifact: `report.json` (text) and `audio.wav` (bytes).
4. `sendReport(report, { endpoint })` uploads it. Use your own endpoint, or see [sending to Tilawa](#sending-reports-to-tilawa).
5. `recorder.dispose()` drops every retained sample.

`source` is required: a short tag for your app (1 to 128 characters). `consent` is required to build a report; see [consent](#consent).

## Send a whole take

Some apps want the whole recording, for example a "send this recitation to help improve Tilawa" button. Use take mode:

```ts
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
```

`mark()` records a moment without building a report, synchronously, so it is safe to call from a tap during the take. `contribution()` then builds one report with the whole take and every marked moment. A contribution with no incidents is fine: clean recitation is useful too.

## Ring mode and take mode

| | Ring (default) | Take |
|---|---|---|
| Option | `retain: { beforeMs: 10_000, afterMs: 5_000 }` | `retain: { take: true, maxMs: 300_000 }` |
| Keeps | The last `beforeMs + afterMs + 30 s` of audio. Older audio is overwritten. | Everything from the start, up to `maxMs`. |
| Memory | About 1.4 MB with the defaults | Up to 9.6 MB for 5 minutes |
| `report()` | A clip `[t - beforeMs, t + afterMs]` | The same |
| `contribution()` | Whatever the ring still holds (`truncatedStart` is true once audio has been overwritten) | The whole take |
| Past the limit | Not applicable | Recording into the buffer stops at `maxMs`; reports set `truncatedEnd` |

Limits: `beforeMs + afterMs` and `maxMs` are each at most 300 seconds. A recorder holds at most 100 incidents. Memory is PCM16 at 16 kHz: 32 KB per second.

## Timing

Everything uses one clock: the [session clock](api.md#the-session-clock), milliseconds of 16 kHz audio delivered to the session. Engine events, audio offsets and incident spans all use it, so they line up without conversion.

- **The moment.** `report()` anchors the clip on, in order: explicit `startMs`/`endMs`; `atMs`; the span of the linked checker events' `timeMs` (`eventId`/`eventIds`); otherwise `session.now()`.
- **The clip** is `[start - beforeMs, end + afterMs]`, clamped at 0.
- **Waiting for the after-context.** If that audio hasn't arrived yet, `report()` waits for it. It resolves early when the session ends or the take buffer is full, and then `audio.truncatedEnd` is true.
- **Old audio.** If the start has already left the ring, the clip starts at the oldest retained sample and `audio.truncatedStart` is true.
- **Slow inference.** The recorder waits for the session's queued audio to be processed (`session.idle()`) before cutting, so the events for the clip are in the report even if inference lags behind capture.
- **Tap latency.** `markedAt` records when the reader acted: `{ sessionMs: session.now(), wallMs: Date.now() }`. The delay between hearing a moment and tapping isn't corrected. Clip context covers it.
- **The VAD.** With `listen(session, { vad: true })`, the session clock skips dropped silence. The clip is exactly what the engine heard, `session.vad` is recorded as true, and the wall-clock gaps are not reconstructed.
- **After the session ends.** The recorder keeps its buffer for `retainAfterEndMs` (default 10 minutes) so readers can review after stopping. Then it expires: later calls reject with `RecorderError` (`reason: "expired"`). `dispose()` ends it sooner.

## The artifact

A report is two files (the types below are abridged; the full ones are exported as `TilawaReport` and `Incident`):

- **`audio.wav`**: canonical 44-byte header, PCM16 little-endian, mono, 16000 Hz.
- **`report.json`**: UTF-8 JSON, at most 4 MiB.

```ts
interface TilawaReport {
  schema: "tilawa.report/1";
  reportId: string;            // uuid v4; the idempotency key for uploads
  sessionId: string;           // session.id
  createdAt: string;           // ISO wall clock
  source: string;              // your app tag, 1-128 chars
  contributor?: string;        // optional pseudonymous id, 1-64 chars of [A-Za-z0-9_-]
  engine: { sdkVersion; modelId; modelVersion; modelSha256: string | null; referenceVersion; eventSchema: 1; tier?; backend? };
  session: { mode: "free" | "memorization"; passage: { surah; fromAyah; toAyah } | null; policy; startedAt; vad: boolean };
  audio: {
    format: "pcm_s16le"; sampleRate: 16000; channels: 1;
    startMs: number;           // session-clock time of the first sample in audio.wav
    durationMs: number;
    sha256: string;            // of audio.wav
    requested: { startMs: number; endMs: number };
    truncatedStart: boolean;   // the request reached back past the retained audio
    truncatedEnd: boolean;     // the session ended, or the take filled, before the request's end
  };
  incidents: Incident[];       // 0..100
  observations: { events: EngineEvent[]; evidence?: PhonemeEvidence[] };
  consent: { policy: string; version: string; send: true; training: boolean; publicDataset: boolean };
}

interface Incident {
  id: string;                  // uuid
  kind: "warning" | "false_warning" | "missed_mistake" | "wrong_position" | "stumble" | "unflagged_sample" | "other";
  startMs: number; endMs: number;                    // session clock
  eventIds: string[];                                // linked checker ids; [] for missed mistakes and stumbles
  markedAt: { sessionMs: number; wallMs: number };
  claim: { question?: string; answer: "yes" | "no" | "unsure" | "skipped" | null; note?: string; text?: string };
}
```

- **Claims and observations stay apart.** `incidents[].claim` is what the reader said. `observations` is what the engine emitted: the `capabilities` event, the checker events in the clip or linked to an incident, and the position events in the clip. Nothing in a report is a verdict.
- **Incident kinds.** Use `warning` when your app showed a warning and asked about it; the answer goes in `claim`. Use `false_warning` when your UI only lets the reader say "this is wrong". `missed_mistake`: the reader says they made a mistake the engine didn't flag. `wrong_position`: the wrong place was shown. `stumble`: a bookmark tapped while reciting. `unflagged_sample`: a moment with no warning, asked about to balance the sample.
- **`claim.note`** is free text. **`claim.text`** is what the reader says they actually recited. Each is at most 2000 characters.
- **Evidence.** `evidence: true` in the recorder options adds the phoneme evidence for the clip. It needs `createTilawa({ recordEvidence: true })`, or there is no evidence to add. Evidence is dropped first if the JSON would exceed 4 MiB.
- **The audio rides alongside.** The report object that `report()`, `contribution()` and `decodeReport()` return has its WAV attached. A copy (spread, `structuredClone`, a JSON round trip) loses it, and `encodeReport` then throws a `TypeError`. `reportAudio(report)` returns the attached bytes.

`engine.modelId` is the model id (`spliced-clean`) and `engine.modelVersion` its registry version (`1-int8`, from `capabilities.modelRelease`; `"unknown"` for an injected model outside the registry). `engine.modelSha256` identifies the model exactly either way.

## Validation

`validateReport(json, wav)` checks an artifact and never throws. It resolves with `{ ok, errors, report? }`. The SDK, the tilawa.dev server and the Python tool `tools/report_validate.py` apply the same rules:

- `schema` is `tilawa.report/1` or `tilawa.report/1.<minor>`. Other majors are rejected.
- `reportId`, `sessionId` and incident ids are lowercase uuids. `createdAt` and `session.startedAt` are ISO timestamps.
- `source` is 1-128 characters; `contributor`, if present, is 1-64 characters of `[A-Za-z0-9_-]`.
- `audio.wav` has the canonical 44-byte header with fmt `(16, 1, 1, 16000, 32000, 2, 16)`, its length matches `durationMs` to within 1 ms, `durationMs` is more than 0 and at most 300 s, and its sha256 matches `audio.sha256`.
- Each incident has `startMs ≤ endMs`, and lies inside the audio, except on a truncated side.
- At most 100 incidents. `claim.note` and `claim.text` at most 2000 characters each. `report.json` at most 4 MiB.
- `consent.send` is `true`, and `training` and `publicDataset` are booleans.

The recorder validates every report it builds, and `sendReport` validates again before uploading. `decodeReport(json, wav)` validates and returns the report with its audio attached, so you can re-send a saved artifact.

## `sendReport`

```ts
const receipt = await sendReport(report, {
  endpoint: "https://reports.example.com/tilawa", // required; nothing is built in
  headers: {},                                    // optional, e.g. an ingestion key from your server
  retries: 3,                                     // default 3: on network errors, 429 and 5xx
  backoffMs: 500,                                 // default 500, doubled per retry
  signal,                                         // optional AbortSignal
});
// receipt: { reportId, receivedAt, deleteToken, duplicate, status }
```

- It POSTs `multipart/form-data` with two parts: `report` (`report.json`, `application/json`) and `audio` (`audio.wav`, `audio/wav`).
- Every retry sends the exact same bytes. Servers deduplicate on `reportId`.
- A `2xx` reply is success. `201` means stored; `200` means the server already had this report (`duplicate: true`). `409` (same id, different bytes, or a deleted id) and other `4xx` replies throw `ReportSendError` with `status` and `body`, without retrying.
- It reads `receivedAt` and `deleteToken` from a JSON reply, if present. Show the delete token to the reader once if your server issues one.
- It throws `ConsentError` if `consent.send` isn't true, and `ReportSendError` (with `status: null`) for an invalid report.

Failed uploads: keep the report object (or `encodeReport`'s output) and call `sendReport` again later. To survive a page reload, store `json` and `wav` yourself and `decodeReport` them.

## Running your own endpoint

The recommended setup: your app sends reports to your own server, and you decide what happens to them. Validate with the same validator:

```ts
import { randomBytes } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { validateReport } from "@tilawa/core/report";

createServer(async (req, res) => {
  if (req.method !== "POST" || req.url !== "/tilawa") return void res.writeHead(404).end();
  // Parse the multipart body with the platform's Request/FormData.
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  const form = await new Request("http://local/", {
    method: "POST",
    headers: { "content-type": req.headers["content-type"] ?? "" },
    body: Buffer.concat(chunks),
  }).formData();
  const report = form.get("report");
  const audio = form.get("audio");
  if (!(report instanceof Blob) || !(audio instanceof Blob)) return void res.writeHead(400).end();

  const json = new Uint8Array(await report.arrayBuffer());
  const wav = new Uint8Array(await audio.arrayBuffer());
  const result = await validateReport(json, wav);
  if (!result.ok || !result.report) {
    res.writeHead(400, { "content-type": "application/json" });
    return void res.end(JSON.stringify({ errors: result.errors }));
  }
  // A real server must also: check consent.version against the versions you accept, refuse a reportId it already
  // has with different bytes (409), rate-limit, and store privately.
  const dir = `reports/${result.report.reportId}`;
  await mkdir(dir, { recursive: true });
  await writeFile(`${dir}/report.json`, json);
  await writeFile(`${dir}/audio.wav`, wav);
  res.writeHead(201, { "content-type": "application/json" });
  res.end(JSON.stringify({
    reportId: result.report.reportId,
    receivedAt: new Date().toISOString(),
    deleteToken: randomBytes(32).toString("base64url"), // store only its hash
  }));
}).listen(8787);
```

A production endpoint should also:

- accept only the consent versions you have actually shown readers (`consent.policy` and `consent.version`),
- be idempotent on `reportId`: `201` the first time, `200` for byte-identical repeats, `409` for different bytes or a deleted id,
- limit request size (`report.json` 4 MiB, `audio.wav` about 9.6 MB) and rate,
- store reports privately, and support deletion (for example with a delete token whose hash you keep),
- from browsers, allow CORS only from your app's origins.

Python services can use `tools/report_validate.py`, which mirrors the TypeScript rules.

## Sending reports to Tilawa

If you would like your readers' reports to help improve the Tilawa model, there are two options:

1. **Share from your own endpoint.** Collect with your own server as above, then share reports with the Tilawa team when your readers' consent allows it.
2. **A scoped ingestion key.** The Tilawa team can issue a per-app key for `https://tilawa.dev/help/api/v2/reports`. It can only upload: it can't list, read or delete. Use it **only from your server**, never in browser code or a shipped app binary, where anyone could extract it. Send it as `headers: { Authorization: "Bearer <key>" }`. Ask at help@tilawa.dev.

The tilawa.dev endpoint only accepts consent versions on its allowlist. Which policy and version third-party apps must show their readers: {{PENDING: decision 4}}. Reports sent there are handled under the [Tilawa privacy policy](privacy.md).

## Consent

Every report carries the reader's consent, and the recorder refuses to build one without it (`ConsentError`):

```ts
const consent: ReportConsent = {
  policy: "my-app-report",   // which policy text the reader saw
  version: "2026-11-01",     // which version of it
  send: true,                // required: they agreed to send this report
  training: false,           // optional: they agreed to model training and testing
  publicDataset: false,      // optional: they agreed to publication in a public dataset
};
```

Pass it once in `createRecorder({ consent })` or per call in `report({ consent })` and `contribution({ consent })`. `training` and `publicDataset` default to `false`.

## Privacy guidance for app developers

Recitation audio is a voice recording, and voices identify people. Treat reports as personal data.

- **Ask first, plainly.** Before sending anything, tell the reader what is sent (a short recording and what the app showed), where it goes, who can hear it, how long it is kept, and how to delete it. Make sending its own choice, separate from any use for training or publication. Optional choices start unticked.
- **Record the consent you showed.** Put your policy name and version in `consent`. Change the version whenever the wording changes, and keep the old wording.
- **Keep the minimum.** Prefer clips (ring mode) to whole takes unless you need them. Don't put names, emails or account ids in `source`, `contributor`, `claim.note` or `claim.text`. If you use `contributor`, make it a random id, not a derived one.
- **Children.** Think about younger readers in your audience. The tilawa.dev contributor page asks people under 16 not to record unless a parent or guardian agrees.
- **Memory only, until sent.** The recorder keeps audio in memory. If you store unsent reports on the device, make it opt-in and give it an expiry.
- **Delete on request.** Give readers a way to delete what they sent, for example the delete token.
- **No secrets in clients.** Never put an ingestion key or admin token in browser code or a shipped app.
- **Your obligations.** You are responsible for meeting the privacy laws that apply to your app and your readers.
