# Help collection service (`/help/api/v2`)

Runbook for the private report collection service (contract §6–7). The code is
in `web/frontend/server/help/v2.ts`, the validator in
`web/frontend/server/help/report-validate.ts`, and the wiring and headers in
`web/frontend/server/app.ts`. It runs in the Dokku app `tilawa`, with
persistent storage mounted at `/app/storage`.

## Routes

| Method + path | Who | Notes |
|---|---|---|
| `POST /help/api/v2/reports` | browser on an allowlisted origin, or `Authorization: Bearer <ingest key>` | multipart `report` (JSON) + `audio` (WAV). Returns `201 {reportId, receivedAt, deleteToken}`. A byte-identical retry returns `200` with the same `reportId`/`receivedAt` and a fresh `deleteToken`; older tokens keep working. The same id with different bytes returns `409`, and so does an id that was deleted. Other errors: `400 {errors:[codes]}`, `401` bad key, `403` origin not allowed, `413` too large, `429` rate limited (`Retry-After`), `507` disk nearly full. |
| `DELETE /help/api/v2/reports/:id` | `X-Delete-Token`, or admin | Soft delete: moves the report to `deleted/<id>`. Missing token `401`, wrong token `403`, unknown id `404`. Repeating a delete returns `200`. |
| `GET /help/api/v2/reports[?include=deleted]` | admin | Report list, folded from `index.jsonl`. |
| `GET /help/api/v2/reports/:id` | admin | `report.json` exactly as received. |
| `GET /help/api/v2/reports/:id/audio` | admin | `audio.wav` (`audio/wav`). |
| `POST /help/api/v2/admin/purge[?retentionDays=N]` | admin | Purges deleted reports older than the retention window. Returns `{purged:[ids]}`. |
| `GET /help/api/v2/legacy/submissions` | admin | Read-only copy of the v1 `submissions.jsonl`, byte for byte. Each row keeps its original `consent` version. |
| `OPTIONS` on the two `reports` paths | anyone | CORS preflight. Allowlisted origins only. POST is allowed on `/reports` and DELETE on `/reports/:id`. No credentials. |
| `GET/HEAD /v1/*` | anyone | Immutable assets from `ASSETS_V1_DIR`: `Cache-Control: public, max-age=31536000, immutable`, `Access-Control-Allow-Origin: *`, CORP `cross-origin`, single-range `Range` support. |

"Admin" means `Authorization: Bearer $HELP_ADMIN_TOKEN` or the signed `/admin`
session cookie (from `ADMIN_PASSWORD`). Anonymous reads get `401`. Nothing on
v1 changes: `/help/api/submit`, `/subscribe`, `/export`, `/subscribers` and
`/audio/:name` behave exactly as before, and `submissions.jsonl` and
`subscribers.jsonl` are only ever appended to by v1.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `HELP_REPORT_CONSENT_VERSIONS` | *(empty)* | Comma-separated list of accepted `consent.version` values. **Empty rejects every report**, so collection stays off until the consent wording is decided (decision 4 proposes `2026-11-report-1`). |
| `HELP_ALLOWED_ORIGINS` | `https://tilawa.dev` | Comma-separated browser origins allowed to POST without a key, and allowed CORS. Add `https://www.tilawa.dev` here if that host serves `/help`. |
| `HELP_INGEST_KEY_HASHES` | *(empty)* | Comma-separated `appId=<sha256 hex of key>` entries. A bare hash gets the app id `key-<hash[:8]>`. |
| `HELP_REPORTS_PER_HOUR` | `20` | Token bucket per IP hash: burst size, refilled evenly over an hour. In memory, so it resets on restart. |
| `HELP_DELETE_RETENTION_DAYS` | `30` | How long a soft-deleted report keeps its audio before `purge` may drop it. |
| `HELP_MIN_FREE_BYTES` | `1073741824` | New reports get `507` when the storage filesystem has less free space than this. |
| `HELP_V2_DIR` | `$HELP_DATA_DIR/v2` | Storage root. `HELP_DATA_DIR` defaults to `./storage/help`, which is `/app/storage/help` in the container. |
| `HELP_ISOLATION` | unset | `1` adds COOP `same-origin` + COEP `require-corp` to `/help*` (needed for threaded WASM inference under the V1 SDK). `/help` self-hosts its fonts and bundles the SDK, so nothing on the page is cross-origin except the model and runtime under `assetBase` (`/v1/`, served with CORP `cross-origin`). |
| `ASSETS_V1_DIR` | `./dist/v1` | Directory served at `/v1/`. Fill it at build or deploy time. Never commit model files. |
| `IP_SALT` | random per boot | Salt for IP hashes, shared by the v1 and v2 rate limiters. v2 never stores IP hashes. |
| `HELP_ADMIN_TOKEN`, `ADMIN_PASSWORD` | unset | Admin bearer and `/admin` password. With neither set, every admin route is closed. |

## Storage layout

```
/app/storage/help/                 # v1, untouched by v2
  submissions.jsonl  subscribers.jsonl  audio/<id>.<ext>
/app/storage/help/v2/
  index.jsonl                      # append-only events: received | deleted | purged
  reports/<reportId>/report.json   # exactly the bytes the client sent
  reports/<reportId>/audio.wav     # canonical PCM16 mono 16 kHz
  reports/<reportId>/receipt.json  # receivedAt, sha256 of json + wav, sha256 of delete tokens, source, ingest
  deleted/<reportId>/...           # same files plus deleted.json {deletedAt, by, purgedAt?}
  tmp/                             # staging for atomic writes; emptied on boot
```

Each write stages all three files in `tmp/`, fsyncs them, and then renames the
directory into `reports/` in one step, so a crash never leaves a half-written
report. The index row comes after the rename. If the process dies between the
two, the next boot re-adds the missing row from `receipt.json`. A per-process
lock serializes commits. Only one app process may write to the store, so do not
scale `web` above 1 without replacing that lock.

Delete tokens are 32 random bytes (base64url), shown to the contributor once.
Only their sha256 is stored. The v2 store holds no IP, user agent or email.
`source` in the index is the ingest key's app id for keyed uploads, and the
report's own `source` tag for browser uploads (the latter is kept as
`claimedSource`).

## Backup

The host disk is about 84% full, so write the backup somewhere other than the
storage volume, and preferably off the host:

```bash
# on the Dokku host
sudo tar -C /var/lib/dokku/data/storage/offline-tarteel -czf - help \
  | ssh backup-host "cat > tilawa-help-$(date -u +%Y%m%dT%H%M%SZ).tgz"
# or locally:
ssh dokku-host "sudo tar -C /var/lib/dokku/data/storage/offline-tarteel -czf - help" \
  > tilawa-help-$(date -u +%Y%m%dT%H%M%SZ).tgz
```

Check a backup with `tar -tzf <file> | grep -c 'v2/reports/.*/audio.wav'` and
`tar -xzOf <file> help/submissions.jsonl | wc -l`. Take one before every deploy
that touches the server.

## Retention and deletion

1. Contributor deletion: the client sends `DELETE /help/api/v2/reports/<id>`
   with `X-Delete-Token`. Requests by email (help@tilawa.dev) are handled by an
   admin with the bearer token: `curl -X DELETE -H "Authorization: Bearer $T" https://tilawa.dev/help/api/v2/reports/<id>`.
2. A deleted report disappears from every admin read at once, and the same id
   cannot be uploaded again.
3. Purge is run by an admin; there is no cron. Run it weekly or so:
   `curl -X POST -H "Authorization: Bearer $T" https://tilawa.dev/help/api/v2/admin/purge`.
   It removes `audio.wav` and `report.json` of reports deleted more than
   `HELP_DELETE_RETENTION_DAYS` ago. `receipt.json` (hashes only) and
   `deleted.json` stay as a tombstone. Pass `?retentionDays=0` to purge every
   deleted report immediately.
4. Backups still hold deleted audio. Rotate backups within the same window, or
   record purged ids (`grep '"event":"purged"' index.jsonl`) to apply on restore.
5. The 24-month limit for unreviewed recordings (proposed in decision 4) is
   not automated yet. Use `GET /reports` with each row's `receivedAt` and admin
   DELETE.

## Rotating the admin token

```bash
NEW=$(openssl rand -base64 32 | tr -d '=+/')
dokku config:set tilawa HELP_ADMIN_TOKEN="$NEW"   # restarts the app
```

The old token stops working on restart. `scripts/help-dashboard.mjs` reads the
token from `dokku config:get` unless `HELP_ADMIN_TOKEN` is set locally, so it
picks up the new one. Update any other script that has a copy (lab data pulls). To rotate `ADMIN_PASSWORD`,
set it the same way. Every `/admin` session is signed with the password, so all
sessions end too.

## What /help sends

The page lives in `web/frontend/help/` (a second Vite page, built to `dist/help/index.html` with its scripts,
styles and fonts under `dist/assets/`). Each take becomes one `contribution()` report: the whole take as
`audio.wav` (capped at 5 minutes, decision 5), the engine's own events in `observations`, and the reader's
answers as `incidents`. Consent policy `tilawa-help-report`, version `2026-11-report-1` (decision 4).

After the reader stops, the page asks about at most 5 moments, in the order they happened: confirmed warnings,
then the reader's "I stumbled here" bookmarks, then suspected or uncertain warnings, then places where we lost or
moved the reader's position. One or two random regions with no warning and no bookmark are always added as the
bias sample. The reader can also mark more moments while listening back. Each answer becomes one incident:

| Incident `kind` | Where it comes from | `claim.question` | What `claim.answer` means |
|---|---|---|---|
| `warning` | a warning we raised (`eventIds` = its checker ids) | `made_mistake` | `no` = the reader says the warning was false; `yes` = they agree they made a mistake |
| `stumble` | a bookmark tapped while reciting (`markedAt` = the tap) | `made_mistake` | as above; `null` with no question when the review never reached it |
| `wrong_position` | a position wobble, or "We showed the wrong ayah" on any moment | `wrong_ayah` | `yes` = we showed the wrong ayah |
| `unflagged_sample` | a random region with no warning or bookmark | `reading_correct` | `yes` = the reader says they read it correctly |
| `missed_mistake` | a moment the reader marked while listening back | `made_mistake` | as above |

`skipped` and `unsure` are recorded as given. `claim.text` / `claim.note` are the optional "what I actually
recited" and free-text boxes. A warning is never treated as the truth about itself, unflagged audio is never
labelled clean, and nothing in the report is inferred: everything in `incidents` is something the reader did
or said. Incident spans are the clip the reader heard (6 s before to 2 s after the moment, 8 s for samples).

Failed uploads keep the encoded report in memory and retry with the same bytes. Retrying after a refresh only
works if the reader ticks "Keep it on this device for 7 days", which stores the encoded report in IndexedDB
(`tilawa-help` / `unsent`). It is deleted after a successful send, on discard, or when it expires. The
success screen shows the delete code once as `<reportId>.<deleteToken>` and offers an immediate delete.

Tests: `npm test` (selection and mapping unit tests in `test/help/`) and `npm run test:help-e2e` (Playwright,
`?mock=1`, real server with a temp store; screenshots and axe results in `test/help-screens/`).

## Third-party apps

Contract §7 gives two options. Neither puts a secret in browser code.

1. **Their own endpoint (recommended).** The app runs its own collector with
   the same validator (`validateReport` from `@tilawa/core/report`, or
   `tools/report_validate.py`), and the SDK's `sendReport` points at it. We
   never see the data unless they share it.
2. **A scoped ingestion key for this endpoint.** The key can only `POST
   /reports`. It cannot list, read or delete. Use it from the app's server,
   never from a browser or a shipped mobile binary.
   ```bash
   KEY=$(openssl rand -base64 32 | tr -d '=+/')
   HASH=$(printf %s "$KEY" | shasum -a 256 | cut -d' ' -f1)
   dokku config:set tilawa HELP_INGEST_KEY_HASHES="quran-app=$HASH,other-app=<hash>"
   # give $KEY to the partner over a secure channel; we keep only $HASH
   ```
   Uploads with `Authorization: Bearer $KEY` are recorded with `source:
   "quran-app"`. To revoke a key, remove its entry and restart. The per-IP rate
   limit applies to keys too, so a partner sending from one server IP may need
   a higher `HELP_REPORTS_PER_HOUR`, or a per-key limit added later.

Browser uploads from `HELP_ALLOWED_ORIGINS` (tilawa.dev) need no key. The
browser's `Origin` header gates them. Non-browser clients can forge that
header, so the rate limit, the validator, consent and the disk guard are the
real protections.
