# 06 Listening review UI (data tab)

Status: resolved
Type: task
Blocked by: none

Spec: `../spec.md` decisions 13, 14, 16. Served as a Modal web endpoint; audio is read from the lab volume.

## Acceptance

- A review queue (for v1: 20 splices plus 20 random cases covering clean stretches, repeats and phone-degraded audio). Each item has playable audio around the case, the labelled text, and for splices the before/after source audio.
- Verdicts: OK, or bad data (label / audio / splice), plus a free-text note kept verbatim. Saved to the volume, not browser storage.
- A running tally against the acceptance bar (no more than 1 bad in 40).
- The same item view and flag buttons are reused by the dashboard drill-down (ticket 11).

## Comments

- Deployed: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/?queue=fixture-review/review/demo.json> (`modal deploy modal/lab_review.py`). With no `?queue=`, the page lists every `/lab/data/*/review/*.json`. Add `&reviewer=<name>` (default `me`). Format, verdict file and item-module API are in `docs/benchmarks/lab-review.md`.
- Load a queue: write it to the `tilawa-lab` volume as `/lab/data/<testSet>/review/<name>.json`, then open `?queue=<testSet>/review/<name>.json`. Ticket 03 should emit the v1 queue (20 splices plus 20 random cases) there. The page checks the queue and lists any problems instead of showing items.
- Queue schema (version 1):
  `{version: 1, testSet, maxBad?: 1, items: [{id, kind: 'splice'|'case', title?, audio: {path, startS, endS, title?}, before?: {...}, after?: {...}, text, label: {type, expected?, said?}, context?}]}`.
  - Paths are relative to `/lab/data`. `before` / `after` are the original, unedited source just before and just after the edit. `text` is the Arabic the labels say is recited in the window.
  - `label.type` is one of `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`, `repeat`, `restart`, `self_correction`, `clean`.
  - Item ids must be unique within a test set.
  - Audio can be WAV in any fixed-frame format, or anything ffmpeg reads (checked with an MP3). The server sends only the requested window, with range requests (a 3 s window is 96 KB of a 406 KB file).
- Verdicts are saved to `/lab/reviews/<testSet>/<reviewer>.json`, keyed by item id.
  - Each record is `{verdict, note, at, queue, history}`. Notes are kept verbatim, and a changed verdict or note moves the old one to `history`.
  - Verdicts: `ok`, `bad_label`, `bad_audio`, `bad_splice`. Only `bad_*` counts against the bar. `listener`, `tracker` and `checker` are accepted for the dashboard's component faults.
  - The header shows reviewed N of total, bad K, and "Pass / Fail / Passing so far · at most 1 bad in N".
- Saving: one container (`max_containers=1`) is the only writer. It writes the file under a lock, replies, and calls `vol.commit()` on a background thread. A save takes about 0.9 s round trip from here; an inline commit took 1–4 s per save. Modal also commits in the background and on container shutdown. Reloads (to see new queues) are skipped while a save is uncommitted.
- Reuse (ticket 11): `import { mountItem, VERDICTS, FAULTS, labelHtml, clipUrl } from '/static/item.js'`. Serve `modal/lab_web/` or call this app's `/audio` and `/api/verdict`.
- Demo set: `modal run modal/lab_review.py` writes `/lab/data/fixture-review/`. It has tone "words" standing in for Al-Fatiha 1:2–1:5, a spliced session with a wrong word, a repeat and a left-out word, a phone copy with noise, echo and a click, and a 5-item queue.
- Checked end to end with curl on the deployed app:
  - the page and `item.js`, the queue list, and the queue with its tally;
  - a WAV slice of the exact length; a 206 range response whose bytes match the full body; 416 for an unsatisfiable range; 400 for a path outside the data folder; 404 for a missing file;
  - verdict upsert with history and an Arabic note kept verbatim; 404 for an unknown item and 400 for an unknown verdict; the bar turns to fail at 2 bad;
  - the reviews file read back with `modal volume get` after commit.
  A browser run checked that space plays, 1 saves and moves on, a note plus 3 saves, `k` goes back with the note shown, and the Arabic is right to left. `tools/test_lab_review.py` has 31 tests for slicing, ranges, path safety, upsert, tally and queue validation. Test verdicts from `curltest` and `browsertest` are left in `/lab/reviews/fixture-review/`.
- Risk: there is no auth beyond the unguessable Modal URL, so anyone with the link can play the audio and write verdicts. Follow-up: Modal's `requires_proxy_auth=True` needs `Modal-Key` / `Modal-Secret` headers, which a browser address bar can't send. For the page that means a small token check in the app (a secret query parameter or cookie), and proxy auth for machine callers such as the dashboard backend.
- Doesn't move a metric, so no scoreboard row.
