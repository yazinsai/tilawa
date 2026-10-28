# Lab listening review

A Modal web page for listening to test set items one at a time and marking each as fine or bad data (`.scratch/lab/issues/06-review-ui.md`). It is used for the test set v1 check (ticket 07) and reused by the dashboard drill-down (ticket 11).

- Page: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/review?queue=fixture-review/review/demo.json> (`/?queue=...` still works; `/` without a queue is the dashboard)
- Deploy: `modal deploy modal/lab_review.py`. Rewrite the demo set: `modal run modal/lab_review.py`.
- Test: `uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_review.py`
- ASR pre-screen (`modal/review_asr.py::prescreen`) writes a smaller queue in this format. It failed calibration on lab-v1.1 (`lab-review-asr-v1.md`), so listening stays fully manual.

Open `?queue=<path under /lab/data>` and optionally `&reviewer=<name>` (default `me`). With no queue, the page lists every `/lab/data/*/review/*.json`. Keys: space plays, `b` / `a` play the original before / after, 1–4 give a verdict, `j` / `k` go next / back.

## Queue

A JSON file on the `tilawa-lab` volume, conventionally `/lab/data/<testSet>/review/<name>.json`. All paths are relative to `/lab/data`; times are seconds in that file.

```json
{
  "version": 1,
  "testSet": "v1",
  "maxBad": 1,
  "items": [
    {
      "id": "s07-e3",
      "kind": "splice",
      "title": "Wrong word planted",
      "audio": { "path": "v1/sessions/s07-phone.wav", "startS": 61.2, "endS": 67.0 },
      "before": { "path": "v1/sources/r12-001.wav", "startS": 40.1, "endS": 43.0 },
      "after": { "path": "v1/sources/r12-001.wav", "startS": 43.0, "endS": 46.2 },
      "text": "رَبِّ الْأَوَّلِينَ الرَّحْمَٰنِ",
      "label": { "type": "wrong_word", "expected": "الْعَالَمِينَ", "said": "الْأَوَّلِينَ" },
      "context": "Al-Fatiha 1:2, word 3 · session s07 at 63.0 s"
    }
  ]
}
```

- `kind`: `splice` (a planted edit) or `case` (any other stretch: clean, repeat, phone copy, an engine failure).
- `audio`: the window to judge. `before` / `after` (optional, splices): the original, unedited source recording just before and just after the edit. Any clip can carry a `title` to rename its button.
- `text`: the Arabic words the labels say are recited in the window.
- `label.type`: `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`, `repeat`, `restart`, `self_correction` or `clean`. `expected` is the reference word(s) and `said` what was planted. It is shown as, for example, "Planted: said X instead of Y".
- `maxBad` (optional, default 1): the acceptance bar, at most this many bad verdicts in the whole queue.
- `title` and `context` are optional. Item ids must be unique within a test set, because verdicts are keyed by test set and id.

Audio is WAV (any fixed-frame format) or anything ffmpeg reads, such as MP3. The server sends only the requested window as a WAV, with range requests.

## Verdicts

`/lab/reviews/<testSet>/<reviewer>.json`:

```json
{ "version": 1, "testSet": "v1", "reviewer": "me", "items": {
  "s07-e3": { "verdict": "bad_splice", "note": "click at the join", "at": "2026-10-28T07:13:03+00:00",
              "queue": "v1/review/check.json", "history": [{ "verdict": "ok", "note": "", "at": "2026-10-28T07:12:53+00:00" }] } } }
```

A verdict is one of `ok` (Sounds right), `bad_label`, `bad_audio` or `bad_splice`. The dashboard adds component faults: `listener`, `tracker`, `checker`. Only the three `bad_*` verdicts count against the bar. Choosing again replaces the verdict, and the earlier verdict and note move to `history`. Notes are kept exactly as typed.

API: `GET /api/queue?path=&reviewer=` returns `{queue, reviews, tally}`, `POST /api/verdict {queue, reviewer, id, verdict, note}` returns `{review, tally}`, and `GET /audio?path=&start=&end=` returns the audio window. `tools/lab_review_core.py` holds the slicing, validation, upsert and tally logic.

## Reusing the item view

```js
import { mountItem, VERDICTS, FAULTS } from '/static/item.js';
const view = mountItem(el, item, { review, verdicts: [...VERDICTS, ...FAULTS], onVerdict: (verdict, note) => save(verdict, note) });
```

`onVerdict` returns a promise of the saved record. The view exposes `play('audio' | 'before' | 'after')`, `choose(index)`, `note()`, `setReview(record)` and `destroy()`. `labelHtml(label)` and `clipUrl(clip)` are also exported.

There is no login, so anyone with the URL can listen to the audio and save verdicts.

## Dashboard

`/` on the same app (`.scratch/lab/issues/11-dashboard.md`). It reads `/lab/ledger.json` and the merged `score.json` of the latest main-line run on the current test set. Every file is parsed once per modification time, and the volume is reloaded at most every 5 s.

- **Header**: the exam overall score on phone audio, its missed and false-warning counts, the test set version, the listening check (latest verdict per item across reviewers on `<base>/review/v1-check.json`), and pending flags. Pending flags are exactly what `lab apply-flags` would apply next.
- **Score**: the attribution bar, three component cards (the wrong calls charged to each, plus its own score with perfect upstream inputs), the teacher reference, experiments with "waiting for your decision" first and the fold or discard command to copy, and the main-line history.
- **Drill-down** (`#/cases/<component>[/studio]`): failing practice cases charged to that component, likely bad data first. Order: more reasons first, then a failed data check, then teacher + student + perfect-hearing checker all failing, then failing in every open experiment. Exam cases are never shown. A case opens the item view with bad-data and fault buttons (keys 1–6).
- **Release**: the release-readiness gates from `docs/benchmarks/scoreboard.md`, as of the deploy.
- No ledger: the page says so, gives `python3 tools/lab.py baseline`, and links recent runs. `?run=<run>` shows any run's `score.json`, and `?ledger=<file>` reads a throwaway ledger at the volume root (as `lab.py --ledger` writes).

A drill-down verdict is keyed by the scorer's case key. It is saved to `/lab/reviews/<base>/<reviewer>.json` with `queue: "<base>/review/dashboard.json"`. The case is also upserted into that queue file, carrying its review window, so `lab apply-flags` finds the window and the session (`session_of` matches `<session>/...` keys).

API: `GET /api/dashboard`, `GET /api/cases?component=&variant=phone|studio`, `GET /api/case?key=&reviewer=`, `POST /api/case-verdict {key, verdict, note, reviewer}`, `GET /api/release`; all take `run` and `ledger`. Logic: `tools/lab_dashboard_core.py`; tests: `uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_dashboard.py`.
