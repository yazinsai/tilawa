# 11 Lab dashboard

Status: resolved
Type: task
Blocked by: 06, 10

Spec: `../spec.md` goal and decisions 1, 8, 9, 14. Modal web endpoint.

## Acceptance

- Header, always visible: the exam overall score on the main line, its missed and false-warning counts, the test set version, and the number of pending flags.
- Three component scores, plus attribution as one bar that adds up to the overall score. The teacher is shown as a ceiling.
- Experiments: component, expected, actual (component and overall, practice; exam after fold-in), verdict, status. Experiments waiting on your decision come first.
- Drill-down per component: failing practice cases, with likely bad data first (decision 14). Clicking a case opens the review item with buttons for bad data or a component fault.
- Release tab: the existing release gates.
- Plain language and counts, no jargon in labels.

## Comments

- Deployed: <https://amal-invest--tilawa-lab-review-web.us-east.modal.run/> (`modal deploy modal/lab_review.py`, the same app as the review page). The review page moved to `/review?queue=...`, and `/?queue=...` still opens it. The no-queue picker is at `/review`. Docs: the "Dashboard" section of `docs/benchmarks/lab-review.md`.
- What's on it:
  - **Header** (always visible): the exam overall score on phone audio (wrong calls per 5 min, big), "X missed + Y false warnings", the wrong-call count and recited minutes, the main-line commit, the test set version with its applied-flag count, the listening check, and "N flags pending" with `python3 tools/lab.py apply-flags` to copy.
    - The listening check takes the latest verdict per item across reviewers on `<base>/review/v1-check.json`. lab-v1.1 shows "passed 40/40".
    - Pending flags come from `lab_ledger.pending_flags`, so the count is exactly what `apply-flags` would apply.
  - **Score tab**: one stacked bar of the attribution split, which adds up to the overall score. Three clickable cards, each with the wrong calls charged to it (missed + false warnings), its own score in plain words and a one-line job description:
    - listener: "Mishears N of every 1,000 sounds";
    - tracker: time lost and wrong jumps on oracle evidence, with the real-evidence numbers under it;
    - checker: wrong calls with perfect hearing and the true place.

    Below the cards:
    - the teacher as "reference, not a target";
    - experiments, with "waiting for your decision" first: component, hypothesis, expected, measured on practice (and after rebase), exam after fold-in, a better/worse/noise verdict, state, a "main line moved since" note, and the fold, discard or run command to copy;
    - main-line history, newest first: each fold's change in exam wrong calls, with re-scores and new test set versions marked.
  - **Drill-down** (`#/cases/<component>`, with a phone/studio toggle): failing practice cases charged to that component. Likely bad data comes first: a failed data check, then teacher, student and perfect-hearing checker all failing, then failing in every open experiment's latest run. Each row shows chips and its latest verdict. Exam cases are never served.
  - **Case view** reuses `mountItem`:
    - the case's audio window, the Arabic said in it (from `<set>/sessions/<id>.json`), and the label (planted wrong, extra or left-out word with its Arabic, or "no mistake");
    - for planted edits, the original before and after;
    - a plain story of what the engine did (each arm: perfect hearing + true place, student's hearing + true place, teacher) and why the case is charged where it is;
    - six buttons: bad label, bad audio, bad splice, listener fault, tracker fault, checker fault.

    Keys: 1–6, space / b / a to play, j / k for next / back, Esc for the list.
  - **Release tab**: the release-readiness section of `docs/benchmarks/scoreboard.md` (13 gates, summary, not-measurable list), baked in at deploy.
- Flags from the drill-down:
  - A verdict is saved to `/lab/reviews/<base>/<reviewer>.json` under the scorer's case key, with `queue: "<base>/review/dashboard.json"`. The case is upserted into that queue with its review window, so `lab apply-flags` finds both.
  - `lab_ledger.session_of` now accepts `<session>/...` keys. It also matches the audio file name by `<session>.` prefix, because the old stem split broke on dotted ids (`lab-v1.1-p01.phone.wav` gave `lab-v1`).
  - Checked: a `bad_label` verdict saved from the page resolves to `lab-v1-p07` 133.44–140.28 s in `pending_flags`, and a fault verdict doesn't count.
- No ledger yet: the page says "No baseline yet", gives `python3 tools/lab.py baseline`, and links the 12 newest runs. `?run=<run>` shows any run's score.json (used for development: `?run=baseline-6643c64-dirty-20261028T131235Z`, exam 150 = 91 + 59, cards 74 / 3 / 73). `?ledger=<file>` reads a throwaway ledger at the volume root, as `lab.py --ledger` writes.
- Speed: small JSON endpoints (dashboard 2.5 KB, a case list 16–30 KB, a case 1 KB). Each file is parsed once per (mtime, size), and the volume is reloaded at most every 5 s. A warm request takes about 1 s from here, the same as a static file, so that's Modal's proxy round trip and not server work. A cold container parses the 760 KB score in about 1 s more.
- Checked end to end on the deployed app:
  - curl on every page and endpoint;
  - 400s for a bad component, variant, run name (`../x`), ledger name, reviewer or verdict, and 404s for an unknown run or case;
  - verdict upsert with history and an Arabic note kept verbatim, read back from the volume after commit;
  - audio windows for the session and for the original source;
  - the review page at `?queue=lab-v1.1/review/v1-check.json` (tally 40/40 pass).

  Browser screenshots covered the score tab, the drill-down, a case (keypress 6 saved and moved to 2/85), the release tab, the no-ledger state, and a throwaway ledger built from the real lab-v1 runs (waiting / open / superseded / folded experiments, 3 history rows).

  Test data was removed afterwards: the `curltest` reviews, `lab-v1/review/dashboard.json`, and `/ledger-dash-test.json`.
- Tests: `tools/test_lab_dashboard.py` (14) and a `session_of` test in `tools/test_lab_ledger.py`. Lab Python tests: 56 passed locally.
- Open:
  - The real `/lab/ledger.json` didn't exist yet, so the live default view shows "No baseline yet". It switches over by itself once `lab baseline` writes the ledger; no redeploy is needed.
  - A deploy from a branch without this change replaces the dashboard, because it's the same Modal app.
  - No auth, same as ticket 06.
  - Doesn't move a metric, so no scoreboard row.
