# Autolab: going live (phase A)

Steps only you can do, in order. Run everything from the repo root. `AL` below stands for
`uv run --python cpython-3.12-macos-aarch64-none --with modal==1.3.4 --with cursor-sdk==1.2.0`.

## 1. Cursor key and model names

```bash
export CURSOR_API_KEY=cursor_...        # Cursor Dashboard → Integrations; put it in your shell profile
$AL python tools/autolab_cursor.py models
```

Copy the exact ids for Opus 5.5, Grok 4.7 and GPT-5.6 Sol into `.scratch/autolab/config.json`. `init` writes that file
with placeholder ids: `models.*`, `executor.model` and `executor.fallbackModel`.

## 2. Smoke test the executor

```bash
git worktree add /tmp/autolab-smoke main
$AL python tools/autolab_cursor.py smoke /tmp/autolab-smoke
git worktree remove --force /tmp/autolab-smoke
```

It should add a comment to a test file and report `finished`. This also confirms that the models accept `tools=[]` and
that a cancel stops a run cleanly (ticket 05).

## 3. Executor isolation (decision 14)

The default config sets the builders' `MODAL_ENVIRONMENT` to `autolab-exec`. Not built yet: the practice-only copy of
the test data for that environment. Until it exists, pick one:

- **Weaker, starts now.** Remove `MODAL_ENVIRONMENT` from `executor.env` in the config. Builders' Modal test runs then
  use your main environment. The data-hash check still pauses everything if `/lab/data` changes.
- **The real thing.** `modal environment create autolab-exec`, a token for it in `executor.env`
  (`"MODAL_TOKEN_ID": "$AUTOLAB_MODAL_TOKEN_ID"`, and the same for the secret), and the practice-data sync. Ask for
  that to be built (a small follow-up on ticket 05).

## 4. Notifications

Install the ntfy app, subscribe to a hard-to-guess topic, and set `notify.topic` (and `notify.dashboard`, the
dashboard URL) in the config.

## 5. Start

```bash
$AL python tools/autolab.py init        # autolab branch and ../tilawa-autolab worktree, practice baseline (~11 min)
$AL python tools/autolab.py loop        # leave running: caffeinate -i $AL python tools/autolab.py loop
```

Keep the watchdog apart from the loop, so a dead loop still pages you. Add this to cron (`crontab -e`):

```
*/10 * * * * cd /Users/rock/ai/projects/tilawa && /usr/bin/env CURSOR_API_KEY=... <AL> python tools/autolab.py watchdog
```

`init`'s baseline also closes ticket 03's last check: its practice counts should match the latest main-line row in
`docs/benchmarks/lab-ledger.json` for the same commit.

## 6. Deploy the dashboard tab

```bash
python3 tools/build_live_review.py && modal deploy modal/lab_review.py
```

Then open the Autolab tab.

## Day to day

- `tools/autolab.py status`, or the dashboard.
- `tools/autolab.py pause [why]` / `resume`.
- `tools/autolab.py match <id> agree|disagree` after reading a shadow decision. About 10 agreements in a row means
  folds can go straight to `main` (ticket 11).
- Take the shadow line's folds: `git merge --ff-only autolab` on `main`, right after the daily rebase, or cherry-pick
  single folds.
- Listening queues and spot checks still come to you, as now.
