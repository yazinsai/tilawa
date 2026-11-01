# 07 Autolab tab and notifications

Status: ready-for-agent
Type: task
Blocked by: 04

The Autolab tab of spec decision 13 on the existing Modal dashboard, plus the notification channel of decision 12.

## Scope

- `GET /api/autolab` in `modal/lab_review.py` reads `state.json` and the tail of `events.jsonl` from the volume. The tab in `modal/lab_web/` shows:
  - agents running now;
  - the queue;
  - recent verdicts, with each gate reason in plain words;
  - each component's practice trend;
  - the portfolio, the coverage map and spend;
  - the heartbeat, red after 3 missed ticks.
- **Notifications:** pick one channel with the user (ntfy is the least setup). Send the daily digest (folds, discards, trend, spend, the listening queue length), `needs-human`, a stale heartbeat, spend caps and paused folding.

## Acceptance

- The tab renders from a recorded `state.json` fixture and from the live volume.
- Each notification kind fires once in a test with a fake sender.

## Comments

**1 Nov 2026, dashboard half built; notifications are separate (`tools/autolab_notify.py`).**

- `tools/autolab_dashboard_core.py`: pure `view(state, events, now, mainline)`. It returns the heartbeat (stale after 3 × `tickSeconds`), the main line's practice brief, needs-human, running and queued experiments (with state, age, the state's time limit and the executor's model), and recent verdicts. Each verdict has every gate check in plain words, the practice change after rebase or on its branch, the exam tripwire word, and the journal anchor `.scratch/autolab/journal.md#<id>` as text. It also returns the per-part practice trend over the autolab ledger's main line (current test set only), the portfolio by part and tier, coverage, spend against `state.caps`, and the last 30 events. Fields are picked by name and `scrub` drops any `exam*` key at any depth (except `examTripwire`), so no exam number can reach the page. `state` None gives the "hasn't started" view.
- `tools/test_autolab_dashboard.py` (11 tests) runs over `tools/fixtures/autolab-state.json`, `autolab-events.jsonl` and `autolab-ledger.json`. The fixture has 8 experiments in every kind of state, a gate fail with three blocking reasons, a fold, a needs-human, a spend cap reached and exam sentinels (777xxx) in the ledger. The tests cover a stale heartbeat (exactly 3 ticks is still fine) and check that no exam number shows up in the output.
- `modal/lab_review.py`: `GET /autolab` (page) and `GET /api/autolab[?root=<folder>]`. The API reads `/lab/autolab/state.json`, the last 500 lines of `events.jsonl` (from at most the last 1 MB) and `ledger.json`'s mainLine. Missing or malformed files give the empty view. `root` is a throwaway folder at the volume root for previews, the same pattern as the dashboard's `ledger`. `autolab_state.py` and `autolab_dashboard_core.py` are added to the image.
- `modal/lab_web/autolab.html|js` with a small `/* Autolab page */` block in `lab.css`. The page polls every 30 s (and again when the tab becomes visible) and keeps open disclosures open across refreshes. The heartbeat sits in the top bar and turns red with an error notice when stale; pause and spend-cap notices are gold. The dashboard's top bar gains an Autolab tab.
- Verified: the module imports; the real `web()` body ran locally over temp `/lab` and `/root` with FastAPI's TestClient (empty view, fixture view, bad `root` → 400, no exam number); the page rendered in a browser for fresh, stale and not-started states. The preview app (`tilawa-lab-review-preview`) shows the empty view from the live volume (no `/lab/autolab` yet), and showed the full fixture view from a throwaway `/autolab-preview-fixture` folder, deleted afterwards. The live app was not deployed.
- Matches `tools/autolab.py` as of 1 Nov: `caps`, `executor.model`, `results.gate`, `results.branchRun.compare` and `results.fold.{compare,gate,why}`. One gap: the slot count lives only in the supervisor's config. Writing `slots` into `state.json` (next to `caps`) shows "4 slots in use of 6"; until then the page shows "4 slots in use".
