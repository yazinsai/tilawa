# 04 The supervisor

Status: claimed
Type: task
Blocked by: 01, 02, 03

`tools/autolab.py`: the reconciling loop of spec decision 3.

## Scope

- **State:** `/lab/autolab/state.json` holds experiments, their states, timestamps, executor handles, spend and counters. `/lab/autolab/events.jsonl` gets one line per transition. Writes go through the same re-read, apply and write pattern as `lab.py`'s `update()`.
- **Tick:** for each experiment, advance it if its condition holds (executor finished, score ready, gate result in). Kill and mark `abandoned` past the state's timeout, saving the executor transcript to `.scratch/autolab/transcripts/<id>.md`. Start new work while there are free slots under the spend caps.
- **Validation** before scoring:
  - the branch builds and its tests pass on Modal (the `lab run --cmd` shard path);
  - the protected-path and literal checks from the gate pass, so a cheating diff fails fast;
  - the card is unchanged since it was accepted;
  - the test data is untouched. Executors hold a Modal token for test runs, and `/lab/data` on the volume is read-only by convention only. So the supervisor keeps a hash of every file under the current test set version's manifest, exclusion list and evidence. It re-checks them before each score, and on a mismatch it pauses everything and notifies the user.
- **Fold queue:** one fold at a time, under a lock in the state file. A rebase conflict is sent back to the executor once with the conflict text; a second failure means `superseded`.
- **Heartbeat:** `state.json.heartbeat` is updated every tick.
- **Commands:** `autolab.py tick` (one pass, for cron), `autolab.py loop --every 120` (phase A), `autolab.py pause|resume`, and `autolab.py status [--json]`.

## Acceptance

- Unit tests over the state machine with a fake executor, fake scoring and a fake gate. They cover each transition, a timeout, a crash between transitions (the next tick finishes the work, never runs it twice), a rebase conflict, and the spend cap.
- A dry run against the real ledger with a do-nothing executor walks one card from `proposed` to `discarded`.

## Comments

Built (1 Nov): `tools/autolab.py` (supervisor and commands), `tools/autolab_state.py` (the document, transitions, timeouts, fold lock, spend), `tools/test_autolab.py` (17 tests over a fake world) and `tools/test_autolab_portfolio.py` (state and portfolio tests).

- Every lab operation goes through `tools/lab.py --main autolab --ledger /autolab/ledger.json --json`. Two flags were added to `lab.py` for this: `fold --gate` merges only when `tools/lab_gate.py` passes on the rebased run, and `baseline --practice-only` keeps the shadow ledger exam-free.
- Long operations run as detached jobs under `.scratch/autolab/run/jobs/` (gitignored), started with `start_new_session`, so they outlive a supervisor restart. The state keeps each job's pid and output path, and the next tick polls it rather than starting it again (tested).
- Validation runs in this order:
  - commit any uncommitted executor work;
  - require at least one change;
  - check the card is unedited;
  - hygiene (protected paths, benchmark literals);
  - the tuning cap;
  - the GPT diff review;
  - the test-data hash (a mismatch pauses everything);
  - build and tests on Modal;
  - for listener cards, a decode of the practice set from the branch, under its own listener name, which is then written into the card for scoring.
- Hygiene, diff-review and test failures send the work back to the executor once with the reason; a second failure discards it.
- Scoring runs the branch practice score, then the gate as a prefilter, then the fold queue (one fold at a time under `foldLock`). A rebase conflict goes back to the executor once.
- Every tick writes a heartbeat. A tick that raises pauses autolab and sends a notification; it doesn't crash the loop.
- Practice scores for executors (spec decision 20) work through request files in `<worktree>/.autolab/`, which the supervisor serves in its own Modal environment, at most 3 per card (tested).
- Daily: the digest, a rebase of `autolab` onto `main` (a conflict pauses), and a practice re-score of the shadow line.
- Deviation: the journal lives on the volume (`/lab/autolab/journal.md`, mirrored in `.scratch/autolab/run/journal.md`) instead of being committed on `autolab`, which avoids a commit on every review.
- Not yet done: a dry run against the real ledger. It needs `CURSOR_API_KEY` and `init`; see `.scratch/autolab/setup.md`.
