# 10 Experiment cards, ledger and fold-in

Status: resolved
Type: task
Blocked by: 08, 09

Spec: `../spec.md` decisions 10–13.

## Acceptance

- Card (in the worktree): component, hypothesis, expected change in counts for the component score and the overall score, listener (default main line).
- `lab new`, `lab run`, `lab status`: each experiment's practice result against the main line, with a verdict of better, worse or noise (a change of 2 or fewer is noise).
- `lab fold <exp>`: rebase onto the main line, re-score practice and exam, and record expected, measured on its branch, and measured after rebase. Merge only on your confirmation. If the gain disappears after rebase, mark it superseded.
- `lab discard <exp>`: archive the card and its results.
- `lab apply-flags`: create the next test set version, then re-score the main line and every open experiment.
- The ledger lives on the lab volume; a summary is committed to `docs/benchmarks/lab-ledger.json` on fold-in.

## Comments

- Landed: `tools/lab.py` (CLI: git here, scoring on Modal), `tools/lab_ledger.py` (pure logic; `tools/test_lab_ledger.py`, 10 tests), scorer `--split practice|exam` and `--exclude <flags json>` (`lab.ts` `shardEntries`, `exclusionsFor`, `scoreLabSession(..., exclude)`; 2 new tests in `test/lab.test.ts`), runner `--merge`, `--run-file` and `LAB_SRC` in `modal/lab_run.py`, and the experiment rule in AGENTS.md (spec decision 21).
- Usage (from any checkout of the CLI; `--main <branch>` defaults to `lab`, `--ledger <volume path>` to `/ledger.json`, or set `LAB_MAIN` / `LAB_LEDGER`):

  ```
  python3 tools/lab.py baseline                 # main line, practice + exam; the first call registers lab-v1 (--manifest under /lab/data)
  python3 tools/lab.py new tighter-omit --component checker --hypothesis 'confirm omissions after 1.5 s' --expect-component -6 --expect-overall -4
  python3 tools/lab.py run ../tilawa-lab-exp-tighter-omit     # practice only, verdict vs the main line
  python3 tools/lab.py status [--all]
  python3 tools/lab.py fold tighter-omit [--yes]   # --yes only when the user approved
  python3 tools/lab.py discard tighter-omit
  python3 tools/lab.py apply-flags [--dry-run]
  ```

- `new` makes `../tilawa-lab-exp-<id>` on branch `exp/<id>` from the main line, writes `lab-experiment.json` (`{id, component, hypothesis, expected: {component, overall}, listener, testSet, created, author}`, `--listener` for a listener experiment) and commits it. The card never reaches the main line: the fold commit removes it, and the ledger keeps a copy. Committing it there would make every later experiment conflict on rebase. The fixture fold test hit this.
- Counts: every row is wrong calls = missed + false warnings on practice. A component's count is the wrong calls the attribution charges to it, so listener + tracker + checker = overall. The details line shows the listener's phoneme errors, the tracker's false jumps and the checker's wrong calls with perfect inputs. Verdict: a change of 2 or fewer is noise, fewer is better. The experiment's verdict is overall's, or `noise overall, <component> better|worse` when only the component moved.
- `run` scores the worktree (its uncommitted changes too, sha `-dirty`) with `--split practice`, one shard per practice session, merged on Modal; exam is never scored or shown. Each result records the main-line score it was measured against (`vs`). `status` compares with that score and says "main line moved since; fold re-checks" when there is a newer one.
- `fold` refuses when the branch had no practice gain, and refuses uncommitted changes. Otherwise it rebases onto the main line (a conflict aborts the rebase) and re-scores practice and exam, re-scoring the main line too in parallel if it moved since its last score. If the rebased practice gain is gone (noise or worse) it marks the experiment superseded and stops. Otherwise it prints expected / on branch / after rebase, the exam counts (numbers only), and asks. On yes it commits `docs/benchmarks/lab-ledger.json` and drops the card in one commit. It then fast-forwards the main line (`merge --ff-only` in the worktree that has it checked out, else `branch -f`), records the new main-line score and removes the worktree (the branch stays). If the main line moves during scoring, it undoes the summary commit and asks you to fold again.
- `discard` marks the experiment discarded. The card and results stay in the ledger, the worktree is removed (refused with uncommitted changes unless `--force`), and the branch stays.
- `apply-flags` reads `/lab/reviews/<base test set>/*.json`, takes each item's latest verdict across reviewers, and keeps the `bad_*` ones not already applied. It finds each item's window in its queue (the record's `queue`) and its session by id prefix, else the audio file stem. It writes `/lab/data/<base>/flags-v<N>.json` (cumulative), registers `<base>.v<N>` as the current test set, and re-scores the main line (practice + exam) and every open experiment (practice) in parallel. The scorer's `--exclude` drops mistakes overlapping a flagged window and false warnings raised inside one, on both audio copies. An entry without a window drops the whole session.
- Ledger (`/lab/ledger.json`, source of truth; every write re-reads first):

  ```
  {version: 1, kind: "lab-ledger", main: "lab", testSet: "<current id>",
   testSets: {"lab-v1": {id, base, version: 1, manifest: "lab-v1/manifest-v1.json", exclude: null | "lab-v1/flags-v2.json", flags, sessions: {practice: [ids], exam: [ids]}, created, previous}},
   mainLine: [{sha, head, run, testSet, listener, excluded, practice: Score, exam: Score, note, experiment?, at}],
   experiments: {"<id>": {card, branch, worktree, state: open|superseded|folded|discarded,
     results: [{kind: branch|rebased, sha, run, testSet, listener, excluded, practice: Score, exam?: Score, vs: {sha, at}, at}],
     decision: null | {state, at, note, testSet}}}}
  Score = {sessions, minutes, wrongCalls, missed, falseWarnings, mistakes, per5Min,
    listener: {wrongCalls, missed, falseWarnings, phonemeErrors, phonemes, per},
    tracker: {wrongCalls, missed, falseWarnings, falseJumps, onPlaceShare},
    checker: {wrongCalls, missed, falseWarnings, alone: {wrongCalls, missed, falseWarnings}}}
  ```

  Version 1 of a test set is its base name; later versions are `<base>.v<N>`. "Waiting for your decision" is shown, not stored: an open experiment with a branch result on the current test set. The flags file is `{version: 1, kind: "lab-flags", testSet: "<base>.v<N>", base, created, excluded: [{session, startS, endS, item, verdict, note, reviewer, at}]}`. The committed summary `docs/benchmarks/lab-ledger.json` is `{version, kind: "lab-ledger-summary", main, testSet, note, rows}`. Rows are main-line changes (`{kind: "main", at, sha, testSet, experiment, note, practice, exam}`) and decided experiments (`{kind: "experiment", at, id, component, hypothesis, expected, state, testSet, note, branch, rebased: {practice, exam}}`), each score `{wrongCalls, missed, falseWarnings, listener, tracker, checker}`. The dashboard (ticket 11) should read the volume ledger.
- Tests: `uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_ledger.py` (10 passed locally). On Modal: `test/lab.test.ts` + `test/lab-engine.test.ts` (19 passed), and core and sdk `tsc -b` are clean (run `lab10-test-d33d7b1-20261028T131237Z`).
- Fixture e2e (`lab-scorer-fixture`, throwaway ledgers, main line `lab-10`), about 2 min per command:
  - `baseline` gave practice 3 (3 missed + 0 false) and exam 2 (2 + 0), matching ticket 08's smoke. `new noop`, then `run`: every row +0, noise, "waiting for your decision".
  - Fold test on a throwaway main line whose checker confirmed nothing (practice 6, exam 3). `fixa` and `fixb` both restored it, and each `run` gave overall -3 and checker -4, better.
  - `fold fixa` answered "n": rebased result recorded, not merged. `fold fixa --yes`: kept its gain (exam 3 to 2), fast-forwarded the checked-out main line with the summary commit, main line now 3.
  - `fold fixb`: the main line had moved, so it was re-scored in parallel. The duplicate commit was dropped in the rebase, the gain was gone (+0 / +0), and fixb was marked superseded, not merged.
  - `apply-flags` with a test review (one `bad_splice` on a missed wrong word, one `ok`) wrote `flags-v2.json` with 1 window and made `lab-scorer-fixture.v2`. The main line went to practice 2 and noop was re-scored (noise). A second call found no pending flags. `discard noop` removed the worktree and kept the branch.
  - Test ledgers, reviews, queues, flags files, branches and worktrees were removed afterwards.
- lab-v1 e2e (main line `lab-10` at d33d7b1, throwaway ledger), about 11 min each, set by the longest session:
  - `baseline` scored 40 shards. Practice 140 wrong calls (82 missed + 58 false warnings) over 18 sessions; exam 118 (67 + 51) over 16.
  - `new noop`, then `run` over 20 shards: overall 140 to 140, listener 54, tracker 2, checker 84, all +0, so noise. Details: 2350 phoneme errors, 13 false jumps, 102 checker wrong calls with perfect inputs.
  - `apply-flags --dry-run` found no reviews yet for lab-v1. `discard noop` worked.
  - These numbers aren't trusted until 07 passes.
- Blocking and open:
  - 6 of 40 lab-v1 sessions aren't scored (p07, p18, e01, e08, e14, e17): "Evidence mismatch". The student evidence ends 0.10 s short of the session (e01: 5337 frames × 40 ms = 213.48 s against 213.58 s), right at the scorer's 0.1 s tolerance in `checkEvidence`; the teacher is 0.02 s short and passes. Fix it in the decoder (pad to the session length) or in the scorer (tolerance of about 2 frames); either is a scorer/decoder contract call for 05/08.
  - The real ledger doesn't exist yet. After `lab-10` merges into `lab`, run `python3 tools/lab.py baseline`. Experiments branched from `lab` before that merge lack the scorer's `--split`.
  - No scoreboard row: the defaults score the same, so no metric moves.
  - Speed: practice runs take about 11 min because one shard per session waits on the longest session with every arm plus the teacher. Splitting long sessions isn't possible, but `--teacher none` on experiment runs would cut about a fifth (the ceiling only matters on the main line).
