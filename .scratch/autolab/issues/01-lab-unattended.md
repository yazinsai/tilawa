# 01 `lab.py` runs unattended

Status: resolved
Type: task
Blocked by: none

`tools/lab.py` assumes a person at the terminal: `fold` prompts, and every command prints prose. The supervisor needs the same operations as calls whose results it can parse.

## Scope

- A global `--json` flag. Each command prints one JSON object on stdout (its result, or `{"error": ...}` with a nonzero exit code) and sends its prose to stderr.
- `fold --no-merge`: rebase, re-score and record the result, but stop before merging. The supervisor applies the gate (ticket 02) and then calls `fold --yes` itself.
- Folding onto the `autolab` shadow branch (spec decision 6) needs no new flag. The existing global options `--main autolab --ledger /autolab/ledger.json` give the shadow line its own branch and its own ledger, so its scores never mix with the main line's.
- `run` and `fold` record the score report's run ID in their JSON output, so the gate can fetch per-session results from `/lab/runs/<run>/score.json`.
- Nothing ever reads stdin when `--json` is set; a missing decision is an error.

## Acceptance

- Existing human use is unchanged without `--json`.
- Tests in `tools/test_lab_fold.py` cover the JSON output of `run`, `fold --no-merge` and `fold --onto`, with git and Modal mocked as the existing tests do.

## Comments

Done (1 Nov).

- **Return values.** Every `cmd_*` in `tools/lab.py` returns a dict, and `main()` prints it as JSON under `--json`, with prose sent to stderr through `redirect_stdout`.
- **`fold` results.** The output carries `state` (`folded`, `waiting` or `superseded`), `run`, `base`, `branchRun`, the practice `compare` and `examTripwire`. The tripwire is `pass`, `blocked`, or `null` for practice-only folds; exam counts are never in the JSON.
- **Prompts.** `ask()` raises in JSON mode instead of prompting.
- **`run --cmd`** returns its exit code instead of calling `sys.exit` inside the command.
- **`lab_gate.exam_tripwire`** now takes the ledger's compact scores, which is what `fold` has.
- **Tests.** There are 5 new tests in `tools/test_lab_fold.py`. The tools suite went from 251 to 271 passing. The 4 failures and the collection errors are unchanged: they come from missing `rapidfuzz`, `modal` and `quran_transcript` in the bare test environment.
- **`--onto`** was dropped as unnecessary (see Scope).
