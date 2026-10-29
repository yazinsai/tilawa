# Tilawa

Offline streaming Quran recitation engine and SDK. See `PLAN.md`.

## Scoreboard

Any change that can move a metric (listener, tracker, checker, labels, eval data) adds a scoreboard row before it merges: run the scoreboard command with a one-line `--note` saying what changed and why, and commit the updated `docs/benchmarks/scoreboard.json`. The browser dashboard reads that file. See `docs/benchmarks/scoreboard.md` for the command.

## Lab experiments

Any change that can move a lab number (listener, tracker, checker) is an experiment with a recorded result (`.scratch/lab/spec.md`): `python3 tools/lab.py new <id> --component ... --hypothesis '...' --expect-component N --expect-overall N` makes its worktree and card, and `python3 tools/lab.py run <worktree>` records its practice result and verdict. The user decides `lab fold` or `lab discard`. Pass `fold --yes` only when the user has approved that fold. Fold-in updates the main line (`lab`). See `tools/lab.py` for every command.

## Compute

Heavy work (audio, decoding, training, scoring, data-dependent tests) runs on Modal; the Mac runs git, the CLIs and small JSON reads. `python3 tools/lab.py run --cmd '...'` runs any TypeScript command on Modal against a worktree's source.

Long jobs (any `modal run` past a minute) go through the shell tool's own background mode, then get polled. A job backgrounded inside a shell call (`nohup ... &`, `cmd &`) dies when that call returns, and its Modal app stops with the local entrypoint: the run stalls with partial outputs and no error. A subagent given a long job needs this rule in its prompt.

Stages skip finished outputs, so a killed run resumes by rerunning the same command. A run is stalled when its Modal app shows 0 tasks, its logs are empty and its output count stops growing: stop the app (`modal app stop <id>`) and rerun.

Code at module level in a Modal file also runs inside the container, where local paths don't exist; read local files only in the image definition or the local entrypoint.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
