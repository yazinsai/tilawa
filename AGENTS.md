# Tilawa

Offline streaming Quran recitation engine and SDK. See `PLAN.md`.

## Release scoreboard

The scoreboard (`docs/benchmarks/scoreboard.md`) holds the release gates, checked before a release rather than per change. Day-to-day changes are lab experiments (below).

## Lab experiments

Any change that can move a lab number (listener, tracker, checker) is an experiment with a recorded result (`.scratch/lab/spec.md`): `python3 tools/lab.py new <id> --component ... --hypothesis '...' --expect-component N --expect-overall N` makes its worktree and card, and `python3 tools/lab.py run <worktree>` records its practice result and verdict. The user decides `lab fold` or `lab discard`. Pass `fold --yes` only when the user has approved that fold. Fold-in updates the main line (`main`). See `tools/lab.py` for every command.

## Compute gotchas

- Heavy work runs on Modal, not the Mac.
- A job backgrounded inside a shell call (`nohup ... &`) dies when the call returns, taking its Modal app with it: the run stalls silently with partial outputs. Use the shell tool's own background mode for long `modal run`s.
- Module-level code in a Modal file also runs inside the container, where local paths don't exist.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.

### Audio failure reviews

When reviewing recitation labels, missed mistakes or false warnings, follow `docs/agents/audio-review.md`: automate incident and component triage first, use a calibrated independent audio opinion when available, and ask the user only for unresolved listening judgments and spot checks.
