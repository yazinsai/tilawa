# Tilawa

Offline streaming Quran recitation engine and SDK. See `PLAN.md`.

## Scoreboard

Any change that can move a metric (listener, tracker, checker, labels, eval data) adds a scoreboard row before it merges: run the scoreboard command with a one-line `--note` saying what changed and why, and commit the updated `docs/benchmarks/scoreboard.json`. The browser dashboard reads that file. See `docs/benchmarks/scoreboard.md` for the command.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) recorded as a `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root, created lazily. See `docs/agents/domain.md`.
