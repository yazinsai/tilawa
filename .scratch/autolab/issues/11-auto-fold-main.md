# 11 Folds go to `main`

Status: needs-triage
Type: task
Blocked by: 04, 07

Spec decision 6: switch from the `autolab` shadow branch to folding on `main` once about 10 consecutive gate decisions in shadow mode match the user's.

## Scope

- **The match record:** each shadow fold and discard gets the user's agree or disagree (one click on the dashboard, or a reply to the digest), stored in `state.json`.
- **The switch** is a config flag, flipped by the user. After it: a daily tag `autolab/YYYY-MM-DD` on `main`, and `autolab.py revert <date>`, which reverts that day's fold commits as one commit and re-scores the main line.
- **The AGENTS.md wording change** of spec decision 15.

## Acceptance

- The revert is tested on a scratch repo with two folds in a day.
