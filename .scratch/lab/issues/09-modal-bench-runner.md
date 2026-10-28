# 09 Modal runner for the TypeScript bench

Status: ready-for-agent
Type: task
Blocked by: none

Spec: `../spec.md` decision 15.

## Acceptance

- A Modal image with Node and pnpm and the workspace dependencies.
- `lab run <worktree>` uploads the worktree source (no push needed), spreads scoring across one container per session, and writes results to the lab volume keyed by experiment, commit and test set version.
- A full practice-set run finishes in a few minutes, and nothing heavy runs on the Mac.
