# 09 Modal runner for the TypeScript bench

Status: resolved
Type: task
Blocked by: none

Spec: `../spec.md` decision 15.

## Acceptance

- A Modal image with Node and pnpm and the workspace dependencies.
- `lab run <worktree>` uploads the worktree source (no push needed), spreads scoring across one container per session, and writes results to the lab volume keyed by experiment, commit and test set version.
- A full practice-set run finishes in a few minutes, and nothing heavy runs on the Mac.

## Comments

- `modal/lab_run.py`: Node 22 + pnpm 10.18 image with `@tilawa/core` and `@tilawa/sdk` dependencies installed from the lockfile (cached layer; reinstalls in-container only if the worktree's lockfile differs). The worktree's tracked and untracked non-ignored files under `packages/` and `tools/` plus root manifests are uploaded as a tarball per run (2.4 MB), so no push is needed.
- Contract for any bench command: `LAB_RUN`, `LAB_SHARD`/`LAB_SHARDS`, `LAB_DATA=/lab/data`, `LAB_OUT=/lab/runs/<run>/<shard>` on the `tilawa-lab` volume; each shard also saves `log.txt` and `status.json`. Run id `<name>-<sha>[-dirty]-<utc>`. Ticket 08's scorer reads its session list from `LAB_DATA` and takes every `LAB_SHARDS`-th session.
- `tools/lab.py run [worktree] --shards N --cmd ...` wraps it; ticket 10 adds the default scorer command.
- Smoke: `vitest run --shard` for core over 4 shards, 507 tests passed, ~97 s wall with a warm image (run `smoke-006a3b1-20261028T070139Z`).
