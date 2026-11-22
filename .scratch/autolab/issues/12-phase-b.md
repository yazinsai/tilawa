# 12 Phase B: the supervisor on Modal, executors in the cloud

Status: needs-triage
Type: task
Blocked by: 04, 05, 11

Spec decision 14. The Mac can be off.

## Scope

- **The supervisor** becomes a Modal cron function (`autolab.py tick` every 2–5 minutes). It keeps a git checkout on the volume and pushes `autolab`, `main` and the `exp/*` branches to `origin` with a deploy key held in a Modal secret.
- **Executors** move to a Cursor Cloud Agents adapter (`cloud=CloudAgentOptions(repos=[...])`) working on `exp/*` branches. The cloud environment needs pnpm and a Modal token for test runs. Set `skip_reviewer_request=True`.
- **Worktrees become branches.** `lab.py` paths that assume local worktrees (`new`, `run`, `fold`, `discard`) work from a fresh clone of the branch instead.

## Acceptance

- A full day running with the Mac off: the dashboard heartbeat stays green, and at least one experiment goes from `proposed` to a verdict.
