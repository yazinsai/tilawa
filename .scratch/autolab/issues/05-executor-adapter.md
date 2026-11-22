# 05 Executor adapter, phase A

Status: ready-for-agent
Type: task
Blocked by: none

One interface, `start(card, worktree, brief) -> handle`, `poll(handle) -> running | finished | failed`, `cancel(handle)` and `collect(handle) -> transcript, summary`. The first implementation uses Cursor SDK local agents (Python `cursor-sdk`, async client, one agent per worktree).

## Scope

- **Model from config** (`.scratch/autolab/config.json`), checked against `Cursor.models.list()` at startup. The default executor is Grok 4.7, with GPT-5.6 Sol as an alternative. An executor that fails twice on a card is not retried with the same model.
- **The brief** holds:
  - the card;
  - the target incidents, with their case pages and `lab-trace-cli` commands;
  - the relevant journal entries;
  - the rules: write the experiment `.md` like the existing ones under `.scratch/lab/experiments/`; run Modal test runs freely; score practice only through the supervisor, at most 3 times per card (spec decision 20); never write evidence or test data; touch no protected path;
  - the time and cost budget.
- **Logging and failures:** the agent ID and run ID are logged as soon as a run starts. A startup failure (`CursorAgentError`) is told apart from a run failure, as the SDK skill describes. Agents are always disposed.
- **Settings:** `setting_sources` left empty, so the executor sees only the brief and repo rules (`AGENTS.md` is read from the worktree).

- **Isolation** (spec decision 14). Executors get a separate Modal environment (`autolab-exec`), their own token and a volume holding only the current test set's practice sessions and evidence, synced by the supervisor at each test set version. `CURSOR_API_KEY` and that token are the only secrets in the executor's environment.

## Acceptance

- A smoke test runs a trivial card ("add a comment to a test file") through start, poll and collect in a scratch worktree.

## Comments

**1 Nov 2026, adapter built; live smoke test pending.** `tools/autolab_cursor.py` (runtime `cursor-local`) and `tools/test_autolab_cursor.py` (14 tests, SDK faked, no network). Built against `cursor-sdk` 1.2.0, with the API checked by introspecting the installed package.

- **Surviving restarts:** `start` writes `<worktree>/.autolab/<card>-<ts>/job.json` and launches `autolab_cursor.py run <job>` with `start_new_session=True`. That runner owns the agent (`Client.launch_bridge(workspace=worktree)`, then `Agent.create(model, api_key, name, local=LocalAgentOptions(cwd=worktree), client)`, `agent.send(brief)`, `run.messages()` written to `transcript.txt` and `messages.jsonl`, then `run.wait()`). It writes `status.json`: starting, then running, then finished or failed, with the agent ID and run ID as soon as they are known. `poll` reads the status file and checks the process group; a dead group with a non-terminal status means failed (`runner-died`). `.autolab/` goes into the main repo's `.git/info/exclude`.
- **Failures:** `failure` is one of:
  - `startup`: `CursorAgentError` before the run existed. The runner exits 1 and records `retryable`.
  - `run-error`: `result.status` was not finished. The runner exits 2.
  - `sdk-error`: `CursorAgentError` mid-stream.
  - `cancelled`.
  - `runner-died`.
  - `crash`.
  The agent is always closed. The cost of each run (`agent.get_usage().cost.charged_cents`) goes into the status file when the backend reports it. For decision 19: the SDK does expose dollar cost per agent, so a dollar cap is possible.
- **Cancel:** SIGTERM to the runner's process group. The runner catches it, calls `run.cancel()` on the live run, closes the agent and exits. After `cancelGraceS` it falls back to an out-of-process `Agent.resume` plus `Agent.get_run(...).cancel()` (guarded by `supports("cancel")`, best effort), then SIGKILL.
- **Isolation (phase A):** `executor.env` overrides, with "$NAME" copied from the supervisor's env, plus `executor.envDrop`. The runner pops `CURSOR_API_KEY` before the bridge starts and passes the key explicitly, so the agent's shell never sees it. A local agent can still read `~/.modal.toml`; real isolation comes in phase B.
- **`prompt_model(config, model, prompt, timeout_s)`:** for the orchestrator and judges. It runs `Agent.create` with `tools=[]` and cwd set to a fresh empty temp dir, then `send` and `wait`. A timer cancels the run at the timeout. Startup failures and non-finished status raise `RuntimeError`.
- **Verified with the real SDK and a bogus key:** the detached runner starts, the bridge launches, and `Agent.create` fails as `startup: Invalid User API Key`. The worktree stays clean, and `check_models` and `prompt_model` both fail clearly.
- **Needs a live `CURSOR_API_KEY`:**
  1. Run `python tools/autolab_cursor.py models` to get the real Grok 4.7 and GPT-5.6 Sol IDs. `grok-4.7` and `gpt-5.6-sol` are placeholders, so put the real IDs in `.scratch/autolab/config.json` under `executor.model` and `executor.fallbackModel`.
  2. Run `python tools/autolab_cursor.py smoke <scratch-worktree>` (needs an interpreter with cursor-sdk, e.g. `uv run --with cursor-sdk==1.2.0`). This is the acceptance test.
  3. Confirm that `tools=[]` is accepted for `prompt_model`.
  4. Confirm that SIGTERM interrupts a blocking `run.messages()` and the cancel is clean.
