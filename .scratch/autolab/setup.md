# Autolab: going live (phase A)

Steps only you can do, in order. Run everything from the repo root. `AL` below stands for
`uv run --python cpython-3.12-macos-aarch64-none --with modal==1.3.4`.

## 1. Keys and models

The loop needs two keys in its environment (the launchd wrapper loads them from `~/.config/tilawa/autolab.env`; a
restart is needed for the loop to see a new one):

- `ANTHROPIC_API_KEY`: the orchestrator (planner, reviewer, strategist: `claude-*` models in `models.*`) runs
  `claude -p --bare` on it, never the desktop login.
- `AZURE_OPENAI_API_KEY`: builders and the judges (judge, diffReviewer: `gpt-*` models) use the Azure deployment of the
  `codex-azure` arm whose `azure.deployment` is the model's id.

## 2. Builders: GPT-6.1 Sol on Azure, through the Codex CLI

Builders come from `executor.runtimes`, a list of arms (default one, `sol`, runtime `codex-azure`). An arm is skipped,
with one notification, until its Azure fields and key are there. Fields of an arm:

| Field | What |
| --- | --- |
| `azure.endpoint` | `https://<resource>.openai.azure.com` (or `.cognitiveservices.azure.com`); `/openai/v1` is added |
| `azure.deployment` | the Azure deployment name; Codex sends it as the model |
| `azure.apiVersion` | leave empty for the v1 API; `preview` for v1 preview features; a dated version (`2025-04-01-preview`) switches to the older `<endpoint>/openai?api-version=...` route |
| `effort` | reasoning effort, default `high` |
| `maxConcurrent` | optional; most builders of this arm at once (an Azure quota) |
| `azure.keyEnv` | optional; the key's variable, default `AZURE_OPENAI_API_KEY` |
| `azure.keyFile` | optional; a file holding the key, read at every tick when the variable isn't set |
| `extraConfig` | optional; more Codex `config.toml` keys, e.g. `{"model_context_window": 400000}` (Codex has no metadata for a deployment name) |
| `startupTimeoutS` | optional; a run with no answer from the model in this long fails as a startup failure (default 300) |

With more than one arm, `executor.split` weighs them (`0` stops new work going to an arm). Each experiment gets its arm
when its builder first starts, balanced per part, and keeps it for retries and bounces. After two failures of its own
arm it moves once to another arm. An arm whose last 3 builders failed to start rests for an hour.

Codex builders never get `MODAL_*` or any `*_API_KEY` but their own, which the runner hands to the codex process only.
Codex runs with its own sandbox off, inside the Seatbelt profile of section 3; its sandbox would stop a worktree from
committing. `CODEX_HOME` is `<worktree>/.autolab/codex-home`, holding only the generated `config.toml`: no ChatGPT login
and nothing from your `~/.codex`.

Check, in order:

```bash
python3 tools/autolab_codex.py config        # the config.toml the arm gets, and whether it's ready
python3 tools/autolab_codex.py probe         # offline: codex against a local 401 server (request shape, clean failure)
git worktree add --detach /tmp/autolab-codex main
python3 tools/autolab_codex.py smoke /tmp/autolab-codex              # live: one trivial card, should report finished
git -C /tmp/autolab-codex checkout -- . && python3 tools/autolab_codex.py smoke-isolation /tmp/autolab-codex
git worktree remove --force /tmp/autolab-codex
```

`smoke-isolation` runs the tries of section 3, plus `env | grep` for `MODAL_*`, `AZURE_OPENAI*` and
`ANTHROPIC_API_KEY`, which must find nothing. Results: `tools/autolab.py ab`, or the Builders table on the Autolab tab.

## 3. Executor isolation (decision 14)

Builders get no Modal credentials. Each runner starts with:

- `HOME` set to `<worktree>/.autolab/home` (`executor.home`, default on). `GIT_CONFIG_GLOBAL`, `NVM_DIR`, `PYENV_ROOT`
  and `UV_PYTHON_INSTALL_DIR` still point at yours, so git, node and uv keep working.
- every `MODAL_*` variable and each `executor.envDrop` pattern (fnmatch, e.g. `"SONIOX_*"`) removed, then
  `executor.env` applied on top.
- a macOS Seatbelt profile (`executor.seatbelt`, default on). It blocks reads and writes of `~/.modal.toml` and
  `~/.modal`, and writes anywhere under your home except the worktree, its git dir, the shared git objects and its own
  branch ref. It also stops reading other processes' arguments and environment (`ps eww`), which would show the loop's
  `ANTHROPIC_API_KEY` and `AZURE_OPENAI_API_KEY`.

Builders run code on Modal through the supervisor instead: `tools/autolab.py modal-run [--shards N] [--gpu G]
[--image node|train] [--timeout MIN] -- '<cmd>'`. It writes a request into the worktree, and the supervisor runs it
in a Modal Sandbox (`modal/autolab_exec.py`). A sandbox has no Modal identity of its own, unlike a function container.
It mounts only `tilawa-lab-exec` read-only at `/lab/data` and `tilawa-lab-scratch` at `/scratch/<id>`, never
`tilawa-lab`. GPUs are for bet cards only, within the card's `gpuUsd`. The limits live in the `exec` config key
(`concurrent`, `maxShards`, `maxGpuShards`, `maxMinutes`, `betMaxMinutes`). The validation tests run the same way.

`tilawa-lab-exec` holds a practice-only copy of the test set: the filtered manifest, practice labels, audio and
evidence for the main listener and the teacher, the filtered flags, checks and decode files, and the listener model.
The copy refuses to commit if any exam id appears in a file name or JSON file. The supervisor refreshes it when the
test set changes. To set it up or redo it by hand:

```bash
uv run --python cpython-3.12-macos-aarch64-none --with modal==1.3.4 modal deploy modal/autolab_exec.py   # after any change to it
$AL python tools/autolab.py sync-exec-data
```

A bet that trains a listener writes `.autolab/model.json` (`{"path": "/scratch/<id>/..."}`). The supervisor copies
that model into `tilawa-lab` `/data/models/exp-<id>-<sha>/` and decodes the practice sessions only.

Check it with a scratch worktree (it needs `AZURE_OPENAI_API_KEY` in the env):

```bash
git worktree add --detach /tmp/autolab-iso main
python3 tools/autolab_codex.py smoke-isolation /tmp/autolab-iso
git worktree remove --force /tmp/autolab-iso
```

The agent runs `modal volume ls tilawa-lab`, reads `~/.modal.toml`, writes outside the worktree and runs `ps eww`; each
of those must fail. git, pnpm, node and `modal-run --help` must work.

What it doesn't stop yet (phase B moves builders to a remote VM):

- A local builder can still read your other files outside the worktree (only the Modal config is hidden), and it has
  full network access.
- Practice scoring and the decode still run the builder's engine and SDK code in `lab_run` with `tilawa-lab` mounted.
  The data-hash check, the hygiene checks and the diff review still guard that path.
- `tilawa-lab-scratch` is shared by all experiments.
- Builders that started before a loop restart keep the environment they started with.

## 4. Notifications

Install the ntfy app, subscribe to a hard-to-guess topic, and set `notify.topic` (and `notify.dashboard`, the
dashboard URL) in the config.

## 5. Start

```bash
$AL python tools/autolab.py init        # autolab branch and ../tilawa-autolab worktree, practice baseline (~11 min)
$AL python tools/autolab.py loop        # leave running: caffeinate -i $AL python tools/autolab.py loop
```

Keep the watchdog apart from the loop, so a dead loop still pages you. Add this to cron (`crontab -e`):

```
*/10 * * * * cd /Users/rock/ai/projects/tilawa && <AL> python tools/autolab.py watchdog
```

`init`'s baseline also closes ticket 03's last check: its practice counts should match the latest main-line row in
`docs/benchmarks/lab-ledger.json` for the same commit.

## 6. Deploy the dashboard tab

```bash
python3 tools/build_live_review.py && modal deploy modal/lab_review.py
```

Then open the Autolab tab.

## Day to day

- `tools/autolab.py status`, or the dashboard.
- `tools/autolab.py ab`: the builder A/B per arm (started, killed by its builder, sent back, gate passes, near misses,
  folds, median practice change, builder hours, handovers).
- `tools/autolab.py pause [why]` / `resume`.
- `tools/autolab.py match <id> agree|disagree` after reading a shadow decision. About 10 agreements in a row means
  folds can go straight to `main` (ticket 11).
- Take the shadow line's folds: `git merge --ff-only autolab` on `main`, right after the daily rebase, or cherry-pick
  single folds.
- Listening queues and spot checks still come to you, as now.
