# 03 Scoring survives its caller

Status: ready-for-human
Type: task
Blocked by: none

`lab.py` scores by running `modal run modal/lab_run.py` as a child process. If the process that started it dies (a shell call returning, a supervisor restart), the Modal app goes with it and the run stalls with partial outputs (AGENTS.md, compute gotchas).

## Scope

- Deploy `lab_run` (`modal deploy`). Its shard function takes the source tarball as an argument; add an entry function that fans out the shards, runs the merge, and returns the run ID.
- In `lab.py`, `score()` calls `modal.Function.from_name("tilawa-lab-run", ...).spawn(...)`, stores the function-call ID in the run file, and polls it. A restarted caller resumes polling from the run file instead of starting again.
- Image drift: the deployed image installs dependencies from the lockfile at deploy time. A shard whose lockfile differs still reinstalls, as now. Redeploying is a supervisor step whenever `modal/lab_run.py` changes on `main`.

## Acceptance

- Killing the local process mid-run and running the same command again picks up the same Modal call and yields the same `score.json`.
- A main-line score through the spawned path matches the current `modal run` path byte for byte on its counts.

## Comments

**1 Nov 2026, built and checked on Modal.**

- `modal/lab_run.py` has a new deployed function, `fanout(src: bytes, cmd: str, shards: int, run: str, merge: str = "") -> dict`, on the app `tilawa-lab-run`. It runs every shard (`starmap` with `return_exceptions`, so a crashed shard comes back as `code -1` and doesn't kill the call), then runs the merge if they all succeed. It returns `{run, shards: [status...], merge: status|None, ok}`. Timeout 6 h, 0.25 CPU. `source(root)` and `run_id(name, sha)` are now shared with `main`, and `modal run modal/lab_run.py --cmd ...` still works (checked with 2 shards). The docstring covers when to redeploy. Deployed with `modal deploy modal/lab_run.py`.
- `tools/lab.py`: `score()` builds the command as before and calls `spawned()`. `spawned()` tars the worktree and spawns `modal.Function.from_name("tilawa-lab-run", "fanout")`. It writes the run file and then polls `FunctionCall.get(timeout=60)`, logging a line each minute and each shard's result to the log. On failure it deletes the run file and raises `Stop` with the last 25 lines of the log. It then reads `/runs/<run>/score.json` as before.
- **Run file:** `$TMPDIR/tilawa-lab-runs/<name>-<key16>.run` holds JSON `{"run", "callId", "key"}`. The key is a SHA-256 over the tarball's file names and contents plus the command, the merge command and the shard count. It ignores file times and gzip headers, so two fresh `git worktree` checkouts of one commit get the same key. The run file sits at a fixed path because callers such as `cmd_fold` pass a throwaway temp log path. When the run file exists with a matching key and the call is still running or finished, the caller resumes polling. When the call has expired (Modal keeps results 7 days) or the run file is in the old plain-text form, it spawns a new call. Resuming a finished call returns the same run, which is fine because scoring is deterministic. A different source or command means a new call.
- **Interpreter:** the `python3` that runs `lab.py` (3.13) has no `modal` package, and the pyenv 3.11 that has it can't parse `lab_ledger.py`. So when `import modal` fails, `spawned()` runs itself in `uv run --no-project --python 3.12 --with 'modal>=1.3'`. Prose goes to stderr and a JSON result comes back on stdout.
- **Input size:** tarballs are about 2.7 MB. Modal stores inputs over 2 MiB in object storage on its own, and spawned inputs always go there; the documented gRPC limit is 100 MB. So the bytes are passed directly and nothing goes through the volume. `modal run` already passed them the same way.
- **Gotcha found:** Modal 1.3.4's `FunctionCall.get(timeout=...)` raises the builtin `TimeoutError` for a running call, not `modal.exception.TimeoutError`. The poll loop catches both.
- **Tests:** `tools/test_lab_score.py` (7 tests, Modal faked) covers a fresh start, a resume, an expired call, the old run file form, a failed shard, a call that raised, and the uv fallback. `tools/test_lab_fold.py` still passes (19 in total).
- **Real check (acceptance 1):** a 2-shard `sleep 120; echo hi > $LAB_OUT/x` with an `ls` merge, from a fresh detached checkout of `e44598d`. After spawn and the first 60 s poll, I killed the whole local process tree. A rerun from a different fresh checkout of the same commit logged `resuming call fc-01M6AJSY66XT2NYS03CDSCYXG8` without spawning, and finished with shards 0 and 1 at exit 0 and the merge at exit 0. `x` and `merge/ls.txt` are on the volume under `/runs/autolab03-check-e44598d-20261101T033510Z/`. A first try from the live main checkout spawned a new call instead, because other agents were editing `tools/` at the same moment, so the source really had changed. That is the intended behavior.
- **Open (acceptance 2):** a main-line score through the spawned path compared with the `modal run` path needs a full 100-shard practice run, which wasn't run here. Suggested check: `python3 tools/lab.py baseline --practice-only`, then compare its counts with the latest ledger entry for the same main sha.
