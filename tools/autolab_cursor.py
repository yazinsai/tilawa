"""Autolab executor on Cursor SDK local agents (.scratch/autolab/spec.md decisions 1, 14, 18, 19; ticket 05), plus the
orchestrator's one-shot model call.

    CursorLocalExecutor(config)                       the `cursor-local` runtime behind tools/autolab_exec.py
    prompt_model(config, model, prompt, timeout_s)    one stateless agent turn in an empty temp dir -> final text

    python3 tools/autolab_cursor.py run <job.json>                the detached runner (start launches it)
    python3 tools/autolab_cursor.py models                        model ids the key can use
    python3 tools/autolab_cursor.py smoke <scratch-worktree> [file]  live check: one trivial card through start/poll/collect
    python3 tools/autolab_cursor.py smoke-isolation <scratch-worktree>  live check: the agent can't reach Modal or write outside

A run survives a supervisor restart: `start` launches this file's runner in its own session (process group), and the
runner owns the agent. Everything the supervisor needs lives in files under `<worktree>/.autolab/<job>/` (job.json,
status.json, transcript.txt, messages.jsonl, runner.log), so poll/cancel/collect need only the stored handle. `.autolab/` is
excluded through the main repo's `.git/info/exclude`, shared by every worktree.

Config, `config["executor"]`: `model` and `fallbackModel` (an id or {"id", "params"}; checked against
`Cursor.models.list()` when the executor is made; a card's `executorModel` overrides `model`), `env` (overrides for the
runner, applied last; a value "$NAME" is copied from the supervisor's NAME), `envDrop` (fnmatch patterns removed; every
MODAL_* always is), `home` (default true: HOME is `<worktree>/.autolab/home`, keeping git's identity through
GIT_CONFIG_GLOBAL), `seatbelt` (default true: the runner runs under `sandbox-exec` with `seatbelt()`'s profile), `python`
(the runner's interpreter argv, default this interpreter, which must have cursor-sdk), `cancelGraceS`.

Isolation, phase A: a builder has no Modal identity. Its environment has no MODAL_* and a HOME without .modal.toml, the
sandbox profile denies reading the real ~/.modal.toml, writing under the real home outside its worktree (so not the
supervisor's code, config or other worktrees) and reading other processes' environments, and the runner pops
CURSOR_API_KEY before the SDK bridge starts (the key goes to the SDK explicitly, so the agent's shell never sees it). Its
Modal work goes through the supervisor (tools/autolab.py modal-run, modal/autolab_exec.py). It can still read the user's
other files and use the network. Phase B cloud agents remove that.
"""

import contextlib
import dataclasses
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_runner as R  # noqa: E402
from autolab_runner import TERMINAL, exclude_autolab, runner_env, seatbelt  # noqa: E402, F401 - re-exported

DEFAULTS = {"model": "grok-4.7", "fallbackModel": "grok-4.7", "cancelGraceS": 30, "home": True, "seatbelt": True}
_now, _hours, _read, _write, _update, _alive = R.now, R.hours, R.read, R.write, R.update, R.alive
_Cancelled, _on_term = R.Cancelled, R.on_term


def _sdk():
    return importlib.import_module("cursor_sdk")


def _api_key():
    key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("CURSOR_API_KEY is not set")
    return key


def _model_id(m):
    return m["id"] if isinstance(m, dict) else m


def available_models(sdk=None):
    sdk, key = sdk or _sdk(), _api_key()
    with tempfile.TemporaryDirectory(prefix="autolab-models-") as tmp, \
            sdk.Client.launch_bridge(workspace=tmp, allow_api_key_env_fallback=False) as client:
        return list(sdk.Cursor.models.list(client=client, api_key=key))


def check_models(cfg, sdk=None):
    """Raise unless every configured model id is in `Cursor.models.list()` for this key; returns the available ids."""
    ids = sorted(m.id for m in available_models(sdk))
    want = [_model_id(cfg[k]) for k in ("model", "fallbackModel") if cfg.get(k)]
    if missing := [m for m in want if m not in ids]:
        raise RuntimeError(f"executor model(s) {missing} not available to this key; available: {ids}")
    return ids


class CursorLocalExecutor:
    def __init__(self, config, check=True, sdk=None):
        self.cfg = {**DEFAULTS, **config.get("executor", {})}
        if check:
            check_models(self.cfg, sdk)

    def start(self, card, worktree, brief):
        worktree = Path(worktree).resolve()
        _api_key()
        exclude_autolab(worktree)
        job_id = f"{card['id']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        d = worktree / ".autolab" / job_id
        d.mkdir(parents=True)
        model = card.get("executorModel") or self.cfg["model"]
        h = {"runtime": "cursor-local", "id": job_id, "worktree": str(worktree), "dir": str(d), "job": str(d / "job.json"),
             "status": str(d / "status.json"), "transcript": str(d / "transcript.txt"), "model": model,
             "started": _now(), "pgid": None, "agentId": None, "runId": None}
        _write(h["job"], {"card": card["id"], "worktree": h["worktree"], "brief": brief, "model": model, "status": h["status"],
                          "transcript": h["transcript"], "messages": str(d / "messages.jsonl")})
        _write(h["status"], {"status": "starting", "card": card["id"], "model": model, "started": h["started"]})
        py = self.cfg.get("python") or [sys.executable]
        home = worktree / ".autolab" / "home" if self.cfg["home"] else None
        h["pgid"] = R.launch(self.cfg, {**h, "worktree": worktree}, [*([py] if isinstance(py, str) else py), str(Path(__file__).resolve()), "run", h["job"]],
                             runner_env(self.cfg, home=home, drop=("AZURE_OPENAI_*",)))
        _update(h["status"], pgid=h["pgid"])
        return h

    def poll(self, handle):
        """Also copies agentId/runId into the handle once the runner has logged them."""
        st = _read(handle["status"])
        for k in ("agentId", "runId"):
            if st.get(k):
                handle[k] = st[k]
        return R.poll(handle)

    def cancel(self, handle):
        R.cancel(handle, self.cfg["cancelGraceS"], last_resort=self._sdk_cancel)

    def _sdk_cancel(self, handle, sdk=None):
        st = _read(handle["status"])
        if not (st.get("agentId") and st.get("runId")):
            return
        try:
            sdk = sdk or _sdk()
            key = _api_key()
            with sdk.Client.launch_bridge(workspace=handle["worktree"], allow_api_key_env_fallback=False) as client:
                opts = sdk.AgentOptions(api_key=key, local=sdk.LocalAgentOptions(cwd=handle["worktree"]))
                with sdk.Agent.resume(st["agentId"], opts, client=client):
                    run = sdk.Agent.get_run(st["runId"], {"runtime": "local", "agentId": st["agentId"],
                                                          "cwd": handle["worktree"]}, client=client)
                    if run.supports("cancel") and run.status == "running":
                        run.cancel()
        except Exception as e:  # noqa: BLE001 - best effort; the process group kill follows
            _update(handle["status"], sdkCancelError=f"{type(e).__name__}: {e}")

    def collect(self, handle):
        st = _read(handle["status"])
        p = Path(handle["transcript"])
        transcript = p.read_text() if p.exists() else ""
        summary = st.get("result") or ""
        if st.get("status") == "failed":
            summary = f"{st.get('failure') or 'failed'}: {st.get('error') or ''}\n{summary}".strip()
        return {"transcript": transcript, "summary": summary,
                "hours": _hours(st.get("started") or handle["started"], st.get("ended") or _now()),
                "agentId": st.get("agentId"), "runId": st.get("runId"), "costCents": st.get("costCents")}


# --- runner (a separate process per job) ---

def _plain(m):
    if dataclasses.is_dataclass(m):
        return dataclasses.asdict(m)
    return dict(m) if hasattr(m, "keys") else {"raw": str(m)}


def _render(m):
    t = getattr(m, "type", None) or (m.get("type") if hasattr(m, "get") else None)
    if t == "assistant":
        return "".join(getattr(b, "text", "") or "" for b in m.message.content)
    if t == "tool_call" and m.status != "running":
        return f"\n[tool {m.name} {m.status}]\n"
    if t == "status":
        return f"\n[status {m.status}]\n"
    return ""


def run(job_path, sdk=None):
    """Drive one job to a terminal status.json; returns the exit code (0 finished, 1 startup failure, 2 run failure)."""
    job = json.loads(Path(job_path).read_text())
    st = job["status"]
    key = os.environ.pop("CURSOR_API_KEY", "")  # before the bridge starts: its shell tools inherit this process's env
    _update(st, status="running", pid=os.getpid(), runnerStarted=_now())
    signal.signal(signal.SIGTERM, _on_term)
    agent = live = None
    code = 2
    try:
        sdk = sdk or _sdk()
        with sdk.Client.launch_bridge(workspace=job["worktree"], allow_api_key_env_fallback=False) as client:
            try:
                agent = sdk.Agent.create(model=job["model"], api_key=key, name=f"autolab {job['card']}", client=client,
                                         local=sdk.LocalAgentOptions(cwd=job["worktree"]))
                _update(st, agentId=agent.agent_id)
                live = agent.send(job["brief"])
                _update(st, runId=live.id)
                with open(job["transcript"], "a") as tr, open(job["messages"], "a") as ms:
                    for m in live.messages():
                        ms.write(json.dumps(_plain(m), default=str) + "\n")
                        tr.write(_render(m))
                        tr.flush()
                        ms.flush()
                res = live.wait()
                ok = res.status == "finished"
                code = 0 if ok else 2
                _update(st, status="finished" if ok else "failed", failure=None if ok else "run-error", runStatus=res.status,
                        result=res.result or "", error=res.error_message, errorCode=res.error_code,
                        durationMs=res.duration_ms, usage=_plain(res.usage) if res.usage else None, ended=_now())
            except _Cancelled:
                if live is not None and live.supports("cancel"):
                    with contextlib.suppress(Exception):  # already terminal, or the bridge is going away
                        live.cancel()
                _update(st, status="failed", failure="cancelled", ended=_now())
            except sdk.CursorAgentError as e:
                code = 1 if live is None else 2
                _update(st, status="failed", failure="startup" if live is None else "sdk-error",
                        error=getattr(e, "message", str(e)), retryable=getattr(e, "is_retryable", False),
                        retryAfter=getattr(e, "retry_after", None), ended=_now())
            finally:
                if agent is not None:
                    with contextlib.suppress(Exception):  # cost is optional and eventually consistent
                        if (cost := agent.get_usage().cost) is not None:
                            _update(st, costCents=cost.charged_cents, rawCostCents=cost.raw_cost_cents)
                    agent.close()
    except _Cancelled:
        if _read(st).get("status") not in TERMINAL:
            _update(st, status="failed", failure="cancelled", ended=_now())
    except Exception as e:  # noqa: BLE001 - bridge launch or a crash; the supervisor reads status.json, not our exit
        if _read(st).get("status") not in TERMINAL:
            _update(st, status="failed", failure="startup" if agent is None else "crash",
                    error=f"{type(e).__name__}: {e}", ended=_now())
        code = 1 if agent is None else 2
    return code


# --- orchestrator one-shot ---

def prompt_model(config, model, prompt, timeout_s=900, sdk=None):
    """One agent turn with no tools, in a fresh empty temp dir (a temp cwd alone wouldn't stop a shell from leaving it)."""
    sdk = sdk or _sdk()
    key = _api_key()
    with tempfile.TemporaryDirectory(prefix="autolab-prompt-") as tmp, \
            sdk.Client.launch_bridge(workspace=tmp, allow_api_key_env_fallback=False) as client:
        try:
            agent = sdk.Agent.create(sdk.AgentOptions(model=model, api_key=key, local=sdk.LocalAgentOptions(cwd=tmp),
                                                      tools=[]), client=client)
        except sdk.CursorAgentError as e:
            raise RuntimeError(f"{_model_id(model)} did not start: {getattr(e, 'message', e)}") from e
        with agent:
            try:
                live = agent.send(prompt)
            except sdk.CursorAgentError as e:
                raise RuntimeError(f"{_model_id(model)} did not start: {getattr(e, 'message', e)}") from e
            timer = threading.Timer(timeout_s, lambda: live.supports("cancel") and live.cancel())
            timer.start()
            try:
                res = live.wait()
            finally:
                timer.cancel()
    if res.status != "finished":
        raise RuntimeError(f"{_model_id(model)} run {res.id} ended {res.status}"
                           + (" (timed out)" if res.status == "cancelled" else "") + f": {res.error_message or ''}")
    return res.result or ""


# --- CLI ---

def _config():
    p = Path(__file__).resolve().parent.parent / ".scratch" / "autolab" / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def smoke(worktree, file=None):
    worktree = Path(worktree).resolve()
    file = file or next(str(p.relative_to(worktree)) for p in sorted((worktree / "tools").glob("test_*.py")))
    ex = CursorLocalExecutor(_config())
    brief = (f"Smoke test of the autolab executor. Add exactly one line, the comment `# autolab smoke test`, as the last "
             f"line of {file}. Change nothing else, run nothing, do not commit. Then reply with one sentence.")
    h = ex.start({"id": "smoke"}, worktree, brief)
    print(json.dumps(h, indent=1))
    deadline = time.time() + 900
    while (state := ex.poll(h)) == "running" and time.time() < deadline:
        time.sleep(5)
    if state == "running":
        ex.cancel(h)
        state = "timeout"
    out = ex.collect(h)
    diff = subprocess.run(["git", "-C", str(worktree), "diff", "--", file], capture_output=True, text=True, check=False).stdout
    print(json.dumps({"state": state, **{k: v for k, v in out.items() if k != "transcript"}}, indent=1))
    print(diff or "(no diff)")
    return 0 if state == "finished" and "autolab smoke test" in diff else 1


def smoke_isolation(worktree):
    """Live check of decision 14: the agent tries each way out; every shell call's exit and first output line are printed
    from the run's messages (not the agent's account). Passes when Modal, the token and the main checkout are out of
    reach and git, node and the autolab CLI still work."""
    worktree = Path(worktree).resolve()
    tries, probe = R.isolation_tries(Path(__file__).resolve().parent.parent, Path.home())
    state, h, _ = R.drive(CursorLocalExecutor(_config()), worktree, "smoke-isolation", R.isolation_brief(tries))
    seen = {}
    for line in Path(h["dir"], "messages.jsonl").read_text().splitlines():
        m = json.loads(line)
        cmd = (m.get("args") or {}).get("command") if m.get("type") == "tool_call" and m.get("status") != "running" else None
        if cmd:
            v = ((m.get("result") or {}).get("value") or {})
            seen[cmd.strip()] = (v.get("exitCode"), ((v.get("stdout") or "") + (v.get("stderr") or "")).strip().splitlines()[:1])
    return R.isolation_report(tries, seen, state, probe)


def main(argv):
    if argv[:1] == ["run"] and len(argv) == 2:
        return run(argv[1])
    if argv[:1] == ["smoke-isolation"] and len(argv) == 2:
        return smoke_isolation(argv[1])
    if argv[:1] == ["models"]:
        for m in available_models():
            print(m.id, "|", m.display_name, "|", ", ".join(v.display_name for v in m.variants))
        return 0
    if argv[:1] == ["smoke"] and len(argv) in (2, 3):
        return smoke(*argv[1:])
    if argv[:1] == ["prompt"]:
        job = json.loads(sys.stdin.read())
        try:
            print(json.dumps({"text": prompt_model(_config(), job["model"], job["prompt"], timeout_s=job.get("timeout", 900))}))
        except Exception as e:  # noqa: BLE001 - the parent decides what a failure means
            print(json.dumps({"error": str(e)}))
            return 1
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
