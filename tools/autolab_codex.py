"""Autolab executor on the Codex CLI with an Azure OpenAI deployment (.scratch/autolab/spec.md decisions 1 and 14): the
`codex-azure` runtime behind tools/autolab_exec.py, used by A/B arms such as GPT-6 Astra next to the Cursor (Grok) arm.

    CodexExecutor(config)                          start/poll/cancel/collect; config["executor"] is one arm's settings
    config_toml(cfg) / base_url(endpoint, ver)     the generated $CODEX_HOME/config.toml and the provider URL
    unready(cfg, environ)                          why this arm can't run now (missing Azure field, key or binary), or None

    python3 tools/autolab_codex.py run <job.json>                     the detached runner (start launches it)
    python3 tools/autolab_codex.py config [--arm NAME]                print the config.toml an arm would get
    python3 tools/autolab_codex.py probe [--arm NAME]                 offline check: codex against a local 401 server
    python3 tools/autolab_codex.py smoke <scratch-worktree> [--arm NAME]           live check: one trivial card
    python3 tools/autolab_codex.py smoke-isolation <scratch-worktree> [--arm NAME] live check: no way out

Same restart-survival design as tools/autolab_cursor.py: `start` launches this file's runner in its own session, the
runner owns `codex exec`, and everything lives in `<worktree>/.autolab/<job>/` (job.json, brief.md, status.json,
transcript.txt, messages.jsonl (codex's JSON events), last-message.txt, codex.log, runner.log), so poll/cancel/collect
need only the stored handle. Cancel is SIGTERM then SIGKILL to the runner's process group, which holds codex too.

Config, one arm (tools/autolab_exec.arm_config merges `executor.codex` defaults, then the arm):
`azure.endpoint` (https://<resource>.openai.azure.com), `azure.deployment` (sent as the model), `azure.apiVersion`
(empty or "v1": the v1 API, <endpoint>/openai/v1 with no api-version; "preview"/"latest": v1 with that api-version; a
dated version like 2025-04-01-preview: <endpoint>/openai with it), `azure.keyEnv` (default AZURE_OPENAI_API_KEY),
`azure.keyFile` (optional; read when keyEnv isn't set, so a key can be added without restarting the loop), `effort`
(model_reasoning_effort, default high), `extraConfig` (more top-level config.toml keys, e.g. model_context_window),
`bin`, `cancelGraceS`, `home`, `seatbelt`, `env`, `envDrop` as for the Cursor runtime.

Isolation, as the Cursor runtime: HOME is <worktree>/.autolab/home, CODEX_HOME is <worktree>/.autolab/codex-home with
only the generated config.toml (no ChatGPT login, no copy of ~/.codex: none of its auth, MCP servers, skills or hooks),
no MODAL_* and no CURSOR_API_KEY, the same Seatbelt profile (CODEX_HOME is inside the worktree, so it's writable). The
key reaches the runner in its environment; the runner pops it and hands it to the codex process only, and
shell_environment_policy keeps it (and any *_API_KEY, *TOKEN*, *SECRET*) out of the agent's shells.

Codex's own sandbox is off (`--dangerously-bypass-approvals-and-sandbox`, approval_policy never): our Seatbelt profile
is the jail. Nesting works on macOS, but codex's workspace-write mode keeps .git read-only and the network off, so the
builder couldn't commit (the git dir is outside the worktree) and two policies would have to agree. Ours already
allows exactly the worktree, its git dir, the shared objects and its own branch ref.
"""

import argparse
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_runner as R  # noqa: E402

DEFAULTS = {"bin": "codex", "effort": "high", "cancelGraceS": 30, "home": True, "seatbelt": True, "startupTimeoutS": 300}
KEY_ENV = "AZURE_OPENAI_API_KEY"
FLAGS = ("--json", "--strict-config", "--dangerously-bypass-approvals-and-sandbox", "--output-last-message", "--cd")
SHELL_EXCLUDE = ["AZURE_*", "*_API_KEY", "*TOKEN*", "*SECRET*", "CURSOR_*", "MODAL_*"]
ACTED = ("agent_message", "command_execution", "file_change", "mcp_tool_call")


def azure(cfg):
    a = cfg.get("azure") or {}
    return {"endpoint": str(a.get("endpoint") or "").strip(), "deployment": str(a.get("deployment") or "").strip(),
            "apiVersion": str(a.get("apiVersion") or "").strip(), "keyEnv": a.get("keyEnv") or KEY_ENV, "keyFile": a.get("keyFile") or ""}


def base_url(endpoint, api_version=""):
    """(base_url, query_params) for Codex's Responses provider: Codex posts to <base_url>/responses?<query_params>."""
    e = endpoint.rstrip("/")
    dated = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}(-preview)?", api_version))
    if not re.search(r"/openai(/v1)?$", e):
        e += "/openai" if dated else "/openai/v1"
    return e, ({"api-version": api_version} if api_version and api_version != "v1" else {})


def api_key(cfg, environ=None):
    a, env = azure(cfg), os.environ if environ is None else environ
    if env.get(a["keyEnv"], "").strip():
        return env[a["keyEnv"]].strip()
    if a["keyFile"]:
        p = Path(a["keyFile"]).expanduser()
        if p.is_file():
            return p.read_text().strip()
    return ""


def codex_bin(cfg):
    return shutil.which(cfg.get("bin") or DEFAULTS["bin"]) or ""


def unready(cfg, environ=None):
    a, name = azure(cfg), cfg.get("name") or "codex"
    if not a["endpoint"]:
        return f"no Azure endpoint (executor.runtimes[{name}].azure.endpoint)"
    if not a["deployment"]:
        return f"no Azure deployment (executor.runtimes[{name}].azure.deployment)"
    if not api_key(cfg, environ):
        return f"${a['keyEnv']} is not set in the loop's environment" + (f" and {a['keyFile']} is empty" if a["keyFile"] else "")
    if not codex_bin(cfg):
        return f"the codex CLI ({cfg.get('bin') or 'codex'}) is not on PATH"
    return None


@functools.cache
def missing_flags(binary):
    """Flags this runtime passes that `codex exec --help` doesn't list (a codex upgrade that renamed one)."""
    out = subprocess.run([binary, "exec", "--help"], capture_output=True, text=True, timeout=30).stdout
    return [f for f in FLAGS if f not in out]


def _toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {_toml(x)}" for k, x in v.items()) + " }"
    raise ValueError(f"can't write {v!r} to config.toml")


def config_toml(cfg):
    """$CODEX_HOME/config.toml for one arm: the Azure provider, the deployment as the model, no approvals, codex's sandbox
    off (ours wraps it), and no key in the agent's shells."""
    a = azure(cfg)
    url, query = base_url(a["endpoint"], a["apiVersion"])
    top = {"model": a["deployment"], "model_provider": "azure", "model_reasoning_effort": cfg.get("effort") or DEFAULTS["effort"],
           "approval_policy": "never", "sandbox_mode": "danger-full-access", **(cfg.get("extraConfig") or {})}
    provider = {"name": "Azure OpenAI", "base_url": url, "env_key": a["keyEnv"], "wire_api": "responses", **({"query_params": query} if query else {})}
    shell = {"inherit": "all", "ignore_default_excludes": False, "exclude": SHELL_EXCLUDE}
    lines = [f"{k} = {_toml(v)}" for k, v in top.items() if not isinstance(v, dict)]
    for title, table in (("model_providers.azure", provider), ("shell_environment_policy", shell),
                         *((k, v) for k, v in top.items() if isinstance(v, dict))):
        lines += ["", f"[{title}]", *(f"{k} = {_toml(v)}" for k, v in table.items())]
    return "\n".join(lines) + "\n"


class CodexExecutor:
    def __init__(self, config, check=True):
        self.cfg = {**DEFAULTS, **config.get("executor", {})}
        if check:
            if why := unready(self.cfg):
                raise RuntimeError(f"codex arm {self.cfg.get('name')}: {why}")
            if gone := missing_flags(codex_bin(self.cfg)):
                raise RuntimeError(f"codex exec no longer takes {', '.join(gone)}; update tools/autolab_codex.py")

    def start(self, card, worktree, brief):
        worktree = Path(worktree).resolve()
        if why := unready(self.cfg):
            raise RuntimeError(f"codex arm {self.cfg.get('name')}: {why}")
        R.exclude_autolab(worktree)
        job_id = f"{card['id']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        d = worktree / ".autolab" / job_id
        d.mkdir(parents=True)
        a = azure(self.cfg)
        model = card.get("executorModel") or a["deployment"]
        if isinstance(model, dict):
            model = model.get("id") or a["deployment"]
        home = worktree / ".autolab" / "home" if self.cfg["home"] else None
        codex_home = worktree / ".autolab" / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text(config_toml({**self.cfg, "azure": {**self.cfg.get("azure", {}), "deployment": model}}))
        h = {"runtime": "codex-azure", "arm": self.cfg.get("name"), "id": job_id, "worktree": str(worktree), "dir": str(d),
             "job": str(d / "job.json"), "status": str(d / "status.json"), "transcript": str(d / "transcript.txt"), "model": model,
             "started": R.now(), "pgid": None, "threadId": None}
        R.write(h["job"], {"card": card["id"], "worktree": h["worktree"], "brief": brief, "model": model, "status": h["status"],
                           "transcript": h["transcript"], "messages": str(d / "messages.jsonl"), "last": str(d / "last-message.txt"),
                           "codexLog": str(d / "codex.log"), "codexHome": str(codex_home), "bin": codex_bin(self.cfg), "keyEnv": a["keyEnv"],
                           "startupTimeoutS": self.cfg["startupTimeoutS"]})
        R.write(h["status"], {"status": "starting", "card": card["id"], "model": model, "started": h["started"]})
        env = R.runner_env(self.cfg, home=home, drop=("CURSOR_API_KEY", "CURSOR_*"))
        env[a["keyEnv"]] = api_key(self.cfg)
        py = self.cfg.get("python") or [sys.executable]
        h["pgid"] = R.launch(self.cfg, {**h, "worktree": worktree}, [*([py] if isinstance(py, str) else py), str(Path(__file__).resolve()), "run", h["job"]], env)
        R.update(h["status"], pgid=h["pgid"])
        return h

    def poll(self, handle):
        st = R.read(handle["status"])
        if st.get("threadId"):
            handle["threadId"] = st["threadId"]
        return R.poll(handle)

    def cancel(self, handle):
        R.cancel(handle, self.cfg["cancelGraceS"])

    def collect(self, handle):
        st = R.read(handle["status"])
        p = Path(handle["transcript"])
        summary = st.get("result") or ""
        if st.get("status") == "failed":
            summary = f"{st.get('failure') or 'failed'}: {st.get('error') or ''}\n{summary}".strip()
        return {"transcript": p.read_text() if p.exists() else "", "summary": summary,
                "hours": R.hours(st.get("started") or handle["started"], st.get("ended") or R.now()),
                "threadId": st.get("threadId"), "usage": st.get("usage")}


# --- runner (a separate process per job) ---

def render(ev):
    """One codex JSON event as transcript text."""
    t, item = ev.get("type"), ev.get("item") or {}
    if t == "item.completed":
        k = item.get("type")
        if k == "agent_message":
            return (item.get("text") or "") + "\n"
        if k == "command_execution":
            return f"\n[shell exit {item.get('exit_code')}] {item.get('command')}\n"
        if k == "file_change":
            return "".join(f"\n[{c.get('kind')} {c.get('path')}]" for c in item.get("changes") or []) + "\n"
        if k == "mcp_tool_call":
            return f"\n[tool {item.get('server')}.{item.get('tool')} {item.get('status')}]\n"
        if k == "error":
            return f"\n[note {item.get('message')}]\n"
        return ""
    if t in ("error", "turn.failed"):
        return f"\n[{t} {ev.get('message') or (ev.get('error') or {}).get('message')}]\n"
    return ""


def run(job_path, popen=None):
    """Drive one job to a terminal status.json; returns the exit code (0 finished, 1 failed before acting, 2 run failure)."""
    job = json.loads(Path(job_path).read_text())
    st = job["status"]
    key = os.environ.pop(job["keyEnv"], "")  # the agent's shells inherit codex's env through its policy, never ours
    R.update(st, status="running", pid=os.getpid(), runnerStarted=R.now())
    signal.signal(signal.SIGTERM, R.on_term)
    brief = Path(job["last"]).with_name("brief.md")
    brief.write_text(job["brief"])
    argv = [job["bin"] or "codex", "exec", "--json", "--strict-config", "--dangerously-bypass-approvals-and-sandbox",
            "--cd", job["worktree"], "--output-last-message", job["last"], "-"]
    env = {**os.environ, "CODEX_HOME": job["codexHome"], job["keyEnv"]: key}
    p, acted, failed, errors, last, usage, answered, silent = None, False, None, [], "", None, threading.Event(), []
    wait_s = job.get("startupTimeoutS", DEFAULTS["startupTimeoutS"])

    def no_answer():
        if not answered.is_set() and p is not None and p.poll() is None:
            silent.append(wait_s)
            p.terminate()

    timer = threading.Timer(wait_s, no_answer)
    timer.daemon = True
    try:
        with open(brief) as stdin, open(job["codexLog"], "ab") as log, open(job["transcript"], "a") as tr, open(job["messages"], "a") as ms:
            p = (popen or subprocess.Popen)(argv, cwd=job["worktree"], env=env, stdin=stdin, stdout=subprocess.PIPE, stderr=log, text=True)
            R.update(st, codexPid=p.pid)
            timer.start()
            for line in p.stdout:
                ms.write(line if line.endswith("\n") else line + "\n")
                ms.flush()
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tr.write(render(ev))
                tr.flush()
                t, item = ev.get("type") or "", ev.get("item") or {}
                if (t.startswith("item.") and item.get("type") not in ("error", None)) or t == "turn.completed":
                    answered.set()
                if t == "thread.started":
                    R.update(st, threadId=ev.get("thread_id"))
                elif t == "item.completed" and item.get("type") in ACTED:
                    acted = True
                    if item.get("type") == "agent_message":
                        last = item.get("text") or last
                elif t == "turn.completed":
                    usage = ev.get("usage")
                elif t == "turn.failed":
                    failed = (ev.get("error") or {}).get("message") or "turn failed"
                elif t == "error":
                    errors.append(ev.get("message") or "")
            code = p.wait()
        timer.cancel()
        final = Path(job["last"]).read_text().strip() if Path(job["last"]).exists() else ""
        if code == 0 and not failed and not silent:
            R.update(st, status="finished", failure=None, result=final or last, exitCode=code, usage=usage, ended=R.now())
            return 0
        # a silent model is a broken arm (wrong endpoint or deployment), not an outage: no outage words in this error
        err = f"no answer from the model in {wait_s} s; check the Azure endpoint and deployment" if silent else \
            failed or (errors[-1] if errors else "") or f"codex exited {code}: " + _tail(job["codexLog"])
        R.update(st, status="failed", failure="run-error" if acted else "startup", error=err[:2000], exitCode=code,
                 result=final or last, usage=usage, codexErrors=errors[-3:], ended=R.now())
        return 2 if acted else 1
    except R.Cancelled:
        timer.cancel()
        _stop(p)
        R.update(st, status="failed", failure="cancelled", ended=R.now())
        return 2
    except Exception as e:  # noqa: BLE001 - codex missing or a crash; the supervisor reads status.json, not our exit
        timer.cancel()
        _stop(p)
        if R.read(st).get("status") not in R.TERMINAL:
            R.update(st, status="failed", failure="startup" if p is None else "crash", error=f"{type(e).__name__}: {e}", ended=R.now())
        return 1 if p is None else 2


def _stop(p):
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()


def _tail(path, n=600):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except FileNotFoundError:
        return ""


# --- CLI ---

def _config():
    p = Path(__file__).resolve().parent.parent / ".scratch" / "autolab" / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def arm_config(name=None):
    """The merged config of a codex arm from .scratch/autolab/config.json (the first codex-azure arm without a name)."""
    import autolab_exec as X  # noqa: PLC0415
    cfg = _config()
    arms = [a for a in X.arms(cfg) if a["runtime"] == "codex-azure" and (name is None or a["name"] == name)]
    if not arms:
        sys.exit(f"no codex-azure arm{' named ' + name if name else ''} in executor.runtimes")
    return X.arm_config(cfg, arms[0])


def probe(config):
    """codex exec against a local server that answers 401 like Azure: shows the URL, auth header and model it sends, and
    that a bad key or endpoint ends as a clean failure. Needs no key or endpoint."""
    from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            seen.append({"path": self.path, "bearer": (self.headers.get("authorization") or "").startswith("Bearer probe-key"),
                         "model": body.get("model"), "effort": (body.get("reasoning") or {}).get("effort")})
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"code":"401","message":"Access denied due to invalid subscription key or wrong API endpoint."}}')

        def log_message(self, *_):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = {**DEFAULTS, **config["executor"]}
    a = azure(cfg)
    cfg["azure"] = {**cfg.get("azure", {}), "endpoint": f"http://127.0.0.1:{srv.server_port}", "deployment": a["deployment"] or "probe-deployment",
                    "keyFile": ""}
    with tempfile.TemporaryDirectory(prefix="autolab-codex-probe-") as tmp:
        wt = Path(tmp) / "wt"
        wt.mkdir()
        subprocess.run(["git", "init", "-q", str(wt)], check=True)
        env = {**R.runner_env(cfg, home=Path(tmp) / "home", drop=("CURSOR_*",)), a["keyEnv"]: "probe-key"}
        R.write(Path(tmp) / "job.json", {"card": "probe", "worktree": str(wt), "brief": "Reply with one word.", "model": cfg["azure"]["deployment"],
                                         "status": f"{tmp}/status.json", "transcript": f"{tmp}/transcript.txt", "messages": f"{tmp}/messages.jsonl",
                                         "last": f"{tmp}/last-message.txt", "codexLog": f"{tmp}/codex.log", "codexHome": f"{tmp}/codex-home",
                                         "bin": codex_bin(cfg), "keyEnv": a["keyEnv"]})
        Path(tmp, "codex-home").mkdir()
        Path(tmp, "codex-home", "config.toml").write_text(config_toml(cfg))
        R.write(Path(tmp) / "status.json", {"status": "starting", "started": R.now()})
        t = time.time()
        p = subprocess.run([*R.jail(cfg, wt), sys.executable, str(Path(__file__).resolve()), "run", f"{tmp}/job.json"], env=env, timeout=300, check=False)
        st = R.read(Path(tmp) / "status.json")
        print(config_toml(cfg))
        print(json.dumps({"runnerExit": p.returncode, "seconds": round(time.time() - t, 1), "status": st.get("status"),
                          "failure": st.get("failure"), "error": (st.get("error") or "")[:300], "requests": seen[:1], "requestCount": len(seen)}, indent=1))
    srv.shutdown()
    want_url, query = base_url(cfg["azure"]["endpoint"], a["apiVersion"])
    path = want_url.split(str(srv.server_port), 1)[1] + "/responses" + (f"?api-version={query['api-version']}" if query else "")
    ok = bool(seen) and seen[0]["path"] == path and seen[0]["bearer"] and seen[0]["model"] == cfg["azure"]["deployment"] \
        and st.get("status") == "failed" and st.get("failure") == "startup"
    print("probe ok: provider block accepted, request shape as expected, 401 ends as a clean startup failure" if ok else f"probe FAILED (wanted {path})")
    return 0 if ok else 1


def smoke(worktree, config):
    worktree = Path(worktree).resolve()
    file = next(str(p.relative_to(worktree)) for p in sorted((worktree / "tools").glob("test_*.py")))
    brief = (f"Smoke test of the autolab executor. Add exactly one line, the comment `# autolab smoke test`, as the last "
             f"line of {file}. Change nothing else, run nothing, do not commit. Then reply with one sentence.")
    state, h, out = R.drive(CodexExecutor(config), worktree, "smoke", brief)
    diff = subprocess.run(["git", "-C", str(worktree), "diff", "--", file], capture_output=True, text=True, check=False).stdout
    print(json.dumps({"state": state, **{k: v for k, v in out.items() if k != "transcript"}}, indent=1))
    print(diff or "(no diff)")
    return 0 if state == "finished" and "autolab smoke test" in diff else 1


def smoke_isolation(worktree, config):
    worktree = Path(worktree).resolve()
    repo = Path(__file__).resolve().parent.parent
    tries, probe_file = R.isolation_tries(repo, Path.home())
    state, h, _ = R.drive(CodexExecutor(config), worktree, "smoke-isolation", R.isolation_brief(tries))
    seen = {}
    for line in Path(h["dir"], "messages.jsonl").read_text().splitlines():
        ev = json.loads(line)
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "command_execution":
            cmd = item.get("command") or ""
            for c in tries:
                if c in cmd:
                    seen[c] = (item.get("exit_code"), (item.get("aggregated_output") or "").strip().splitlines()[:1])
    return R.isolation_report(tries, seen, state, probe_file)


def main(argv):
    if argv[:1] == ["run"] and len(argv) == 2:
        return run(argv[1])
    p = argparse.ArgumentParser(prog="autolab_codex.py")
    p.add_argument("op", choices=("config", "probe", "smoke", "smoke-isolation"))
    p.add_argument("worktree", nargs="?")
    p.add_argument("--arm")
    a = p.parse_args(argv)
    cfg = arm_config(a.arm)
    if a.op == "config":
        print(config_toml({**DEFAULTS, **cfg["executor"]}), end="")
        why = unready({**DEFAULTS, **cfg["executor"]})
        print(f"# {'not ready: ' + why if why else 'ready'}", file=sys.stderr)
        return 0
    if a.op == "probe":
        return probe(cfg)
    if not a.worktree:
        sys.exit(f"{a.op} needs a scratch worktree")
    return smoke(a.worktree, cfg) if a.op == "smoke" else smoke_isolation(a.worktree, cfg)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
