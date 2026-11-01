# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_codex.py"""

import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_codex as K  # noqa: E402
import autolab_exec as X  # noqa: E402

FAKE = """#!{py}
import json, os, sys, time
args = sys.argv[1:]
prompt = sys.stdin.read()
out = os.environ.get("FAKE_CODEX_OUT")
if out:
    json.dump({{"argv": args, "prompt": prompt, "env": dict(os.environ)}}, open(out, "w"))
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
say = lambda d: print(json.dumps(d), flush=True)
say({{"type": "thread.started", "thread_id": "th-1"}})
say({{"type": "turn.started"}})
if mode == "hang":
    time.sleep(60)
if mode == "silent":
    say({{"type": "error", "message": "Reconnecting... 1/5 (error sending request for url)"}})
    time.sleep(60)
if mode == "fail":
    say({{"type": "error", "message": "Reconnecting... 1/5 (unexpected status 401 Unauthorized)"}})
    say({{"type": "turn.failed", "error": {{"message": "unexpected status 401 Unauthorized: wrong API endpoint"}}}})
    sys.exit(1)
say({{"type": "item.completed", "item": {{"id": "i1", "type": "command_execution", "command": "bash -lc 'ls'", "exit_code": 0, "aggregated_output": "a"}}}})
if mode == "die":
    sys.exit(3)
say({{"type": "item.completed", "item": {{"id": "i2", "type": "file_change", "changes": [{{"path": "x.py", "kind": "update"}}]}}}})
say({{"type": "item.completed", "item": {{"id": "i3", "type": "agent_message", "text": "Changed x.py."}}}})
say({{"type": "turn.completed", "usage": {{"input_tokens": 10, "output_tokens": 5}}}})
open(args[args.index("--output-last-message") + 1], "w").write("Changed x.py.")
"""


@pytest.fixture
def codex(tmp_path):
    p = tmp_path / "bin" / "codex"
    p.parent.mkdir()
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "wt"
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    return r


def arm(codex, **kw):
    return {"executor": {"name": "astra", "runtime": "codex-azure", "bin": codex, "seatbelt": False,
                         "azure": {"endpoint": "https://res.openai.azure.com", "deployment": "astra-dep", "apiVersion": ""}, **kw}}


# --- config ---

def test_base_url_follows_the_api_version():
    assert K.base_url("https://r.openai.azure.com/") == ("https://r.openai.azure.com/openai/v1", {})
    assert K.base_url("https://r.openai.azure.com", "v1") == ("https://r.openai.azure.com/openai/v1", {})
    assert K.base_url("https://r.openai.azure.com", "preview") == ("https://r.openai.azure.com/openai/v1", {"api-version": "preview"})
    assert K.base_url("https://r.openai.azure.com", "2025-04-01-preview") == ("https://r.openai.azure.com/openai", {"api-version": "2025-04-01-preview"})
    assert K.base_url("https://r.cognitiveservices.azure.com/openai/v1", "2025-04-01-preview")[0] == "https://r.cognitiveservices.azure.com/openai/v1"


def test_config_toml_has_the_azure_provider_and_no_approvals():
    cfg = {**K.DEFAULTS, **arm("codex", effort="xhigh", extraConfig={"model_context_window": 400000})["executor"]}
    cfg["azure"]["apiVersion"] = "2025-04-01-preview"
    doc = tomllib.loads(K.config_toml(cfg))
    assert doc["model"] == "astra-dep" and doc["model_provider"] == "azure" and doc["model_reasoning_effort"] == "xhigh"
    assert doc["approval_policy"] == "never" and doc["sandbox_mode"] == "danger-full-access" and doc["model_context_window"] == 400000
    p = doc["model_providers"]["azure"]
    assert p == {"name": "Azure OpenAI", "base_url": "https://res.openai.azure.com/openai", "env_key": "AZURE_OPENAI_API_KEY",
                 "wire_api": "responses", "query_params": {"api-version": "2025-04-01-preview"}}
    assert "AZURE_*" in doc["shell_environment_policy"]["exclude"] and not doc["shell_environment_policy"]["ignore_default_excludes"]
    assert "requires_openai_auth" not in p and "key" not in json.dumps(doc).lower().replace("env_key", "").replace("_api_key", "")
    assert "query_params" not in tomllib.loads(K.config_toml({**cfg, "azure": {**cfg["azure"], "apiVersion": ""}}))["model_providers"]["azure"]
    assert tomllib.loads(K.config_toml({**cfg, "effort": None}))["model_reasoning_effort"] == "high"


def test_unready_names_what_is_missing(codex, tmp_path):
    cfg = arm(codex)["executor"]
    assert "endpoint" in K.unready({**cfg, "azure": {"deployment": "d"}}, {"AZURE_OPENAI_API_KEY": "k"})
    assert "deployment" in K.unready({**cfg, "azure": {"endpoint": "https://e"}}, {"AZURE_OPENAI_API_KEY": "k"})
    assert "AZURE_OPENAI_API_KEY" in K.unready(cfg, {})
    assert "not on PATH" in K.unready({**cfg, "bin": "no-such-codex-xyz"}, {"AZURE_OPENAI_API_KEY": "k"})
    assert K.unready(cfg, {"AZURE_OPENAI_API_KEY": "k"}) is None
    key = tmp_path / "key"
    key.write_text("from-file\n")
    with_file = {**cfg, "azure": {**cfg["azure"], "keyFile": str(key)}}
    assert K.unready(with_file, {}) is None and K.api_key(with_file, {}) == "from-file"


# --- start / poll / collect ---

def test_start_writes_config_and_launches_a_jailed_runner_with_the_key_only_for_codex(codex, repo, monkeypatch, tmp_path):
    seen = {}
    real = subprocess.Popen
    monkeypatch.setattr(K.subprocess, "Popen", lambda argv, **kw: real(argv, **kw) if argv[0] == "git" else seen.update(argv=argv, **kw) or NS(pid=4242))
    for k, v in {"AZURE_OPENAI_API_KEY": "az-key", "MODAL_TOKEN_ID": "t", "CURSOR_API_KEY": "c", "SONIOX_API_KEY": "s"}.items():
        monkeypatch.setenv(k, v)
    real_home = tmp_path / "realhome"
    (real_home / ".codex").mkdir(parents=True)
    (real_home / ".codex" / "auth.json").write_text('{"tokens": "chatgpt"}')
    monkeypatch.setenv("HOME", str(real_home))
    h = K.CodexExecutor(arm(codex, envDrop=["SONIOX_*"]), check=False).start({"id": "checker-x"}, repo, "do the card")
    assert seen["argv"][-2:] == ["run", h["job"]] and seen["argv"][-3].endswith("autolab_codex.py") and seen["start_new_session"]
    env = seen["env"]
    assert env["AZURE_OPENAI_API_KEY"] == "az-key" and not [k for k in env if k.startswith(("MODAL_", "CURSOR_", "SONIOX_"))]
    assert env["HOME"] == str(repo.resolve() / ".autolab" / "home")
    job = json.loads(Path(h["job"]).read_text())
    ch = Path(job["codexHome"])
    assert ch == repo.resolve() / ".autolab" / "codex-home" and sorted(p.name for p in ch.iterdir()) == ["config.toml"]
    assert tomllib.loads((ch / "config.toml").read_text())["model"] == "astra-dep" and "az-key" not in (ch / "config.toml").read_text()
    assert "az-key" not in Path(h["job"]).read_text() and job["brief"] == "do the card" and job["keyEnv"] == "AZURE_OPENAI_API_KEY"
    assert h["runtime"] == "codex-azure" and h["arm"] == "astra" and h["pgid"] == 4242 and json.loads(json.dumps(h)) == h
    assert json.loads(Path(h["status"]).read_text())["status"] == "starting"
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_start_refuses_an_incomplete_arm(codex, repo, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
        K.CodexExecutor(arm(codex), check=False).start({"id": "x"}, repo, "b")


def wait(ex, h, s=20):
    deadline = time.time() + s
    while (state := ex.poll(h)) == "running" and time.time() < deadline:
        time.sleep(0.1)
    return state


def test_a_real_detached_run_finishes_and_survives_a_supervisor_restart(codex, repo, monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("MODAL_TOKEN_ID", "t")
    monkeypatch.setenv("FAKE_CODEX_OUT", str(tmp_path / "codex-saw.json"))
    h = K.CodexExecutor(arm(codex), check=False).start({"id": "checker-x"}, repo, "do the card")
    stored = json.loads(json.dumps(h))
    assert wait(K.CodexExecutor(arm(codex), check=False), stored) == "finished"  # a fresh executor from the stored dict
    out = K.CodexExecutor(arm(codex), check=False).collect(stored)
    assert out["summary"] == "Changed x.py." and out["threadId"] == "th-1" and out["usage"]["output_tokens"] == 5 and out["hours"] >= 0
    assert "[shell exit 0] bash -lc 'ls'" in out["transcript"] and "[update x.py]" in out["transcript"]
    saw = json.loads((tmp_path / "codex-saw.json").read_text())
    assert saw["prompt"] == "do the card" and saw["env"]["AZURE_OPENAI_API_KEY"] == "az-key"
    assert saw["env"]["CODEX_HOME"] == str(repo.resolve() / ".autolab" / "codex-home") and not [k for k in saw["env"] if k.startswith("MODAL_")]
    for flag in ("--json", "--strict-config", "--dangerously-bypass-approvals-and-sandbox"):
        assert flag in saw["argv"]
    assert saw["argv"][saw["argv"].index("--cd") + 1] == str(repo.resolve()) and saw["argv"][-1] == "-"
    lines = Path(h["dir"], "messages.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["type"] == "thread.started" and len(lines) == 6


# --- runner, in process ---

def job(tmp_path, codex):
    d = tmp_path / ".autolab" / "j"
    d.mkdir(parents=True)
    j = {"card": "x", "worktree": str(tmp_path), "brief": "do it", "model": "astra-dep", "status": str(d / "status.json"),
         "transcript": str(d / "transcript.txt"), "messages": str(d / "messages.jsonl"), "last": str(d / "last-message.txt"),
         "codexLog": str(d / "codex.log"), "codexHome": str(d / "ch"), "bin": codex, "keyEnv": "AZURE_OPENAI_API_KEY"}
    (d / "job.json").write_text(json.dumps(j))
    (d / "status.json").write_text(json.dumps({"status": "starting", "started": "2026-11-01T00:00:00+00:00"}))
    return str(d / "job.json"), j


def status(j):
    return json.loads(Path(j["status"]).read_text())


def test_runner_writes_running_then_finished_and_drops_the_key_from_its_own_env(codex, tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    path, j = job(tmp_path, codex)
    seen = []
    real = subprocess.Popen

    def popen(argv, **kw):
        seen.append(status(j)["status"])
        assert kw["env"]["AZURE_OPENAI_API_KEY"] == "az-key" and "AZURE_OPENAI_API_KEY" not in os.environ
        return real(argv, **kw)

    assert K.run(path, popen) == 0
    st = status(j)
    assert seen == ["running"] and st["status"] == "finished" and st["result"] == "Changed x.py." and st["threadId"] == "th-1" and st["ended"]


def test_runner_failures_say_whether_codex_got_going(codex, tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("FAKE_CODEX_MODE", "fail")
    path, j = job(tmp_path / "a", codex)
    assert K.run(path) == 1
    st = status(j)
    assert st["status"] == "failed" and st["failure"] == "startup" and "401 Unauthorized" in st["error"]
    monkeypatch.setenv("FAKE_CODEX_MODE", "die")
    path, j = job(tmp_path / "b", codex)
    assert K.run(path) == 2
    st = status(j)
    assert st["failure"] == "run-error" and "exited 3" in st["error"]
    path, j = job(tmp_path / "c", str(tmp_path / "no-codex"))
    assert K.run(path) == 1 and status(j)["failure"] == "startup"


def test_a_model_that_never_answers_is_a_startup_failure_not_an_outage(codex, tmp_path, monkeypatch):
    import autolab as A
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("FAKE_CODEX_MODE", "silent")
    path, j = job(tmp_path, codex)
    Path(path).write_text(json.dumps({**j, "startupTimeoutS": 1}))
    t = time.time()
    assert K.run(path) == 1 and time.time() - t < 15
    st = status(j)
    assert st["failure"] == "startup" and "no answer from the model in 1 s" in st["error"] and "error sending request" in st["codexErrors"][0]
    out = K.CodexExecutor(arm(codex), check=False).collect({"status": j["status"], "transcript": j["transcript"], "started": "2026-11-01T00:00:00+00:00"})
    assert not A.outage(out["summary"])
    assert A.outage("startup: exceeded retry limit, last status: 429 Too Many Requests")


def test_poll_marks_a_dead_runner_and_cancel_kills_the_process_group(codex, repo, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("FAKE_CODEX_MODE", "hang")
    ex = K.CodexExecutor(arm(codex, cancelGraceS=5), check=False)
    h = ex.start({"id": "x"}, repo, "b")
    deadline = time.time() + 10
    while not json.loads(Path(h["status"]).read_text()).get("codexPid") and time.time() < deadline:
        time.sleep(0.1)
    codex_pid = json.loads(Path(h["status"]).read_text())["codexPid"]
    assert ex.poll(h) == "running"
    ex.cancel(h)
    assert ex.poll(h) == "failed" and json.loads(Path(h["status"]).read_text())["failure"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.killpg(h["pgid"], 0)
    with pytest.raises(ProcessLookupError):
        os.kill(codex_pid, 0)
    dead = subprocess.Popen(["true"])
    dead.wait()
    h2 = {**h, "pgid": dead.pid}
    Path(h2["status"]).write_text(json.dumps({"status": "running"}))
    assert ex.poll(h2) == "failed" and json.loads(Path(h2["status"]).read_text())["failure"] == "runner-died"


# --- arms in autolab_exec ---

CFG = {"executor": {"runtime": "cursor-local", "model": "grok", "fallbackModel": "grok-2", "envDrop": ["SONIOX_*"],
                    "codex": {"effort": "medium", "azure": {"apiVersion": "preview"}},
                    "runtimes": [{"name": "grok", "runtime": "cursor-local"},
                                 {"name": "astra", "runtime": "codex-azure", "effort": "high", "azure": {"endpoint": "https://e", "deployment": "astra-dep"}}],
                    "split": {"grok": 1, "astra": 3}}}


def test_arms_merge_shared_settings_and_keep_the_old_single_runtime_config():
    grok, astra = X.arms(CFG)
    g, a = X.arm_config(CFG, grok)["executor"], X.arm_config(CFG, astra)["executor"]
    assert X.arm_models(g) == ("grok", "grok-2") and g["envDrop"] == ["SONIOX_*", "AZURE_OPENAI_API_KEY"]
    assert X.arm_models(a) == ("astra-dep", "astra-dep") and a["effort"] == "high" and a["azure"]["apiVersion"] == "preview"
    assert "model" not in a and a["envDrop"] == ["SONIOX_*"] and "runtimes" not in a
    assert X.weights(CFG) == {"grok": 1.0, "astra": 3.0}
    legacy = {"executor": {"runtime": "cursor-local", "model": "grok", "fallbackModel": "grok-2"}}
    assert X.arms(legacy) == [{"name": "cursor-local", "runtime": "cursor-local"}]
    assert X.arm_models(X.arm_config(legacy, X.arms(legacy)[0])["executor"]) == ("grok", "grok-2")


def test_unready_per_runtime(monkeypatch):
    grok, astra = X.arms(CFG)
    assert "CURSOR_API_KEY" in X.unready(CFG, grok, {}) and X.unready(CFG, grok, {"CURSOR_API_KEY": "c"}) is None
    monkeypatch.setattr(K, "codex_bin", lambda cfg: "/bin/codex")
    assert "AZURE_OPENAI_API_KEY" in X.unready(CFG, astra, {}) and X.unready(CFG, astra, {"AZURE_OPENAI_API_KEY": "k"}) is None
    empty = {**astra, "azure": {"endpoint": "", "deployment": ""}}
    assert "endpoint" in X.unready(CFG, empty, {"AZURE_OPENAI_API_KEY": "k"})


def test_make_picks_codex_without_checks():
    assert isinstance(X.make(CFG, X.arms(CFG)[1], check=False), K.CodexExecutor)
