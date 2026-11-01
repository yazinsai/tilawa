# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_cursor.py"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_cursor as C  # noqa: E402
import autolab_exec  # noqa: E402


class AgentError(Exception):
    def __init__(self, message, is_retryable=False):
        super().__init__(message)
        self.message, self.is_retryable, self.retry_after = message, is_retryable, None


class FakeRun:
    def __init__(self, sdk, messages, status="finished", result="all done"):
        self.sdk, self.id, self.status, self._messages, self._status, self._result = sdk, "run-1", "running", messages, status, result
        self.cancelled = False

    def messages(self):
        for m in self._messages:
            yield m() if callable(m) else m

    def wait(self):
        self.status = "cancelled" if self.cancelled else self._status
        return NS(id=self.id, status=self.status, result=self._result, error_message="boom" if self._status == "error" else None,
                  error_code=None, duration_ms=1234, usage=None)

    def supports(self, op):
        return True

    def cancel(self):
        self.cancelled = True


class FakeAgent:
    def __init__(self, sdk, run):
        self.sdk, self.agent_id, self.run, self.closed = sdk, "agent-1", run, False

    def send(self, message):
        self.sdk.sent.append(message)
        if self.sdk.send_error:
            raise self.sdk.send_error
        return self.run

    def get_usage(self):
        return NS(cost=NS(charged_cents=42, raw_cost_cents=40))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeClient:
    def __init__(self, sdk, workspace):
        sdk.bridges.append(workspace)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def fake_sdk(messages=(), status="finished", create_error=None, send_error=None, models=("grok-4.7", "gpt-5.6-sol"), on_create=None):
    sdk = NS(sent=[], created=[], bridges=[], agents=[], send_error=send_error, CursorAgentError=AgentError)
    sdk.run = FakeRun(sdk, list(messages), status)

    def create(options=None, **kw):
        sdk.created.append(options or NS(**kw))
        if on_create:
            on_create(options or NS(**kw))
        if create_error:
            raise create_error
        a = FakeAgent(sdk, sdk.run)
        sdk.agents.append(a)
        return a

    sdk.Agent = NS(create=create)
    sdk.Client = NS(launch_bridge=lambda workspace=None, **_: FakeClient(sdk, workspace))
    sdk.Cursor = NS(models=NS(list=lambda **_: [NS(id=m, display_name=m, variants=()) for m in models]))
    sdk.LocalAgentOptions = lambda **kw: NS(**kw)
    sdk.AgentOptions = lambda **kw: NS(**kw)
    return sdk


def assistant(text):
    return NS(type="assistant", message=NS(content=[NS(type="text", text=text)]))


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("CURSOR_API_KEY", "cursor_test")


REAL_POPEN = subprocess.Popen


def fake_popen(runner):
    """Fakes only the runner launch; git and test helpers still get real processes."""
    return lambda argv, **kw: REAL_POPEN(argv, **kw) if argv[0] in ("git", "true", "sleep") else runner(argv, **kw)


def dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def executor(**cfg):
    return C.CursorLocalExecutor({"executor": cfg}, check=False)


# --- start / poll / collect ---

def test_start_writes_job_and_launches_detached_runner(repo, key, monkeypatch):
    seen = {}
    monkeypatch.setattr(C.subprocess, "Popen", fake_popen(lambda argv, **kw: seen.update(argv=argv, **kw) or NS(pid=4242)))
    monkeypatch.setenv("EXEC_MODAL_ID", "tok-id")
    monkeypatch.setenv("SONIOX_API_KEY", "secret")
    ex = executor(env={"MODAL_ENVIRONMENT": "autolab-exec", "MODAL_TOKEN_ID": "$EXEC_MODAL_ID"}, envDrop=["SONIOX_API_KEY"])
    h = ex.start({"id": "checker-x"}, repo, "do the card")
    job = json.loads(Path(h["job"]).read_text())
    assert job["brief"] == "do the card" and job["model"] == "grok-4.7" and job["worktree"] == str(repo.resolve())
    assert seen["argv"][-2:] == ["run", h["job"]] and seen["argv"][-3].endswith("autolab_cursor.py")
    assert seen["start_new_session"] and seen["cwd"] == repo.resolve()
    env = seen["env"]
    assert env["MODAL_ENVIRONMENT"] == "autolab-exec" and env["MODAL_TOKEN_ID"] == "tok-id" and "SONIOX_API_KEY" not in env
    assert env["CURSOR_API_KEY"] == "cursor_test"
    assert h["pgid"] == 4242 and Path(h["dir"]).parent == repo.resolve() / ".autolab"
    assert json.loads(Path(h["status"]).read_text())["status"] == "starting"
    assert ".autolab/" in (repo / ".git" / "info" / "exclude").read_text().split("\n")
    assert json.loads(json.dumps(h)) == h
    ex.start({"id": "checker-y", "executorModel": "gpt-5.6-sol"}, repo, "b")
    assert (repo / ".git" / "info" / "exclude").read_text().count(".autolab/") == 1
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=False).stdout == ""


def test_start_needs_key_and_named_env(repo, monkeypatch):
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
        executor().start({"id": "x"}, repo, "b")
    monkeypatch.setenv("CURSOR_API_KEY", "k")
    monkeypatch.delenv("NOPE", raising=False)
    monkeypatch.setattr(C.subprocess, "Popen", fake_popen(lambda *a, **k: NS(pid=1)))
    with pytest.raises(RuntimeError, match="NOPE"):
        executor(env={"MODAL_TOKEN_ID": "$NOPE"}).start({"id": "x"}, repo, "b")


def handle(tmp_path, status, pgid):
    (tmp_path / "status.json").write_text(json.dumps(status))
    return {"status": str(tmp_path / "status.json"), "transcript": str(tmp_path / "transcript.txt"), "pgid": pgid,
            "started": "2026-11-01T00:00:00+00:00", "worktree": str(tmp_path)}


def test_poll_maps_status_and_dead_pids(tmp_path):
    ex = executor()
    assert ex.poll(handle(tmp_path, {"status": "finished"}, dead_pid())) == "finished"
    assert ex.poll(handle(tmp_path, {"status": "failed"}, os.getpgid(0))) == "failed"
    h = handle(tmp_path, {"status": "running", "agentId": "a-9", "runId": "r-9"}, os.getpgid(0))
    assert ex.poll(h) == "running" and (h["agentId"], h["runId"]) == ("a-9", "r-9")
    for st in ("running", "starting"):
        h = handle(tmp_path, {"status": st}, dead_pid())
        assert ex.poll(h) == "failed"
        assert json.loads(Path(h["status"]).read_text())["failure"] == "runner-died"


def test_collect_returns_transcript_summary_hours(tmp_path):
    h = handle(tmp_path, {"status": "finished", "started": "2026-11-01T00:00:00+00:00", "ended": "2026-11-01T01:30:00+00:00",
                          "result": "added the guard", "agentId": "a", "runId": "r", "costCents": 42}, 1)
    Path(h["transcript"]).write_text("hello\n[tool edit completed]\n")
    out = executor().collect(h)
    assert out["transcript"].startswith("hello") and out["summary"] == "added the guard" and out["hours"] == 1.5
    assert out["costCents"] == 42
    h = handle(tmp_path, {"status": "failed", "failure": "startup", "error": "401", "started": "2026-11-01T00:00:00+00:00",
                          "ended": "2026-11-01T00:06:00+00:00"}, 1)
    out = executor().collect(h)
    assert out["summary"] == "startup: 401" and out["hours"] == 0.1


def test_cancel_marks_dead_runner_cancelled(tmp_path):
    h = handle(tmp_path, {"status": "running"}, dead_pid())
    executor().cancel(h)
    assert json.loads(Path(h["status"]).read_text())["failure"] == "cancelled"


def test_cancel_terminates_the_process_group(tmp_path):
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    h = handle(tmp_path, {"status": "running"}, p.pid)
    t = time.time()
    executor(cancelGraceS=5).cancel(h)
    assert time.time() - t < 4 and p.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(p.pid, 0)
    assert json.loads(Path(h["status"]).read_text())["status"] == "failed"


# --- runner ---

def job(tmp_path):
    d = tmp_path / ".autolab" / "j"
    d.mkdir(parents=True)
    j = {"card": "checker-x", "worktree": str(tmp_path), "brief": "do it", "model": "grok-4.7", "status": str(d / "status.json"),
         "transcript": str(d / "transcript.txt"), "messages": str(d / "messages.jsonl")}
    (d / "job.json").write_text(json.dumps(j))
    (d / "status.json").write_text(json.dumps({"status": "starting", "started": "2026-11-01T00:00:00+00:00"}))
    return str(d / "job.json"), j


def status(j):
    return json.loads(Path(j["status"]).read_text())


def test_runner_writes_running_then_finished(tmp_path, key):
    path, j = job(tmp_path)
    seen = []
    sdk = fake_sdk([assistant("Done: "), NS(type="tool_call", name="edit", status="completed"), assistant("added it.")],
                   on_create=lambda o: seen.append(status(j)["status"]))
    assert C.run(path, sdk) == 0
    st = status(j)
    assert seen == ["running"] and st["status"] == "finished" and (st["agentId"], st["runId"]) == ("agent-1", "run-1")
    assert st["result"] == "all done" and st["costCents"] == 42 and st["ended"]
    assert "Done: " in Path(j["transcript"]).read_text() and "[tool edit completed]" in Path(j["transcript"]).read_text()
    assert len(Path(j["messages"]).read_text().splitlines()) == 3
    opts = sdk.created[0]
    assert opts.api_key == "cursor_test" and opts.model == "grok-4.7" and opts.local.cwd == str(tmp_path)
    assert not getattr(opts.local, "setting_sources", None)
    assert "CURSOR_API_KEY" not in os.environ and sdk.bridges == [str(tmp_path)] and sdk.agents[0].closed
    assert sdk.sent == ["do it"]


def test_runner_startup_failure(tmp_path, key):
    path, j = job(tmp_path)
    assert C.run(path, fake_sdk(create_error=AgentError("bad key", is_retryable=False))) == 1
    st = status(j)
    assert st["status"] == "failed" and st["failure"] == "startup" and st["error"] == "bad key" and "agentId" not in st
    path, j = job(tmp_path / "b")
    sdk = fake_sdk(send_error=AgentError("rate", is_retryable=True))
    assert C.run(path, sdk) == 1
    assert status(j)["failure"] == "startup" and status(j)["retryable"] and sdk.agents[0].closed


def test_runner_error_status(tmp_path, key):
    path, j = job(tmp_path)
    sdk = fake_sdk([assistant("trying")], status="error")
    assert C.run(path, sdk) == 2
    st = status(j)
    assert st["status"] == "failed" and st["failure"] == "run-error" and st["runStatus"] == "error" and st["error"] == "boom"
    assert sdk.agents[0].closed


def test_runner_sigterm_cancels_the_live_run(tmp_path, key):
    path, j = job(tmp_path)
    old = signal.getsignal(signal.SIGTERM)
    try:
        sdk = fake_sdk([assistant("a"), lambda: os.kill(os.getpid(), signal.SIGTERM) or assistant("b"), assistant("c")])
        C.run(path, sdk)
    finally:
        signal.signal(signal.SIGTERM, old)
    st = status(j)
    assert st["status"] == "failed" and st["failure"] == "cancelled" and sdk.run.cancelled and sdk.agents[0].closed


# --- orchestrator one-shot and models ---

def test_prompt_model_runs_in_an_empty_temp_dir(key):
    seen = {}

    def look(o):
        cwd = Path(o.local.cwd)
        seen.update(cwd=cwd, empty=cwd.is_dir() and not any(cwd.iterdir()), tools=o.tools)

    sdk = fake_sdk(on_create=look)
    assert C.prompt_model({}, "gpt-5.6-sol", "judge this", sdk=sdk) == "all done"
    assert seen["empty"] and seen["tools"] == [] and not seen["cwd"].exists() and Path.cwd() not in seen["cwd"].parents
    assert sdk.bridges == [str(seen["cwd"])] and sdk.sent == ["judge this"] and sdk.agents[0].closed


def test_prompt_model_raises_on_startup_failure_and_error_status(key):
    with pytest.raises(RuntimeError, match="did not start"):
        C.prompt_model({}, "m", "p", sdk=fake_sdk(create_error=AgentError("401")))
    with pytest.raises(RuntimeError, match="ended error"):
        C.prompt_model({}, "m", "p", sdk=fake_sdk(status="error"))


def test_models_checked_at_startup(key):
    C.CursorLocalExecutor({"executor": {"model": "grok-4.7"}}, sdk=fake_sdk())
    with pytest.raises(RuntimeError, match="not available.*grok-4.7"):
        C.CursorLocalExecutor({"executor": {"model": "grok-9"}}, sdk=fake_sdk())


def test_make_picks_cursor_local(monkeypatch):
    monkeypatch.setattr(C, "check_models", lambda cfg, sdk=None: [])
    assert isinstance(autolab_exec.make({"executor": {"runtime": "cursor-local"}}), C.CursorLocalExecutor)
    assert "cursor_sdk" not in sys.modules


# --- isolation (spec decision 14) ---

def test_runner_env_drops_modal_and_moves_home(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / ".gitconfig").write_text("[user]\n\tname = r\n")
    base = {"HOME": str(real), "MODAL_TOKEN_ID": "t", "MODAL_PROFILE": "p", "SONIOX_API_KEY": "s", "PATH": "/bin"}
    home = tmp_path / "wt/.autolab/home"
    env = C.runner_env({"envDrop": ["SONIOX_*"], "env": {"X": "1"}}, base, home=home)
    assert not [k for k in env if k.startswith(("MODAL_", "SONIOX_"))]
    assert env["HOME"] == str(home) and home.is_dir() and env["GIT_CONFIG_GLOBAL"] == str(real / ".gitconfig")
    assert "NVM_DIR" not in env and env["PATH"] == "/bin" and env["X"] == "1"
    assert C.runner_env({}, base)["HOME"] == str(real)


@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="macOS sandbox")
def test_the_seatbelt_profile_keeps_a_builder_in_its_worktree(tmp_path):
    home = (tmp_path / "home").resolve()
    main, wt = home / "main", home / "wt"
    home.mkdir()
    (home / ".modal.toml").write_text("token_id = 'x'")
    git = lambda *a, cwd=main: subprocess.run(["git", "-c", "user.name=a", "-c", "user.email=a@b", *a], cwd=cwd, capture_output=True, check=True)  # noqa: E731
    main.mkdir()
    git("init", "-q", "-b", "main")
    git("commit", "-q", "--allow-empty", "-m", "base")
    git("worktree", "add", "-q", "-b", "exp/x", str(wt))
    profile = C.seatbelt(wt, home=home)
    sh = lambda c: subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, "bash", "-c", c], cwd=wt, capture_output=True, text=True).returncode  # noqa: E731
    assert sh(f"cat {home}/.modal.toml") != 0
    assert sh(f"touch {main}/x") != 0 and sh(f"touch {home}/.zshrc") != 0
    assert sh("git -c user.name=a -c user.email=a@b commit -q --allow-empty -m mine") == 0
    assert sh("git branch -f main HEAD") != 0
    assert sh(f"touch {wt}/.autolab-x && cat {main}/.git/HEAD") == 0
    assert sh("ps eww -o command= -p 1") != 0
