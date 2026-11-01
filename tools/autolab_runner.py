"""What every local builder runtime shares (.scratch/autolab/spec.md decision 14): the job files under
`<worktree>/.autolab/<job>/`, the detached runner's process group, and the builder's isolation (environment and macOS
Seatbelt profile). tools/autolab_cursor.py and tools/autolab_codex.py use it; neither imports the other.

    runner_env(cfg, base, home, drop)   the runner's environment: no MODAL_*, nothing in envDrop or `drop`, HOME moved
    seatbelt(worktree, home)            the sandbox profile: no Modal token, no writes under the real home outside the
                                        worktree, no reading other processes' arguments and environment
    jail(cfg, worktree)                 the sandbox-exec argv prefix ([] when off or not on macOS)
    launch(cfg, h, argv, env)           start a runner in its own session under the jail; returns its pgid
    poll(h) / cancel(h, grace) / hours  shared by every runtime's poll/cancel/collect
"""

import contextlib
import fnmatch
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = ("finished", "failed")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def hours(a, b):
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600, 4)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, doc):
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, path)


def update(path, **kv):
    doc = {**read(path), **kv, "updated": now()}
    write(path, doc)
    return doc


def alive(pgid):
    try:
        os.waitpid(pgid, os.WNOHANG)  # reap our own child so a zombie doesn't read as alive
    except ChildProcessError:
        pass
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def exclude_autolab(worktree):
    """Add `.autolab/` to the main repo's info/exclude (worktrees read the common dir's)."""
    common = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    path = (Path(worktree) / common).resolve() / "info" / "exclude"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    if ".autolab/" not in text.split("\n"):
        path.write_text(text + ("" if not text or text.endswith("\n") else "\n") + ".autolab/\n")


def runner_env(cfg, base=None, home=None, drop=()):
    """The runner's (and so the agent's shell's) environment: no MODAL_*, nothing in `envDrop` or `drop` (fnmatch
    patterns), HOME at `home` with git's identity and the read-only tool installs kept, then `env`."""
    env = dict(os.environ if base is None else base)
    pats = [*cfg.get("envDrop", []), *drop]
    for name in [k for k in env if k.startswith("MODAL_") or any(fnmatch.fnmatchcase(k, p) for p in pats)]:
        del env[name]
    if home:
        real = Path(env.get("HOME") or Path.home())
        Path(home).mkdir(parents=True, exist_ok=True)
        keep = {"GIT_CONFIG_GLOBAL": real / ".gitconfig", "NVM_DIR": real / ".nvm", "PYENV_ROOT": real / ".pyenv",
                "UV_PYTHON_INSTALL_DIR": real / ".local/share/uv/python"}
        env.update({k: str(v) for k, v in keep.items() if k not in env and v.exists()}, HOME=str(home))
    for name, value in cfg.get("env", {}).items():
        if isinstance(value, str) and value.startswith("$"):
            if value[1:] not in env:
                raise RuntimeError(f"executor env {name} wants ${value[1:]}, which is not set")
            value = env[value[1:]]
        env[name] = str(value)
    return env


def seatbelt(worktree, home=None):
    """A macOS sandbox profile for the runner's process tree (the agent and its shells inherit it): no reading the Modal
    token, no writing under the real home except this worktree and what its commits need (the objects and its own
    branch), and no reading other processes' arguments and environment (the supervisor's holds the API keys)."""
    git = lambda *a: subprocess.run(["git", "-C", str(worktree), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    wt, home = Path(worktree).resolve(), Path(home or Path.home()).resolve()
    common, gitdir = (wt / git("rev-parse", "--git-common-dir")).resolve(), (wt / git("rev-parse", "--git-dir")).resolve()
    branch = git("branch", "--show-current")
    q = lambda kind, p: f'({kind} "{p}")'  # noqa: E731
    writable = [q("subpath", wt), q("subpath", gitdir), q("subpath", common / "objects")]
    if branch:
        writable += [q("literal", common / d / "refs/heads" / f"{branch}{s}") for d in ("", "logs") for s in ("", ".lock")]
    return "\n".join([
        "(version 1)", "(allow default)",
        f"(deny file-write* {q('subpath', home)})",
        f"(allow file-write* {' '.join(writable)})",
        f"(deny file-read* file-write* {q('literal', home / '.modal.toml')} {q('subpath', home / '.modal')})",
        '(deny sysctl-read (sysctl-name "kern.procargs2"))',
    ])


def jail(cfg, worktree):
    return [SANDBOX_EXEC, "-p", seatbelt(worktree)] if cfg.get("seatbelt", True) and Path(SANDBOX_EXEC).exists() else []


def launch(cfg, h, argv, env, popen=None):
    """Starts `argv` in its own session (process group) under the jail, logging to <dir>/runner.log; returns the pgid."""
    with open(Path(h["dir"]) / "runner.log", "ab") as log:
        p = (popen or subprocess.Popen)([*jail(cfg, h["worktree"]), *argv], cwd=h["worktree"], env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid


def poll(handle):
    st = read(handle["status"])
    if st.get("status") in TERMINAL:
        return st["status"]
    if handle.get("pgid") and alive(handle["pgid"]):
        return "running"
    update(handle["status"], status="failed", failure="runner-died", ended=now())
    return "failed"


def cancel(handle, grace_s, last_resort=None):
    """SIGTERM to the runner's process group (the runner stops its agent and exits), then SIGKILL after `grace_s`."""
    pgid = handle.get("pgid")
    if pgid and alive(pgid):
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.time() + grace_s
        while time.time() < deadline and alive(pgid):
            time.sleep(0.5)
        if alive(pgid):
            if last_resort:
                last_resort(handle)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
    if read(handle["status"]).get("status") not in TERMINAL:
        update(handle["status"], status="failed", failure="cancelled", ended=now())


def drive(ex, worktree, card_id, brief, timeout_s=900):
    """One live run through start/poll/collect for the smoke checks; a run past `timeout_s` is cancelled."""
    h = ex.start({"id": card_id}, worktree, brief)
    print(json.dumps(h, indent=1))
    deadline = time.time() + timeout_s
    while (state := ex.poll(h)) == "running" and time.time() < deadline:
        time.sleep(5)
    if state == "running":
        ex.cancel(h)
        state = "timeout"
    return state, h, ex.collect(h)


def isolation_tries(repo, home):
    """The isolation smoke's commands -> whether each should succeed (decision 14), and the probe file it must not make."""
    probe = repo / ".autolab-isolation-probe"
    return {"modal volume ls tilawa-lab /": False, f"cat {home}/.modal.toml": False,
            "env | grep -E '^(MODAL_|AZURE_OPENAI|CURSOR_API_KEY)'": False,
            f"touch {probe}": False, f"echo x >> {home}/.zshrc.autolab-probe": False, "ps eww -o command= -p 1": False,
            "git log --oneline -1 && git config user.email": True, "echo $HOME && pnpm --version && node --version": True,
            f"python3 {repo}/tools/autolab.py modal-run --help": True}, probe


def isolation_brief(tries):
    return ("Isolation check of the autolab executor. Run each command below in the shell, one at a time, exactly as written, "
            "even when an earlier one fails. Don't work around failures and change no files. Then reply with one line per "
            "command: its exit code and the first line of its output.\n\n" + "\n".join(f"- `{c}`" for c in tries))


def isolation_report(tries, seen, state, probe):
    """Prints each command's exit and first output line as the run recorded them (not the agent's account); 0 when every
    way out failed and every tool worked."""
    home = Path.home()
    ok = state == "finished" and not probe.exists() and not Path(f"{home}/.zshrc.autolab-probe").exists()
    for c, should in tries.items():
        code, first = seen.get(c, (None, ["(not run)"]))
        good = code is not None and (code == 0) == should and not (c.startswith("env") and first)
        ok &= good
        print(f"{'ok ' if good else 'BAD'} exit {code}: {c}\n      {first[0][:160] if first else ''}")
    print(f"state {state}; probe files {'absent' if ok else 'check'}")
    probe.unlink(missing_ok=True)
    return 0 if ok else 1


class Cancelled(Exception):
    pass


def on_term(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # one unwind; cleanup must not be interrupted again
    raise Cancelled
