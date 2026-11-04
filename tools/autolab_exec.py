"""Autolab executors (.scratch/autolab/spec.md decisions 1 and 14; ticket 05). One interface; the supervisor never
knows which runtime implements it.

    start(card, worktree, brief) -> handle      a JSON-serializable dict, stored in the state document
    poll(handle) -> "running" | "finished" | "failed"
    cancel(handle) -> None
    collect(handle) -> {"transcript": str, "summary": str, "hours": float}

A handle must survive a supervisor restart: poll/cancel/collect work from the stored dict alone. Every handle names its
`runtime`, and the supervisor dispatches poll/cancel/collect by it.

Builder arms (the A/B between runtimes): `config["executor"]["runtimes"]` is a list of arms, each {"name", "runtime",
...its settings}, and `executor.split` weighs them ({name: weight}, default equal). Without `runtimes`,
`executor.runtime` (default codex-azure) is one arm named after its runtime. An arm's
settings override `executor`'s shared ones (env, envDrop, home, seatbelt, cancelGraceS, and `executor.codex` for
codex-azure arms).

    arms(config) -> [arm]                   arm_config(config, arm) -> a config whose "executor" is that arm's merge
    unready(config, arm, environ) -> why    None when the arm can start a builder now
    arm_models(cfg) -> (model, fallback)    what a builder of that arm runs on its first start, and on a retry
"""

import os
from typing import Protocol

RUNTIMES = ("codex-azure", "fake")


class Executor(Protocol):
    def start(self, card: dict, worktree: str, brief: str) -> dict: ...
    def poll(self, handle: dict) -> str: ...
    def cancel(self, handle: dict) -> None: ...
    def collect(self, handle: dict) -> dict: ...


class FakeExecutor:
    """For tests and dry runs: `script(card, worktree)` runs at start (it may write files or commit); polls report
    `finished` after `polls` calls, or `failed` when the script raises."""

    def __init__(self, script=None, polls: int = 1, runtime: str = "fake"):
        self.script, self.polls, self.seen, self.cancelled, self.runtime, self.started = script, polls, {}, [], runtime, []

    def start(self, card, worktree, brief):
        err = None
        self.started.append(card)
        try:
            if self.script:
                self.script(card, worktree)
        except Exception as e:  # noqa: BLE001 - a failing script is a failed executor run
            err = str(e)
        h = {"runtime": self.runtime, "id": f"fake-{card['id']}", "error": err}
        self.seen[h["id"]] = 0
        return h

    def poll(self, handle):
        if handle.get("error"):
            return "failed"
        self.seen[handle["id"]] = self.seen.get(handle["id"], 0) + 1
        return "finished" if self.seen[handle["id"]] >= self.polls else "running"

    def cancel(self, handle):
        self.cancelled.append(handle["id"])

    def collect(self, handle):
        return {"transcript": f"fake run {handle['id']}", "summary": handle.get("error") or "done", "hours": 0.0}


def arms(config: dict) -> list[dict]:
    ex = config.get("executor", {})
    if not ex.get("runtimes"):
        rt = ex.get("runtime", "codex-azure")
        return [{"name": ex.get("name") or rt, "runtime": rt}]
    return [{**a, "name": a.get("name") or a.get("runtime", "codex-azure"), "runtime": a.get("runtime", "codex-azure")} for a in ex["runtimes"]]


def weights(config: dict) -> dict:
    split = config.get("executor", {}).get("split") or {}
    return {a["name"]: float(split.get(a["name"], 1)) for a in arms(config)}


def key_env(arm: dict) -> str | None:
    return ((arm.get("azure") or {}).get("keyEnv") or "AZURE_OPENAI_API_KEY") if arm.get("runtime") == "codex-azure" else None


def arm_config(config: dict, arm: dict) -> dict:
    ex = config.get("executor", {})
    base = {k: v for k, v in ex.items() if k not in ("runtimes", "split", "codex")}
    if arm["runtime"] == "codex-azure":
        shared = ex.get("codex") or {}
        base = {k: v for k, v in base.items() if k not in ("model", "fallbackModel")}  # the arm's model is its Azure deployment
        merged = {**base, **shared, **arm, "azure": {**(shared.get("azure") or {}), **(arm.get("azure") or {})}}
    else:
        merged = {**base, **arm}
    others = sorted({k for a in arms(config) if a["name"] != arm["name"] and (k := key_env(a)) and k != key_env(arm)})
    merged["envDrop"] = [*merged.get("envDrop", []), *others]
    return {**config, "executor": merged}


def arm_models(cfg: dict) -> tuple:
    """`cfg` is arm_config(...)["executor"]."""
    if cfg.get("runtime") == "codex-azure":
        dep = (cfg.get("azure") or {}).get("deployment") or ""
        return cfg.get("model") or dep, cfg.get("fallbackModel") or cfg.get("model") or dep
    return cfg.get("model"), cfg.get("fallbackModel") or cfg.get("model")


def unready(config: dict, arm: dict, environ=None) -> str | None:
    env = os.environ if environ is None else environ
    cfg = arm_config(config, arm)["executor"]
    if arm["runtime"] == "codex-azure":
        import autolab_codex  # noqa: PLC0415
        return autolab_codex.unready(cfg, env)
    return None if arm["runtime"] == "fake" else f"unknown runtime {arm['runtime']!r} (one of {', '.join(RUNTIMES)})"


def make(config: dict, arm: dict | None = None, check: bool = True) -> Executor:
    """The executor of `arm` (default: the first arm): "fake" or "codex-azure". `check` False skips the
    startup checks (model ids, codex flags): enough to poll, cancel and collect stored handles."""
    arm = arm or arms(config)[0]
    cfg, runtime = arm_config(config, arm), arm["runtime"]
    if runtime == "fake":
        return FakeExecutor()
    if runtime == "codex-azure":
        from autolab_codex import CodexExecutor  # noqa: PLC0415
        return CodexExecutor(cfg, check=check)
    raise ValueError(f"unknown executor runtime {runtime}")
