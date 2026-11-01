"""Autolab executors (.scratch/autolab/spec.md decisions 1 and 14; ticket 05). One interface; the supervisor never
knows which runtime implements it.

    start(card, worktree, brief) -> handle      a JSON-serializable dict, stored in the state document
    poll(handle) -> "running" | "finished" | "failed"
    cancel(handle) -> None
    collect(handle) -> {"transcript": str, "summary": str, "hours": float}

A handle must survive a supervisor restart: poll/cancel/collect work from the stored dict alone.
"""

from typing import Protocol


class Executor(Protocol):
    def start(self, card: dict, worktree: str, brief: str) -> dict: ...
    def poll(self, handle: dict) -> str: ...
    def cancel(self, handle: dict) -> None: ...
    def collect(self, handle: dict) -> dict: ...


class FakeExecutor:
    """For tests and dry runs: `script(card, worktree)` runs at start (it may write files or commit); polls report
    `finished` after `polls` calls, or `failed` when the script raises."""

    def __init__(self, script=None, polls: int = 1):
        self.script, self.polls, self.seen, self.cancelled = script, polls, {}, []

    def start(self, card, worktree, brief):
        err = None
        try:
            if self.script:
                self.script(card, worktree)
        except Exception as e:  # noqa: BLE001 - a failing script is a failed executor run
            err = str(e)
        h = {"runtime": "fake", "id": f"fake-{card['id']}", "error": err}
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


def make(config: dict) -> Executor:
    """The executor named by config["executor"]["runtime"]: "fake" or "cursor-local"."""
    runtime = config.get("executor", {}).get("runtime", "cursor-local")
    if runtime == "fake":
        return FakeExecutor()
    if runtime == "cursor-local":
        from autolab_cursor import CursorLocalExecutor  # noqa: PLC0415 - cursor-sdk is only needed for real runs
        return CursorLocalExecutor(config)
    raise ValueError(f"unknown executor runtime {runtime}")
