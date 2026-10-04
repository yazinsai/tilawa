"""Unseen-word scoreboard for acted correction labels (seen / unseen split).

Help-page prompts repeat the same verses across speakers, so a test label on a
word that is also an acted-dev label measures memorisation, not detection.
Every correction result is reported on both halves; go/kill reads U_test.

U_test: in-scope acted-test labels whose ``surah:ayah:word`` is not the word
of any acted-dev word label. Dev ``skip_ayah`` labels name an ayah, not a
word, so they ban nothing; a test ``skip_ayah`` label is seen only if its
recorded word key is some dev word label's key. On the current export this is
62 of the 207 in-scope test labels.

Scoring matches ``acted_eval.py``: a flag hits a label on the same
surah:ayah with its word within ±1 (``skip_ayah``: anywhere in the ayah),
any issue kind. Clean false flags count every error issue on the guard sets.

Private data only: prints aggregates, never clip ids or per-clip rows.

    ../.venv/bin/python scripts/unseen_scoreboard.py \\
        --labels /tmp/correction_eval/acted_located.jsonl \\
        --runs rules=/tmp/replay/rules,cand=/tmp/replay/cand
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acted_eval import IN_SCOPE, hits, load_jsonl, wilson  # noqa: E402

ERROR_KINDS = frozenset({
    "possible_omission", "possible_substitution", "possible_vowel",
    "possible_skipped_ayah", "unclear_ayah",
})
GUARDS = (("help-clean", "test"), ("tlog-dev", None), ("v1", None))


def word_key(label: dict) -> tuple[int, int, int] | None:
    if label.get("word_index") is None:
        return None
    return (int(label["surah"]), int(label["ayah"]), int(label["word_index"]))


def dev_word_keys(labels: list[dict]) -> set[tuple[int, int, int]]:
    return {
        k for r in labels
        if r.get("split") == "dev" and r.get("mapped") and r["label_kind"] != "skip_ayah"
        and (k := word_key(r)) is not None
    }


def test_labels(labels: list[dict], split: str = "test") -> list[dict]:
    """In-scope mapped labels of ``split``, each tagged ``unseen``."""
    banned = dev_word_keys(labels)
    out = []
    for r in labels:
        if r.get("split") != split or not r.get("mapped") or r["label_kind"] not in IN_SCOPE:
            continue
        out.append({**r, "unseen": word_key(r) not in banned})
    return out


def split_counts(labels: list[dict]) -> dict[str, Counter]:
    rows = test_labels(labels)
    return {
        "all": Counter(r["label_kind"] for r in rows),
        "unseen": Counter(r["label_kind"] for r in rows if r["unseen"]),
    }


def caught(issues: list[dict], label: dict) -> bool:
    return any(hits(i, label) for i in issues if i.get("kind") in ERROR_KINDS)


def score(issues_by_id: dict[str, list[dict]], labels: list[dict], split: str = "test") -> dict:
    """caught/n per kind on all, seen and unseen labels of ``split``."""
    out = {part: Counter() for part in ("all_n", "all_c", "seen_n", "seen_c", "unseen_n", "unseen_c")}
    for lab in test_labels(labels, split):
        issues = issues_by_id.get(str(lab["id"]))
        if issues is None:
            continue
        c = caught(issues, lab)
        part = "unseen" if lab["unseen"] else "seen"
        for p in ("all", part):
            out[f"{p}_n"][lab["label_kind"]] += 1
            out[f"{p}_c"][lab["label_kind"]] += int(c)
    return out


def guard_ff(run_dir: Path) -> dict[str, tuple[int, float]]:
    ff = {}
    for name, split in GUARDS:
        rows = [r for r in load_jsonl(run_dir / f"{name}.jsonl")
                if not r.get("error") and (split is None or r.get("split") == split)]
        n = sum(1 for r in rows for i in r.get("issues") or [] if i.get("kind") in ERROR_KINDS)
        ff[name] = (n, sum(float(r.get("duration_s") or 0) for r in rows) / 60.0)
    return ff


def fmt(c: int, n: int) -> str:
    ci = wilson(c, n)
    return f"{c}/{n}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def render(runs: list[tuple[str, dict, dict]]) -> str:
    lines = ["| run | all | seen | unseen (U_test) | FF help clean test / TLOG clean dev / v1 |",
             "|---|---|---|---|---|"]
    for name, sc, ff in runs:
        tot = {p: sum(sc[p].values()) for p in sc}
        ffs = " / ".join(str(ff[g][0]) for g, _ in GUARDS)
        lines.append(f"| {name} | {fmt(tot['all_c'], tot['all_n'])} | {fmt(tot['seen_c'], tot['seen_n'])} | "
                     f"{fmt(tot['unseen_c'], tot['unseen_n'])} | {ffs} |")
    lines += ["", "| run | " + " | ".join(f"{k} unseen" for k in IN_SCOPE) + " |",
              "|---|" + "---|" * len(IN_SCOPE)]
    for name, sc, _ff in runs:
        lines.append(f"| {name} | " + " | ".join(
            f"{sc['unseen_c'][k]}/{sc['unseen_n'][k]}" for k in IN_SCOPE) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--runs", default="", help="name=dir,... replay/live dirs with help-acted.jsonl etc.")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    labels = load_jsonl(args.labels)
    counts = split_counts(labels)
    print(f"in-scope test labels {sum(counts['all'].values())}, unseen {sum(counts['unseen'].values())}: "
          + ", ".join(f"{k} {counts['unseen'][k]}/{counts['all'][k]}" for k in IN_SCOPE))
    runs = []
    for spec in filter(None, args.runs.split(",")):
        name, d = spec.split("=", 1)
        rows = load_jsonl(Path(d) / "help-acted.jsonl")
        issues = {str(r["id"]): r.get("issues") or [] for r in rows if not r.get("error")}
        runs.append((name, score(issues, labels, args.split), guard_ff(Path(d))))
    if runs:
        print(render(runs))


if __name__ == "__main__":
    main()
