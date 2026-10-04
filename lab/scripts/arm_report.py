"""Per-arm table: every variant vs v3 and vs its A0 counterpart, with paired
bootstrap CIs on headline and madd-free headline PER (clip-level resampling).

Reads `promotion_gates.py --json` outputs (gates_<name>.json) and the mirrored
eval dirs. Variant names follow `<arm>-ep{1,2}[-a0.5|-a0.7]`; the A0 counterpart
of `a0e-ep2-a0.5` is `a0-ep2-a0.5`.

  ../.venv/bin/python scripts/arm_report.py --eval-dir /tmp/phase0/eval --gates-dir /tmp/phase0 \\
      --arm a0e --qlab-text /tmp/qlab
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB / "scripts"))

SUFFIXES = ("ep1", "ep1-a0.5", "ep1-a0.7", "ep2", "ep2-a0.5", "ep2-a0.7")


def per_clip_errors(eval_dir: Path, name: str, golds: dict | None) -> dict[str, tuple[int, int]]:
    from promotion_gates import collapse, load

    out = {}
    for r in load(eval_dir / name / "qlab.jsonl"):
        if golds is None:
            g = r["headline"]
            out[r["id"]] = (g["sub"] + g["ins"] + g["del"], g["ref_len"])
        else:
            from Levenshtein import distance

            ref, hyp = collapse(golds[r["id"]]), collapse(r["hyp"])
            out[r["id"]] = (distance("".join(chr(0x100 + x) for x in ref), "".join(chr(0x100 + x) for x in hyp)), len(ref))
    return out


def boot_ci(a: dict, b: dict, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(delta, lo, hi) of PER(a) - PER(b) in pp over shared clips."""
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)

    def per(d, s):
        return 100 * sum(d[i][0] for i in s) / max(sum(d[i][1] for i in s), 1)

    delta = per(a, ids) - per(b, ids)
    ds = []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        ds.append(per(a, s) - per(b, s))
    ds.sort()
    return round(delta, 3), round(ds[int(0.025 * n)], 3), round(ds[int(0.975 * n)], 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--gates-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base-arm", default="a0")
    ap.add_argument("--qlab-text", required=True)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from promotion_gates import madd_free_golds

    ev = Path(args.eval_dir)
    golds = madd_free_golds(Path(args.qlab_text), {})
    cache: dict = {}

    def errs(name, mf):
        key = (name, mf)
        if key not in cache:
            cache[key] = per_clip_errors(ev, name, golds if mf else None)
        return cache[key]

    rows = []
    for suf in SUFFIXES:
        name, a0 = f"{args.arm}-{suf}", f"{args.base_arm}-{suf}"
        gp = Path(args.gates_dir) / f"gates_{name}.json"
        if not gp.is_file():
            continue
        g = json.loads(gp.read_text())
        c = g["metrics"]["cand"]
        row = {
            "variant": name,
            "headline": c["headline"]["per"],
            "maddfree": c.get("maddfree_headline", {}).get("per"),
            "drops": c["heldout_multi"]["ayahs_dropped"],
            "ins_holdout": c["holdout_nonclean"]["ins"],
            "ins_tlog_dev": c["tlog_dev_nonclean"]["ins"],
            "dev_single_mf": c.get("maddfree_dev_single", {}).get("per"),
            "tracker": (g.get("tracker", {}).get("cand") or {}).get("seq_acc"),
            "correction": (g.get("correction", {}).get("cand") or {}).get("overall"),
            "clean_ff": (g.get("correction", {}).get("cand") or {}).get("clean_false_flags_per_min"),
            "gates_failed": [k for k, v in g["gates"].items() if not v],
            "promote": g["promote"],
            "ci_vs_v3": boot_ci(errs(name, False), errs("v3", False)),
            "ci_mf_vs_v3": boot_ci(errs(name, True), errs("v3", True)),
        }
        if (ev / a0 / "qlab.jsonl").is_file() and a0 != name:
            row["ci_vs_a0"] = boot_ci(errs(name, False), errs(a0, False))
            row["ci_mf_vs_a0"] = boot_ci(errs(name, True), errs(a0, True))
        rows.append(row)
    for r in rows:
        print(json.dumps(r))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
