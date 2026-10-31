"""A spot-check listening queue drawn from a test set's triage queue (ticket 20): the sessions new in this version only
(those kept from an earlier one were heard there), every case the two models disagree on, then seeded samples of the
unsure cases (edits and repeats before clean windows), the seam checks (inserted audio first: every bad splice so far
put donor audio in) and the auto-accepted spot checks. maxBad is spec decision 16's 1 per 40.

    python3 tools/lab_spot_check.py --test-set lab-v1.5      # reads /lab/data/<set>/review/triage.json, writes review/spot-check.json
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

QUOTA = {"disagree": None, "unsure-edit": 12, "unsure-clean": 6, "seam": 10, "spot": 6}  # None: all
INSERTED = ("wrong_word", "added_word", "mutashabihat")
SEED = 20


def pool(item: dict) -> str:
    t = item["triage"]
    first = t["reasons"][0]
    if first == "unsure":
        return "unsure-clean" if t["class"] == "clean" else "unsure-edit"
    return first


def spot_check(queue: dict, own: str, seed: int = SEED, quota: dict = QUOTA) -> dict:
    """`own`: the id prefix of this version's sessions (`<testSet>-`)."""
    rng = random.Random(f"{seed}:spot-check:{queue['testSet']}")
    items = [i for i in queue["items"] if i["id"].startswith(own)]
    out = []
    for name, n in quota.items():
        xs = sorted((i for i in items if pool(i) == name), key=lambda i: i["id"])
        rng.shuffle(xs)
        if name == "seam":
            xs.sort(key=lambda i: i["triage"]["class"] not in INSERTED)
        elif name == "unsure-edit":
            by = {}
            for i in xs:
                by.setdefault(i["triage"]["class"], []).append(i)
            xs = [g[k] for k in range(max(map(len, by.values()), default=0)) for _, g in sorted(by.items()) if k < len(g)]
        out += xs if n is None else xs[:n]
    return {"version": 1, "testSet": queue["testSet"], "maxBad": max(1, len(out) // 40), "items": out,
            "spotCheck": {"from": "review/triage.json", "seed": seed, "quota": {k: v if v is not None else "all" for k, v in quota.items()},
                          "pools": {k: sum(pool(i) == k for i in items) for k in quota}, "own": own}}


def main():
    import lab
    from lab_review_core import validate_queue

    p = argparse.ArgumentParser()
    p.add_argument("--test-set", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    q = lab.vol_get(f"/data/{a.test_set}/review/triage.json")
    doc = spot_check(q, f"{a.test_set}-", a.seed)
    problems = validate_queue(doc)
    if problems:
        sys.exit(f"spot check: {problems[:5]}")
    lab.vol_put(f"/data/{a.test_set}/review/spot-check.json", doc)
    print(f"/lab/data/{a.test_set}/review/spot-check.json: {len(doc['items'])} items, maxBad {doc['maxBad']}; pools {doc['spotCheck']['pools']}")


if __name__ == "__main__":
    main()
