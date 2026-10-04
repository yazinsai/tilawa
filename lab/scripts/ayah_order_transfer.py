"""Ayah-window-disjoint k-fold for the ayah-order check (dev labels only).

Acted dev takes are grouped by prompt passage. Two passages share a group when
their detector windows overlap (same surah, ayahs [first-2 .. last+4]), so no
window straddles a fold. Groups holding a dev skip_ayah label are shuffled
(``random.Random(0)``) and dealt round-robin into k folds (k = 5 at >= 15
groups, else the largest k in {4, 3, 2} that leaves >= 2 groups on every fit
side). Groups without a skip_ayah label stay on every fit side.

For fold i, the detection params (``detect_grid``) and flag thresholds (``grid``
plus the frozen rel 0.04) are fit on the other folds and scored on fold i:

1. no new issue on help clean dev or v1 (a global constraint, as in the
   verse-pair protocol; TLOG is never used);
2. most skip_ayah labels caught (engine or check) on the fit side;
3. fewest new flags that hit no label on the fit side;
4. fewest knobs off the frozen point.

In-sample is the same fit on every group, scored on every group. Test labels
are never read (``--split`` is fixed to dev). Aggregates only.

    ../.venv/bin/python scripts/ayah_order_transfer.py --run /tmp/aos/run/a0w-nopass --passage tracker
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acted_eval import hits, load_jsonl  # noqa: E402
from ayah_order import DEFAULTS, RULE, Corpus, detect_take, flags_of  # noqa: E402
from ayah_order_eval import detect_grid, grid  # noqa: E402
from unseen_scoreboard import ERROR_KINDS  # noqa: E402

BACK, AHEAD = 2, 4


def passage(row: dict) -> tuple[int, int, int] | None:
    vs = row.get("expected_verses") or []
    if not vs or len({int(v["surah"]) for v in vs}) != 1:
        return None
    ay = [int(v["ayah"]) for v in vs]
    return int(vs[0]["surah"]), min(ay), max(ay)


def groups_of(rows: list[dict]) -> dict[str, str]:
    """clip id -> group id (smallest 'surah:ayah' of the merged passages)."""
    ps = sorted({p for r in rows if (p := passage(r))})
    parent = {p: p for p in ps}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for a in ps:
        for b in ps:
            if a < b and a[0] == b[0] and a[1] - BACK <= b[2] + AHEAD and b[1] - BACK <= a[2] + AHEAD:
                parent[find(a)] = find(b)
    comp = defaultdict(list)
    for p in ps:
        comp[find(p)].append(p)
    name = {}
    for members in comp.values():
        gid = min(f"{s:03d}:{a:03d}" for s, a, _e in members)
        for p in members:
            name[p] = gid
    return {str(r["id"]): name[p] if (p := passage(r)) else f"none:{r['id']}" for r in rows}


def rules() -> list[dict]:
    out = grid()
    for margin in (3, 5, 8, 99):
        for fit in (0.2, 0.25, 0.3, 0.4):
            for between in (0, 8):
                for min_post in (0, 8):
                    out.append({"margin": margin, "rel": 0.04, "fit": fit, "between": between, "min_post": min_post})
    return out


def off_frozen(dp: dict, rule: dict) -> int:
    n = sum(1 for k, v in dp.items() if DEFAULTS.get(k) != v)
    return n + sum(1 for k in ("margin", "rel", "fit", "between", "min_post") if RULE.get(k, 0) != rule.get(k, 0))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--passage", choices=("tracker", "expected"), default="tracker")
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    acted = [r for r in load_jsonl(args.run / "help-acted.jsonl") if not r.get("error") and r.get("split") == "dev"]
    clean = [r for r in load_jsonl(args.run / "help-clean.jsonl") if not r.get("error") and r.get("split") == "dev"]
    clean += [r for r in load_jsonl(args.run / "v1.jsonl") if not r.get("error")]
    labs = [lab for lab in load_jsonl(args.labels)
            if lab.get("split") == "dev" and lab.get("mapped") and lab["label_kind"] == "skip_ayah"]
    ids = {str(r["id"]) for r in acted}
    labs = [lab for lab in labs if str(lab["id"]) in ids]
    by_id = defaultdict(list)
    for lab in labs:
        by_id[str(lab["id"])].append(lab)
    gid = groups_of(acted)
    lab_groups = sorted({gid[str(lab["id"])] for lab in labs})
    n_groups = len({g for g in gid.values() if not g.startswith("none:")})
    print(f"dev acted takes {len(acted)}, skip_ayah labels {len(labs)}, passage groups {n_groups}, "
          f"labelled groups {len(lab_groups)}, clean takes {len(clean)}")

    corpus = Corpus()
    rl = rules()
    combos = []          # (dp, rule)
    caught = []          # [combo] -> np.array over labels (bool)
    wrong = []           # [combo] -> dict group -> wrong new flags
    newtp = []           # [combo] -> dict group -> correct new flags
    clean_new = []
    base_caught = np.array([any(i.get("kind") in ERROR_KINDS and hits(i, lab)
                                for i in next(r for r in acted if str(r["id"]) == str(lab["id"]))["issues"])
                            for lab in labs])
    for dp in detect_grid():
        p = {**DEFAULTS, **dp}
        ca = {str(r["id"]): detect_take(corpus, r, args.passage, p) for r in acted}
        cc = [detect_take(corpus, r, args.passage, p) for r in clean]
        for rule in rl:
            nclean = 0
            for r, c in zip(clean, cc):
                for f in flags_of(c, rule):
                    if not any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                               and int(i.get("ayah", 0)) == f["ayah"] for i in r["issues"]):
                        nclean += 1
            if nclean:
                continue
            new = {}
            w, t = defaultdict(int), defaultdict(int)
            for r in acted:
                rid = str(r["id"])
                fl = [f for f in flags_of(ca[rid], rule)
                      if not any(i.get("kind") in ERROR_KINDS and int(i.get("surah", 0)) == f["surah"]
                                 and int(i.get("ayah", 0)) == f["ayah"] for i in r["issues"])]
                new[rid] = fl
                for f in fl:
                    ok = any(hits(f, lab) for lab in by_id.get(rid, []))
                    (t if ok else w)[gid[rid]] += 1
            combos.append((dp, rule))
            caught.append(base_caught | np.array([any(hits(f, lab) for f in new[str(lab["id"])]) for lab in labs]))
            wrong.append(w)
            newtp.append(t)
            clean_new.append(nclean)
        print("detect", json.dumps(dp), "legal combos so far", len(combos), flush=True)

    lab_g = np.array([gid[str(lab["id"])] for lab in labs])
    caught_m = np.array(caught)
    off = np.array([off_frozen(dp, r) for dp, r in combos])

    def fit(fit_groups: set[str], ties: bool = False):
        mask = np.isin(lab_g, list(fit_groups))
        rec = caught_m[:, mask].sum(axis=1)
        wr = np.array([sum(v for g, v in w.items() if g in fit_groups or g not in lab_groups) for w in wrong])
        order = np.lexsort((off, wr, -rec))
        best = int(order[0])
        if ties:
            return np.where((rec == rec[best]) & (wr == wr[best]))[0]
        return best

    all_g = set(lab_groups)
    ins = fit(all_g)
    ins_rec = int(caught_m[ins].sum())
    base_rec = int(base_caught.sum())
    G = len(lab_groups)
    k = 5 if G >= 15 else next((kk for kk in (4, 3, 2) if G - -(-G // kk) >= 2), None)
    if k is None:
        raise SystemExit("too few labelled groups for a fold eval")
    shuffled = list(lab_groups)
    random.Random(0).shuffle(shuffled)
    folds = [shuffled[i::k] for i in range(k)]
    held = np.zeros(len(labs), dtype=bool)
    held_tp = held_wrong = 0
    worst = best_case = 0
    picks = []
    for i, fg in enumerate(folds):
        j = fit(all_g - set(fg))
        m = np.isin(lab_g, fg)
        tie = fit(all_g - set(fg), ties=True)
        per = caught_m[np.ix_(tie, np.where(m)[0])].sum(axis=1)
        worst += int(per.min())
        best_case += int(per.max())
        print(f"fold {i}: {len(tie)} combos tie on the fit side; their held-out catches {int(per.min())}..{int(per.max())}")
        held[m] = caught_m[j, m]
        held_tp += sum(newtp[j].get(g, 0) for g in fg)
        held_wrong += sum(wrong[j].get(g, 0) for g in fg)
        picks.append(combos[j])
        print(f"fold {i}: groups {len(fg)}, labels {int(m.sum())}, caught {int(caught_m[j, m].sum())} "
              f"(engine {int(base_caught[m].sum())}), pick {json.dumps(combos[j][0])} {json.dumps(combos[j][1])}")
    ho_rec = int(held.sum())
    ins_tp = sum(newtp[ins].values())
    ins_wrong = sum(wrong[ins].get(g, 0) for g in lab_groups)
    rng = np.random.default_rng(0)
    ratios, gains = [], []
    for _ in range(args.boot):
        gs = rng.choice(lab_groups, size=G, replace=True)
        idx = np.concatenate([np.where(lab_g == g)[0] for g in gs])
        a, b, e = held[idx].sum(), caught_m[ins][idx].sum(), base_caught[idx].sum()
        if b:
            ratios.append(a / b)
        if b - e:
            gains.append((a - e) / (b - e))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    glo, ghi = np.percentile(gains, [2.5, 97.5])
    print(f"k={k}; engine {base_rec}/{len(labs)}; in-sample {ins_rec}/{len(labs)} "
          f"(pick {json.dumps(combos[ins][0])} {json.dumps(combos[ins][1])}, new {ins_tp}/{ins_tp + ins_wrong}); "
          f"held-out {ho_rec}/{len(labs)} (new {held_tp}/{held_tp + held_wrong})")
    print(f"recall ratio {ho_rec / max(1, ins_rec):.3f} [{lo:.2f}, {hi:.2f}]; "
          f"gain ratio {(ho_rec - base_rec) / max(1, ins_rec - base_rec):.3f} [{glo:.2f}, {ghi:.2f}]; "
          f"clean new flags on help clean dev + v1 at every pick: {clean_new[ins]} / "
          f"{[clean_new[combos.index(p)] for p in picks]}; legal combos {len(combos)}")
    print(f"without the frozen-point tie-break: held-out {worst}..{best_case}/{len(labs)}, "
          f"recall ratio {worst / max(1, ins_rec):.3f}..{best_case / max(1, ins_rec):.3f}, "
          f"gain ratio {(worst - base_rec) / max(1, ins_rec - base_rec):.3f}..")
    flat = int((caught_m.sum(axis=1) == ins_rec).sum())
    print(f"legal combos tied with the in-sample recall: {flat}")


if __name__ == "__main__":
    main()
