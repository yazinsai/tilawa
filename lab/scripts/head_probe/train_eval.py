"""Train the pre-declared slip heads and score test once.

Configs (fixed before this script looks at test labels):
  encoders: shipped int8, a0w-ep1-a0.5 int8
  heads: L2 logistic on mean+max frames, MLP 1024-64-1, frame-level max-MIL
  data: acted dev + clean pool, and the same plus dev-only splices
  thresholds: out-of-fold clean-pool rates in OPS_PER_MIN

Kill/go uses the best test recall gain among those operating points whose
false-flag *counts* are no higher than the rules-only counts on help clean
test, TLOG clean dev and v1. Go if that gain is >= +0.08 and its speaker
bootstrap CI excludes 0. Kill if it is < +0.05 or the CI includes 0.
The +0.10/min band is reported separately and is not the go rule.

Prints aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HEADS, IN_SCOPE, OPS_PER_MIN, hits_label, k_for_rate, threshold_at_k,
)

SET_HELP_ACTED, SET_HELP_CLEAN, SET_EA, SET_TLOG_TRAIN, SET_TLOG_DEV, SET_V1, SET_SYNTH = range(7)
KIND_NAME = ["skip_word", "substitution", "vowel", "skip_ayah", "repeat", "tajweed"]
RNG = np.random.default_rng(0)


def _load(path: Path):
    z = np.load(path)
    frame_path = path.with_suffix(".frames.f16")
    nframes = frame_path.stat().st_size // (512 * 2)
    frames = np.memmap(frame_path, dtype=np.float16, mode="r", shape=(nframes, 512))
    return z, frames


def _fit_linear(X: np.ndarray, y: np.ndarray, kind: str):
    if len(y) == 0 or int(y.sum()) == 0 or int(y.sum()) == len(y):
        return ("const", float(y.mean()) if len(y) else 0.0)
    mu = X.mean(0)
    sd = np.where(X.std(0) < 1e-6, 1.0, X.std(0))
    xs = (X - mu) / sd
    if kind == "logistic":
        model = LogisticRegression(
            C=0.01, max_iter=120, class_weight="balanced", solver="lbfgs", tol=1e-3,
        )
        model.fit(xs, y)
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", alpha=1e-3, batch_size=256,
            learning_rate_init=1e-3, max_iter=12, random_state=0,
        )
        n_pos = max(1, int(y.sum()))
        n_neg = max(1, int((y == 0).sum()))
        model.fit(xs, y, sample_weight=np.where(y == 1, n_neg / n_pos, 1.0))
    return ("model", model, mu.astype(np.float32), sd.astype(np.float32))


def _predict_linear(pack, X: np.ndarray) -> np.ndarray:
    if pack[0] == "const":
        return np.full(len(X), pack[1], np.float32)
    _tag, model, mu, sd = pack
    return model.predict_proba((X - mu) / sd)[:, 1].astype(np.float32)


def _bag(frames, f0, f1, abs_a, span0, span1, i: int) -> np.ndarray:
    sl = np.asarray(frames[int(f0[i]):int(f1[i])], dtype=np.float32)
    if len(sl) == 0 or int(span0[i]) < 0:
        return sl
    lo = max(int(span0[i]), int(abs_a[i]))
    hi = min(int(span1[i]), int(abs_a[i]) + len(sl))
    if hi <= lo:
        return sl
    return sl[lo - int(abs_a[i]):hi - int(abs_a[i])]


def _fit_frame(frames, f0, f1, abs_a, span0, span1, y, train_idx: np.ndarray):
    pos = train_idx[y[train_idx] == 1]
    neg = train_idx[y[train_idx] == 0]
    rng = np.random.default_rng(0)
    rng.shuffle(neg)
    neg_x = []
    for i in neg:
        if len(neg_x) >= 40000:
            break
        sl = np.asarray(frames[int(f0[i]):int(f1[i])], dtype=np.float32)
        if len(sl) == 0:
            continue
        neg_x.append(sl[int(rng.integers(0, len(sl)))])
    bags = [_bag(frames, f0, f1, abs_a, span0, span1, int(i)) for i in pos]
    bags = [b for b in bags if len(b)]
    if not bags or not neg_x:
        return ("const", float(y[train_idx].mean()) if len(train_idx) else 0.0)
    neg_x = np.stack(neg_x).astype(np.float32)
    # 3 rounds of max-instance MIL. Round 0 uses every positive frame (capped).
    chosen = []
    for bag in bags:
        if len(chosen) >= 20000:
            break
        chosen.append(bag if len(bag) <= 8 else bag[:: max(1, len(bag) // 8)])
    pos_x = np.concatenate(chosen, 0)[:20000]
    pack = None
    for round_i in range(3):
        x = np.concatenate([pos_x, neg_x], 0)
        yy = np.concatenate([np.ones(len(pos_x), np.int8), np.zeros(len(neg_x), np.int8)])
        pack = _fit_linear(x, yy, "logistic")
        if round_i == 2:
            break
        picked = []
        for bag in bags:
            pr = _predict_linear(pack, bag)
            picked.append(bag[int(pr.argmax())])
        pos_x = np.stack(picked).astype(np.float32)
    return pack


def _score_frame(pack, frames, f0, f1, idx: np.ndarray) -> np.ndarray:
    out = np.zeros(len(idx), np.float32)
    for n0 in range(0, len(idx), 512):
        batch = idx[n0:n0 + 512]
        parts, lengths = [], []
        for i in batch:
            sl = np.asarray(frames[int(f0[i]):int(f1[i])], dtype=np.float32)
            parts.append(sl)
            lengths.append(len(sl))
        total = sum(lengths)
        if total == 0:
            continue
        cat = np.concatenate([p for p in parts if len(p)], 0)
        prob = _predict_linear(pack, cat)
        k = 0
        for j, n in enumerate(lengths):
            out[n0 + j] = float(prob[k:k + n].max()) if n else 0.0
            k += n
    return out


def _metrics(z, scores, thr: float | None):
    """Per-clip (in_scope, caught, tp, nflags) plus FF counts. scores is all-words or None."""
    clip = z["clip"]
    nclips = len(z["clip_dur"])
    flagged = [set() for _ in range(nclips)]
    surah, ayah, word = z["surah"], z["ayah"], z["word"]
    for c, s, a, w in zip(z["issue_clip"], z["issue_s"], z["issue_a"], z["issue_w"]):
        flagged[int(c)].add((int(s), int(a), int(w)))
    if scores is not None and thr is not None:
        for i in np.where(scores >= thr)[0]:
            flagged[int(clip[i])].add((int(surah[i]), int(ayah[i]), int(word[i])))
    rows = []
    per_kind_hit = {k: 0 for k in IN_SCOPE}
    per_kind_n = {k: 0 for k in IN_SCOPE}
    ff = {"help": 0, "tlog": 0, "v1": 0}
    minutes = {"help": 0.0, "tlog": 0.0, "v1": 0.0}
    tp = fl = caught = nlab = 0
    for c in range(nclips):
        sett = int(z["clip_set"][c])
        test = int(z["clip_test"][c])
        nflags = len(flagged[c]) + int(z["clip_extra"][c])
        lab = None
        if int(z["clip_mapped"][c]) and int(z["clip_kind"][c]) >= 0:
            lab = {
                "mapped": True,
                "surah": int(z["clip_lab_s"][c]),
                "ayah": int(z["clip_lab_a"][c]),
                "word_index": None if int(z["clip_lab_w"][c]) < 0 else int(z["clip_lab_w"][c]),
                "label_kind": KIND_NAME[int(z["clip_kind"][c])],
            }
        hit_n = 0
        is_caught = 0
        if lab:
            hit_n = sum(1 for s, a, w in flagged[c] if hits_label(s, a, w, lab))
            is_caught = int(hit_n > 0)
        inscope = int(z["clip_in_scope"][c]) and test and sett == SET_HELP_ACTED
        if sett == SET_HELP_CLEAN and test:
            ff["help"] += nflags
            minutes["help"] += float(z["clip_dur"][c]) / 60.0
            tp += hit_n
            fl += nflags
        elif sett == SET_HELP_ACTED and test:
            if lab and lab["label_kind"] in per_kind_n:
                per_kind_n[lab["label_kind"]] += 1
                per_kind_hit[lab["label_kind"]] += is_caught
            tp += hit_n
            fl += nflags
            if inscope:
                caught += is_caught
                nlab += 1
        elif sett == SET_TLOG_DEV:
            ff["tlog"] += nflags
            minutes["tlog"] += float(z["clip_dur"][c]) / 60.0
        elif sett == SET_V1:
            ff["v1"] += nflags
            minutes["v1"] += float(z["clip_dur"][c]) / 60.0
        if (sett in (SET_HELP_ACTED, SET_HELP_CLEAN) and test):
            rows.append({
                "group": int(z["clip_group"][c]),
                "inscope": int(inscope),
                "caught": int(is_caught if inscope else 0),
                "tp": int(hit_n if sett == SET_HELP_ACTED else 0),
                "fl": int(nflags),
            })
    prec = tp / fl if fl else 0.0
    rec = caught / nlab if nlab else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "P": prec, "R": rec, "F1": f1, "caught": caught, "n": nlab, "tp": tp, "fl": fl,
        "ff": ff, "minutes": minutes, "rows": rows,
        "per_kind": {k: [per_kind_hit[k], per_kind_n[k]] for k in IN_SCOPE},
    }


def _f1_r(rows, ids):
    ins = caught = tp = fl = 0
    for i in ids:
        r = rows[i]
        ins += r["inscope"]
        caught += r["caught"]
        tp += r["tp"]
        fl += r["fl"]
    prec = tp / fl if fl else 0.0
    rec = caught / ins if ins else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, rec


def _bootstrap(base_rows, cand_rows, n: int = 2000):
    by: dict[int, list[int]] = {}
    for i, row in enumerate(base_rows):
        by.setdefault(row["group"], []).append(i)
    groups = list(by)
    rng = np.random.default_rng(0)
    d_f, d_r = [], []
    for _ in range(n):
        ids = [c for g in rng.choice(groups, len(groups)) for c in by[int(g)]]
        f1_c, r_c = _f1_r(cand_rows, ids)
        f1_b, r_b = _f1_r(base_rows, ids)
        d_f.append(f1_c - f1_b)
        d_r.append(r_c - r_b)
    f1_c, r_c = _f1_r(cand_rows, range(len(cand_rows)))
    f1_b, r_b = _f1_r(base_rows, range(len(base_rows)))
    return {
        "dR": r_c - r_b,
        "dR_lo": float(np.percentile(d_r, 2.5)),
        "dR_hi": float(np.percentile(d_r, 97.5)),
        "dF1": f1_c - f1_b,
        "dF1_lo": float(np.percentile(d_f, 2.5)),
        "dF1_hi": float(np.percentile(d_f, 97.5)),
    }


def _loc(z, scores, test: int) -> dict:
    clip = z["clip"]
    by: dict[int, list[int]] = {}
    for i, c in enumerate(clip):
        by.setdefault(int(c), []).append(i)
    aucs = []
    top1 = n = 0
    for c, idxs in by.items():
        if int(z["clip_set"][c]) != SET_HELP_ACTED or int(z["clip_test"][c]) != test:
            continue
        if not int(z["clip_mapped"][c]) or int(z["clip_kind"][c]) < 0:
            continue
        if KIND_NAME[int(z["clip_kind"][c])] == "skip_ayah":
            continue
        if len(idxs) < 3:
            continue
        lab = [i for i in idxs
               if int(z["surah"][i]) == int(z["clip_lab_s"][c])
               and int(z["ayah"][i]) == int(z["clip_lab_a"][c])
               and int(z["word"][i]) == int(z["clip_lab_w"][c])]
        if len(lab) != 1 or not np.isfinite(scores[lab[0]]):
            continue
        s = float(scores[lab[0]])
        others = np.array([float(scores[i]) for i in idxs if i != lab[0] and np.isfinite(scores[i])])
        if len(others) < 2:
            continue
        aucs.append(float(np.mean(s > others)))
        top1 += int(s > others.max())
        n += 1
    return {"auc": float(np.mean(aucs)) if aucs else None, "top1": top1, "n": n}


def _ff_line(m):
    parts = []
    for key in ("help", "tlog", "v1"):
        minutes = m["minutes"][key]
        flags = m["ff"][key]
        rate = flags / minutes if minutes else 0.0
        parts.append(f"{key} {rate:.3f}({flags})")
    return " ".join(parts)


def _rate(m, key):
    minutes = m["minutes"][key]
    return (m["ff"][key] / minutes) if minutes else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heads", default=",".join(HEADS))
    args = ap.parse_args()
    z, frames = _load(args.units)
    y = z["y"].astype(np.int8)
    ign = z["ign"].astype(bool)
    clip = z["clip"]
    sett = z["clip_set"]
    test = z["clip_test"]
    # Dev rows the head may train on. Synth is optional per arm.
    dev_clip = (test == 0) & np.isin(sett, [SET_HELP_ACTED, SET_HELP_CLEAN, SET_EA, SET_TLOG_TRAIN, SET_SYNTH])
    # Held-out clip groups must not appear in training.
    train_groups = set(z["clip_group"][dev_clip & np.isin(sett, [SET_HELP_ACTED, SET_HELP_CLEAN, SET_EA, SET_TLOG_TRAIN, SET_SYNTH])].tolist())
    test_groups = set(z["clip_group"][(test == 1) & (sett == SET_HELP_ACTED)].tolist())
    if train_groups & test_groups:
        raise SystemExit("group ids overlap dev and test")
    names = ["help-acted", "help-clean", "ea", "tlog-train", "tlog-dev", "v1", "synth"]
    for sid, name in enumerate(names):
        wmask = sett[clip] == sid
        print(f"set {name} words={int(wmask.sum())} pos={int(y[wmask].sum())}", flush=True)
    calib_n = int((
        ((sett[clip] == SET_HELP_CLEAN) & (test[clip] == 0))
        | (sett[clip] == SET_EA) | (sett[clip] == SET_TLOG_TRAIN)
    ).sum())
    if calib_n < 50000:
        raise SystemExit(f"calibration words {calib_n} < 50000; do not score test")
    print(f"calibration_words={calib_n}", flush=True)
    x_word = np.concatenate([
        z["mean"].astype(np.float32), z["amax"].astype(np.float32),
    ], axis=1)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    base = _metrics(z, None, None)
    print(
        f"rules P={base['P']:.3f} R={base['R']:.3f} F1={base['F1']:.3f} "
        f"caught={base['caught']}/{base['n']} ff {_ff_line(base)}",
        flush=True,
    )
    summary = {
        "encoder": args.encoder,
        "words": int(len(y)),
        "positives": int(y.sum()),
        "rules": {k: base[k] for k in ("P", "R", "F1", "caught", "n", "ff", "minutes", "per_kind")},
        "configs": [],
    }
    results_for_verdict = []

    for synth in (False, True):
        for head in heads:
            use_clip = dev_clip if synth else (dev_clip & (sett != SET_SYNTH))
            train_word = use_clip[clip] & ~ign
            if head != "rules" and int(y[train_word].sum()) < 5:
                print(f"skip {head} synth={int(synth)} positives={int(y[train_word].sum())}", flush=True)
                continue
            groups = z["clip_group"][clip[train_word]]
            idx = np.where(train_word)[0]
            oof = np.full(len(y), np.nan, np.float32)
            n_splits = min(5, len(np.unique(groups)))
            print(f"fit {args.encoder} {head} synth={int(synth)} train={len(idx)} pos={int(y[idx].sum())} folds={n_splits}", flush=True)
            if head == "frame":
                pack_for = lambda tr: _fit_frame(  # noqa: E731
                    frames, z["f0"], z["f1"], z["abs_a"], z["span0"], z["span1"], y, idx[tr],
                )
                score_idx = lambda pack, ii: _score_frame(pack, frames, z["f0"], z["f1"], ii)  # noqa: E731
            else:
                pack_for = lambda tr, head=head: _fit_linear(x_word[idx[tr]], y[idx[tr]], head)  # noqa: E731
                score_idx = lambda pack, ii, head=head: _predict_linear(pack, x_word[ii])  # noqa: E731
            if n_splits >= 2 and len(np.unique(y[idx])) > 1:
                for tr, te in GroupKFold(n_splits).split(idx, y[idx], groups):
                    pack = pack_for(tr)
                    held = set(groups[te].tolist())
                    pred = np.where(use_clip[clip] & np.isin(z["clip_group"][clip], list(held)))[0]
                    if len(pred):
                        oof[pred] = score_idx(pack, pred)
            labelled = train_word & np.isfinite(oof)
            dev_auc = dev_ap = None
            if labelled.any() and len(np.unique(y[labelled])) > 1:
                dev_auc = float(roc_auc_score(y[labelled], oof[labelled]))
                dev_ap = float(average_precision_score(y[labelled], oof[labelled]))
            print(f"  dev-oof auc={dev_auc} ap={dev_ap}", flush=True)
            calib_clips = np.where(
                ((sett == SET_HELP_CLEAN) & (test == 0)) | (sett == SET_EA) | (sett == SET_TLOG_TRAIN)
            )[0]
            minutes = float(z["clip_dur"][calib_clips].sum()) / 60.0
            calib_words = np.where(
                np.isin(clip, calib_clips) & (z["rule"] == 0) & np.isfinite(oof)
            )[0]
            order = calib_words[np.argsort(-oof[calib_words])]
            desc = oof[order]
            final = pack_for(np.arange(len(idx)))
            # Test and guard clips only. Dev scores stay out-of-fold.
            word_set = sett[clip]
            word_test = test[clip]
            score_at = np.where(
                ((word_set == SET_HELP_ACTED) & (word_test == 1))
                | ((word_set == SET_HELP_CLEAN) & (word_test == 1))
                | (word_set == SET_TLOG_DEV)
                | (word_set == SET_V1)
            )[0]
            # Dev acted localisation reads OOF; test/guard read the refit.
            scores = oof.copy()
            pred = score_idx(final, score_at)
            scores[score_at] = pred
            loc_dev = _loc(z, oof, test=0)
            loc_test = _loc(z, scores, test=1)
            ops = []
            for rate in OPS_PER_MIN:
                thr = threshold_at_k(desc, k_for_rate(rate, minutes))
                m = _metrics(z, scores, float(thr))
                boot = _bootstrap(base["rows"], m["rows"])
                ff_ok = all(m["ff"][k] <= base["ff"][k] for k in ("help", "tlog", "v1"))
                band = all(_rate(m, k) <= _rate(base, k) + 0.10 + 1e-9 for k in ("help", "tlog", "v1"))
                op = {
                    "rate": rate, "thr": round(float(thr), 4),
                    "P": round(m["P"], 3), "R": round(m["R"], 3), "F1": round(m["F1"], 3),
                    "caught": m["caught"], "n": m["n"],
                    "ff": m["ff"], "ff_rate": {k: round(_rate(m, k), 3) for k in m["ff"]},
                    "dR": round(boot["dR"], 3), "dR_lo": round(boot["dR_lo"], 3), "dR_hi": round(boot["dR_hi"], 3),
                    "dF1": round(boot["dF1"], 3), "dF1_lo": round(boot["dF1_lo"], 3), "dF1_hi": round(boot["dF1_hi"], 3),
                    "ff_ok": ff_ok, "band_0_1": band, "per_kind": m["per_kind"],
                }
                ops.append(op)
                results_for_verdict.append({**op, "head": head, "synth": synth})
                print(
                    f"  op={rate:.2f} P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
                    f"dR={boot['dR']:+.3f}[{boot['dR_lo']:+.3f},{boot['dR_hi']:+.3f}] "
                    f"ff {_ff_line(m)} ff_ok={int(ff_ok)}",
                    flush=True,
                )
            summary["configs"].append({
                "head": head, "synth": synth, "dev_auc": dev_auc, "dev_ap": dev_ap,
                "calib_minutes": round(minutes, 2), "calib_words": int(len(calib_words)),
                "loc_dev": loc_dev, "loc_test": loc_test, "ops": ops,
            })
            print(f"  loc dev {loc_dev} test {loc_test}", flush=True)

    matched = [r for r in results_for_verdict if r["ff_ok"]]
    band = [r for r in results_for_verdict if r["band_0_1"]]

    def _best(rows):
        return max(rows, key=lambda r: (r["dR"], r["dR_lo"])) if rows else None

    best = _best(matched)
    best_band = _best(band)
    if best is None:
        call = "kill"
        why = "no pre-declared operating point kept FF counts <= rules-only on help, TLOG dev and v1"
    elif best["dR"] < 0.05 or best["dR_lo"] <= 0:
        call = "kill"
        why = "best matched-FF test recall gain is under +0.05 or its CI includes 0"
    elif best["dR"] >= 0.08 and best["dR_lo"] > 0:
        call = "go"
        why = "matched-FF test recall gain is at least +0.08 and the CI excludes 0"
    else:
        call = "neither"
        why = "matched-FF gain is between the kill and go cutoffs"
    summary["verdict"] = call
    summary["why"] = why
    summary["best_matched"] = best
    summary["best_plus_0_1"] = best_band
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Drop per-clip rows if any slipped in. Keep numbers and head names only.
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"VERDICT {call}: {why}", flush=True)
    if best:
        print(
            f"best matched head={best['head']} synth={int(best['synth'])} op={best['rate']} "
            f"dR={best['dR']:+.3f}[{best['dR_lo']:+.3f},{best['dR_hi']:+.3f}] R={best['R']}",
            flush=True,
        )
    if best_band:
        print(
            f"best +0.1/min head={best_band['head']} synth={int(best_band['synth'])} op={best_band['rate']} "
            f"dR={best_band['dR']:+.3f}[{best_band['dR_lo']:+.3f},{best_band['dR_hi']:+.3f}] R={best_band['R']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
