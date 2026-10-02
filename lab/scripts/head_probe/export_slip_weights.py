"""Fit the shipped slip head and write its weights.

Same logistic as ``train_eval.py``: L2, C=0.01, class-weighted, on the
concatenated mean and max of encoder frames. Training rows are acted dev
speakers plus the clean calibration pool (help clean dev, EveryAyah dev,
TLOG clean train). No splices. Test speakers never enter the fit or the
threshold. The two cutoffs are the probe's: zero extra OOF clean flags
(``strict``) and +0.10 extra clean flags per minute (``high``).

Prints aggregates. The JSON is weights and those two thresholds only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eval import (  # noqa: E402
    SET_EA,
    SET_HELP_ACTED,
    SET_HELP_CLEAN,
    SET_SYNTH,
    SET_TLOG_DEV,
    SET_TLOG_TRAIN,
    SET_V1,
    _bootstrap,
    _fit_linear,
    _load,
    _metrics,
    _predict_linear,
    _rate,
)

OPS = (0.0, 0.10)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "packages" / "core" / "src" / "recitation" / "slip-weights.json"


def _f32(a: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(a, dtype=np.float32)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=Path, required=True, help="a0w units npz from build_units.py, no synth")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    z, _frames = _load(args.units)
    y = z["y"].astype(np.int8)
    ign = z["ign"].astype(bool)
    clip = z["clip"]
    sett = z["clip_set"]
    test = z["clip_test"]
    dev_clip = (test == 0) & np.isin(sett, [SET_HELP_ACTED, SET_HELP_CLEAN, SET_EA, SET_TLOG_TRAIN])
    if int((sett[clip] == SET_SYNTH).sum()):
        raise SystemExit("units contain splices; the shipped head is trained without them")
    train_groups = set(z["clip_group"][dev_clip].tolist())
    test_groups = set(z["clip_group"][(test == 1) & (sett == SET_HELP_ACTED)].tolist())
    if train_groups & test_groups:
        raise SystemExit("group ids overlap dev and test")
    train_word = dev_clip[clip] & ~ign
    idx = np.where(train_word)[0]
    if int(y[idx].sum()) < 5:
        raise SystemExit(f"too few positives: {int(y[idx].sum())}")
    x_word = np.concatenate([z["mean"].astype(np.float32), z["amax"].astype(np.float32)], axis=1)
    groups = z["clip_group"][clip[idx]]
    n_splits = min(5, len(np.unique(groups)))
    oof = np.full(len(y), np.nan, np.float32)
    for tr, te in GroupKFold(n_splits).split(idx, y[idx], groups):
        pack = _fit_linear(x_word[idx[tr]], y[idx[tr]], "logistic")
        held = set(groups[te].tolist())
        pred_at = np.where(dev_clip[clip] & np.isin(z["clip_group"][clip], list(held)))[0]
        if len(pred_at):
            oof[pred_at] = _predict_linear(pack, x_word[pred_at])
    labelled = train_word & np.isfinite(oof)
    dev_auc = float(roc_auc_score(y[labelled], oof[labelled])) if len(np.unique(y[labelled])) > 1 else None
    calib_clips = np.where(
        ((sett == SET_HELP_CLEAN) & (test == 0)) | (sett == SET_EA) | (sett == SET_TLOG_TRAIN)
    )[0]
    minutes = float(z["clip_dur"][calib_clips].sum()) / 60.0
    calib_words = np.where(np.isin(clip, calib_clips) & (z["rule"] == 0) & np.isfinite(oof))[0]
    if len(calib_words) < 50000:
        raise SystemExit(f"calibration words {len(calib_words)} < 50000")
    order = calib_words[np.argsort(-oof[calib_words])]
    desc = oof[order]
    final = _fit_linear(x_word[idx], y[idx], "logistic")
    if final[0] != "model":
        raise SystemExit("logistic fit collapsed to a constant")
    _tag, model, mu, sd = final
    word_set = sett[clip]
    word_test = test[clip]
    score_at = np.where(
        ((word_set == SET_HELP_ACTED) & (word_test == 1))
        | ((word_set == SET_HELP_CLEAN) & (word_test == 1))
        | (word_set == SET_TLOG_DEV)
        | (word_set == SET_V1)
    )[0]
    scores = oof.copy()
    scores[score_at] = _predict_linear(final, x_word[score_at])
    base = _metrics(z, None, None)
    from common import k_for_rate, threshold_at_k  # noqa: E402

    thresholds = {}
    for rate, name in ((0.0, "strict"), (0.10, "high")):
        thr = float(threshold_at_k(desc, k_for_rate(rate, minutes)))
        thresholds[name] = thr
        m = _metrics(z, scores, thr)
        boot = _bootstrap(base["rows"], m["rows"])
        ff = " ".join(
            f"{k} {_rate(m, k):.3f}({m['ff'][k]})" for k in ("help", "tlog", "v1")
        )
        print(
            f"{name} thr={thr:.6f} P={m['P']:.3f} R={m['R']:.3f} "
            f"caught={m['caught']}/{m['n']} dR={boot['dR']:+.3f}"
            f"[{boot['dR_lo']:+.3f},{boot['dR_hi']:+.3f}] ff {ff} "
            f"kind={m['per_kind']}",
            flush=True,
        )
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    payload = {
        "encoder": "a0w-ep1-a0.5",
        "head": "logistic",
        "splices": False,
        "dim": 512,
        "C": 0.01,
        "mu": _f32(mu),
        "sd": _f32(sd),
        "coef": _f32(coef),
        "intercept": float(np.float32(intercept)),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "train": {
            "words": int(len(idx)),
            "positives": int(y[idx].sum()),
            "calib_words": int(len(calib_words)),
            "calib_minutes": round(minutes, 2),
            "dev_auc": None if dev_auc is None else round(dev_auc, 3),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out.name} positives={payload['train']['positives']} "
        f"calib_words={payload['train']['calib_words']} dev_auc={payload['train']['dev_auc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
