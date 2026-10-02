"""Score the head probe. Aggregates only."""
import json, pickle, sys
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

rows = pickle.load(open("/tmp/headprobe/units.pkl", "rb"))
LABELS = {r["id"]: r for r in map(json.loads, open("/tmp/correction_eval/acted_located.jsonl"))}
IN_SCOPE = {"skip_word", "substitution", "vowel", "skip_ayah", "repeat"}
ERR = {"possible_omission", "possible_substitution", "possible_vowel", "possible_skipped_ayah", "unclear_ayah"}
BASE = {}
for s in ("help-acted", "help-clean"):
    for l in open(f"/tmp/tc/rec_enc/{s}.jsonl"):
        r = json.loads(l); BASE[r["id"]] = (s, r)
DUR = {cid: r["duration_s"] for cid, (s, r) in BASE.items()}


def feats(x, which):
    parts = []
    if "rule" in which: parts.append(x["rule"])
    if "lp" in which: parts.append(x["lpf"])
    if "enc" in which: parts.append(x["enc"])
    return np.concatenate(parts)


def model(C):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))


def hit(flag, lab):
    if flag[0] != lab["ayah"]:
        return False
    if lab["label_kind"] == "skip_ayah":
        return True
    return lab.get("word_index") is not None and abs(flag[1] - int(lab["word_index"])) <= 1


def evaluate(split, head_flags):
    """head_flags: cid -> set of (ayah, w). Returns metrics of rules ∪ head."""
    caught = defaultdict(int); n = defaultdict(int); tp = fl = 0; clean_flags = 0; clean_min = 0.0
    for cid, (s, r) in BASE.items():
        if r["split"] != split:
            continue
        flags = {(i["ayah"], i["word"]) for i in r["issues"] if i["kind"] in ERR}
        flags |= head_flags.get(cid, set())
        if s == "help-clean":
            clean_flags += len(flags); clean_min += DUR[cid] / 60; fl += len(flags); continue
        lab = LABELS[cid]
        k = lab["label_kind"]
        n[k] += 1
        h = [f for f in flags if hit(f, lab)]
        caught[k] += bool(h); tp += len(h); fl += len(flags)
    ins = sum(n[k] for k in IN_SCOPE); c_ins = sum(caught[k] for k in IN_SCOPE)
    P = tp / fl if fl else 0; R = c_ins / ins
    return dict(P=round(P, 3), R=round(R, 3), F1=round(2 * P * R / (P + R), 3) if P + R else 0,
                caught=c_ins, n=ins, ff=round(clean_flags / clean_min, 3), clean_flags=clean_flags,
                per_kind={k: f"{caught[k]}/{n[k]}" for k in sorted(n)})


def rule_flagged(cid):
    s, r = BASE[cid]
    return {(i["ayah"], i["word"]) for i in r["issues"] if i["kind"] in ERR}


def run(which, C):
    use = lambda x: not x["ign"]
    dev = [x for x in rows if x["split"] == "dev" and use(x)]
    test = [x for x in rows if x["split"] == "test" and use(x)]
    Xd = np.stack([feats(x, which) for x in dev]); yd = np.array([x["y"] for x in dev])
    gd = np.array([x["spk"] for x in dev])
    oof = np.zeros(len(dev))
    for tr, te in GroupKFold(5).split(Xd, yd, gd):
        oof[te] = model(C).fit(Xd[tr], yd[tr]).predict_proba(Xd[te])[:, 1]
    m = model(C).fit(Xd, yd)
    Xt = np.stack([feats(x, which) for x in test]); yt = np.array([x["y"] for x in test])
    st = m.predict_proba(Xt)[:, 1]
    res = dict(which=which, C=C, dev_oof_auc=round(roc_auc_score(yd, oof), 3), dev_oof_ap=round(average_precision_score(yd, oof), 3),
               test_auc=round(roc_auc_score(yt, st), 3), test_ap=round(average_precision_score(yt, st), 3))
    # Thresholds from OOF dev: k extra head flags on dev clean words (not already rule-flagged).
    clean_dev = sorted((oof[i] for i, x in enumerate(dev) if x["set"] == "help-clean"
                        and (x["ayah"], x["w"]) not in rule_flagged(x["cid"])), reverse=True)
    ops = {}
    for k in (0, 1, 2, 4, 8):
        thr = clean_dev[k] + 1e-9 if k < len(clean_dev) else 0
        def flags(units, scores):
            out = defaultdict(set)
            for x, s in zip(units, scores):
                if s >= thr: out[x["cid"]].add((x["ayah"], x["w"]))
            return out
        ops[k] = dict(thr=round(float(thr), 4), dev_oof=evaluate("dev", flags(dev, oof)), test=evaluate("test", flags(test, st)))
    res["ops"] = ops
    return res


if __name__ == "__main__":
    print("rules only (current defaults, no passage, LP replay):")
    print(" dev ", evaluate("dev", {}))
    SHOW = len(sys.argv) > 1
    if SHOW: print(" test", evaluate("test", {}))
    CONFIGS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [[w, C] for w in (["rule", "lp"], ["enc"], ["rule", "lp", "enc"]) for C in ([0.1, 1.0] if "enc" not in w else [0.0003, 0.001, 0.01])]
    for which, C in CONFIGS:
        which = tuple(which)
        if True:
            r = run(which, C)
            print(f"\n== {'+'.join(which)} C={C}: dev OOF AUC {r['dev_oof_auc']} AP {r['dev_oof_ap']}" + (f" | test AUC {r['test_auc']} AP {r['test_ap']}" if SHOW else ""))
            for k, o in r["ops"].items():
                d, t = o["dev_oof"], o["test"]
                print(f"  k={k} thr={o['thr']}: dev R {d['R']} P {d['P']} ff {d['ff']} ({d['caught']}/{d['n']}) {d['per_kind']}" + (f" | "
                      f"test R {t['R']} P {t['P']} F1 {t['F1']} ff {t['ff']} ({t['caught']}/{t['n']}) {t['per_kind']}" if SHOW else ""))
