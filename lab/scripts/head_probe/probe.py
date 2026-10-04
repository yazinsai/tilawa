"""Offline probe: does a small head on frozen encoder frames separate slipped
words from clean ones better than the CTC-derived rule features?

Private: reads /tmp only, prints aggregates only.
"""
import hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/lab")
from shared.audio import load_audio

LP = Path("/tmp/lp/shipped_enc")
TR = Path("/tmp/tc/trace")
BASE = Path("/tmp/tc/rec_enc")
LABELS = {r["id"]: r for r in map(json.loads, open("/tmp/correction_eval/acted_located.jsonl"))}
MAN = {s["id"]: s for s in json.load(open("/tmp/help/manifest.json"))["samples"]}
HEAD_KINDS = {"substitution", "vowel", "tajweed", "skip_word"}
STATES = 5


def key_of(cid):
    a = load_audio(f"/tmp/help/{cid}.wav").astype("float32")
    return hashlib.sha1(a.tobytes()).hexdigest()[:16], len(a) / 16000


def load_mat(p, width):
    b = p.read_bytes()
    n, per = np.frombuffer(b[:8], "<u4")
    return np.frombuffer(b[8:], "<f4").reshape(n * per // width, width)


def latest_verdicts(trace):
    v = {}
    for op in trace or []:
        for row in op.get("v") or []:
            v[(row[1], row[2])] = row  # (ayah, word) -> packed verdict (last wins)
    return v


def rule_feats(row):
    if row is None:
        return np.full(16, np.nan)
    st = np.zeros(STATES); st[row[4]] = 1
    nums = [np.nan if x is None else float(x) for x in row[5:15]]
    return np.concatenate([st, nums, [1.0]])


def units_for(rec):
    track = rec.get("trackPost") or rec.get("track") or []
    words = defaultdict(list)
    for ch, fr, ayah, w in track:
        words[(ayah, w)].append(fr)
    keys = sorted(words, key=lambda k: min(words[k]))
    out = []
    for i, k in enumerate(keys):
        fs = words[k]
        out.append((k, min(fs), max(fs) + 1))
        # skipped words between consecutive tracked words in the same ayah
        if i + 1 < len(keys) and keys[i + 1][0] == k[0] and keys[i + 1][1] > k[1] + 1:
            a, b = max(fs) + 1, min(words[keys[i + 1]])
            for w in range(k[1] + 1, keys[i + 1][1]):
                out.append(((k[0], w), a, max(a + 1, b)))
    return out


def pool(enc, lp, a, b):
    a = max(0, a - 1); b = min(len(enc), b + 1)
    if b <= a:
        a, b = max(0, min(a, len(enc) - 1)), max(1, min(a, len(enc) - 1) + 1)
    e = enc[a:b]; l = lp[a:b]
    p = np.exp(l)
    ent = -(p * l).sum(1).mean()
    blank = l[:, 250].mean()
    nonblank = l[:, :250].max(1)
    lpf = [b - a, ent, blank, nonblank.mean(), nonblank.min(), l.max(1).mean()]
    return np.concatenate([e.mean(0), e.max(0)]), np.array(lpf)


def build():
    rows = []
    for setname in ("help-acted", "help-clean"):
        for line in open(TR / f"{setname}.jsonl"):
            rec = json.loads(line)
            cid = rec["id"]
            spk = MAN[cid].get("speaker") or MAN[cid].get("speaker_id") or MAN[cid].get("user") or cid
            key, dur = key_of(cid)
            lp = load_mat(LP / f"{key}.f32", 251)
            enc = load_mat(LP / f"{key}.enc.f32", 512)
            assert len(lp) == len(enc), (len(lp), len(enc))
            lab = LABELS.get(cid) if setname == "help-acted" else None
            verd = latest_verdicts(rec.get("trace"))
            for (ayah, w), a, b in units_for(rec):
                e, lpf = pool(enc, lp, a, b)
                y, near, ign = 0, False, False
                if lab and lab.get("mapped") and lab.get("word_index") is not None and ayah == lab["ayah"]:
                    d = abs(w - int(lab["word_index"]))
                    near = d <= 1
                    if lab["label_kind"] in HEAD_KINDS:
                        y = int(d == 0); ign = (0 < d <= 1)
                    else:
                        ign = d <= 1
                if lab and lab["label_kind"] == "skip_ayah" and ayah == lab["ayah"]:
                    ign = True
                rows.append(dict(cid=cid, set=setname, split=rec["split"], spk=spk, dur=dur, ayah=ayah, w=w,
                                 y=y, near=near, ign=ign, enc=e, lpf=lpf, rule=rule_feats(verd.get((ayah, w)))))
    return rows


if __name__ == "__main__":
    import pickle
    rows = build()
    pickle.dump(rows, open("/tmp/headprobe/units.pkl", "wb"))
    for sp in ("dev", "test"):
        r = [x for x in rows if x["split"] == sp]
        print(sp, "units", len(r), "pos", sum(x["y"] for x in r), "clean-set units", sum(x["set"] == "help-clean" for x in r))
