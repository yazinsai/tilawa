"""Per-clip scoring for streaming Zipformer2-CTC ONNX exports.

Greedy PER with a sub/ins/del split, plus a CTC forced-alignment score of the
reference so a label filter is not only "what the model already decodes".
numpy + onnxruntime + Levenshtein only; the fbank frontend is the caller's.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from Levenshtein import editops

BLANK = 250
CLEAN_MAX = 0.10
MISLABEL_MIN = 0.35
DRAIN_FRAMES = 200


def _ids_str(ids) -> str:
    return "".join(chr(0x100 + int(i)) for i in ids)


class StreamingZipformer:
    """Cache-aware streaming CTC export (T=61 / hop=48 for chunk 24)."""

    def __init__(self, model_path: str, T: int = 61, hop: int = 48, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.T, self.hop = int(T), int(hop)
        self.state_meta = []
        for inp in self.sess.get_inputs():
            if inp.name == "x":
                continue
            dtype = np.int64 if "int64" in inp.type else np.float32
            # symbolic batch dims ("N") are 1 for single-stream decode
            self.state_meta.append((inp.name, [d if isinstance(d, int) else 1 for d in inp.shape], dtype))
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def log_probs(self, frames: np.ndarray) -> np.ndarray:
        states = {n: np.zeros(s, dtype=d) for n, s, d in self.state_meta}
        pad = np.zeros((DRAIN_FRAMES, frames.shape[1]), dtype=np.float32)
        feats = np.concatenate([frames.astype(np.float32, copy=False), pad], axis=0)
        chunks = []
        i = 0
        while i + self.T <= feats.shape[0]:
            outs = dict(zip(self.out_names, self.sess.run(self.out_names, {"x": feats[None, i : i + self.T], **states})))
            lp = outs["log_probs"]
            chunks.append(lp.reshape(lp.shape[-2], lp.shape[-1]))
            for n in states:
                states[n] = outs[f"new_{n}"]
            i += self.hop
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 251), np.float32)


def greedy_ids(lp: np.ndarray, blank: int = BLANK) -> list[int]:
    out, prev = [], None
    for u in lp.argmax(-1).tolist():
        if u != blank and u != prev:
            out.append(int(u))
        prev = u
    return out


@dataclass
class EditCounts:
    sub: int
    ins: int
    dele: int
    ref_len: int

    @property
    def err(self) -> int:
        return self.sub + self.ins + self.dele

    @property
    def per(self) -> float:
        return self.err / max(self.ref_len, 1)


def edit_counts(ref_ids, hyp_ids) -> EditCounts:
    ops = editops(_ids_str(ref_ids), _ids_str(hyp_ids))
    c = {"replace": 0, "insert": 0, "delete": 0}
    for op, _, _ in ops:
        c[op] += 1
    return EditCounts(sub=c["replace"], ins=c["insert"], dele=c["delete"], ref_len=len(ref_ids))


def ctc_forced_logprob(lp: np.ndarray, ref_ids, blank: int = BLANK) -> float:
    """log p(ref | audio) under CTC (sum over alignments). -inf if infeasible."""
    T = lp.shape[0]
    L = len(ref_ids)
    if L == 0:
        return float(lp[:, blank].sum())
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = np.asarray(ref_ids, dtype=np.int64)
    S = ext.size
    # skip transition s-2 -> s allowed for labels differing from s-2
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    neg = -np.inf
    alpha = np.full(S, neg)
    alpha[0] = lp[0, ext[0]]
    if S > 1:
        alpha[1] = lp[0, ext[1]]
    for t in range(1, T):
        a1 = np.concatenate(([neg], alpha[:-1]))
        a2 = np.where(skip, np.concatenate(([neg, neg], alpha[:-2])), neg)
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + lp[t, ext]
    tail = np.logaddexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1]
    return float(tail)


def ctc_viterbi_spans(lp: np.ndarray, ref_ids, blank: int = BLANK) -> list[tuple[int, int]] | None:
    """Best CTC alignment of ``ref_ids``: (first, last) output frame per label. None if infeasible."""
    T, L = lp.shape[0], len(ref_ids)
    if L == 0 or T < L:
        return None
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = np.asarray(ref_ids, dtype=np.int64)
    S = ext.size
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    neg = -np.inf
    delta = np.full(S, neg)
    delta[0] = lp[0, ext[0]]
    delta[1] = lp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        a0 = delta
        a1 = np.concatenate(([neg], delta[:-1]))
        a2 = np.where(skip, np.concatenate(([neg, neg], delta[:-2])), neg)
        stack = np.stack([a0, a1, a2])
        back[t] = stack.argmax(0)
        delta = stack.max(0) + lp[t, ext]
    s = S - 1 if delta[S - 1] >= delta[S - 2] else S - 2
    if not np.isfinite(delta[s]):
        return None
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    spans: list[list[int]] = [[-1, -1] for _ in range(L)]
    for t, st in enumerate(states):
        if st % 2 == 1:
            k = st // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t
    return [tuple(x) for x in spans]


def best_path_logprob(lp: np.ndarray) -> float:
    return float(lp.max(-1).sum())


def bucket(per: float, clean_max: float = CLEAN_MAX, mislabel_min: float = MISLABEL_MIN) -> str:
    if per <= clean_max:
        return "clean"
    if per > mislabel_min:
        return "mislabel"
    return "suspect"


@dataclass
class ClipScore:
    id: str
    surah: int
    ayah: int
    duration: float
    ref_len: int
    hyp_len: int
    sub: int
    ins: int
    dele: int
    per: float
    frames: int
    forced_lp_per_token: float
    forced_gap_per_frame: float
    bucket: str
    hyp: str = ""
    alt_key: str = ""
    alt_per: float = math.nan

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("per", "forced_lp_per_token", "forced_gap_per_frame", "alt_per"):
            v = d[k]
            d[k] = None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(v, 5)
        return d


def score_clip(
    lp: np.ndarray,
    ref_ids,
    *,
    clip_id: str,
    surah: int,
    ayah: int,
    duration: float,
    tokens: list[str] | None = None,
) -> ClipScore:
    hyp = greedy_ids(lp)
    ec = edit_counts(ref_ids, hyp)
    flp = ctc_forced_logprob(lp, ref_ids)
    gap = (best_path_logprob(lp) - flp) / max(lp.shape[0], 1) if math.isfinite(flp) else math.inf
    return ClipScore(
        id=clip_id,
        surah=surah,
        ayah=ayah,
        duration=round(float(duration), 3),
        ref_len=ec.ref_len,
        hyp_len=len(hyp),
        sub=ec.sub,
        ins=ec.ins,
        dele=ec.dele,
        per=ec.per,
        frames=int(lp.shape[0]),
        forced_lp_per_token=flp / max(len(ref_ids), 1),
        forced_gap_per_frame=gap,
        bucket=bucket(ec.per),
        hyp="".join(tokens[i] for i in hyp) if tokens else "",
    )
