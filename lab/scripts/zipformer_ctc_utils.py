"""Pure helpers for icefall Zipformer2-CTC training/export (no k2, icefall, or GPU).

Blank permutation
-----------------
The reference Zipformer CTC vocab (`tokens.txt`) is 251 symbols with
``<blank>`` last (id 250). icefall CTC assumes blank id 0.

Train-time mapping is the rotation ``icefall_id = (ref_id + 1) % 251`` (so blank
250 → 0, phoneme 0 → 1, …, phoneme 249 → 250). After training, permute the CTC
output Linear's rows (weight + bias) with the inverse rotation so the exported
ONNX emits logits in reference order with blank at 250.

``permute_ctc_head`` is that row permutation: ``W_perm[ref] = W_ice[(ref+1)%251]``.
Verified by ``softmax(x @ W_perm.T + b_perm)[ref] == softmax(x @ W.T + b)[ice]``.

Quran-Lab v3.1 ``.pt`` stores the CTC head in **reference** order (blank=250);
``export_quran_streaming_onnx.py`` does **not** permute on export (loads
``ctc_head`` as-is). Fine-tune load therefore applies ``inverse_permute_ctc_head``
so icefall trains with blank=0, and the existing export permutation restores
reference order.

Streaming T / hop
-----------------
icefall ``export-onnx-streaming-ctc.py``::

    pad_length = 7 + 2 * 3   # Conv2dSubsampling (T-7)//2 plus ConvNeXt right pad
    hop = chunk_size * 2     # decode_chunk_len at 10 ms fbank rate
    T = hop + pad_length     # encoder_embed consumes T frames per chunk

``chunk_size=24`` → hop=48, T=61, matching the reference ``zipformer-io.json``.

CNN kernels are the reference ONNX ``metadata_props`` (and icefall Zipformer
default): ``31,31,15,15,15,31``. Causal conv cache last-dim is ``kernel//2``
(15,15,7,7,7,15), which is what ``zipformer-io.json`` stores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

VOCAB_SIZE = 251
REF_BLANK_ID = 250
ICEFALL_BLANK_ID = 0
# QuranTTS is NPL-1.2 — not in the shipped mix. Ablate with --sources qurantts.
# Leak-free manifests (phase0_modal.py leak-audit --write-rx) + v3-filtered TLOG
# (phase0_modal.py tlog-merge). The raw everyayah/everyayah_multi manifests hold
# the three q-lab held-out EveryAyah reciters; train() refuses them.
DEFAULT_TRAIN_SOURCES = "everyayah_rx,qua_rx,iqra_rx,retasy_rx,tlog_clean_v3"
# Conv2dSubsampling left context 7 frames + ConvNeXt right pad 3 (×2 at 100 Hz).
PAD_LENGTH = 7 + 2 * 3  # 13
# chunk_size at 50 Hz after encoder_embed; 24 → T=61, hop=48 (reference I/O).
DEFAULT_EXPORT_CHUNK_SIZE = 24
DEFAULT_LEFT_CONTEXT_FRAMES = 256
# v3/v3.1 were trained only at these 50 Hz chunk sizes (checkpoint context_profiles).
DEFAULT_TRAIN_CHUNK_SIZES = "8,16,24"
DEFAULT_TRAIN_LEFT_CONTEXT = "128,256"
DEFAULT_BASE_LR = 0.005
DEFAULT_NUM_EPOCHS = 5
DEFAULT_AVG = 3
DEFAULT_WARMUP_BATCHES = 500.0
DEFAULT_INIT_FROM = "/vol/reference/zipformer_p_arabic_v3.1.pt"
REF_HF_REPO = "Quran-Lab/zipformer_p-arabic-v3"
REF_PT_NAME = "zipformer_p_arabic_v3.1.pt"
CTC_ICEFALL_WEIGHT_KEY = "ctc_output.1.weight"
CTC_ICEFALL_BIAS_KEY = "ctc_output.1.bias"

# Quran-Lab custom AsrModel → icefall AsrModel (CTC-only).
_QURANLAB_KEY_REMAP = (
    ("sub.", "encoder_embed."),
    ("ctc_head.", "ctc_output.1."),
)
_QURANLAB_DROP_PREFIXES = (
    "decoder.",
    "joiner.",
    "simple_am_proj.",
    "simple_lm_proj.",
)

ARCH_FLAGS = [
    "--causal",
    "1",
    "--use-ctc",
    "1",
    "--use-transducer",
    "0",
    "--num-encoder-layers",
    "2,2,3,4,3,2",
    "--feedforward-dim",
    "512,768,1024,1536,1024,768",
    "--encoder-dim",
    "192,256,384,512,384,256",
    "--encoder-unmasked-dim",
    "192,192,256,256,256,192",
    "--cnn-module-kernel",
    "31,31,15,15,15,31",
    "--downsampling-factor",
    "1,2,4,8,4,2",
    "--num-heads",
    "4,4,4,8,4,4",
    "--query-head-dim",
    "32",
    "--value-head-dim",
    "12",
    "--pos-dim",
    "192",
]

_ORT_DTYPE = {
    "tensor(float)": "float32",
    "tensor(float32)": "float32",
    "float": "float32",
    "float32": "float32",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "int64": "int64",
    "int32": "int32",
}


def ref_to_icefall_ids(ids: int | Sequence[int] | np.ndarray) -> int | list[int] | np.ndarray:
    """Rotate reference ids so blank 250 maps to icefall blank 0."""
    if isinstance(ids, (int, np.integer)):
        return int((int(ids) + 1) % VOCAB_SIZE)
    arr = np.asarray(ids)
    out = (arr + 1) % VOCAB_SIZE
    if isinstance(ids, np.ndarray):
        return out.astype(ids.dtype, copy=False)
    return out.tolist()


def icefall_to_ref_ids(ids: int | Sequence[int] | np.ndarray) -> int | list[int] | np.ndarray:
    """Inverse of :func:`ref_to_icefall_ids`."""
    if isinstance(ids, (int, np.integer)):
        return int((int(ids) - 1) % VOCAB_SIZE)
    arr = np.asarray(ids)
    out = (arr - 1) % VOCAB_SIZE
    if isinstance(ids, np.ndarray):
        return out.astype(ids.dtype, copy=False)
    return out.tolist()


def permute_ctc_head(
    weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Permute CTC Linear rows from icefall order (blank=0) to reference (blank=250).

    ``weight`` is ``[vocab, hidden]``. After permutation,
    ``softmax(x @ W_perm.T + b_perm)[ref_id] == softmax(x @ W.T + b)[icefall_id]``.
    """
    w = np.asarray(weight)
    b = np.asarray(bias)
    if w.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected weight dim0={VOCAB_SIZE}, got {w.shape}")
    if b.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected bias dim0={VOCAB_SIZE}, got {b.shape}")
    perm = (np.arange(VOCAB_SIZE) + 1) % VOCAB_SIZE
    return w[perm].copy(), b[perm].copy()


def inverse_permute_ctc_head(
    weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Permute CTC Linear rows from reference (blank=250) to icefall (blank=0).

    Inverse of :func:`permute_ctc_head`. icefall row 0 is reference row 250.
    """
    w = np.asarray(weight)
    b = np.asarray(bias)
    if w.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected weight dim0={VOCAB_SIZE}, got {w.shape}")
    if b.shape[0] != VOCAB_SIZE:
        raise ValueError(f"expected bias dim0={VOCAB_SIZE}, got {b.shape}")
    perm = (np.arange(VOCAB_SIZE) - 1) % VOCAB_SIZE
    return w[perm].copy(), b[perm].copy()


def remap_quranlab_key(key: str) -> str | None:
    """Map a Quran-Lab v3.1 ``model`` key onto icefall AsrModel, or ``None`` to drop.

    Drops leftover transducer heads (``decoder`` / ``joiner`` / ``simple_*``);
    icefall ``--use-transducer 0`` does not instantiate them. ``sub.*`` is their
    Conv2dSubsampling; icefall calls it ``encoder_embed.*``. ``ctc_head`` is a
    bare Linear → icefall ``ctc_output.1`` (Sequential dropout, linear, log-softmax).
    """
    for prefix in _QURANLAB_DROP_PREFIXES:
        if key == prefix[:-1] or key.startswith(prefix):
            return None
    for src, dst in _QURANLAB_KEY_REMAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    return key


def remap_quranlab_state_keys(keys: Iterable[str]) -> dict[str, str | None]:
    """``old_key → new_key | None`` for every key in ``keys``."""
    return {k: remap_quranlab_key(k) for k in keys}


FT_CHECKPOINT_STATE_KEYS = frozenset({"model", "model_avg"})


def parse_export_interp(spec: str) -> tuple[str, str, float, str]:
    """Parse ``INIT_PT:FT_PT:ALPHA[:model|model_avg]``.

    Alpha is the last float field; an optional 4th token selects the icefall
    checkpoint tensor dict (default ``model``). Paths may contain colons.
    """
    text = str(spec).strip()
    ft_state_key = "model"
    rest = text
    head, sep, tail = text.rpartition(":")
    if sep and tail in FT_CHECKPOINT_STATE_KEYS:
        ft_state_key = tail
        rest = head
    parts = rest.rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(
            "export-interp must be INIT_PT:FT_PT:ALPHA[:model|model_avg], "
            f"got {spec!r}"
        )
    init_pt, ft_pt, alpha_s = parts
    try:
        alpha = float(alpha_s)
    except ValueError as exc:
        raise ValueError(
            f"export-interp alpha is not a float: {alpha_s!r}"
        ) from exc
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"export-interp alpha must be in [0, 1], got {alpha}")
    return init_pt, ft_pt, alpha, ft_state_key


def checkpoint_top_keys(blob: object) -> list[str]:
    """Sorted top-level keys of an icefall/Quran-Lab checkpoint wrapper."""
    if not isinstance(blob, dict):
        return []
    return sorted(str(k) for k in blob.keys())


def select_checkpoint_state(blob: object, key: str) -> dict:
    """Pull ``blob[key]`` (``model`` or ``model_avg``). Fail loud if missing."""
    if key not in FT_CHECKPOINT_STATE_KEYS:
        raise ValueError(
            f"ft state key must be one of {sorted(FT_CHECKPOINT_STATE_KEYS)}, got {key!r}"
        )
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint is {type(blob)}, not a dict")
    top = checkpoint_top_keys(blob)
    if key not in blob:
        raise KeyError(
            f"checkpoint has no {key!r}; top-level keys: {top}"
        )
    raw = blob[key]
    if not isinstance(raw, dict):
        raise TypeError(
            f"checkpoint[{key!r}] is {type(raw)}, not a state dict; "
            f"top-level keys: {top}"
        )
    return raw


def detect_state_space(keys: Iterable[str]) -> str:
    """``reference`` if ``ctc_head.*``/``sub.*``; ``icefall`` if ``ctc_output.1.*``.

    Raises if both (or neither) families are present so a second inverse-permute
    cannot silently rotate an already-icefall CTC head.
    """
    key_list = [str(k) for k in keys]
    has_ref = any(
        k.startswith("ctc_head.") or k.startswith("sub.") for k in key_list
    )
    has_ice = any(k.startswith("ctc_output.1.") for k in key_list)
    if has_ref and has_ice:
        raise ValueError(
            "ambiguous state-dict space: both reference markers "
            "(ctc_head.*/sub.*) and icefall markers (ctc_output.1.*) present"
        )
    if has_ref:
        return "reference"
    if has_ice:
        return "icefall"
    raise ValueError(
        "cannot detect state-dict space: need ctc_head.*/sub.* (reference) "
        "or ctc_output.1.* (icefall)"
    )


def icefall_state_from_reference(
    init_sd: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Remap Quran-Lab keys and inverse-permute CTC rows to icefall blank=0 order."""
    out: dict[str, np.ndarray] = {}
    for key, value in init_sd.items():
        new_key = remap_quranlab_key(key)
        if new_key is None:
            continue
        if new_key in out:
            raise RuntimeError(f"key collision after remap: {key} -> {new_key}")
        out[new_key] = np.asarray(value)
    if CTC_ICEFALL_WEIGHT_KEY not in out or CTC_ICEFALL_BIAS_KEY not in out:
        ctc_keys = [k for k in out if "ctc" in k]
        raise RuntimeError(
            f"CTC Linear missing after remap: have {ctc_keys} "
            f"(need {CTC_ICEFALL_WEIGHT_KEY}, {CTC_ICEFALL_BIAS_KEY})"
        )
    weight, bias = inverse_permute_ctc_head(
        out[CTC_ICEFALL_WEIGHT_KEY], out[CTC_ICEFALL_BIAS_KEY]
    )
    out[CTC_ICEFALL_WEIGHT_KEY] = weight
    out[CTC_ICEFALL_BIAS_KEY] = bias
    return out


def interpolate_state_dicts(
    init_sd: dict[str, np.ndarray],
    ft_sd: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """WiSE-FT blend ``alpha * ft + (1 - alpha) * init`` in icefall key/CTC order.

    Detects init space from keys: reference (``ctc_head.*``/``sub.*``) is remapped
    and inverse-permuted via :func:`icefall_state_from_reference`; icefall
    (``ctc_output.1.*``) is used as-is. Ambiguous/unknown spaces raise.
    Matching keys only; missing/extra keys or shape mismatch raise.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    space = detect_state_space(init_sd)
    init = (
        icefall_state_from_reference(init_sd)
        if space == "reference"
        else {k: np.asarray(v) for k, v in init_sd.items()}
    )
    ft = {k: np.asarray(v) for k, v in ft_sd.items()}
    only_init = sorted(set(init) - set(ft))
    only_ft = sorted(set(ft) - set(init))
    if only_init or only_ft:
        raise KeyError(
            "state dict key mismatch before interpolation: "
            f"only_init={only_init[:24]} only_ft={only_ft[:24]} "
            f"n_only_init={len(only_init)} n_only_ft={len(only_ft)}"
        )
    out: dict[str, np.ndarray] = {}
    for key in init:
        left, right = init[key], ft[key]
        if left.shape != right.shape:
            raise ValueError(
                f"shape mismatch for {key}: init {left.shape} vs ft {right.shape}"
            )
        blended = (1.0 - float(alpha)) * left.astype(np.float64) + float(alpha) * right.astype(
            np.float64
        )
        out[key] = blended.astype(left.dtype, copy=False)
    return out


def arch_train_flags() -> list[str]:
    """Architecture + causal chunk flags for icefall ``train.py``."""
    return [
        *ARCH_FLAGS,
        "--chunk-size",
        DEFAULT_TRAIN_CHUNK_SIZES,
        "--left-context-frames",
        DEFAULT_TRAIN_LEFT_CONTEXT,
    ]


def icefall_train_flags(
    *,
    world_size: int = 1,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    start_epoch: int = 1,
    exp_dir: str = "/vol/exp/run",
    base_lr: float = DEFAULT_BASE_LR,
    max_duration: int = 1200,
    sources: str = DEFAULT_TRAIN_SOURCES,
    source_weights: str = "",
    limit_cuts: int = 0,
    smoke: bool = False,
    init_from: str = DEFAULT_INIT_FROM,
) -> list[str]:
    """Argv for icefall ``zipformer/train.py`` (no executable). ``init_from`` is
    recorded for tests; the Modal job loads it via env, not as an icefall flag.
    """
    del init_from  # loaded via ZIPFORMER_INIT_FROM, not train.py argv
    limited = smoke or int(limit_cuts) > 0
    return [
        "--world-size",
        str(world_size),
        "--num-epochs",
        str(num_epochs),
        "--start-epoch",
        str(start_epoch),
        "--exp-dir",
        str(exp_dir),
        "--bpe-model",
        "/app/tokens.txt",
        "--use-fp16",
        "1",
        "--base-lr",
        str(base_lr),
        "--max-duration",
        str(max_duration),
        "--full-libri",
        "1",
        "--enable-musan",
        "0",
        "--num-workers",
        "2" if smoke else "8",
        "--drop-last",
        "0" if limited else "1",
        "--num-buckets",
        "4" if limited else "30",
        "--manifest-dir",
        "/vol/manifests",
        "--sources",
        sources,
        "--source-weights",
        source_weights,
        "--limit-cuts",
        str(limit_cuts),
        *arch_train_flags(),
    ]


def parse_source_list(sources: str | Sequence[str]) -> list[str]:
    if isinstance(sources, str):
        return [s.strip() for s in sources.split(",") if s.strip()]
    return [str(s).strip() for s in sources if str(s).strip()]


def parse_source_weights(spec: str) -> dict[str, float]:
    """Parse ``a=1.5,b=2`` into ``{a: 1.5, b: 2.0}``. Empty → ``{}``.

    Factors multiply that source's hours-weight in ``CutSet.mux`` (default 1.0).
    """
    text = str(spec or "").strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        name, sep, raw = token.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(
                "source-weights must be comma-separated source=factor "
                f"(e.g. everyayah_multi=2.5,a=1.5), got {spec!r}"
            )
        try:
            factor = float(raw.strip())
        except ValueError as exc:
            raise ValueError(
                f"source-weights factor for {name!r} is not a float: {raw!r}"
            ) from exc
        if factor < 0:
            raise ValueError(
                f"source-weights factor for {name!r} must be >= 0, got {factor}"
            )
        if name in out:
            raise ValueError(f"duplicate source-weights key: {name!r}")
        out[name] = factor
    return out


def scale_mux_weights(
    hours: Sequence[float],
    sources: Sequence[str],
    overrides: dict[str, float] | None = None,
) -> list[float]:
    """``hours[i] * overrides.get(sources[i], 1.0)`` for ``CutSet.mux``."""
    if len(hours) != len(sources):
        raise ValueError(
            f"hours/sources length mismatch: {len(hours)} vs {len(sources)}"
        )
    factors = overrides or {}
    return [
        float(h) * float(factors.get(str(src), 1.0))
        for h, src in zip(hours, sources)
    ]


def fbank_cuts_name(source: str) -> str:
    return f"{source}_cuts_fbank.jsonl.gz"


def missing_fbank_sources(
    sources: str | Sequence[str],
    manifest_dir: str | Path,
) -> list[str]:
    """Requested sources whose ``{source}_cuts_fbank.jsonl.gz`` is absent."""
    root = Path(manifest_dir)
    return [
        s
        for s in parse_source_list(sources)
        if not (root / fbank_cuts_name(s)).is_file()
    ]


def resolve_train_sources(
    sources: str | Sequence[str],
    manifest_dir: str | Path,
    *,
    smoke: bool = False,
    synthetic: bool = False,
) -> tuple[list[str], bool]:
    """Return ``(src_list, use_synthetic)``.

    Full runs (not ``--smoke``, not ``--synthetic``) fail loud if any
    requested source is missing. Never drop a subset and never fall
    through to synthetic. Smoke may fall back to synthetic. Explicit
    ``synthetic=True`` skips the check.
    """
    src_list = parse_source_list(sources)
    if synthetic:
        return ["synthetic"], True
    missing = missing_fbank_sources(src_list, manifest_dir)
    if not missing:
        return src_list, False
    if smoke:
        return ["synthetic"], True
    avail = sorted(p.name for p in Path(manifest_dir).glob("*_cuts_fbank.jsonl.gz"))
    raise FileNotFoundError(
        "missing staged fbank cuts for sources "
        f"{missing}; requested={src_list} available={avail}. "
        "Wait for prepare to finish, or pass --synthetic / --smoke."
    )


def compute_T_hop(chunk_size: int, pad_length: int = PAD_LENGTH) -> tuple[int, int]:
    """Return ``(T, hop)`` for icefall streaming Zipformer export.

    ``hop = chunk_size * 2`` (fbank frames consumed per decode step).
    ``T = hop + pad_length`` with ``pad_length = 7 + 2*3 = 13``.
    ``chunk_size=24`` yields the reference ``(61, 48)``.
    """
    hop = int(chunk_size) * 2
    t = hop + int(pad_length)
    return t, hop


def write_icefall_tokens(tokens: Sequence[str], path: str | Path) -> None:
    """Write icefall-ordered ``tokens.txt``: ``<blk> 0`` then phonemes at 1..250."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<blk> 0"]
    n = 1
    for tok in tokens:
        if tok in ("<blank>", "<blk>"):
            continue
        lines.append(f"{tok} {n}")
        n += 1
    if n != VOCAB_SIZE:
        raise ValueError(
            f"expected {VOCAB_SIZE - 1} non-blank tokens, wrote {n - 1}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def icefall_token_list(ref_tokens: Sequence[str]) -> list[str]:
    """``['<blk>', *phonemes]`` — icefall order, length 251."""
    phonemes = [t for t in ref_tokens if t not in ("<blank>", "<blk>")]
    if len(phonemes) != VOCAB_SIZE - 1:
        raise ValueError(f"expected {VOCAB_SIZE - 1} phonemes, got {len(phonemes)}")
    return ["<blk>", *phonemes]


def _as_list(shape: Any) -> list[int]:
    if shape is None:
        return []
    out = []
    for d in list(shape):
        if d is None or d == "N" or isinstance(d, str):
            out.append(1 if d in (None, "N") else d)
        else:
            out.append(int(d))
    return out


def _dtype_str(typ: Any) -> str:
    if typ is None:
        return "float32"
    if isinstance(typ, type):
        typ = getattr(typ, "__name__", str(typ))
    s = str(typ).lower()
    if s in _ORT_DTYPE:
        return _ORT_DTYPE[s]
    if "int64" in s:
        return "int64"
    if "int32" in s:
        return "int32"
    return "float32"


def _node_meta(node: Any) -> dict:
    name = getattr(node, "name", None) or node["name"]
    shape = getattr(node, "shape", None)
    if shape is None and isinstance(node, dict):
        shape = node.get("shape") or node.get("dims")
    typ = getattr(node, "type", None)
    if typ is None and isinstance(node, dict):
        typ = node.get("type") or node.get("dtype")
    return {"name": name, "dims": _as_list(shape), "dtype": _dtype_str(typ)}


def io_json_from_session(
    inputs_meta: Iterable[Any],
    outputs_meta: Iterable[Any],
    *,
    model: str,
    T: int,
    hop: int,
    feature_dim: int = 80,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    """Build a ``zipformer-io.json`` dict from ONNX Runtime session metadata."""
    inputs = [_node_meta(n) for n in inputs_meta]
    outputs = [_node_meta(n) for n in outputs_meta]
    return {
        "model": model,
        "T": int(T),
        "hop": int(hop),
        "featureDim": int(feature_dim),
        "vocabSize": int(vocab_size),
        "inputs": inputs,
        "outputs": outputs,
    }


def diff_io_json(ours: dict, ref: dict) -> list[str]:
    """Human-readable name/dims diffs. Names-only mismatches are 'NAME' prefixed."""
    diffs: list[str] = []
    our_in = {i["name"]: i for i in ours.get("inputs", [])}
    ref_in = {i["name"]: i for i in ref.get("inputs", [])}
    extra = sorted(set(our_in) - set(ref_in))
    missing = sorted(set(ref_in) - set(our_in))
    if extra:
        diffs.append(f"NAME extra inputs: {extra}")
    if missing:
        diffs.append(f"NAME missing inputs: {missing}")
    for name in sorted(set(our_in) & set(ref_in)):
        a, b = our_in[name], ref_in[name]
        if a.get("dims") != b.get("dims"):
            diffs.append(f"dims {name}: {a.get('dims')} vs ref {b.get('dims')}")
        if a.get("dtype") != b.get("dtype"):
            diffs.append(f"dtype {name}: {a.get('dtype')} vs ref {b.get('dtype')}")
    for key in ("T", "hop", "featureDim", "vocabSize"):
        if key in ref and ours.get(key) != ref.get(key):
            diffs.append(f"{key}: {ours.get(key)} vs ref {ref.get(key)}")
    return diffs


def io_names_match(ours: dict, ref: dict) -> bool:
    our_names = [i["name"] for i in ours.get("inputs", [])]
    ref_names = [i["name"] for i in ref.get("inputs", [])]
    return our_names == ref_names


def io_inputs_match(ours: dict, ref: dict) -> bool:
    """True iff input names and dims match the reference, in order."""
    our = ours.get("inputs", [])
    theirs = ref.get("inputs", [])
    if len(our) != len(theirs):
        return False
    for a, b in zip(our, theirs):
        if a.get("name") != b.get("name"):
            return False
        if list(a.get("dims") or []) != list(b.get("dims") or []):
            return False
    return True


class IcefallPhonemeEncoder:
    """Duck-types SentencePieceProcessor.encode for icefall ``train.py``.

    Wraps :class:`shared.phoneme_labels.PhonemeTokenizer` (reference ids) and
    rotates to icefall ids so CTC blank is 0.
    """

    def __init__(self, ref_tokens: Sequence[str]):
        from shared.phoneme_labels import PhonemeTokenizer

        self.ref_tokens = list(ref_tokens)
        self.icefall_tokens = icefall_token_list(ref_tokens)
        self._tok = PhonemeTokenizer(self.ref_tokens)
        self.blank_id = ICEFALL_BLANK_ID
        self.vocab_size = VOCAB_SIZE

    def get_piece_size(self) -> int:
        return VOCAB_SIZE

    def piece_to_id(self, piece: str) -> int:
        if piece in (" ", "<blk>", "<blank>"):
            return ICEFALL_BLANK_ID
        try:
            return self.icefall_tokens.index(piece)
        except ValueError:
            return ICEFALL_BLANK_ID

    def encode(self, texts, out_type=int):
        def one(text: str):
            ids = ref_to_icefall_ids(self._tok.encode(text))
            if not isinstance(ids, list):
                ids = list(ids)
            if out_type is str or out_type == str:
                return [self.icefall_tokens[i] for i in ids]
            return ids

        if isinstance(texts, str):
            return one(texts)
        return [one(t) for t in texts]


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_plain(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-serialisable Python types.

    Python ``type`` objects (the k2 CPU stub's ``k2.with_cuda`` is a class,
    and ONNX Runtime ``NodeArg.type`` can be a dtype class) are ``str()``'d.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, type):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(k): json_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_plain(v) for v in value]
    if isinstance(value, np.generic):
        return json_plain(value.item())
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return json_plain(value.item())
    return str(value)


def build_metadata(
    *,
    run: str,
    epoch: int,
    avg: int,
    chunk: int,
    left: int,
    icefall_sha: str,
    k2: str,
    param_count: int | None,
    fp32_bytes: int,
    int8_bytes: int,
    T: int,
    hop: int,
    io_diff: Sequence[Any] | None = None,
    init_meta: dict | None = None,
) -> dict:
    """Build the export ``metadata.json`` dict (always ``json.dumps``-able)."""
    meta: dict[str, Any] = {
        "run": run,
        "epoch": epoch,
        "avg": avg,
        "chunk": chunk,
        "left": left,
        "icefall_sha": icefall_sha,
        "k2": k2,
        "param_count": param_count,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "T": T,
        "hop": hop,
        "io_diff": list(io_diff or []),
    }
    if init_meta is not None:
        meta["init_meta"] = init_meta
    return json_plain(meta)
