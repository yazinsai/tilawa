"""Rabah Quran CTC experiment with layer-pruned and int8 variants.

Variants exposed via list_models():
- 12-layer-int8
- 8-layer-int8
- 6-layer-int8

The module prefers local pruned checkpoints under:
  data/rabah-ctc-pruned-{layers}/
If unavailable, it loads the upstream model and prunes at runtime.
"""

import io
import math
import os
import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from Levenshtein import ratio as lev_ratio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

# Import ctc_scorer from sibling experiment
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

RABAH_MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"
LOCAL_SOURCE = PROJECT_ROOT / "data" / "rabah-ctc-source"
LOCAL_PRUNED_TEMPLATE = PROJECT_ROOT / "data" / "rabah-ctc-pruned-{layers}"
LOCAL_FINETUNED_TEMPLATE = PROJECT_ROOT / "data" / "rabah-ctc-pruned-{layers}l-{strategy}-finetuned"
ALLOW_REMOTE_DOWNLOAD = os.getenv("ALLOW_REMOTE_MODELS", "0") == "1"

TOP_K = 100
CHUNK_SIZE = 50
TOP_SURAHS = 5
MAX_SPAN = 4
SKIP_SPAN_THRESHOLD = 0.15

VARIANTS = {
    # Fine-tuned variants (preferred — trained CTC head on pruned representations)
    "8-layer-ft-es-int8": {"layers": 8, "strategy": "evenly_spaced", "finetuned": True, "int8": True},
    "8-layer-ft-fn-int8": {"layers": 8, "strategy": "first_n", "finetuned": True, "int8": True},
    "12-layer-ft-es-int8": {"layers": 12, "strategy": "evenly_spaced", "finetuned": True, "int8": True},
    "6-layer-ft-es-int8": {"layers": 6, "strategy": "evenly_spaced", "finetuned": True, "int8": True},
    # Naive-pruned baselines (no fine-tuning after pruning)
    "12-layer-int8": {"layers": 12, "int8": True},
    "8-layer-int8": {"layers": 8, "int8": True},
    "6-layer-int8": {"layers": 6, "int8": True},
}

_models: dict[str, Wav2Vec2ForCTC] = {}
_processors: dict[str, Wav2Vec2Processor] = {}
_devices: dict[str, str] = {}
_model_sizes: dict[str, int] = {}
_db = None


def list_models() -> list[str]:
    return list(VARIANTS.keys())


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _estimate_state_dict_size(model: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def _can_use_dynamic_int8() -> bool:
    supported = set(torch.backends.quantized.supported_engines)
    # Try setting qnnpack explicitly (fixes macOS ARM where default engine is unavailable)
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    engine = torch.backends.quantized.engine
    return engine in {"qnnpack", "fbgemm", "x86"} and engine in supported


def _evenly_spaced_indices(total_layers: int, keep_layers: int) -> list[int]:
    if keep_layers >= total_layers:
        return list(range(total_layers))
    if keep_layers <= 1:
        return [total_layers - 1]

    raw = [
        round(i * (total_layers - 1) / (keep_layers - 1))
        for i in range(keep_layers)
    ]

    fixed = []
    prev = -1
    for idx in raw:
        idx = max(idx, prev + 1)
        fixed.append(min(idx, total_layers - 1))
        prev = fixed[-1]
    return fixed


def _prune_model_in_place(model: Wav2Vec2ForCTC, layers: int) -> None:
    current_layers = len(model.wav2vec2.encoder.layers)
    if layers >= current_layers:
        return
    keep_indices = _evenly_spaced_indices(current_layers, layers)
    model.wav2vec2.encoder.layers = torch.nn.ModuleList(
        [model.wav2vec2.encoder.layers[i] for i in keep_indices]
    )
    model.config.num_hidden_layers = layers
    model.wav2vec2.config.num_hidden_layers = layers


def _resolve_model_source(
    layers: int, strategy: str | None = None, finetuned: bool = False
) -> tuple[str, bool, int]:
    """Return (source_path_or_id, already_pruned, on_disk_size)."""
    # Check fine-tuned path first
    if finetuned and strategy:
        ft_dir = Path(str(LOCAL_FINETUNED_TEMPLATE).format(layers=layers, strategy=strategy))
        if ft_dir.exists():
            return str(ft_dir), True, _dir_size_bytes(ft_dir)

    # Fall back to naive-pruned
    pruned_dir = Path(str(LOCAL_PRUNED_TEMPLATE).format(layers=layers))
    if pruned_dir.exists():
        return str(pruned_dir), True, _dir_size_bytes(pruned_dir)
    if LOCAL_SOURCE.exists():
        return str(LOCAL_SOURCE), False, _dir_size_bytes(LOCAL_SOURCE)
    if not ALLOW_REMOTE_DOWNLOAD:
        raise FileNotFoundError(
            "No local Rabah checkpoint found. Build local models first: "
            ".venv/bin/python scripts/build_rabah_pruned_models.py --layers 12 8 6 --save-source. "
            "Set ALLOW_REMOTE_MODELS=1 to permit Hugging Face downloads."
        )
    return RABAH_MODEL_ID, False, 0


def _ensure_loaded(model_name: str):
    global _db
    if model_name in _models:
        return

    if model_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{model_name}'. Choices: {list_models()}")

    cfg = VARIANTS[model_name]
    layers = cfg["layers"]
    use_int8 = cfg["int8"]
    strategy = cfg.get("strategy")
    finetuned = cfg.get("finetuned", False)

    source, already_pruned, on_disk_size = _resolve_model_source(layers, strategy, finetuned)

    print(f"Loading Rabah CTC from {source} (target layers={layers}, int8={use_int8})...")
    processor = Wav2Vec2Processor.from_pretrained(source)
    model = Wav2Vec2ForCTC.from_pretrained(source)

    if not already_pruned:
        _prune_model_in_place(model, layers)

    quantized_ok = False
    if use_int8 and _can_use_dynamic_int8():
        # Dynamic int8 quantization is CPU-only; keep this variant on CPU.
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        device = "cpu"
        quantized_ok = True
    else:
        if use_int8:
            print("Warning: int8 quantization backend unavailable; using fp32 pruned model.")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model.eval()
    model.to(device)

    if use_int8 and quantized_ok:
        size_bytes = _estimate_state_dict_size(model)
    elif on_disk_size > 0:
        size_bytes = on_disk_size
    else:
        size_bytes = _estimate_state_dict_size(model)

    _models[model_name] = model
    _processors[model_name] = processor
    _devices[model_name] = device
    _model_sizes[model_name] = size_bytes

    if _db is None:
        _db = QuranDB()


def _tokenize_for_ctc(text: str, processor: Wav2Vec2Processor) -> list[int]:
    encoded = processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _levenshtein_candidates(text: str) -> list[dict]:
    normalized = normalize_arabic(text)
    spaceless = normalized.replace(" ", "")

    scored = []
    for verse in _db.verses:
        score_regular = lev_ratio(normalized, verse["text_clean"])
        score_spaceless = lev_ratio(spaceless, verse["text_clean"].replace(" ", ""))
        best = max(score_regular, score_spaceless)
        scored.append({**verse, "score": best, "text": verse["text_uthmani"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_K]


def _build_span_candidates(single_results: list[tuple[dict, float]]) -> list[dict]:
    seen = set()
    top_surahs = []
    for candidate, _ in single_results:
        surah = candidate["surah"]
        if surah in seen:
            continue
        seen.add(surah)
        top_surahs.append(surah)
        if len(top_surahs) >= TOP_SURAHS:
            break

    span_candidates = []
    for surah_num in top_surahs:
        verses = _db.get_surah(surah_num)
        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                span_candidates.append(
                    {
                        "surah": surah_num,
                        "ayah": chunk[0]["ayah"],
                        "ayah_end": chunk[-1]["ayah"],
                        "text_clean": " ".join(v["text_clean"] for v in chunk),
                        "text_uthmani": " ".join(v["text_uthmani"] for v in chunk),
                        "text": " ".join(v["text_uthmani"] for v in chunk),
                    }
                )
    return span_candidates


def transcribe(audio_path: str, model_name: str = "8-layer-int8") -> str:
    _ensure_loaded(model_name)

    model = _models[model_name]
    processor = _processors[model_name]
    device = _devices[model_name]

    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]
    return normalize_arabic(processor.decode(pred_ids.cpu().numpy()))


def predict(audio_path: str, model_name: str = "8-layer-int8") -> dict:
    _ensure_loaded(model_name)

    model = _models[model_name]
    processor = _processors[model_name]
    device = _devices[model_name]

    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    logits_cpu = logits.cpu()
    blank_id = processor.tokenizer.pad_token_id

    pred_ids = torch.argmax(logits_cpu, dim=-1)[0]
    transcript = normalize_arabic(processor.decode(pred_ids.numpy()))

    if not transcript.strip():
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }

    candidates = _levenshtein_candidates(transcript)
    token_fn = lambda text: _tokenize_for_ctc(text, processor)
    single_results = score_candidates(logits_cpu, candidates, token_fn, blank_id=blank_id)

    if not single_results:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": transcript,
        }

    span_results = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
        for i in range(0, len(span_candidates), CHUNK_SIZE):
            chunk = span_candidates[i:i + CHUNK_SIZE]
            scored = score_candidates(logits_cpu, chunk, token_fn, blank_id=blank_id)
            span_results.extend(scored)
        span_results.sort(key=lambda x: x[1])

    best_candidate, best_loss = single_results[0]
    if span_results and span_results[0][1] < best_loss:
        best_candidate, best_loss = span_results[0]

    confidence = math.exp(-best_loss) if math.isfinite(best_loss) else 0.0

    return {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(confidence, 4),
        "transcript": transcript,
    }


def model_size(model_name: str = "8-layer-int8") -> int:
    _ensure_loaded(model_name)
    return _model_sizes[model_name]
