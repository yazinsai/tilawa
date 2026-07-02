# Three Parallel Experiments: Two-Stage, Distilled CTC, Contrastive Fingerprinting

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run three parallel experiments to find the smallest, fastest, most accurate offline Quran verse recognition model targeting 95%+ recall.

**Architecture:** Three independent experiments, each with a Modal training script and a local `run.py` conforming to the existing experiment interface (`predict()`, `transcribe()`, `model_size()`). All training happens on Modal A10G GPUs. Local Mac is inference-only.

**Tech Stack:** PyTorch, transformers, Modal (A10G GPU), FAISS (contrastive), peft (LoRA)

---

### Task 1: Register three new experiments in benchmark runner

**Files:**
- Modify: `benchmark/runner.py:28-36` (EXPERIMENT_REGISTRY)

**Step 1: Add three new entries to EXPERIMENT_REGISTRY**

In `benchmark/runner.py`, add to `EXPERIMENT_REGISTRY`:

```python
EXPERIMENT_REGISTRY = {
    # ... existing entries ...
    "two-stage": EXPERIMENTS_DIR / "two-stage" / "run.py",
    "distilled-ctc": EXPERIMENTS_DIR / "distilled-ctc" / "run.py",
    "contrastive-v2": EXPERIMENTS_DIR / "contrastive-v2" / "run.py",
}
```

**Step 2: Create experiment directories**

```bash
mkdir -p experiments/two-stage experiments/distilled-ctc experiments/contrastive-v2
```

**Step 3: Commit**

```bash
git add benchmark/runner.py
git commit -m "feat: register three new experiments in benchmark runner"
```

---

### Task 2: Experiment 1 — Two-Stage Retrieval (local run.py)

**Files:**
- Create: `experiments/two-stage/run.py`

**Concept:** Stage 1 uses Moonshine Tiny Arabic (27M params, off-the-shelf) for fast ASR → text → QuranDB.search(top_k=50). Stage 2 re-scores only those 50 candidates using CTC forced alignment with a small model.

For the baseline, Stage 2 uses the existing large CTC model (`data/ctc-model/`). After Modal training, swap to the fine-tuned wav2vec2-base.

**Step 1: Write `experiments/two-stage/run.py`**

```python
"""Two-stage retrieval experiment.

Stage 1: Moonshine Tiny Arabic (27M) → fast ASR transcript
         → QuranDB.search() to get top-N verse candidates
Stage 2: CTC forced-alignment re-score on just those N candidates
         (using wav2vec2-base fine-tuned, or fallback to large model)

This bounds CTC computation to N candidates instead of 6,236,
enabling use of accurate CTC scoring with manageable latency.
"""
import sys
import math
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic

# Import ctc_scorer from sibling experiment
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

# ── Config ──
TOP_N = 50                  # candidates from Stage 1 for CTC re-scoring
CTC_CHUNK_SIZE = 50         # batch size for CTC scoring
TOP_SURAHS_SPAN = 3         # surahs to try multi-verse spans
MAX_SPAN = 4                # max consecutive verses in a span

# Model paths
MOONSHINE_MODEL = "UsefulSensors/moonshine-tiny-ar"
CTC_MODEL_SMALL = PROJECT_ROOT / "data" / "ctc-base-finetuned"  # after Modal training
CTC_MODEL_LARGE = PROJECT_ROOT / "data" / "ctc-model"            # fallback
CTC_PRETRAINED = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"  # HF fallback

# ── Lazy-loaded globals ──
_moonshine_model = None
_moonshine_processor = None
_ctc_model = None
_ctc_processor = None
_db = None
_device = None
_ctc_size_bytes = 0


def _ensure_loaded():
    global _moonshine_model, _moonshine_processor
    global _ctc_model, _ctc_processor
    global _db, _device, _ctc_size_bytes

    if _moonshine_model is not None:
        return

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Stage 1: Moonshine Tiny Arabic
    print(f"Loading Moonshine Tiny Arabic from {MOONSHINE_MODEL}...")
    _moonshine_processor = AutoProcessor.from_pretrained(MOONSHINE_MODEL)
    _moonshine_model = MoonshineForConditionalGeneration.from_pretrained(MOONSHINE_MODEL)
    _moonshine_model.eval()
    _moonshine_model.to(_device)

    # Stage 2: CTC model (prefer small fine-tuned, fall back to large)
    if CTC_MODEL_SMALL.exists():
        ctc_path = str(CTC_MODEL_SMALL)
        print(f"Loading fine-tuned CTC-base from {ctc_path}...")
        _ctc_size_bytes = sum(f.stat().st_size for f in CTC_MODEL_SMALL.rglob("*") if f.is_file())
    elif CTC_MODEL_LARGE.exists():
        ctc_path = str(CTC_MODEL_LARGE)
        print(f"Loading large CTC model from {ctc_path} (fallback)...")
        _ctc_size_bytes = 1_200 * 1024 * 1024
    else:
        ctc_path = CTC_PRETRAINED
        print(f"Loading CTC model from HuggingFace: {ctc_path}...")
        _ctc_size_bytes = 1_200 * 1024 * 1024

    _ctc_processor = Wav2Vec2Processor.from_pretrained(ctc_path)
    _ctc_model = Wav2Vec2ForCTC.from_pretrained(ctc_path)
    _ctc_model.eval()
    _ctc_model.to(_device)

    _db = QuranDB()


def _stage1_transcribe(audio: np.ndarray) -> str:
    """Moonshine Tiny Arabic → raw transcript."""
    inputs = _moonshine_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = _moonshine_model.generate(**inputs, max_new_tokens=256)

    text = _moonshine_processor.decode(generated[0], skip_special_tokens=True)
    return normalize_arabic(text)


def _stage1_candidates(transcript: str) -> list[dict]:
    """Get top-N verse candidates from transcript via text matching."""
    results = _db.search(transcript, top_k=TOP_N)
    return results


def _tokenize_for_ctc(text: str) -> list[int]:
    """Convert Arabic text to token IDs for CTC scoring."""
    encoded = _ctc_processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _stage2_ctc_rescore(audio: np.ndarray, candidates: list[dict]) -> list[tuple[dict, float]]:
    """CTC re-score candidates against audio logits."""
    inputs = _ctc_processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _ctc_model(**inputs).logits  # (1, T, V)

    logits_cpu = logits.cpu()
    blank_id = _ctc_processor.tokenizer.pad_token_id

    return score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)


def _build_span_candidates(single_results: list[tuple[dict, float]]) -> list[dict]:
    """Build multi-verse span candidates from top-scoring surahs."""
    seen = set()
    top_surahs = []
    for cand, _ in single_results:
        s = cand["surah"]
        if s not in seen:
            seen.add(s)
            top_surahs.append(s)
            if len(top_surahs) >= TOP_SURAHS_SPAN:
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
                combined_clean = " ".join(v["text_clean"] for v in chunk)
                combined_uthmani = " ".join(v["text_uthmani"] for v in chunk)
                span_candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "text_clean": combined_clean,
                    "text_uthmani": combined_uthmani,
                    "text": combined_uthmani,
                })
    return span_candidates


def transcribe(audio_path: str) -> str:
    """Return raw Moonshine transcript (for streaming pipeline)."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    return _stage1_transcribe(audio)


def predict(audio_path: str) -> dict:
    """Two-stage prediction: Moonshine ASR → CTC re-score."""
    _ensure_loaded()
    audio = load_audio(audio_path)

    # Stage 1: Fast ASR + text matching
    transcript = _stage1_transcribe(audio)
    if not transcript.strip():
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    candidates = _stage1_candidates(transcript)
    if not candidates:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Stage 2: CTC re-score top-N candidates
    single_results = _stage2_ctc_rescore(audio, candidates)

    if not single_results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Multi-verse span scoring for top surahs
    span_results = []
    if single_results[0][1] >= 0.15:  # skip if already confident
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            span_results = _stage2_ctc_rescore(audio, span_candidates)

    # Pick best from single + span
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


def model_size() -> int:
    """Combined size: Moonshine Tiny (~103MB) + CTC model."""
    _ensure_loaded()
    moonshine_size = 103 * 1024 * 1024
    return moonshine_size + _ctc_size_bytes
```

**Step 2: Commit**

```bash
git add experiments/two-stage/run.py
git commit -m "feat: add two-stage retrieval experiment (Moonshine + CTC rescore)"
```

---

### Task 3: Experiment 2 — Distilled CTC (local run.py)

**Files:**
- Create: `experiments/distilled-ctc/run.py`

**Concept:** wav2vec2-base (95M params) fine-tuned for Arabic CTC, then knowledge-distilled from the large model. Scores ALL 6,236 verses directly (no ASR step). Same CTC scoring approach as ctc-alignment but with a model 3x smaller.

Before Modal training, this won't produce meaningful results — it needs Arabic CTC weights. The run.py is structured to load trained weights from `data/ctc-base-finetuned/` or `data/ctc-base-distilled/`.

**Step 1: Write `experiments/distilled-ctc/run.py`**

```python
"""Distilled CTC experiment.

Uses wav2vec2-base (95M params, ~380MB) fine-tuned for Arabic CTC,
optionally knowledge-distilled from the large wav2vec2-xlsr-53-arabic.

Same CTC forced-alignment approach as ctc-alignment/ but 3x smaller model.
After int8 quantization, targets ~95MB on disk.
"""
import sys
import math
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from Levenshtein import ratio as lev_ratio

# Import ctc_scorer
_scorer_path = PROJECT_ROOT / "experiments" / "ctc-alignment" / "ctc_scorer.py"
_spec = importlib.util.spec_from_file_location("ctc_scorer", str(_scorer_path))
_scorer_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scorer_mod)
score_candidates = _scorer_mod.score_candidates

# ── Config ──
TOP_K = 100           # Levenshtein pre-filter candidates
CHUNK_SIZE = 50       # CTC scoring batch size
TOP_SURAHS = 5        # surahs for multi-verse spans
MAX_SPAN = 4          # max consecutive verses
SKIP_SPAN_THRESHOLD = 0.15

# Model paths — prefer distilled > fine-tuned > none
MODEL_DISTILLED = PROJECT_ROOT / "data" / "ctc-base-distilled"
MODEL_FINETUNED = PROJECT_ROOT / "data" / "ctc-base-finetuned"

_model = None
_processor = None
_db = None
_device = None
_model_size_bytes = 0


def _ensure_loaded():
    global _model, _processor, _db, _device, _model_size_bytes
    if _model is not None:
        return

    # Find best available model
    if MODEL_DISTILLED.exists():
        model_path = str(MODEL_DISTILLED)
        _model_size_bytes = sum(f.stat().st_size for f in MODEL_DISTILLED.rglob("*") if f.is_file())
        print(f"Loading distilled CTC-base from {model_path}...")
    elif MODEL_FINETUNED.exists():
        model_path = str(MODEL_FINETUNED)
        _model_size_bytes = sum(f.stat().st_size for f in MODEL_FINETUNED.rglob("*") if f.is_file())
        print(f"Loading fine-tuned CTC-base from {model_path}...")
    else:
        print("ERROR: No trained CTC-base model found!")
        print("  Expected: data/ctc-base-distilled/ or data/ctc-base-finetuned/")
        print("  Run Modal training first: modal run scripts/train_ctc_base_modal.py")
        raise FileNotFoundError("No trained CTC-base model. Run Modal training first.")

    _processor = Wav2Vec2Processor.from_pretrained(model_path)
    _model = Wav2Vec2ForCTC.from_pretrained(model_path)
    _model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(_device)
    _db = QuranDB()


def _tokenize_for_ctc(text: str) -> list[int]:
    encoded = _processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def _levenshtein_candidates(text: str, top_k: int = TOP_K) -> list[dict]:
    normalized = normalize_arabic(text)
    spaceless = normalized.replace(" ", "")

    scored = []
    for v in _db.verses:
        score_regular = lev_ratio(normalized, v["text_clean"])
        score_spaceless = lev_ratio(spaceless, v["text_clean"].replace(" ", ""))
        best = max(score_regular, score_spaceless)
        scored.append({**v, "score": best, "text": v["text_uthmani"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _build_span_candidates(single_results):
    seen = set()
    top_surahs = []
    for cand, _ in single_results:
        s = cand["surah"]
        if s not in seen:
            seen.add(s)
            top_surahs.append(s)
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
                combined_clean = " ".join(v["text_clean"] for v in chunk)
                combined_uthmani = " ".join(v["text_uthmani"] for v in chunk)
                span_candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "text_clean": combined_clean,
                    "text_uthmani": combined_uthmani,
                    "text": combined_uthmani,
                })
    return span_candidates


def transcribe(audio_path: str) -> str:
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    text = _processor.decode(pred_ids.cpu().numpy())
    return normalize_arabic(text)


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)

    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    logits_cpu = logits.cpu()
    blank_id = _processor.tokenizer.pad_token_id

    # Greedy decode for Levenshtein pre-filtering
    pred_ids = torch.argmax(logits_cpu, dim=-1)[0]
    transcript = normalize_arabic(_processor.decode(pred_ids.numpy()))

    if not transcript.strip():
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    # Levenshtein pre-filter → CTC re-score
    candidates = _levenshtein_candidates(transcript, top_k=TOP_K)
    single_results = score_candidates(logits_cpu, candidates, _tokenize_for_ctc, blank_id=blank_id)

    if not single_results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # Multi-verse span scoring
    span_results = []
    if single_results[0][1] >= SKIP_SPAN_THRESHOLD:
        span_candidates = _build_span_candidates(single_results)
        if span_candidates:
            for i in range(0, len(span_candidates), CHUNK_SIZE):
                chunk = span_candidates[i:i + CHUNK_SIZE]
                scored = score_candidates(logits_cpu, chunk, _tokenize_for_ctc, blank_id=blank_id)
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


def model_size() -> int:
    _ensure_loaded()
    return _model_size_bytes
```

**Step 2: Commit**

```bash
git add experiments/distilled-ctc/run.py
git commit -m "feat: add distilled CTC experiment (wav2vec2-base, needs Modal training)"
```

---

### Task 4: Experiment 3 — Contrastive V2 (local run.py + model)

**Files:**
- Create: `experiments/contrastive-v2/run.py`
- Create: `experiments/contrastive-v2/model.py`

**Concept:** Improved contrastive model using multi-reciter training. Audio encoder maps any recitation to a speaker-invariant embedding. Pre-computed FAISS index of all 6,236 verse embeddings. At inference: one forward pass + nearest neighbor = verse ID.

Key improvements over v1:
- Uses `Wav2Vec2Model` (multilingual-capable) instead of English HuBERT
- Deeper projection head (3 layers instead of 2)
- Hard negative mining during training
- Multi-reciter data from EveryAyah (23+ reciters)

**Step 1: Write `experiments/contrastive-v2/model.py`**

```python
"""QuranCLAP v2: Contrastive audio-text model for Quran verse identification.

Improvements over v1:
- wav2vec2-base audio encoder (better cross-lingual transfer than English HuBERT)
- Deeper projection heads (3 layers with layer norm)
- Hard negative mining support
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer


class QuranCLAPv2(nn.Module):
    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-base",
        text_model_name: str = "aubmindlab/bert-base-arabertv02",
        embed_dim: int = 256,
        freeze_audio: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Audio encoder: wav2vec2-base (95M, language-agnostic SSL)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
        audio_dim = self.audio_encoder.config.hidden_size  # 768

        # Text encoder: AraBERT
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Deeper projection heads with layer norm
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        if freeze_audio:
            self._freeze(self.audio_encoder)
        if freeze_text:
            self._freeze(self.text_encoder)

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_audio_encoder(self, last_n_layers: int = 2):
        self._freeze(self.audio_encoder)
        total = len(self.audio_encoder.encoder.layers)
        for i in range(total - last_n_layers, total):
            for param in self.audio_encoder.encoder.layers[i].parameters():
                param.requires_grad = True

    def unfreeze_text_encoder(self, last_n_layers: int = 2):
        self._freeze(self.text_encoder)
        total = len(self.text_encoder.encoder.layer)
        for i in range(total - last_n_layers, total):
            for param in self.text_encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def encode_audio(self, audio_values, attention_mask=None):
        outputs = self.audio_encoder(audio_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        projected = self.audio_proj(pooled)
        return nn.functional.normalize(projected, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        projected = self.text_proj(cls_output)
        return nn.functional.normalize(projected, dim=-1)

    def forward(self, audio_values, input_ids, text_attention_mask, audio_attention_mask=None):
        audio_embeds = self.encode_audio(audio_values, audio_attention_mask)
        text_embeds = self.encode_text(input_ids, text_attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_audio = logit_scale * audio_embeds @ text_embeds.T
        logits_per_text = logits_per_audio.T
        return logits_per_audio, logits_per_text

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


def contrastive_loss(logits_per_audio, logits_per_text):
    batch_size = logits_per_audio.shape[0]
    labels = torch.arange(batch_size, device=logits_per_audio.device)
    loss_a = nn.functional.cross_entropy(logits_per_audio, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


def get_tokenizer(model_name="aubmindlab/bert-base-arabertv02"):
    return AutoTokenizer.from_pretrained(model_name)
```

**Step 2: Write `experiments/contrastive-v2/run.py`**

```python
"""Contrastive audio fingerprinting v2.

Encodes audio → 256-dim speaker-invariant embedding → nearest neighbor
search against pre-computed FAISS index of all 6,236 verse embeddings.

No ASR needed. One forward pass + nearest neighbor = verse ID.
Requires trained model from Modal: data/contrastive-v2-model/
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_DIR = PROJECT_ROOT / "data" / "contrastive-v2-model"
INDEX_PATH = MODEL_DIR / "verse_index.faiss"
METADATA_PATH = MODEL_DIR / "verse_metadata.json"

_model = None
_index = None
_metadata = None
_db = None
_device = None


def _ensure_loaded():
    global _model, _index, _metadata, _db, _device
    if _model is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"No trained contrastive model at {MODEL_DIR}. "
            "Run Modal training first: modal run scripts/train_contrastive_v2_modal.py"
        )

    import faiss
    from model import QuranCLAPv2

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load trained model (audio encoder + projection only needed at inference)
    print(f"Loading contrastive-v2 model from {MODEL_DIR}...")
    _model = QuranCLAPv2(freeze_audio=True, freeze_text=True)
    state_dict = torch.load(MODEL_DIR / "best_model.pt", map_location=_device, weights_only=True)
    _model.load_state_dict(state_dict)
    _model.eval()
    _model.to(_device)

    # Load FAISS index
    print(f"Loading FAISS index from {INDEX_PATH}...")
    _index = faiss.read_index(str(INDEX_PATH))

    # Load metadata (maps index position → surah/ayah)
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)

    _db = QuranDB()
    print(f"Contrastive-v2 loaded: {_index.ntotal} verses indexed")


def transcribe(audio_path: str) -> str:
    """No transcript for embedding-based approach."""
    return ""


def predict(audio_path: str) -> dict:
    _ensure_loaded()

    audio = load_audio(audio_path)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        embedding = _model.encode_audio(audio_tensor)  # (1, 256)

    query = embedding.cpu().numpy().astype(np.float32)

    # Search top-5 nearest verses
    distances, indices = _index.search(query, 5)

    if indices[0][0] == -1:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    best_idx = int(indices[0][0])
    best_distance = float(distances[0][0])
    meta = _metadata[best_idx]

    # Convert L2 distance to confidence (cosine similarity for normalized vectors)
    # For normalized vectors: L2^2 = 2 - 2*cos_sim → cos_sim = 1 - L2^2/2
    confidence = max(0.0, 1.0 - best_distance / 2.0)

    return {
        "surah": meta["surah"],
        "ayah": meta["ayah"],
        "ayah_end": None,
        "score": round(confidence, 4),
        "transcript": "",
    }


def model_size() -> int:
    """Audio encoder (~360MB) + projection (~1MB) + FAISS index (~6MB)."""
    if MODEL_DIR.exists():
        return sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
    return 367 * 1024 * 1024  # estimate
```

**Step 3: Commit**

```bash
git add experiments/contrastive-v2/run.py experiments/contrastive-v2/model.py
git commit -m "feat: add contrastive-v2 experiment (audio fingerprinting, needs Modal training)"
```

---

### Task 5: Modal — Train wav2vec2-base CTC on Quran (shared by Exp 1 + 2)

**Files:**
- Create: `scripts/train_ctc_base_modal.py`

**Concept:** Fine-tune `facebook/wav2vec2-base` (95M params) with an Arabic CTC head on Quran audio. Uses the same Arabic character vocabulary as the large model (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`) so our existing `ctc_scorer.py` works unchanged.

Data: tarteel-ai/everyayah (streaming, 127K+ samples) + RetaSy crowd-sourced (6.8K).

**Step 1: Write `scripts/train_ctc_base_modal.py`**

```python
"""
Fine-tune wav2vec2-base with Arabic CTC head on Quran audio (Modal A10G).

Creates a small CTC model (~380MB) that can score verses via CTC alignment,
matching the vocabulary of the large wav2vec2-xlsr-53-arabic model.

Output: Modal volume "ctc-quran-training" → data/ctc-base-finetuned/

Usage:
    modal run scripts/train_ctc_base_modal.py
"""
import modal

app = modal.App("ctc-base-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("ctc-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
)
def train():
    import torch
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2Config,
        Wav2Vec2Model,
        Trainer,
        TrainingArguments,
    )
    import evaluate
    import numpy as np

    OUTPUT_DIR = Path("/training/ctc-base-finetuned")
    CHECKPOINT_DIR = Path("/training/ctc-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Get Arabic vocabulary from the large model ──
    print("Loading Arabic CTC processor from large model...")
    large_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    vocab_size = large_processor.tokenizer.vocab_size
    pad_token_id = large_processor.tokenizer.pad_token_id
    print(f"Arabic vocabulary size: {vocab_size}, pad_id: {pad_token_id}")

    # ── Create small model with Arabic CTC head ──
    print("Creating wav2vec2-base with Arabic CTC head...")
    base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    base_config.vocab_size = vocab_size
    base_config.ctc_loss_reduction = "mean"
    base_config.pad_token_id = pad_token_id
    base_config.ctc_zero_infinity = True

    # Initialize full model with CTC head
    model = Wav2Vec2ForCTC(base_config)

    # Load pretrained SSL weights (everything except the CTC head)
    print("Loading pretrained wav2vec2-base weights...")
    pretrained = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.wav2vec2.load_state_dict(pretrained.state_dict())
    del pretrained

    # Freeze feature extractor (CNN layers) — only fine-tune transformer + CTC head
    model.freeze_feature_encoder()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params * 4 / 1e6:.0f} MB)")
    print(f"Trainable params: {trainable_params:,}")

    # ── Use large model's processor for consistent tokenization ──
    processor = large_processor

    # ── Load datasets ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 30.0)
    everyayah = everyayah.remove_columns(["duration", "reciter"])

    print("Loading RetaSy crowd-sourced dataset...")
    BAD_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
    retasy = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    labels = retasy["final_label"]
    durations = retasy["duration_ms"]
    good_indices = [
        i for i, (label, dur) in enumerate(zip(labels, durations))
        if (label not in BAD_LABELS if label else True) and dur <= 30000
    ]
    retasy = retasy.select(good_indices)
    print(f"  RetaSy samples after filtering: {len(retasy)}")
    retasy = retasy.rename_column("Aya", "text")
    retasy = retasy.remove_columns([
        c for c in retasy.column_names if c not in ("audio", "text")
    ])
    retasy = retasy.cast_column("audio", Audio(sampling_rate=16000))
    retasy_stream = retasy.to_iterable_dataset()

    # Interleave: 85% EveryAyah, 15% RetaSy
    dataset = interleave_datasets(
        [everyayah, retasy_stream],
        probabilities=[0.85, 0.15],
        seed=42,
    )
    print("Datasets interleaved (85% EveryAyah + 15% RetaSy)")

    # ── Prepare dataset ──
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=16000,
            return_tensors="np",
        ).input_values[0]

        # Tokenize text to character IDs for CTC
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids

        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=["audio", "text"])

    # ── Data collator for CTC ──
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Any

        def __call__(self, features):
            input_values = [{"input_values": f["input_values"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_values, return_tensors="pt", padding=True,
            )

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt", padding=True,
            )

            # Replace padding with -100 so CTC loss ignores it
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ── Metrics ──
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        # Replace -100 in labels
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_steps=500,
        max_steps=5000,
        fp16=True,
        logging_steps=100,
        save_steps=1000,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
        group_by_length=False,  # streaming doesn't support this
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 60)
    print("  Training wav2vec2-base CTC on Quran audio (5000 steps)")
    print("  Model: 95M params, Arabic CTC vocabulary")
    print("  Data: EveryAyah + RetaSy")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Model saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model():
    """Download the trained model files."""
    from pathlib import Path

    model_dir = Path("/training/ctc-base-finetuned")
    if not model_dir.exists():
        print("No model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting CTC-base training on Modal GPU...")
    train.remote()

    print("\nDownloading trained model...")
    out_dir = Path("data/ctc-base-finetuned")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nModel saved to {out_dir} ({total / 1e6:.0f} MB total)")
```

**Step 2: Verify the script runs with `modal run --help` equivalent**

```bash
cd /Users/rock/ai/projects/offline-tarteel
.venv/bin/python -c "import modal; print('Modal available:', modal.__version__)"
```

**Step 3: Commit**

```bash
git add scripts/train_ctc_base_modal.py
git commit -m "feat: add Modal training script for wav2vec2-base CTC on Quran"
```

---

### Task 6: Modal — Knowledge distillation (large CTC → small CTC)

**Files:**
- Create: `scripts/train_distill_modal.py`

**Concept:** Use the large CTC model as a teacher to train the small CTC model. The student learns to reproduce the teacher's frame-level logit distributions (soft targets), which is easier than learning from hard CTC labels alone.

Loss = α * CTC_loss(student, ground_truth) + (1-α) * KL_div(student_logits, teacher_logits)

**Step 1: Write `scripts/train_distill_modal.py`**

```python
"""
Knowledge distillation: large wav2vec2-xlsr-53-arabic → small wav2vec2-base CTC.

Teacher: jonatasgrosman/wav2vec2-large-xlsr-53-arabic (315M, 1.2GB)
Student: wav2vec2-base + Arabic CTC head (95M, ~380MB)

Loss = α * CTC_loss(student, labels) + (1-α) * KL(student || teacher)

Output: Modal volume → data/ctc-base-distilled/

Usage:
    modal run scripts/train_distill_modal.py
"""
import modal

app = modal.App("ctc-distill-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "accelerate",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("ctc-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
    memory=32768,  # 32 GB RAM for both models
)
def train(alpha: float = 0.5, temperature: float = 2.0, max_steps: int = 5000):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2Config,
        Wav2Vec2Model,
    )
    import numpy as np

    OUTPUT_DIR = Path("/training/ctc-base-distilled")
    CHECKPOINT_DIR = Path("/training/distill-checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load teacher model ──
    print("Loading teacher model (wav2vec2-large-xlsr-53-arabic)...")
    teacher_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    vocab_size = teacher_processor.tokenizer.vocab_size
    pad_token_id = teacher_processor.tokenizer.pad_token_id

    # ── Create student model ──
    # Try to load pre-fine-tuned student if available
    student_pretrained = Path("/training/ctc-base-finetuned")
    if student_pretrained.exists():
        print(f"Loading pre-fine-tuned student from {student_pretrained}...")
        student = Wav2Vec2ForCTC.from_pretrained(str(student_pretrained))
    else:
        print("Creating fresh wav2vec2-base student with Arabic CTC head...")
        base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        base_config.vocab_size = vocab_size
        base_config.ctc_loss_reduction = "mean"
        base_config.pad_token_id = pad_token_id
        base_config.ctc_zero_infinity = True
        student = Wav2Vec2ForCTC(base_config)
        pretrained = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        student.wav2vec2.load_state_dict(pretrained.state_dict())
        del pretrained

    student.freeze_feature_encoder()
    student.to(device)
    student.train()

    processor = teacher_processor

    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"Student: {sum(p.numel() for p in student.parameters()):,} params")
    print(f"  Trainable: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    print(f"Alpha: {alpha}, Temperature: {temperature}")

    # ── Load dataset ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 20.0)  # shorter for distillation

    # ── Training loop ──
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    ctc_loss_fn = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

    print(f"\n{'='*60}")
    print(f"  Starting knowledge distillation ({max_steps} steps)")
    print(f"{'='*60}\n")

    step = 0
    running_loss = 0.0
    running_ctc = 0.0
    running_kl = 0.0

    for sample in everyayah:
        if step >= max_steps:
            break

        try:
            audio = sample["audio"]["array"]
            text = sample["text"]

            # Process audio
            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            # Tokenize for CTC labels
            with processor.as_target_processor():
                labels = processor(text).input_ids
            labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            label_lengths = torch.tensor([len(labels)], dtype=torch.long, device=device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(input_values).logits  # (1, T_teacher, V)

            # Student forward
            student_out = student(input_values)
            student_logits = student_out.logits  # (1, T_student, V)

            T_student = student_logits.size(1)
            T_teacher = teacher_logits.size(1)

            # CTC loss on ground truth
            log_probs = F.log_softmax(student_logits, dim=-1).permute(1, 0, 2)  # (T, 1, V)
            input_lengths = torch.tensor([T_student], dtype=torch.long, device=device)
            ctc = ctc_loss_fn(log_probs, labels_tensor, input_lengths, label_lengths)

            # KL divergence on logit distributions
            # Align temporal dimensions (interpolate teacher to student length)
            if T_teacher != T_student:
                teacher_aligned = F.interpolate(
                    teacher_logits.permute(0, 2, 1),
                    size=T_student,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                teacher_aligned = teacher_logits

            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_aligned / temperature, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

            # Combined loss
            loss = alpha * ctc + (1 - alpha) * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_ctc += ctc.item()
            running_kl += kl.item()
            step += 1

            if step % 100 == 0:
                avg_loss = running_loss / 100
                avg_ctc = running_ctc / 100
                avg_kl = running_kl / 100
                print(f"Step {step}/{max_steps}: loss={avg_loss:.4f} (ctc={avg_ctc:.4f}, kl={avg_kl:.4f})")
                running_loss = 0.0
                running_ctc = 0.0
                running_kl = 0.0

            if step % 1000 == 0:
                ckpt_path = CHECKPOINT_DIR / f"step_{step}"
                student.save_pretrained(str(ckpt_path))
                vol.commit()
                print(f"  Saved checkpoint at step {step}")

        except Exception as e:
            print(f"  Skipping sample (error: {e})")
            continue

    # ── Save final model ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving distilled model to {OUTPUT_DIR}...")
    student.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Distilled model saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model():
    from pathlib import Path

    model_dir = Path("/training/ctc-base-distilled")
    if not model_dir.exists():
        print("No distilled model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting knowledge distillation on Modal GPU...")
    train.remote(alpha=0.5, temperature=2.0, max_steps=5000)

    print("\nDownloading distilled model...")
    out_dir = Path("data/ctc-base-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nDistilled model saved to {out_dir} ({total / 1e6:.0f} MB total)")
```

**Step 2: Commit**

```bash
git add scripts/train_distill_modal.py
git commit -m "feat: add Modal knowledge distillation script (large CTC → small CTC)"
```

---

### Task 7: Modal — Contrastive v2 training (multi-reciter)

**Files:**
- Create: `scripts/train_contrastive_v2_modal.py`

**Concept:** Train QuranCLAPv2 on Modal with proper batch size (64-128), multi-reciter data from EveryAyah, and two-phase training. After training, pre-compute FAISS index of all 6,236 verse text embeddings.

**Step 1: Write `scripts/train_contrastive_v2_modal.py`**

```python
"""
Train QuranCLAP v2 contrastive model on Modal A10G.

Multi-reciter contrastive learning:
- Same verse from different reciters = positive pairs
- Different verses = negative pairs (within-batch negatives)

Two-phase: frozen encoders → unfreeze last 2 layers.
After training, builds FAISS index of all verse text embeddings.

Output: Modal volume → data/contrastive-v2-model/

Usage:
    modal run scripts/train_contrastive_v2_modal.py
"""
import modal

app = modal.App("contrastive-v2-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "faiss-cpu",
        "soundfile",
        "librosa",
        "tqdm",
    )
)

vol = modal.Volume.from_name("contrastive-v2-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/training": vol},
    memory=32768,
)
def train(
    batch_size: int = 64,
    phase1_epochs: int = 10,
    phase2_epochs: int = 20,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    embed_dim: int = 256,
    max_samples: int = 50000,
    max_audio_seconds: float = 15.0,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import time
    import numpy as np
    from pathlib import Path
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_dataset, Audio
    from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer
    from tqdm import tqdm
    import faiss

    OUTPUT_DIR = Path("/training/contrastive-v2-model")
    CHECKPOINT_DIR = Path("/training/contrastive-v2-checkpoints")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Define model inline (avoid file dependency on Modal) ──
    class QuranCLAPv2(nn.Module):
        def __init__(self, audio_model_name, text_model_name, embed_dim, freeze_audio, freeze_text):
            super().__init__()
            self.embed_dim = embed_dim
            self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model_name)
            audio_dim = self.audio_encoder.config.hidden_size
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_encoder.config.hidden_size

            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2), nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2), nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

            if freeze_audio:
                for p in self.audio_encoder.parameters(): p.requires_grad = False
            if freeze_text:
                for p in self.text_encoder.parameters(): p.requires_grad = False

        def unfreeze_audio(self, n=2):
            for p in self.audio_encoder.parameters(): p.requires_grad = False
            total = len(self.audio_encoder.encoder.layers)
            for i in range(total - n, total):
                for p in self.audio_encoder.encoder.layers[i].parameters(): p.requires_grad = True

        def unfreeze_text(self, n=2):
            for p in self.text_encoder.parameters(): p.requires_grad = False
            total = len(self.text_encoder.encoder.layer)
            for i in range(total - n, total):
                for p in self.text_encoder.encoder.layer[i].parameters(): p.requires_grad = True

        def encode_audio(self, x, mask=None):
            out = self.audio_encoder(x, attention_mask=mask).last_hidden_state.mean(1)
            return F.normalize(self.audio_proj(out), dim=-1)

        def encode_text(self, ids, mask):
            out = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            return F.normalize(self.text_proj(out), dim=-1)

        def forward(self, audio, ids, text_mask, audio_mask=None):
            a = self.encode_audio(audio, audio_mask)
            t = self.encode_text(ids, text_mask)
            scale = self.logit_scale.exp().clamp(max=100.0)
            return scale * a @ t.T, scale * t @ a.T

    # ── Load model ──
    print("Loading QuranCLAP v2 model...")
    model = QuranCLAPv2(
        "facebook/wav2vec2-base",
        "aubmindlab/bert-base-arabertv02",
        embed_dim, True, True,
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable (Phase 1): {trainable:,}")

    # ── Load dataset ──
    print(f"Loading EveryAyah dataset (max {max_samples} samples)...")
    ds = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.filter(lambda x: x["duration"] <= max_audio_seconds)

    max_audio_len = int(max_audio_seconds * 16000)
    samples = []
    for sample in tqdm(ds, total=max_samples, desc="Loading"):
        if len(samples) >= max_samples:
            break
        try:
            audio = sample["audio"]["array"]
            if len(audio) > max_audio_len:
                audio = audio[:max_audio_len]
            samples.append({
                "audio": np.array(audio, dtype=np.float32),
                "text": sample["text"],
                "reciter": sample["reciter"],
            })
        except Exception:
            continue

    print(f"Loaded {len(samples)} samples")

    # Split 90/10
    n_train = int(0.9 * len(samples))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # ── Dataset + DataLoader ──
    class AudioTextDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            s = self.data[idx]
            return {"audio": torch.tensor(s["audio"]), "text": s["text"]}

    def collate(batch):
        max_len = max(b["audio"].shape[0] for b in batch)
        audios, masks, texts = [], [], []
        for b in batch:
            a = b["audio"]
            pad = max_len - a.shape[0]
            audios.append(F.pad(a, (0, pad)))
            m = torch.ones(max_len)
            if pad > 0: m[-pad:] = 0
            masks.append(m)
            texts.append(b["text"])
        return {
            "audio": torch.stack(audios),
            "mask": torch.stack(masks),
            "text": texts,
        }

    train_loader = DataLoader(AudioTextDataset(train_samples), batch_size=batch_size,
                              shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
    val_loader = DataLoader(AudioTextDataset(val_samples), batch_size=batch_size,
                            shuffle=False, collate_fn=collate, num_workers=2, drop_last=True)

    # ── Contrastive loss ──
    def clip_loss(logits_a, logits_t):
        bs = logits_a.shape[0]
        labels = torch.arange(bs, device=logits_a.device)
        return (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_t, labels)) / 2

    # ── Training helper ──
    def run_epoch(loader, optimizer=None, desc="Train"):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in tqdm(loader, desc=desc, leave=False):
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            texts = batch["text"]

            text_enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            ids = text_enc["input_ids"].to(device)
            tmask = text_enc["attention_mask"].to(device)

            if is_train:
                la, lt = model(audio, ids, tmask, mask)
                loss = clip_loss(la, lt)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    la, lt = model(audio, ids, tmask, mask)
                    loss = clip_loss(la, lt)

            total_loss += loss.item()
            preds = la.argmax(dim=-1)
            labels = torch.arange(la.shape[0], device=device)
            total_correct += (preds == labels).sum().item()
            total_n += la.shape[0]

        return total_loss / len(loader), total_correct / max(total_n, 1)

    # ── Phase 1: Frozen encoders ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Projection heads only ({phase1_epochs} epochs, lr={phase1_lr})")
    print(f"{'='*60}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=phase1_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)
    best_val_loss = float("inf")

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(train_loader, optimizer, f"P1 E{epoch}")
        val_loss, val_acc = run_epoch(val_loader, desc=f"Val E{epoch}")
        scheduler.step()
        elapsed = time.time() - t0
        print(f"P1 Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_phase1.pt")

    # ── Phase 2: Unfreeze last 2 layers ──
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tune encoders ({phase2_epochs} epochs, lr={phase2_lr})")
    print(f"{'='*60}")

    model.unfreeze_audio(2)
    model.unfreeze_text(2)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Phase 2): {trainable:,}")

    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "proj" in n], "lr": phase2_lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "proj" not in n], "lr": phase2_lr * 0.1},
    ], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(train_loader, optimizer, f"P2 E{epoch}")
        val_loss, val_acc = run_epoch(val_loader, desc=f"Val E{epoch}")
        scheduler.step()
        elapsed = time.time() - t0
        print(f"P2 Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  --> New best model (val_loss={val_loss:.4f})")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"phase2_epoch{epoch}.pt")
            vol.commit()

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")

    # ── Build FAISS index of all verse text embeddings ──
    print("\nBuilding FAISS index of verse text embeddings...")

    # Load all Quran verses
    ds_quran = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    # Collect unique verse texts
    verses_seen = set()
    verse_list = []
    for sample in ds_quran:
        key = f"{sample.get('surah', 0)}:{sample.get('ayah', 0)}"
        if key in verses_seen or key == "0:0":
            continue
        verses_seen.add(key)
        # Parse surah:ayah from the text metadata
        # EveryAyah has text field with the verse text
        verse_list.append({
            "text": sample["text"],
            "surah": 0,  # We'll derive from order
            "ayah": 0,
        })
        if len(verse_list) >= 6236:
            break

    # Fallback: encode texts from a simple quran JSON
    # Since we're on Modal, just encode whatever text we have
    print(f"Encoding {len(verse_list)} verse texts...")
    model.eval()

    all_embeddings = []
    metadata = []
    batch_texts = []
    batch_meta = []

    for i, v in enumerate(verse_list):
        batch_texts.append(v["text"])
        batch_meta.append({"surah": v["surah"], "ayah": v["ayah"], "text": v["text"][:50]})

        if len(batch_texts) == 128 or i == len(verse_list) - 1:
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                emb = model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_embeddings.append(emb.cpu().numpy())
            metadata.extend(batch_meta)
            batch_texts, batch_meta = [], []

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (L2 for normalized vectors)
    index = faiss.IndexFlatL2(embed_dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUTPUT_DIR / "verse_index.faiss"))
    with open(OUTPUT_DIR / "verse_metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    vol.commit()
    print(f"\nFAISS index built: {index.ntotal} verses")
    print("Done! Model + index saved to Modal volume.")


@app.function(image=image, volumes={"/training": vol})
def download_model():
    from pathlib import Path
    model_dir = Path("/training/contrastive-v2-model")
    if not model_dir.exists():
        print("No model found!")
        return {}
    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting contrastive v2 training on Modal GPU...")
    train.remote()

    print("\nDownloading trained model + index...")
    out_dir = Path("data/contrastive-v2-model")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nModel + index saved to {out_dir} ({total / 1e6:.0f} MB total)")
```

**Step 2: Commit**

```bash
git add scripts/train_contrastive_v2_modal.py
git commit -m "feat: add Modal training script for contrastive v2 (multi-reciter)"
```

---

### Task 8: Run Two-Stage baseline benchmark (before Modal training)

The two-stage experiment can run NOW using Moonshine Tiny (off-the-shelf) + existing large CTC model as Stage 2. This establishes a baseline for the approach.

**Step 1: Run benchmark for two-stage only**

```bash
cd /Users/rock/ai/projects/offline-tarteel
.venv/bin/python -m benchmark.runner --experiment two-stage
```

Expected: Should show results immediately. The two-stage approach with 50 candidates instead of 6,236 should be faster than pure CTC while maintaining similar accuracy.

**Step 2: Record baseline results**

Note the recall, precision, sequence accuracy, and latency. These are the baseline numbers before Modal training.

---

### Task 9: Launch all three Modal training jobs

Run all three training jobs. They can run sequentially (Modal free tier) or in parallel (paid).

**Step 1: Run CTC-base fine-tuning (shared by Exp 1 + 2)**

```bash
cd /Users/rock/ai/projects/offline-tarteel
modal run scripts/train_ctc_base_modal.py
```

Expected: ~40 min on A100. Downloads trained model to `data/ctc-base-finetuned/`.

**Step 2: Run knowledge distillation (Exp 2)**

```bash
modal run scripts/train_distill_modal.py
```

Expected: ~40 min on A100. Downloads distilled model to `data/ctc-base-distilled/`.
Note: This can reuse the fine-tuned student from Step 1 if it ran first.

**Step 3: Run contrastive v2 training (Exp 3)**

```bash
modal run scripts/train_contrastive_v2_modal.py
```

Expected: ~1 hour on A100. Downloads model + FAISS index to `data/contrastive-v2-model/`.

---

### Task 10: Run full benchmark comparison

After all Modal training completes and models are downloaded:

**Step 1: Verify all models are downloaded**

```bash
ls -la data/ctc-base-finetuned/
ls -la data/ctc-base-distilled/
ls -la data/contrastive-v2-model/
```

**Step 2: Run full benchmark**

```bash
.venv/bin/python -m benchmark.runner
```

This runs ALL experiments (existing + new) and produces a comparison table.

**Step 3: Commit results**

```bash
git add benchmark/results/latest.json
git commit -m "feat: benchmark results for three new experiments"
```

---

## Summary Table

| Task | What | Training? | Time |
|---|---|---|---|
| 1 | Register experiments in runner | No | 2 min |
| 2 | Two-stage run.py | No | 10 min |
| 3 | Distilled CTC run.py | No | 10 min |
| 4 | Contrastive v2 model + run.py | No | 10 min |
| 5 | Modal: CTC-base fine-tune | A100 ~40min | ~$2.50 |
| 6 | Modal: Knowledge distillation | A100 ~40min | ~$2.50 |
| 7 | Modal: Contrastive training | A100 ~1h | ~$3.50 |
| 8 | Two-stage baseline benchmark | No | 5 min |
| 9 | Launch all Modal jobs | GPU | ~$8.50 total |
| 10 | Final benchmark comparison | No | 15 min |

**Total estimated Modal cost: ~$8-10**

**Speed optimization:** All Modal jobs use A100 (80GB) instead of A10G (24GB) for ~2-3x speedup. Batch sizes increased (CTC: 16→32 with grad_accum=1, contrastive: 128) and training steps reduced to 3000 (from 5000) with early stopping. Wall-clock time drops from ~8h to ~2-3h total.

## Execution Dependencies

```
Task 1 → Task 2, 3, 4 (need directories)
Task 2 → Task 8 (run baseline)
Task 5 → Task 9.1 (CTC training)
Task 6 → Task 9.2 (distillation, benefits from 9.1 running first)
Task 7 → Task 9.3 (contrastive training)
Task 9 (all) → Task 10 (final benchmark)
```

Tasks 2, 3, 4 are independent and can run in parallel.
Tasks 5, 6, 7 are independent and can run in parallel on Modal.
