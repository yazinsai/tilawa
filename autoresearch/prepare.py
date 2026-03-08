"""Data preparation, model setup, export, and evaluation for FastConformer phoneme CTC.

Adapted from scripts/train_fastconformer_phoneme_modal.py and
scripts/export_phoneme_onnx_modal.py to run directly on a GPU machine (no Modal).

Usage:
    python autoresearch/prepare.py          # prepare data to /workspace/data
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHONEME_VOCAB = [
    # Vowels (12)
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    # Consonants (28)
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    # Geminated (28)
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    # Word boundary (1)
    "|",
]
NUM_CLASSES = len(PHONEME_VOCAB)  # 69
BLANK_ID = NUM_CLASSES  # 69 = blank; total outputs = 70

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
IQRA_TRAIN_DATASET = "IqraEval/Iqra_train"
IQRA_TTS_DATASET = "IqraEval/Iqra_TTS"
RETASY_DATASET_ID = "RetaSy/quranic_audio_dataset"
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
TLOG_DATASET_ID = "tarteel-ai/tlog"

DATA_DIR = "/workspace/data"
REPO_DIR = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback if package is absent.

    NeMo imports kaldialign in context-biasing paths even when that code path
    is not used for our training/inference setup.
    """
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"

        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _enable_ctc_only_training_step(model) -> None:
    """Patch NeMo hybrid model to optimize only CTC loss.

    NeMo's default hybrid training step always computes RNNT loss, which
    triggers numba CUDA kernels requiring libNVVM. For our pipeline we only
    consume CTC transcripts, so CTC-only adaptation is sufficient and avoids
    the RNNT/NVVM runtime path.
    """
    import torch

    def _ctc_training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        log_probs = self.ctc_decoder(encoder_output=encoded)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )
        loss_value = self.add_auxiliary_losses(ctc_loss)

        if hasattr(self, "_trainer") and self._trainer is not None:
            sample_id = self._trainer.global_step
            log_every_n_steps = max(int(self._trainer.log_every_n_steps), 1)
        else:
            sample_id = batch_nb
            log_every_n_steps = 1

        logs = {
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            "train_ctc_loss": ctc_loss,
            "train_loss": loss_value,
        }

        if (sample_id + 1) % log_every_n_steps == 0:
            self.ctc_wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            ctc_wer, _, _ = self.ctc_wer.compute()
            self.ctc_wer.reset()
            logs["training_batch_wer_ctc"] = ctc_wer

        self.log_dict(logs)
        return {"loss": loss_value}

    model.training_step = types.MethodType(_ctc_training_step, model)
    model.ctc_loss_weight = 1.0
    try:
        from omegaconf import open_dict

        with open_dict(model.cfg):
            model.cfg.ctc_loss_weight = 1.0
    except Exception:
        pass


def _enable_ctc_only_validation_step(model) -> None:
    """Patch NeMo hybrid model to use CTC-only validation."""
    def _ctc_validation_step(self, batch, batch_nb, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        log_probs = self.ctc_decoder(encoder_output=encoded)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )
        self.log("val_loss", ctc_loss, prog_bar=True, sync_dist=True)
        return {"val_loss": ctc_loss}

    model.validation_step = types.MethodType(_ctc_validation_step, model)


def _safe_text(text: str) -> str:
    """Normalize whitespace in a phoneme string."""
    text = (text or "").strip()
    return " ".join(text.split())


@dataclass
class BuildStats:
    written: int = 0
    skipped_empty: int = 0
    skipped_duration: int = 0
    bytes_written: int = 0


def _build_quran_phoneme_lookup(quran_data_dir: str | None = None) -> dict[str, str | None]:
    """Build normalized_text -> phoneme_string lookup from quran data.

    Returns dict where value is phoneme string if unique mapping,
    or None if multiple distinct phoneme strings map to the same normalized text.
    """
    if quran_data_dir is None:
        quran_data_dir = REPO_DIR

    # normalizer lives in shared/
    shared_dir = str(Path(REPO_DIR) / "shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    from normalizer import normalize_arabic

    data_dir = Path(quran_data_dir) / "data"
    quran = json.loads((data_dir / "quran.json").read_text(encoding="utf-8"))
    phonemes = json.loads((data_dir / "quran_phonemes.json").read_text(encoding="utf-8"))

    # Build (surah, ayah) -> phoneme_string
    phoneme_map: dict[tuple[int, int], str] = {}
    for entry in phonemes:
        key = (int(entry["surah"]), int(entry["ayah"]))
        phoneme_map[key] = entry["phonemes"]

    # Build normalized_text -> phoneme_string, marking ambiguous entries
    lookup: dict[str, str | None] = {}
    ambiguous_count = 0
    for verse in quran:
        key = (int(verse["surah"]), int(verse["ayah"]))
        ph = phoneme_map.get(key)
        if ph is None:
            continue
        norm = normalize_arabic(verse.get("text_clean", ""))
        if not norm:
            continue
        if norm in lookup:
            if lookup[norm] is not None and lookup[norm] != ph:
                lookup[norm] = None  # ambiguous
                ambiguous_count += 1
        else:
            lookup[norm] = ph

    print(f"Quran phoneme lookup: {len(lookup)} entries, {ambiguous_count} ambiguous")
    return lookup


def _build_verse_phoneme_map(quran_data_dir: str | None = None) -> dict[tuple[int, int], str]:
    """Build (surah, ayah) -> phoneme_string lookup for TLOG mapping."""
    if quran_data_dir is None:
        quran_data_dir = REPO_DIR

    data_dir = Path(quran_data_dir) / "data"
    phonemes = json.loads((data_dir / "quran_phonemes.json").read_text(encoding="utf-8"))
    phoneme_map: dict[tuple[int, int], str] = {}
    for entry in phonemes:
        phoneme_map[(int(entry["surah"]), int(entry["ayah"]))] = entry["phonemes"]
    print(f"Verse phoneme map: {len(phoneme_map)} entries")
    return phoneme_map


# ---------------------------------------------------------------------------
# _PhonemeTokenizer
# ---------------------------------------------------------------------------

class _PhonemeTokenizer:
    """Minimal tokenizer that maps space-separated phoneme strings to IDs."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self._token_to_id = {t: i for i, t in enumerate(vocab)}
        self._id_to_token = {i: t for i, t in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.pad_id = 0
        self.bos_id = None
        self.eos_id = None
        self.unk_id = None

    @property
    def text_to_ids(self):
        return self._text_to_ids

    def _text_to_ids(self, text: str) -> list[int]:
        tokens = text.strip().split()
        ids = []
        for t in tokens:
            if t in self._token_to_id:
                ids.append(self._token_to_id[t])
            # Skip unknown tokens silently
        return ids

    def ids_to_text(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i in self._id_to_token:
                tokens.append(self._id_to_token[i])
        return " ".join(tokens)

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token.get(i, "") for i in ids]

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id.get(t, 0) for t in tokens]

    def text_to_tokens(self, text: str) -> list[str]:
        return text.strip().split()

    def __len__(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# setup_model
# ---------------------------------------------------------------------------

def setup_model(freeze_encoder_layers: int = 10, freeze_preprocessor: bool = True):
    """Load base FastConformer, replace CTC head with phoneme vocab, freeze layers.

    Returns a configured NeMo model ready for CTC-only phoneme training.
    """
    import torch
    import torch.nn as nn

    _install_kaldialign_fallback()
    from nemo.collections.asr.losses.ctc import CTCLoss as NemoCTCLoss
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from omegaconf import open_dict

    print(f"Loading base model: {BASE_MODEL_ID}")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ------------------------------------------------------------------
    # Replace CTC decoder head with phoneme vocabulary
    # ------------------------------------------------------------------
    vocab_size = len(PHONEME_VOCAB) + 1  # +1 for CTC blank at last index

    print(f"Replacing CTC decoder: {vocab_size} classes (69 phonemes + blank)")

    if hasattr(model, "ctc_decoder"):
        old_decoder = model.ctc_decoder
        if hasattr(old_decoder, "decoder_layers") and len(old_decoder.decoder_layers) > 0:
            last_layer = old_decoder.decoder_layers[-1]
            if hasattr(last_layer, "in_channels"):
                in_features = last_layer.in_channels
            elif hasattr(last_layer, "in_features"):
                in_features = last_layer.in_features
            else:
                in_features = 512
            old_decoder.decoder_layers[-1] = nn.Conv1d(in_features, vocab_size, kernel_size=1)
            print(f"  Replaced decoder_layers[-1]: Conv1d({in_features}, {vocab_size})")
        elif hasattr(old_decoder, "fc"):
            in_features = old_decoder.fc.in_features
            old_decoder.fc = nn.Linear(in_features, vocab_size)
            print(f"  Replaced fc: Linear({in_features}, {vocab_size})")
        else:
            in_features = model.cfg.encoder.get("d_model", 512)
            model.ctc_decoder = nn.Linear(in_features, vocab_size)
            print(f"  Replaced entire ctc_decoder: Linear({in_features}, {vocab_size})")
    else:
        raise RuntimeError("Model has no ctc_decoder attribute. Cannot replace CTC head.")

    # ------------------------------------------------------------------
    # Replace CTC loss module with correct blank index
    # ------------------------------------------------------------------
    if hasattr(model, "ctc_loss"):
        new_blank = len(PHONEME_VOCAB)  # blank at last index
        model.ctc_loss = NemoCTCLoss(
            num_classes=new_blank, zero_infinity=True
        )
        print(f"  Replaced CTC loss module with blank={new_blank} (vocab_size={vocab_size})")

    # ------------------------------------------------------------------
    # Install phoneme tokenizer
    # ------------------------------------------------------------------
    phoneme_tokenizer = _PhonemeTokenizer(PHONEME_VOCAB)
    model.tokenizer = phoneme_tokenizer

    # Update ctc_wer decoding vocabulary
    try:
        if hasattr(model, "ctc_wer") and hasattr(model.ctc_wer, "decoding"):
            model.ctc_wer.decoding.labels_map = {i: p for i, p in enumerate(PHONEME_VOCAB)}
    except Exception:
        pass

    # Update config vocabulary references
    with open_dict(model.cfg):
        model.cfg.labels = list(PHONEME_VOCAB)

    # ------------------------------------------------------------------
    # Enable CTC-only training (skip RNNT loss)
    # ------------------------------------------------------------------
    _enable_ctc_only_training_step(model)
    _enable_ctc_only_validation_step(model)
    print("CTC-only finetune mode enabled (RNNT loss path disabled).")

    # ------------------------------------------------------------------
    # Freeze layers
    # ------------------------------------------------------------------
    if freeze_preprocessor and hasattr(model, "preprocessor"):
        for p in model.preprocessor.parameters():
            p.requires_grad = False
        print("Preprocessor frozen.")

    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        total_layers = len(model.encoder.layers)
        freeze_n = max(0, min(int(freeze_encoder_layers), total_layers))
        for idx, layer in enumerate(model.encoder.layers):
            req_grad = idx >= freeze_n
            for p in layer.parameters():
                p.requires_grad = req_grad
        print(f"Encoder freeze: first {freeze_n}/{total_layers} layers")
    else:
        print("Warning: encoder layers not found; skipping partial encoder freeze.")

    # Freeze RNNT decoder and joint (not used)
    if hasattr(model, "decoder"):
        for p in model.decoder.parameters():
            p.requires_grad = False
    if hasattr(model, "joint"):
        for p in model.joint.parameters():
            p.requires_grad = False
    print("RNNT decoder/joint frozen (not used for phoneme CTC).")

    # Keep CTC decoding active
    try:
        model.change_decoding_strategy(decoder_type="ctc", verbose=False)
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Params total={total_params:,} "
        f"trainable={trainable_params:,} "
        f"({(trainable_params / max(total_params, 1)) * 100:.2f}%)"
    )

    return model


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

def prepare_data(
    data_dir: str = DATA_DIR,
    tlog_per_verse: int = 5,
    retasy_ratio: float = 0.2,
    force_rebuild: bool = False,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
) -> dict:
    """Download and prepare all training data to local disk.

    Downloads Iqra train/val, TTS, RetaSy, TLOG from HuggingFace.
    Writes JSONL manifests to data_dir/manifests/ and audio to data_dir/audio/.
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    base = Path(data_dir)
    manifests_dir = base / "manifests"
    audio_root = base / "audio"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    metadata_path = manifests_dir / "data_metadata.json"

    if train_manifest.exists() and val_manifest.exists() and not force_rebuild:
        print("Manifests already exist, skipping rebuild.")
        return {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "metadata": str(metadata_path),
        }

    manifests_dir.mkdir(parents=True, exist_ok=True)
    (audio_root / "iqra_train").mkdir(parents=True, exist_ok=True)
    (audio_root / "iqra_val").mkdir(parents=True, exist_ok=True)
    (audio_root / "iqra_tts").mkdir(parents=True, exist_ok=True)

    def write_audio(audio_obj, out_path: Path) -> float:
        if isinstance(audio_obj, dict) and audio_obj.get("bytes") is not None:
            raw = audio_obj["bytes"]
            out_path.write_bytes(raw)
            try:
                with sf.SoundFile(io.BytesIO(raw)) as f:
                    return float(len(f) / float(f.samplerate))
            except Exception:
                return -1.0

        if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
            arr = audio_obj["array"]
            sr = int(audio_obj.get("sampling_rate", 16000))
            sf.write(out_path, arr, sr)
            return float(len(arr) / max(sr, 1))

        raise ValueError(f"Unsupported audio object format: {type(audio_obj)}")

    # ------------------------------------------------------------------
    # 1) IqraEval/Iqra_train  (train + dev/validation splits)
    # ------------------------------------------------------------------
    print(f"Loading {IQRA_TRAIN_DATASET} ...")

    def build_iqra_train_split(split_name: str, manifest_path: Path, subdir: str):
        stats = BuildStats()

        try:
            ds = load_dataset(IQRA_TRAIN_DATASET, split=split_name, streaming=True)
        except Exception:
            if split_name == "dev":
                ds = load_dataset(IQRA_TRAIN_DATASET, split="validation", streaming=True)
            elif split_name == "validation":
                ds = load_dataset(IQRA_TRAIN_DATASET, split="dev", streaming=True)
            else:
                raise
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        with manifest_path.open("w", encoding="utf-8") as mf:
            for idx, sample in enumerate(ds):
                text = _safe_text(
                    sample.get("phoneme_aug")
                    or sample.get("phoneme")
                    or ""
                )
                if not text:
                    stats.skipped_empty += 1
                    continue

                duration = float(sample.get("duration", -1.0))
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    stats.skipped_duration += 1
                    continue

                out_file = audio_root / subdir / f"{split_name}_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = duration if duration > 0 else measured
                if effective_duration <= 0:
                    effective_duration = measured

                if effective_duration <= 0:
                    stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                if effective_duration < min_duration or effective_duration > max_duration:
                    stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                stats.written += 1
                stats.bytes_written += out_file.stat().st_size
                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": text,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if stats.written % 5000 == 0:
                    print(
                        f"[{split_name}] written={stats.written:,} "
                        f"skipped_dur={stats.skipped_duration:,} "
                        f"skipped_empty={stats.skipped_empty:,}"
                    )

        return stats

    train_stats = build_iqra_train_split("train", train_manifest, "iqra_train")
    print(
        f"Iqra_train train: written={train_stats.written:,} "
        f"skipped_dur={train_stats.skipped_duration:,} "
        f"skipped_empty={train_stats.skipped_empty:,}"
    )

    val_stats = build_iqra_train_split("dev", val_manifest, "iqra_val")
    print(
        f"Iqra_train val: written={val_stats.written:,} "
        f"skipped_dur={val_stats.skipped_duration:,} "
        f"skipped_empty={val_stats.skipped_empty:,}"
    )

    # ------------------------------------------------------------------
    # 2) IqraEval/Iqra_TTS  (all samples appended to train manifest)
    # ------------------------------------------------------------------
    tts_stats = BuildStats()
    print(f"Loading {IQRA_TTS_DATASET} ...")

    try:
        tts_ds = load_dataset(IQRA_TTS_DATASET, split="train", streaming=True)
        tts_ds = tts_ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        with train_manifest.open("a", encoding="utf-8") as mf:
            for idx, sample in enumerate(tts_ds):
                text = _safe_text(
                    sample.get("phoneme_mis")
                    or sample.get("phoneme")
                    or ""
                )
                if not text:
                    tts_stats.skipped_empty += 1
                    continue

                duration = float(sample.get("duration", -1.0))
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    tts_stats.skipped_duration += 1
                    continue

                out_file = audio_root / "iqra_tts" / f"tts_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = duration if duration > 0 else measured
                if effective_duration <= 0:
                    effective_duration = measured

                if effective_duration <= 0:
                    tts_stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                if effective_duration < min_duration or effective_duration > max_duration:
                    tts_stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                tts_stats.written += 1
                tts_stats.bytes_written += out_file.stat().st_size
                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": text,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if tts_stats.written % 1000 == 0:
                    print(
                        f"[iqra_tts] written={tts_stats.written:,} "
                        f"skipped_dur={tts_stats.skipped_duration:,} "
                        f"skipped_empty={tts_stats.skipped_empty:,}"
                    )

    except Exception as exc:
        print(f"Warning: failed to load {IQRA_TTS_DATASET}: {exc}")
        print("Continuing without TTS data.")

    print(
        f"Iqra_TTS: written={tts_stats.written:,} "
        f"skipped_dur={tts_stats.skipped_duration:,} "
        f"skipped_empty={tts_stats.skipped_empty:,}"
    )

    # ------------------------------------------------------------------
    # 3) RetaSy noisy-domain blend (mapped to canonical verse phonemes)
    # ------------------------------------------------------------------
    shared_dir = str(Path(REPO_DIR) / "shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    from normalizer import normalize_arabic

    phoneme_lookup = _build_quran_phoneme_lookup()

    iqra_train_count = train_stats.written + tts_stats.written
    effective_retasy_cap = int(iqra_train_count * retasy_ratio)

    retasy_meta = {
        "total_seen": 0,
        "filtered_bad_label": 0,
        "filtered_duration": 0,
        "mapped": 0,
        "unmapped": 0,
        "ambiguous": 0,
        "skipped_empty": 0,
        "retasy_train_written": 0,
        "retasy_val_written": 0,
        "effective_cap": effective_retasy_cap,
    }

    (audio_root / "retasy").mkdir(parents=True, exist_ok=True)

    print(f"Loading {RETASY_DATASET_ID} (cap={effective_retasy_cap})...")
    try:
        retasy_ds = load_dataset(RETASY_DATASET_ID, split="train")
        retasy_ds = retasy_ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        retasy_written_total = 0
        with train_manifest.open("a", encoding="utf-8") as train_mf, \
             val_manifest.open("a", encoding="utf-8") as val_mf:
            for idx, sample in enumerate(retasy_ds):
                if retasy_written_total >= effective_retasy_cap:
                    break

                retasy_meta["total_seen"] += 1

                label = sample.get("final_label")
                if label in BAD_RETASY_LABELS:
                    retasy_meta["filtered_bad_label"] += 1
                    continue

                aya_text = (sample.get("Aya") or "").strip()
                if not aya_text:
                    retasy_meta["skipped_empty"] += 1
                    continue

                norm_text = normalize_arabic(aya_text)
                phoneme_str = phoneme_lookup.get(norm_text)
                if norm_text not in phoneme_lookup:
                    retasy_meta["unmapped"] += 1
                    continue
                if phoneme_str is None:
                    retasy_meta["ambiguous"] += 1
                    continue

                retasy_meta["mapped"] += 1

                duration = float(sample.get("duration_ms", 0)) / 1000.0
                if duration <= 0:
                    duration = -1.0
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    retasy_meta["filtered_duration"] += 1
                    continue

                out_file = audio_root / "retasy" / f"retasy_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = duration if duration > 0 else measured
                if effective_duration <= 0:
                    effective_duration = measured

                if effective_duration <= 0 or effective_duration < min_duration or effective_duration > max_duration:
                    retasy_meta["filtered_duration"] += 1
                    out_file.unlink(missing_ok=True)
                    continue

                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": _safe_text(phoneme_str),
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"

                # Deterministic split: idx % 50 == 0 -> validation (2%)
                if idx % 50 == 0:
                    val_mf.write(line)
                    retasy_meta["retasy_val_written"] += 1
                else:
                    train_mf.write(line)
                    retasy_meta["retasy_train_written"] += 1

                retasy_written_total += 1

                if retasy_written_total % 1000 == 0:
                    print(
                        f"[retasy] written={retasy_written_total:,} "
                        f"(train={retasy_meta['retasy_train_written']:,} "
                        f"val={retasy_meta['retasy_val_written']:,}) "
                        f"unmapped={retasy_meta['unmapped']:,} "
                        f"ambiguous={retasy_meta['ambiguous']:,}"
                    )

    except Exception as exc:
        print(f"Warning: failed to load {RETASY_DATASET_ID}: {exc}")
        print("Continuing without RetaSy data.")

    print(f"RetaSy: {json.dumps(retasy_meta, indent=2)}")

    # ------------------------------------------------------------------
    # 4) TLOG (Tarteel app recordings, mapped via surah:ayah filenames)
    # ------------------------------------------------------------------
    verse_phoneme_map = _build_verse_phoneme_map()

    effective_tlog_cap = int(iqra_train_count * 0.75)

    tlog_meta = {
        "total_seen": 0,
        "mapped": 0,
        "unmapped": 0,
        "filtered_duration": 0,
        "filtered_unclean": 0,
        "tlog_train_written": 0,
        "tlog_val_written": 0,
        "effective_cap": effective_tlog_cap,
        "verses_seen": 0,
    }

    (audio_root / "tlog").mkdir(parents=True, exist_ok=True)

    print(f"Loading {TLOG_DATASET_ID} clean split (cap={effective_tlog_cap})...")
    try:
        tlog_ds = load_dataset(TLOG_DATASET_ID, split="clean", streaming=True)
        tlog_ds = tlog_ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        tlog_written_total = 0
        verse_counts: dict[tuple[int, int], int] = {}
        MAX_PER_VERSE = tlog_per_verse

        with train_manifest.open("a", encoding="utf-8") as train_mf, \
             val_manifest.open("a", encoding="utf-8") as val_mf:
            for idx, sample in enumerate(tlog_ds):
                if tlog_written_total >= effective_tlog_cap:
                    break

                tlog_meta["total_seen"] += 1

                # Skip unclean samples
                if not sample.get("is_clean", True):
                    tlog_meta["filtered_unclean"] += 1
                    continue

                # Extract surah:ayah from audio filename (pattern: {surah}_{ayah}_{id}.flac)
                audio_obj = sample.get("audio", {})
                audio_path = audio_obj.get("path", "") if isinstance(audio_obj, dict) else ""
                parts = audio_path.split("_")
                if len(parts) < 3:
                    tlog_meta["unmapped"] += 1
                    continue

                try:
                    surah = int(parts[0])
                    ayah = int(parts[1])
                except (ValueError, IndexError):
                    tlog_meta["unmapped"] += 1
                    continue

                verse_key = (surah, ayah)
                phoneme_str = verse_phoneme_map.get(verse_key)
                if phoneme_str is None:
                    tlog_meta["unmapped"] += 1
                    continue

                tlog_meta["mapped"] += 1

                # Limit per-verse to maintain diversity
                current_count = verse_counts.get(verse_key, 0)
                if current_count >= MAX_PER_VERSE:
                    continue
                verse_counts[verse_key] = current_count + 1

                out_file = audio_root / "tlog" / f"tlog_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = measured

                if effective_duration <= 0 or effective_duration < min_duration or effective_duration > max_duration:
                    tlog_meta["filtered_duration"] += 1
                    out_file.unlink(missing_ok=True)
                    continue

                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": _safe_text(phoneme_str),
                }
                line = json.dumps(row, ensure_ascii=False) + "\n"

                # Deterministic split: idx % 50 == 0 -> validation (2%)
                if idx % 50 == 0:
                    val_mf.write(line)
                    tlog_meta["tlog_val_written"] += 1
                else:
                    train_mf.write(line)
                    tlog_meta["tlog_train_written"] += 1

                tlog_written_total += 1

                if tlog_written_total % 2000 == 0:
                    print(
                        f"[tlog] written={tlog_written_total:,} "
                        f"(train={tlog_meta['tlog_train_written']:,} "
                        f"val={tlog_meta['tlog_val_written']:,}) "
                        f"verses={len(verse_counts):,} "
                        f"unmapped={tlog_meta['unmapped']:,}"
                    )

        tlog_meta["verses_seen"] = len(verse_counts)

    except Exception as exc:
        print(f"Warning: failed to load {TLOG_DATASET_ID}: {exc}")
        import traceback
        traceback.print_exc()
        print("Continuing without TLOG data.")

    print(f"TLOG: {json.dumps(tlog_meta, indent=2)}")

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    metadata = {
        "iqra_train_dataset": IQRA_TRAIN_DATASET,
        "iqra_tts_dataset": IQRA_TTS_DATASET,
        "retasy_dataset": RETASY_DATASET_ID,
        "tlog_dataset": TLOG_DATASET_ID,
        "phoneme_vocab_size": len(PHONEME_VOCAB),
        "min_duration": min_duration,
        "max_duration": max_duration,
        "tlog_per_verse": tlog_per_verse,
        "retasy_ratio": retasy_ratio,
        "iqra_train": train_stats.__dict__,
        "iqra_val": val_stats.__dict__,
        "iqra_tts": tts_stats.__dict__,
        "retasy": retasy_meta,
        "tlog": tlog_meta,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Data preparation complete.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "metadata": str(metadata_path),
    }


# ---------------------------------------------------------------------------
# export_onnx_q8
# ---------------------------------------------------------------------------

def export_onnx_q8(checkpoint_path: str, output_dir: str) -> str:
    """Export a NeMo checkpoint to ONNX fp32, then quantize to uint8.

    Also writes phoneme_vocab.json. Returns path to the q8 ONNX file.
    """
    import os
    import tarfile

    import torch

    _install_kaldialign_fallback()
    import nemo.collections.asr as nemo_asr
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "fastconformer_phoneme.onnx"
    q8_path = out_dir / "fastconformer_phoneme_q8.onnx"
    vocab_path = out_dir / "phoneme_vocab.json"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Using checkpoint: {checkpoint_path}")

    # Load base model
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)

    # Move model to CPU for export
    model = model.cpu()

    # Replace CTC head (ctc_decoder, not decoder which is RNNT)
    vocab_size = NUM_CLASSES + 1  # 69 phonemes + blank = 70
    old_decoder = model.ctc_decoder
    if hasattr(old_decoder, "decoder_layers") and len(old_decoder.decoder_layers) > 0:
        last_layer = old_decoder.decoder_layers[-1]
        in_features = getattr(last_layer, "in_channels", getattr(last_layer, "in_features", 512))
        old_decoder.decoder_layers[-1] = torch.nn.Conv1d(in_features, vocab_size, kernel_size=1)
        print(f"Replaced CTC head: Conv1d({in_features}, {vocab_size})")
    else:
        in_features = model.cfg.encoder.get("d_model", 512)
        model.ctc_decoder = torch.nn.Linear(in_features, vocab_size)
        print(f"Replaced entire ctc_decoder: Linear({in_features}, {vocab_size})")

    # Load fine-tuned weights from .nemo checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with tarfile.open(checkpoint_path, "r:gz") as tar:
                tar.extractall(tmpdir)
        except tarfile.ReadError:
            with tarfile.open(checkpoint_path, "r:") as tar:
                tar.extractall(tmpdir)

        weights_path = Path(tmpdir) / "model_weights.ckpt"
        if not weights_path.exists():
            for p in Path(tmpdir).rglob("*.ckpt"):
                print(f"  Found weights: {p}")
                weights_path = p
                break
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys:    {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()

    # Export ONNX (CTC-only)
    model.set_export_config({"decoder_type": "ctc"})
    print(f"Exporting ONNX to: {onnx_path}")
    model.export(str(onnx_path))
    print(f"ONNX export: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Quantize to uint8
    print("Quantizing to uint8...")
    quantize_dynamic(str(onnx_path), str(q8_path), weight_type=QuantType.QUInt8)
    print(f"Quantized:   {q8_path.stat().st_size / 1e6:.1f} MB")

    # Save phoneme vocabulary
    vocab = {}
    for i, token in enumerate(PHONEME_VOCAB):
        vocab[str(i)] = token
    vocab[str(BLANK_ID)] = "<blank>"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary: {vocab_path} ({len(vocab)} tokens)")

    return str(q8_path)


# ---------------------------------------------------------------------------
# evaluate_accuracy
# ---------------------------------------------------------------------------

def evaluate_accuracy(checkpoint_path: str) -> tuple[int, int]:
    """Export checkpoint to ONNX q8, deploy to web frontend, run accuracy test.

    Returns (correct, total) tuple parsed from validate-streaming.ts output.
    """
    frontend_public = Path(REPO_DIR) / "web" / "frontend" / "public"
    frontend_dir = Path(REPO_DIR) / "web" / "frontend"

    with tempfile.TemporaryDirectory() as tmpdir:
        q8_path = export_onnx_q8(checkpoint_path, tmpdir)

        # Copy q8 ONNX and vocab to web frontend
        shutil.copy2(q8_path, frontend_public / "fastconformer_phoneme_q8.onnx")
        shutil.copy2(Path(tmpdir) / "phoneme_vocab.json", frontend_public / "phoneme_vocab.json")
        print(f"Copied q8 model and vocab to {frontend_public}")

    # Fix onnxruntime-node version conflict: @huggingface/transformers bundles
    # an older onnxruntime-node (1.21) that clashes with the top-level 1.24.
    # Removing the nested copy forces Node to use the top-level one.
    nested_ort = frontend_dir / "node_modules" / "@huggingface" / "transformers" / "node_modules" / "onnxruntime-node"
    if nested_ort.exists():
        shutil.rmtree(nested_ort)
        print(f"Removed conflicting nested onnxruntime-node: {nested_ort}")

    # Run the non-streaming accuracy test
    print("Running accuracy test (non-streaming)...")
    result = subprocess.run(
        ["npx", "tsx", "test/validate-streaming.ts", "--no-streaming"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        timeout=1800,  # 96 test cases × ~30s each
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"Test stderr:\n{stderr}")
        print(f"Test stdout:\n{stdout}")
        raise RuntimeError(f"Accuracy test failed with return code {result.returncode}")

    print(stdout)

    # Parse accuracy from final summary line, e.g. "44/53" or "Overall: 44/53"
    match = re.search(r"(\d+)/(\d+)", stdout.strip().split("\n")[-1])
    if not match:
        # Try searching all lines from the end
        for line in reversed(stdout.strip().split("\n")):
            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                break

    if not match:
        raise RuntimeError(f"Could not parse accuracy from test output:\n{stdout}")

    correct = int(match.group(1))
    total = int(match.group(2))
    print(f"Accuracy: {correct}/{total} ({correct / max(total, 1) * 100:.1f}%)")
    return (correct, total)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_data()
