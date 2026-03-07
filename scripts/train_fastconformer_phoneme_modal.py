"""Fine-tune NVIDIA Arabic FastConformer with phoneme CTC head for robust Quran phoneme ASR.

Training target:
  nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0

Data sources:
  IqraEval/Iqra_train  (train/dev splits, uses phoneme_aug field)
  IqraEval/Iqra_TTS    (uses phoneme_mis field, appended to train)

The CTC vocabulary is replaced with a 70-token Arabic phoneme set (vowels,
consonants, geminated forms). The pipeline is intended for robust browser-side
Quran phoneme recognition, so the default training schedule unfreezes more of
the encoder and trains longer than the initial clean-transcription baseline.

Artifacts are written to Modal volume "fastconformer-phoneme-training":
  /training/<output_name>/
    manifests/train_manifest.jsonl
    manifests/val_manifest.jsonl
    checkpoints/...
    model/model.nemo
    model/training_metadata.json

Usage:
  modal run --detach scripts/train_fastconformer_phoneme_modal.py \\
    --output-name fastconformer-phoneme-v1

  modal run scripts/train_fastconformer_phoneme_modal.py \\
    --download-only \\
    --output-name fastconformer-phoneme-v1

  modal run scripts/train_fastconformer_phoneme_modal.py \\
    --prepare-only \\
    --output-name fastconformer-phoneme-v1
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App("fastconformer-phoneme-ctc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "datasets>=3.0,<4.0",
        "lightning>=2.4,<3.0",
        "omegaconf>=2.3,<3.0",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("fastconformer-phoneme-training", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
IQRA_TRAIN_DATASET = "IqraEval/Iqra_train"
IQRA_TTS_DATASET = "IqraEval/Iqra_TTS"
RETASY_DATASET_ID = "RetaSy/quranic_audio_dataset"
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
TLOG_DATASET_ID = "tarteel-ai/tlog"

# Image with quran data files for RetaSy phoneme mapping
_repo_root = Path(__file__).resolve().parent.parent
prepare_image = (
    image
    .add_local_file(str(_repo_root / "data" / "quran.json"), remote_path="/quran_data/quran.json")
    .add_local_file(str(_repo_root / "data" / "quran_phonemes.json"), remote_path="/quran_data/quran_phonemes.json")
    .add_local_file(str(_repo_root / "shared" / "normalizer.py"), remote_path="/quran_data/normalizer.py")
)

# ---------------------------------------------------------------------------
# Phoneme vocabulary (69 tokens, CTC blank added by model)
# ---------------------------------------------------------------------------
# CTC blank is the last index (69).  The blank token is handled
# internally by NeMo; this list defines the output label set.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback if package is absent.

    NeMo imports `kaldialign` in context-biasing paths even when that code path
    is not used for our training/inference setup.
    """
    import sys
    import types

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
    import torch

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


def _build_quran_phoneme_lookup(quran_data_dir: str = "/quran_data") -> dict[str, str | None]:
    """Build normalized_text → phoneme_string lookup from quran data.

    Returns dict where value is phoneme string if unique mapping,
    or None if multiple distinct phoneme strings map to the same normalized text.
    """
    import sys
    sys.path.insert(0, quran_data_dir)
    from normalizer import normalize_arabic

    data_dir = Path(quran_data_dir)
    quran = json.loads((data_dir / "quran.json").read_text(encoding="utf-8"))
    phonemes = json.loads((data_dir / "quran_phonemes.json").read_text(encoding="utf-8"))

    # Build (surah, ayah) → phoneme_string
    phoneme_map: dict[tuple[int, int], str] = {}
    for entry in phonemes:
        key = (int(entry["surah"]), int(entry["ayah"]))
        phoneme_map[key] = entry["phonemes"]

    # Build normalized_text → phoneme_string, marking ambiguous entries
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


def _build_verse_phoneme_map(quran_data_dir: str = "/quran_data") -> dict[tuple[int, int], str]:
    """Build (surah, ayah) → phoneme_string lookup for TLOG mapping."""
    data_dir = Path(quran_data_dir)
    phonemes = json.loads((data_dir / "quran_phonemes.json").read_text(encoding="utf-8"))
    phoneme_map: dict[tuple[int, int], str] = {}
    for entry in phonemes:
        phoneme_map[(int(entry["surah"]), int(entry["ayah"]))] = entry["phonemes"]
    print(f"Verse phoneme map: {len(phoneme_map)} entries")
    return phoneme_map


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

@app.function(
    image=prepare_image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 10,
    volumes={"/training": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def prepare_data(
    output_name: str = "fastconformer-phoneme-v2",
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    force_rebuild: bool = False,
    max_retasy_samples: int = 0,
    max_tlog_samples: int = 0,
    max_tlog_per_verse: int = 5,
):
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    base = Path(f"/training/{output_name}")
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
        """Process one split of Iqra_train into a JSONL manifest."""
        stats = BuildStats()

        # The dataset may use "dev" or "validation" for the held-out split
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
                # Use phoneme_aug (actual spoken phoneme sequence)
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

    # Try "dev" first, fall back to "validation"
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
                # Use phoneme_mis (mispronunciation phoneme sequence)
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
    import sys
    sys.path.insert(0, "/quran_data")
    from normalizer import normalize_arabic

    phoneme_lookup = _build_quran_phoneme_lookup("/quran_data")

    # Default RetaSy cap: 20% of clean train count
    iqra_train_count = train_stats.written + tts_stats.written
    effective_retasy_cap = max_retasy_samples if max_retasy_samples > 0 else int(iqra_train_count * 0.2)

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

                # Deterministic split: idx % 50 == 0 → validation (2%)
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
    verse_phoneme_map = _build_verse_phoneme_map("/quran_data")

    # Default TLOG cap: 30% of clean train count (larger than RetaSy since mapping is reliable)
    effective_tlog_cap = max_tlog_samples if max_tlog_samples > 0 else int(iqra_train_count * 0.75)

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
        # Track per-verse counts to limit oversampling (max 5 per verse)
        verse_counts: dict[tuple[int, int], int] = {}
        MAX_PER_VERSE = max_tlog_per_verse

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

                # Deterministic split: idx % 50 == 0 → validation (2%)
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
        "iqra_train": train_stats.__dict__,
        "iqra_val": val_stats.__dict__,
        "iqra_tts": tts_stats.__dict__,
        "retasy": retasy_meta,
        "tlog": tlog_meta,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    vol.commit()
    print("Data preparation complete.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "metadata": str(metadata_path),
    }


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    timeout=60 * 60 * 12,
    volumes={"/training": vol},
)
def train(
    output_name: str = "fastconformer-phoneme-v2",
    train_batch_size: int = 32,
    grad_accum: int = 2,
    max_steps: int = 8000,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    freeze_encoder_layers: int = 10,
    freeze_preprocessor: bool = True,
    val_check_interval: int = 250,
    num_workers: int = 8,
    early_stopping_patience: int = 6,
    enable_augmentation: bool = True,
):
    import lightning.pytorch as pl
    import torch
    import torch.nn as nn
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from omegaconf import open_dict

    _install_kaldialign_fallback()
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    base = Path(f"/training/{output_name}")
    manifests_dir = base / "manifests"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    metadata_path = manifests_dir / "data_metadata.json"
    checkpoints_dir = base / "checkpoints"
    output_dir = base / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Missing manifests. Run prepare_data first.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load base hybrid model
    # ------------------------------------------------------------------
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ------------------------------------------------------------------
    # Replace CTC decoder head with phoneme vocabulary
    # ------------------------------------------------------------------
    vocab_size = len(PHONEME_VOCAB) + 1  # +1 for CTC blank at last index

    print(f"Replacing CTC decoder: {vocab_size} classes (70 phonemes + blank)")

    if hasattr(model, "ctc_decoder"):
        old_decoder = model.ctc_decoder
        # NeMo ConvASRDecoder stores projection in decoder_layers
        if hasattr(old_decoder, "decoder_layers") and len(old_decoder.decoder_layers) > 0:
            last_layer = old_decoder.decoder_layers[-1]
            if hasattr(last_layer, "in_channels"):
                in_features = last_layer.in_channels
            elif hasattr(last_layer, "in_features"):
                in_features = last_layer.in_features
            else:
                # FastConformer large uses d_model=512
                in_features = 512
            old_decoder.decoder_layers[-1] = nn.Conv1d(in_features, vocab_size, kernel_size=1)
            print(f"  Replaced decoder_layers[-1]: Conv1d({in_features}, {vocab_size})")
        elif hasattr(old_decoder, "fc"):
            in_features = old_decoder.fc.in_features
            old_decoder.fc = nn.Linear(in_features, vocab_size)
            print(f"  Replaced fc: Linear({in_features}, {vocab_size})")
        else:
            # Fallback: replace the entire ctc_decoder with a simple linear projection
            # Determine input features from encoder config
            in_features = model.cfg.encoder.get("d_model", 512)
            model.ctc_decoder = nn.Linear(in_features, vocab_size)
            print(f"  Replaced entire ctc_decoder: Linear({in_features}, {vocab_size})")
    else:
        raise RuntimeError("Model has no ctc_decoder attribute. Cannot replace CTC head.")

    # ------------------------------------------------------------------
    # Replace CTC loss module with correct blank index
    # ------------------------------------------------------------------
    # NeMo's CTCLoss inherits from torch.nn.CTCLoss and was initialized
    # with blank=1024 (original BPE vocab size). We must replace it entirely
    # since the blank index is baked into the torch module's constructor.
    if hasattr(model, "ctc_loss"):
        from nemo.collections.asr.losses.ctc import CTCLoss as NemoCTCLoss
        new_blank = len(PHONEME_VOCAB)  # blank at last index
        model.ctc_loss = NemoCTCLoss(
            num_classes=new_blank, zero_infinity=True
        )
        print(f"  Replaced CTC loss module with blank={new_blank} (vocab_size={vocab_size})")

    # ------------------------------------------------------------------
    # Build a character-level tokenizer mapping for the phoneme vocab.
    # NeMo's data pipeline calls model.tokenizer to encode text labels.
    # For char-level CTC with a hybrid model, we install a minimal
    # tokenizer that maps each phoneme token to an integer ID.
    # ------------------------------------------------------------------
    phoneme_to_id = {p: i for i, p in enumerate(PHONEME_VOCAB)}

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

    phoneme_tokenizer = _PhonemeTokenizer(PHONEME_VOCAB)

    # Install the phoneme tokenizer on the model
    model.tokenizer = phoneme_tokenizer

    # Also update the ctc_wer decoding vocabulary so WER logging works
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

    # ------------------------------------------------------------------
    # Configure data loaders
    # ------------------------------------------------------------------
    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.is_tarred = False
        model.cfg.train_ds.tarred_audio_filepaths = None
        model.cfg.train_ds.batch_size = int(train_batch_size)
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.num_workers = int(num_workers)
        model.cfg.train_ds.pin_memory = True
        model.cfg.train_ds.max_duration = 30.0
        model.cfg.train_ds.min_duration = 0.3
        model.cfg.train_ds.shuffle_n = 2048
        model.cfg.train_ds.use_start_end_token = False

        # Augmentation (train only)
        if enable_augmentation:
            model.cfg.train_ds.augmentor = {
                "speed": {"prob": 0.3, "sr": 16000, "resample_type": "kaiser_fast",
                          "min_speed_rate": 0.9, "max_speed_rate": 1.1, "num_rates": 5},
                "gain": {"prob": 0.3, "min_gain_dbfs": -10, "max_gain_dbfs": 5},
                "white_noise": {"prob": 0.3, "min_level": -80, "max_level": -50},
                "shift": {"prob": 0.2, "min_shift_ms": -200.0, "max_shift_ms": 200.0},
                "silence": {"prob": 0.2, "min_start_silence_secs": 0.0, "max_start_silence_secs": 0.4,
                             "min_end_silence_secs": 0.0, "max_end_silence_secs": 0.5},
            }

        model.cfg.validation_ds.manifest_filepath = str(val_manifest)
        model.cfg.validation_ds.batch_size = int(train_batch_size)
        model.cfg.validation_ds.shuffle = False
        model.cfg.validation_ds.num_workers = int(num_workers)
        model.cfg.validation_ds.pin_memory = True
        model.cfg.validation_ds.max_duration = 30.0
        model.cfg.validation_ds.min_duration = 0.3
        model.cfg.validation_ds.use_start_end_token = False

        model.cfg.test_ds.manifest_filepath = str(val_manifest)
        model.cfg.test_ds.batch_size = int(train_batch_size)
        model.cfg.test_ds.shuffle = False
        model.cfg.test_ds.num_workers = int(num_workers)
        model.cfg.test_ds.pin_memory = True
        model.cfg.test_ds.use_start_end_token = False

        model.cfg.tokenizer.dir = "/tmp"
        model.cfg.optim.lr = float(learning_rate)
        if "sched" in model.cfg.optim and "warmup_steps" in model.cfg.optim.sched:
            model.cfg.optim.sched.warmup_steps = int(warmup_steps)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="phoneme-{step:06d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        every_n_train_steps=int(val_check_interval),
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=int(max_steps),
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(checkpoints_dir),
        gradient_clip_val=1.0,
        accumulate_grad_batches=int(grad_accum),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        val_check_interval=int(val_check_interval),
        limit_val_batches=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=int(early_stopping_patience),
                mode="min",
                verbose=True,
            ),
            checkpoint_callback,
        ],
    )

    model.set_trainer(trainer)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    print("\n" + "=" * 72)
    print("FastConformer Phoneme CTC fine-tune")
    print(f"output_name:        {output_name}")
    print(f"train_manifest:     {train_manifest}")
    print(f"val_manifest:       {val_manifest}")
    print(f"phoneme_vocab_size: {len(PHONEME_VOCAB)}")
    print(f"ctc_classes:        {vocab_size} (incl. blank)")
    print(f"batch/train:        {train_batch_size}")
    print(f"grad_accum:         {grad_accum}")
    print(f"max_steps:          {max_steps}")
    print(f"learning_rate:      {learning_rate}")
    print(f"warmup_steps:       {warmup_steps}")
    print(f"freeze_layers:      {freeze_encoder_layers}")
    print(f"freeze_preprocessor:{freeze_preprocessor}")
    print("=" * 72 + "\n")

    trainer.fit(model)

    # Log best checkpoint
    best_ckpt = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score
    print(f"Best checkpoint: {best_ckpt} (val_loss={best_score})")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    nemo_path = output_dir / "model.nemo"
    model.save_to(str(nemo_path))

    train_meta = {
        "base_model": BASE_MODEL_ID,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "phoneme_vocab": PHONEME_VOCAB,
        "phoneme_vocab_size": len(PHONEME_VOCAB),
        "ctc_classes": vocab_size,
        "train_batch_size": train_batch_size,
        "grad_accum": grad_accum,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "freeze_encoder_layers": freeze_encoder_layers,
        "freeze_preprocessor": freeze_preprocessor,
        "early_stopping_patience": early_stopping_patience,
        "enable_augmentation": enable_augmentation,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "data_metadata_path": str(metadata_path),
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "best_val_loss": float(best_score) if best_score is not None else None,
        "total_checkpoints_saved": len(checkpoint_callback.best_k_models),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(train_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vol.commit()
    print(f"Saved fine-tuned model to {nemo_path}")
    return {"model_path": str(nemo_path), "metadata_path": str(output_dir / "training_metadata.json")}


# ---------------------------------------------------------------------------
# TLOG quality filter (inference-based)
# ---------------------------------------------------------------------------

def _levenshtein_ratio_tokens(a: list[str], b: list[str]) -> float:
    """Token-level levenshtein similarity ratio."""
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    return 1.0 - prev[la] / max(la, lb)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4,
    timeout=60 * 60 * 6,
    volumes={"/training": vol},
)
def filter_tlog_quality(
    output_name: str,
    model_source: str = "fastconformer-phoneme-v3-retasy-aug-u8",
    min_ratio: float = 0.3,
    batch_size: int = 16,
):
    """Run CTC inference on TLOG samples and filter by phoneme match quality.

    Uses an existing trained checkpoint to score each TLOG sample against its
    expected phoneme string.  Samples below min_ratio are removed from the
    training manifest.  A quality report is saved for Tarteel review.
    """
    import os
    import shutil
    import tarfile
    import tempfile

    import soundfile as sf
    import torch

    _install_kaldialign_fallback()
    import nemo.collections.asr as nemo_asr

    vol.reload()

    # ------------------------------------------------------------------
    # Load model (same pattern as export script)
    # ------------------------------------------------------------------
    checkpoint_path = f"/training/{model_source}/model/model.nemo"
    if not os.path.exists(checkpoint_path):
        # Try to find any available checkpoint on the volume
        vol_root = Path("/training")
        available = sorted(vol_root.rglob("model.nemo")) if vol_root.exists() else []
        if available:
            checkpoint_path = str(available[0])
            model_source = checkpoint_path.split("/training/")[1].split("/model/")[0]
            print(f"Requested model not found, using: {checkpoint_path}")
        else:
            print(
                f"WARNING: No trained model checkpoint found on volume. "
                f"Skipping TLOG quality filter. Train will use unfiltered data."
            )
            return {"skipped": True, "reason": "no_checkpoint"}

    print(f"Loading base model: {BASE_MODEL_ID}")
    model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL_ID)
    model = model.cuda()

    # Replace CTC head with phoneme vocab
    vocab_size = len(PHONEME_VOCAB) + 1
    old_decoder = model.ctc_decoder
    if hasattr(old_decoder, "decoder_layers") and len(old_decoder.decoder_layers) > 0:
        last_layer = old_decoder.decoder_layers[-1]
        in_features = getattr(last_layer, "in_channels", getattr(last_layer, "in_features", 512))
        old_decoder.decoder_layers[-1] = torch.nn.Conv1d(in_features, vocab_size, kernel_size=1).cuda()
    else:
        in_features = model.cfg.encoder.get("d_model", 512)
        model.ctc_decoder = torch.nn.Linear(in_features, vocab_size).cuda()

    # Load trained weights
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
                weights_path = p
                break

        state_dict = torch.load(weights_path, map_location="cuda")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model.eval()
    blank_id = len(PHONEME_VOCAB)  # 69

    # ------------------------------------------------------------------
    # Read manifest and separate TLOG vs non-TLOG entries
    # ------------------------------------------------------------------
    base = Path(f"/training/{output_name}")
    manifest_path = base / "manifests" / "train_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    tlog_entries: list[dict] = []
    other_entries: list[str] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "/tlog/" in entry.get("audio_filepath", ""):
                tlog_entries.append(entry)
            else:
                other_entries.append(line)

    print(f"Manifest: {len(other_entries)} non-TLOG, {len(tlog_entries)} TLOG samples")

    if not tlog_entries:
        print("No TLOG samples found in manifest. Nothing to filter.")
        return {}

    # ------------------------------------------------------------------
    # Run inference in batches
    # ------------------------------------------------------------------
    results: list[dict] = []
    kept_lines: list[str] = []
    ratio_buckets = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    stats = {"total": len(tlog_entries), "good": 0, "uncertain": 0, "bad": 0}

    for batch_start in range(0, len(tlog_entries), batch_size):
        batch_entries = tlog_entries[batch_start : batch_start + batch_size]
        signals = []
        lengths = []

        for entry in batch_entries:
            audio_data, sr = sf.read(entry["audio_filepath"], dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            # Resample to 16kHz if needed (TLOG files may be at original rate)
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            t = torch.tensor(audio_data, dtype=torch.float32)
            signals.append(t)
            lengths.append(t.shape[0])

        # Pad to max length in batch
        max_len = max(lengths)
        padded = torch.zeros(len(signals), max_len)
        for i, sig in enumerate(signals):
            padded[i, : sig.shape[0]] = sig
        signal_lens = torch.tensor(lengths, dtype=torch.int64)

        with torch.no_grad():
            padded = padded.cuda()
            signal_lens = signal_lens.cuda()
            encoded, encoded_len = model.forward(
                input_signal=padded, input_signal_length=signal_lens,
            )
            log_probs = model.ctc_decoder(encoder_output=encoded)
            # log_probs: [B, T, V] or [B, V, T] depending on decoder type
            if log_probs.shape[-1] != vocab_size:
                log_probs = log_probs.transpose(1, 2)  # [B, T, V]

        for i, entry in enumerate(batch_entries):
            # CTC greedy decode
            probs_i = log_probs[i, : encoded_len[i].item()]  # [T, V]
            ids = probs_i.argmax(dim=-1).cpu().tolist()  # [T]
            decoded_ids = []
            prev = -1
            for t in ids:
                if t != prev and t != blank_id:
                    decoded_ids.append(t)
                prev = t
            decoded_tokens = [PHONEME_VOCAB[idx] for idx in decoded_ids if idx < len(PHONEME_VOCAB)]
            expected_tokens = entry["text"].split()

            ratio = _levenshtein_ratio_tokens(decoded_tokens, expected_tokens)

            # Classify
            bucket_idx = min(int(ratio * 10), 9)
            bucket_key = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"
            ratio_buckets[bucket_key] += 1

            if ratio >= 0.5:
                status = "good"
                stats["good"] += 1
                kept_lines.append(json.dumps(entry, ensure_ascii=False))
            elif ratio >= min_ratio:
                status = "uncertain"
                stats["uncertain"] += 1
                kept_lines.append(json.dumps(entry, ensure_ascii=False))
            else:
                status = "bad"
                stats["bad"] += 1
                # Not added to kept_lines

            results.append({
                "filepath": entry["audio_filepath"],
                "ratio": round(ratio, 4),
                "status": status,
                "decoded": " ".join(decoded_tokens[:20]),  # truncate for report
                "expected": " ".join(expected_tokens[:20]),
            })

        processed = batch_start + len(batch_entries)
        if processed % (batch_size * 10) == 0 or processed == len(tlog_entries):
            print(
                f"[filter] {processed}/{len(tlog_entries)} "
                f"good={stats['good']} uncertain={stats['uncertain']} bad={stats['bad']}"
            )

    # ------------------------------------------------------------------
    # Write filtered manifest (backup original first)
    # ------------------------------------------------------------------
    backup_path = base / "manifests" / "train_manifest_unfiltered.jsonl"
    shutil.copy2(manifest_path, backup_path)
    print(f"Backed up original manifest to {backup_path}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for line in other_entries:
            f.write(line + "\n")
        for line in kept_lines:
            f.write(line + "\n")

    print(f"Filtered manifest: {len(other_entries) + len(kept_lines)} samples "
          f"(removed {stats['bad']} bad TLOG samples)")

    # ------------------------------------------------------------------
    # Write quality report
    # ------------------------------------------------------------------
    report = {
        "model_source": model_source,
        "min_ratio_threshold": min_ratio,
        "summary": stats,
        "ratio_distribution": ratio_buckets,
        "bad_samples": [r for r in results if r["status"] == "bad"],
        "uncertain_samples": [r for r in results if r["status"] == "uncertain"],
    }
    report_path = base / "manifests" / "tlog_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Quality report: {report_path}")

    vol.commit()
    return stats


# ---------------------------------------------------------------------------
# download_model
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model(output_name: str = "fastconformer-phoneme-v1"):
    """Download model artifacts from Modal volume."""
    model_dir = Path(f"/training/{output_name}/model")
    if not model_dir.exists():
        print(f"No model directory found at: {model_dir}")
        return {}

    files: dict[str, bytes] = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            data = f.read_bytes()
            files[rel] = data
            print(f"  {rel}: {len(data):,} bytes")
    return files


# ---------------------------------------------------------------------------
# main (local entrypoint)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    output_name: str = "fastconformer-phoneme-v2",
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    force_rebuild_data: bool = False,
    train_batch_size: int = 32,
    grad_accum: int = 2,
    max_steps: int = 8000,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
    freeze_encoder_layers: int = 10,
    freeze_preprocessor: bool = True,
    val_check_interval: int = 250,
    num_workers: int = 8,
    early_stopping_patience: int = 6,
    enable_augmentation: bool = True,
    max_retasy_samples: int = 0,
    max_tlog_samples: int = 0,
    max_tlog_per_verse: int = 5,
    download_only: bool = False,
    download_after_train: bool = False,
    prepare_only: bool = False,
    train_only: bool = False,
    filter_tlog: bool = False,
    filter_model_source: str = "fastconformer-phoneme-v3-retasy-aug-u8",
    filter_min_ratio: float = 0.3,
):
    local_out_dir = Path("data") / output_name

    if download_only:
        print(f"Downloading model artifacts for {output_name}...")
        local_out_dir.mkdir(parents=True, exist_ok=True)
        files = download_model.remote(output_name=output_name)
        if not files:
            print("No files to download.")
            return

        total = 0
        for rel, data in files.items():
            out_path = local_out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            total += len(data)
            print(f"  saved {rel} ({len(data):,} bytes)")

        print(f"\nDownloaded to {local_out_dir} ({total / 1e6:.1f} MB)")
        return

    if not train_only:
        print("Preparing manifests/audio on Modal volume...")
        prepare_data.remote(
            output_name=output_name,
            min_duration=min_duration,
            max_duration=max_duration,
            force_rebuild=force_rebuild_data,
            max_retasy_samples=max_retasy_samples,
            max_tlog_samples=max_tlog_samples,
            max_tlog_per_verse=max_tlog_per_verse,
        )

    if filter_tlog:
        print("\nFiltering TLOG samples by inference quality...")
        filter_stats = filter_tlog_quality.remote(
            output_name=output_name,
            model_source=filter_model_source,
            min_ratio=filter_min_ratio,
        )
        print(f"Filter results: {filter_stats}")

    if prepare_only:
        print("\nData preparation complete. Skipping training (--prepare-only).")
        return

    print("\nStarting GPU fine-tuning job...")
    train.remote(
        output_name=output_name,
        train_batch_size=train_batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        freeze_encoder_layers=freeze_encoder_layers,
        freeze_preprocessor=freeze_preprocessor,
        val_check_interval=val_check_interval,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        enable_augmentation=enable_augmentation,
    )

    if not download_after_train:
        print(
            "\nTraining job submitted. Skipping local download "
            "(set --download-after-train to fetch artifacts automatically)."
        )
        return

    print(f"\nDownloading model artifacts for {output_name}...")
    local_out_dir.mkdir(parents=True, exist_ok=True)
    files = download_model.remote(output_name=output_name)
    if not files:
        print("No files to download.")
        return

    total = 0
    for rel, data in files.items():
        out_path = local_out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        total += len(data)
        print(f"  saved {rel} ({len(data):,} bytes)")

    print(f"\nDownloaded to {local_out_dir} ({total / 1e6:.1f} MB)")
