# Contrastive Audio-Text Model (QuranCLAP)

A CLIP/CLAP-style contrastive learning experiment for offline Quran verse identification. Maps Quranic audio and Arabic verse text into a shared embedding space, enabling direct audio-to-verse matching without ASR.

## Approach

Instead of transcribing audio to text (ASR pipeline), this approach learns a joint embedding space where audio recordings and their corresponding verse texts are nearby. At inference time, the model encodes the query audio and finds the nearest verse text embedding via cosine similarity.

### Architecture

```
Audio (16kHz waveform) --> HuBERT-base --> mean pool --> projection MLP --> L2-normalize --> [256-d]
                                                                                               |
                                                                                          cosine sim
                                                                                               |
Arabic verse text      --> AraBERT      --> CLS token --> projection MLP --> L2-normalize --> [256-d]
```

- **Audio Encoder**: `facebook/hubert-base-ls960` (HuBERT-base, 95M params) -- a self-supervised speech model pretrained on 960h of English speech (LibriSpeech)
- **Text Encoder**: `aubmindlab/bert-base-arabertv02` (AraBERT v2, 135M params) -- a BERT model pretrained on Arabic text
- **Projection Heads**: 2-layer MLPs (768 -> 256 -> 256) with GELU activation, one for each modality
- **Loss**: Symmetric InfoNCE (CLIP-style contrastive loss) with learnable temperature
- **Total Parameters**: 230M (525K trainable with frozen encoders)

### Training Strategy

Two-phase training (only Phase 1 completed):

1. **Phase 1**: Freeze both encoders, train only projection heads (525K params)
2. **Phase 2** (not run): Unfreeze last 2 transformer layers of each encoder, fine-tune with differential learning rates

## Results

### Training (Phase 1 -- Projection Heads Only)

| Metric | Value |
|--------|-------|
| Data | Alafasy recitations, surahs 1-5 (789 verses) |
| Train/Val split | 552 / 118 samples |
| Batch size | 4 |
| Epochs | 5 |
| Training time | 28.6 minutes (Apple MPS) |
| Best val loss | 1.076 |
| Best val batch accuracy | 44.9% (random = 25%) |
| Trainable params | 525K / 230M total |

Training loss decreased steadily from 1.29 to 1.11 over 5 epochs. The within-batch accuracy of 44.9% (vs 25% random) shows the projection heads ARE learning to discriminate audio-text pairs within small batches.

### Evaluation -- Retrieval Accuracy

Retrieval task: given an audio clip, find the correct verse among N candidates.

| Model | Candidates | Top-1 | Top-5 | Top-10 |
|-------|-----------|-------|-------|--------|
| Zero-shot (no training) | 789 (surahs 1-5) | 0.1% | 0.5% | 0.8% |
| **Trained (Phase 1)** | **789 (surahs 1-5)** | **0.1%** | **0.6%** | **1.6%** |
| Zero-shot (no training) | 6,236 (all verses) | 0.0% | 0.0% | 0.0% |
| Trained (Phase 1) | 6,236 (all verses) | 0.0% | 0.3% | 0.5% |

### Analysis

**Why the model barely works despite learning within batches:**

1. **Domain mismatch**: HuBERT was pretrained on English speech. Arabic Quran recitation has very different phonetics, prosody, and melodic qualities (tajweed). The frozen encoder extracts features that don't capture Arabic phoneme distinctions well.

2. **Modality gap is too large**: HuBERT encodes acoustic features (pitch, rhythm, spectral content) while AraBERT encodes semantic/orthographic features. A 2-layer projection MLP cannot bridge this fundamental gap -- it can only learn a linear-ish mapping between two incompatible representation spaces.

3. **Small batch contrastive ≠ global retrieval**: With batch size 4, the model only needs to discriminate among 4 options (44.9% accuracy). But retrieval requires discriminating among 789-6,236 candidates. The projection heads learn rough clustering but lack the resolution for fine-grained verse-level discrimination.

4. **Single reciter, no augmentation**: Training used only Alafasy's recitations, providing no speaker diversity. The model overfits to his voice rather than learning verse-specific acoustic patterns.

5. **Insufficient training**: 5 epochs over 552 samples (2,760 total examples) is tiny for a contrastive model. CLIP trained on 400M image-text pairs. Even audio-specific CLAP models train on millions of audio clips.

## What Would Make This Work

For this approach to be viable, it would need:

### Minimum Requirements
- **Arabic speech encoder**: Replace HuBERT with a model pretrained on Arabic speech (e.g., Whisper's encoder, or a HuBERT model fine-tuned on Arabic/Quran data)
- **Multi-reciter data**: Train on 10+ reciters (available via Quran-MD dataset on HuggingFace, ~30 reciters)
- **Phase 2 fine-tuning**: Unfreeze encoder layers to adapt the representations
- **Larger batch sizes**: At least 32-64 for effective contrastive learning (requires GPU with 24GB+ VRAM)
- **More epochs**: 50-100 epochs minimum

### Ideal Setup
- **Pretrain on Arabic audio-text pairs** from non-Quran sources first (Arabic audiobooks, lectures)
- **Hard negative mining**: Sample negatives that are acoustically similar (same surah, adjacent verses) rather than random
- **Multiple loss functions**: Add triplet loss or proxy-NCA in addition to InfoNCE
- **Data augmentation**: Speed perturbation, pitch shifting, noise injection
- **GPU training**: A100 or similar for batch sizes of 256+ and 100+ epochs

### Estimated Resources
- Training time: ~4-8 hours on A100 GPU (vs 28 min on MPS for this tiny experiment)
- Training data: 6,236 verses x 10+ reciters = 62K+ audio-text pairs
- Model size: ~230M params (same architecture) or ~500M with larger encoders

## Comparison with ASR Approach

| Aspect | Contrastive (this) | ASR-based (Whisper) |
|--------|-------------------|---------------------|
| Accuracy (top-1) | ~0% (untrained) | 56-85% (depending on setup) |
| Approach | Direct audio-to-verse matching | Transcribe -> text match |
| Latency | Single forward pass (~100ms) | Transcription (~2-3s) + search |
| Model size | 230M (HuBERT + AraBERT + projections) | 39M-1.5B (Whisper variants) |
| Training data needed | Millions of paired samples | Already pretrained |
| Offline capable | Yes (after training) | Yes (with local model) |
| Handles poor audio | Potentially robust (learned features) | Poor (transcription degrades) |

**Verdict**: The ASR-based approach wins decisively for this use case. Whisper already understands Arabic speech, and combining it with text search gives high accuracy out of the box. The contrastive approach is an interesting research direction but would require significant investment (GPU compute, multi-reciter data, Arabic speech pretrained models) to become competitive.

## File Structure

```
contrastive/
  model.py          -- QuranCLAP architecture (HuBERT + AraBERT + projections + InfoNCE loss)
  dataset.py        -- Dataset loaders (EveryAyah local audio + quran.json, or HuggingFace streaming)
  train.py          -- Two-phase training script with MPS support
  evaluate.py       -- Evaluation: zero-shot baseline and trained model retrieval accuracy
  evaluation_results.json -- Full results
  checkpoints/      -- Saved model weights
  logs/             -- Training logs (JSON)
  README.md         -- This file
```

## Reproducing

```bash
# From the contrastive/ directory, with the project venv activated:

# Train (Phase 1 only, surahs 1-5, ~30 min on Apple MPS)
python train.py --source everyayah --batch-size 4 --phase1-epochs 5 --phase2-epochs 0 --max-surah 5

# Evaluate zero-shot vs trained
python evaluate.py --mode both --checkpoint checkpoints/best_phase1.pt

# Evaluate zero-shot only
python evaluate.py --mode zero_shot
```
