"""
Train QuranCLAP v2 contrastive model on Modal A100-80GB.

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
    .apt_install("ffmpeg", "libsndfile1-dev")
    .pip_install(
        "torch",
        "transformers>=4.40,<5",
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
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    volumes={"/training": vol},
    memory=32768,
)
def train(
    batch_size: int = 32,
    phase1_epochs: int = 6,
    phase2_epochs: int = 12,
    phase1_lr: float = 2e-3,
    phase2_lr: float = 2e-4,
    embed_dim: int = 256,
    max_samples: int = 30000,
    max_audio_seconds: float = 15.0,
):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    GRAD_ACCUM = 4  # effective batch = batch_size * GRAD_ACCUM = 128

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

    import io
    import soundfile as sf

    max_audio_len = int(max_audio_seconds * 16000)
    samples = []
    n_errors = 0
    for sample in tqdm(ds, total=max_samples, desc="Loading"):
        if len(samples) >= max_samples:
            break
        try:
            audio_data = sample["audio"]
            # Streaming mode may return {"array": [...]} or {"bytes": b"..."}
            if isinstance(audio_data, dict) and "array" in audio_data:
                audio = np.array(audio_data["array"], dtype=np.float32)
            elif isinstance(audio_data, dict) and "bytes" in audio_data:
                # Manually decode WAV bytes
                wav_bytes = audio_data["bytes"]
                audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                raise ValueError(f"Unexpected audio format: {type(audio_data)}")

            if len(audio) > max_audio_len:
                audio = audio[:max_audio_len]
            samples.append({
                "audio": np.array(audio, dtype=np.float32),
                "text": sample["text"],
                "reciter": sample["reciter"],
            })
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  Error loading sample #{n_errors}: {type(e).__name__}: {e}")
            continue

    print(f"Loaded {len(samples)} samples ({n_errors} errors)")

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
                              shuffle=True, collate_fn=collate, num_workers=8, drop_last=True)
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

        if is_train:
            optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            texts = batch["text"]

            text_enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            ids = text_enc["input_ids"].to(device)
            tmask = text_enc["attention_mask"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if is_train:
                    la, lt = model(audio, ids, tmask, mask)
                    loss = clip_loss(la, lt) / GRAD_ACCUM
                else:
                    with torch.no_grad():
                        la, lt = model(audio, ids, tmask, mask)
                        loss = clip_loss(la, lt)

            if is_train:
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Undo the /GRAD_ACCUM scaling for reporting
            reported_loss = loss.item() * GRAD_ACCUM if is_train else loss.item()
            total_loss += reported_loss
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
        # EveryAyah has text field with the verse text
        verse_list.append({
            "text": sample["text"],
            "surah": 0,  # We'll derive from order
            "ayah": 0,
        })
        if len(verse_list) >= 6236:
            break

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
