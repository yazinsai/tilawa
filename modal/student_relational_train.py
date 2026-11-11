"""Matched relational/hint encoder fine-tunes, called directly on the train image.

PYTHONPATH=/app/tools:$PYTHONPATH python modal/student_relational_train.py --recipe /scratch/.../recipe.json \
    --mode relational --out /student/runs/relational

The recipe is produced only after auditing supplied training audio and speakers.
It fixes both arms' data, augmentation, updates and selection; no practice data
is consumed here. ckpt.pt is validation-selected, for the existing export wrapper.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from student_relational import batch_objective


def validate_recipe(recipe):
    """Require an auditable training-only plan; this cannot prove its provenance."""
    if recipe.get("schema") != 1 or recipe.get("selection") != "minimum-validation-ctc":
        raise ValueError("Require training-only CTC checkpoint selection")
    if (recipe.get("steps", 0) <= 0 or recipe.get("validate_every", 0) <= 0
            or not math.isfinite(recipe.get("temperature", 0)) or recipe["temperature"] <= 0
            or not math.isfinite(recipe.get("learning_rate", 0)) or recipe["learning_rate"] <= 0
            or recipe.get("real_pause_examples", 0) <= 0):
        raise ValueError("Freeze a finite matched schedule and retain audited real pauses")
    parts = [set(recipe.get(k, [])) for k in ("training_speakers", "validation_speakers", "excluded_speakers")]
    if any(not p for p in parts) or any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("Require disjoint training-speaker partitions and release exclusions")
    items = recipe.get("items", [])
    if not items or len({r['key'] for r in items}) != len(items):
        raise ValueError("Require unique source item identities")
    parents = {}
    for item in items:
        recording = item.get("recording")
        if not isinstance(recording, str) or not recording:
            raise ValueError("Require audited parent recording identity, distinct from clip key")
        identity = (item["speaker"], item["source"])
        if recording in parents and parents[recording] != identity:
            raise ValueError("A parent recording cannot change speaker or source")
        parents[recording] = identity
        partition = item["partition"]
        if partition not in ("training", "validation") or item["speaker"] not in parts[partition == "validation"]:
            raise ValueError("Item speaker differs from its frozen training partition")
        if item.get("original_split") != "train" or item.get("source") not in ("authorized-quran", "help-none"):
            raise ValueError("Only authorized transcripts and original training-speaker none takes")
        for name in ("audio_sha256",):
            value = item.get(name, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Require source audio fingerprint")
    for partition in ("training", "validation"):
        sources = {r["source"] for r in items if r["partition"] == partition}
        if sources != {"authorized-quran", "help-none"}:
            raise ValueError("Both partitions need ordinary Quran and none-labelled phone recordings")
    for name in ("student_sha256", "teacher_sha256"):
        value = recipe.get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Require initial model fingerprints")
    weights = recipe.get("weights", {})
    if (set(weights) != {"ctc", "kd", "preservation", "distillation"}
            or any(not math.isfinite(w) or w <= 0 for w in weights.values())):
        raise ValueError("Require positive matched acoustic and preservation weights")
    train_items = [r for r in items if r["partition"] == "training"]
    plans = recipe.get("training_batches", [])
    if (not plans or any(not b or len(set(b)) != len(b) or min(b) < 0
                         or max(b) >= len(train_items) for b in plans)
            or set(i for b in plans for i in b) != set(range(len(train_items)))):
        raise ValueError("Batch plan must cover every frozen training recording")
    return items


def validate_help_row(item, labels, reference):
    row = labels[item["help_id"]]
    if (row.get("split") != "train" or row.get("kind") != "none" or row.get("scored") is not True
            or row.get("events") != [] or row.get("non_errors") != [] or row.get("unsupported") != []
            or row.get("excluded") or row.get("speaker") != item["speaker"]):
        raise ValueError("Help item is not an eligible none-labelled training take")
    transcript = "".join(reference[a] for a in row["ayahs"])
    if item["transcript"] != transcript:
        raise ValueError("Help transcript differs from its none-labelled reference")
    return row["wav"]


def model_fingerprint(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TeacherFeatures:
    """Singleton fresh teacher inference, on precisely the augmented item PCM.

    The audited phoneme projection's input hook provides its final acoustic
    features. We never approximate hidden features using output probabilities.
    """
    def __init__(self, recipe, device):
        import torch
        from quran_muaalem import Muaalem
        from student import TOKENS

        self.teacher = Muaalem("obadx/muaalem-model-v3_2", device=str(device), dtype=torch.float32)
        self.teacher.model.eval().float().requires_grad_(False)
        alphabet = self.teacher.multi_level_tokenizer.id_to_vocab["phonemes"]
        if [alphabet[i] for i in range(1, len(TOKENS) + 1)] != list(TOKENS):
            raise ValueError("Teacher alphabet differs from shipped primary")
        if model_fingerprint(self.teacher.model) != recipe["teacher_sha256"]:
            raise ValueError("Teacher differs from audited state")
        self.device = device
        self.feature_module = self.teacher.model.get_submodule(recipe["teacher_feature_module"])

    def __call__(self, key, wave):
        import torch
        captured = []
        def capture(module, inputs):
            captured.append(inputs[0].detach().clone())
        handle = self.feature_module.register_forward_pre_hook(capture)
        try:
            inputs = self.teacher.processor([wave.cpu().numpy()], sampling_rate=16000,
                                            return_tensors="pt", padding=False)
            if not bool(inputs["attention_mask"].all()):
                raise ValueError("Singleton teacher item unexpectedly contains batch padding")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.teacher.model(**inputs, return_dict=False)[0]["phonemes"]
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != outputs.shape[:2]:
            raise ValueError("Audited teacher feature tap differs from phoneme head clock")
        return {"key": key, "features": captured[0][0].float(), "log_probs": outputs[0].float().log_softmax(-1)}


class TrainingItems:
    def __init__(self, recipe, partition):
        import numpy as np
        validate_recipe(recipe)
        self.recipe = recipe
        self.items = [r for r in recipe["items"] if r["partition"] == partition]
        self.training = partition == "training"
        self.aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
        self.waves = []
        help_root = Path(recipe["help_root"]).resolve()
        if not help_root.is_relative_to(Path("/lab/data/help-train")):
            raise ValueError("Help input must come from supplied training corpus")
        labels = {r["id"]: r for r in (json.loads(s) for s in (help_root / "labels.jsonl").read_text().splitlines())}
        reference = {f"{a['s']}:{a['a']}": a["p"] for a in json.loads(Path(recipe["reference_index"]).read_text())["ayahs"]}
        for item in self.items:
            if item["source"] == "help-none":
                path = (help_root / validate_help_row(item, labels, reference)).resolve()
                if not path.is_relative_to(help_root):
                    raise ValueError("Help audio leaves training corpus")
                data = path.read_bytes()
            else:
                # Explicit authorized recording input, never /lab practice audio.
                path = Path(item["audio"]).resolve()
                if not path.is_relative_to(Path("/student")) or path.is_relative_to(Path("/student/eval")):
                    raise ValueError("Quran audio must be from audited training input")
                data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item["audio_sha256"]:
                raise ValueError("Audio differs from frozen source fingerprint")
            import soundfile as sf
            wave, sr = sf.read(io.BytesIO(data), dtype="float32")
            if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
                raise ValueError("Expected finite, mono model-rate training PCM")
            self.waves.append(wave)

    def batch(self, indices, step):
        import numpy as np
        import torch
        import torchaudio
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        feats, waves, targets, keys, recordings = [], [], [], [], []
        for index in indices:
            item = self.items[index]
            wave = self.waves[index].copy()
            # Private, per-recording RNG gives matched arms identical audio and
            # masks independent of projection initialization or training RNG.
            seed = int.from_bytes(hashlib.sha256(f"{self.recipe['seed']}:{step}:{item['key']}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            if self.training:
                cfg = self.recipe["augmentation"]
                if rng.random() < cfg["rir_p"]:
                    wave = reverb(wave, pick(self.aug["rir"], rng))
                if rng.random() < cfg["noise_p"]:
                    wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
                wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
            wave = torch.from_numpy(np.ascontiguousarray(wave))
            features = torchaudio.compliance.kaldi.fbank(wave[None], num_mel_bins=80, sample_frequency=16000,
                                                         dither=0.0, energy_floor=1e-10, high_freq=-400)
            if self.training and self.recipe["augmentation"]["spec_aug"]:
                features = spec_augment(features, rng)
            feats.append(features)
            waves.append(wave)
            targets.append(torch.tensor(phoneme_ids(item["transcript"]), dtype=torch.long))
            keys.append(item["key"])
            recordings.append(item["recording"])
        return {"feats": torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=-23.0259),
                "feat_lens": torch.tensor([len(f) for f in feats]), "waves": waves, "keys": keys,
                "recordings": recordings,
                "targets": torch.cat(targets), "target_lens": torch.tensor([len(t) for t in targets])}


def run(recipe, mode, out):
    import torch
    import torch.nn.functional as F
    from student_relational import deterministic_features
    from student_train import DEFAULTS, build_model

    validate_recipe(recipe)
    torch.set_num_threads(4)
    torch.manual_seed(recipe["seed"])
    device = torch.device("cuda")
    ckpath = Path(recipe["initial_checkpoint"])
    if hashlib.sha256(ckpath.read_bytes()).hexdigest() != recipe["student_sha256"]:
        raise ValueError("Initial checkpoint differs from audited primary")
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    expected_aug = {k: ck["cfg"].get(k, DEFAULTS[k]) for k in ("rir_p", "noise_p", "snr_db", "spec_aug")}
    # JSON turns the SNR tuple into a list; otherwise require the shipped recipe.
    expected_aug["snr_db"] = list(expected_aug["snr_db"])
    if recipe["augmentation"] != expected_aug:
        raise ValueError("Matched arms must retain the primary augmentation recipe")
    model = build_model(ck["cfg"]["size"]).to(device)
    model.load_state_dict(ck["model"])
    # Training geometry fixed to the deployed streaming-equivalent mask.
    model.streaming_eval(16, 128)
    anchor = build_model(ck["cfg"]["size"]).to(device).eval().requires_grad_(False)
    anchor.load_state_dict(ck["model"])
    anchor.streaming_eval(16, 128)
    teacher = TeacherFeatures(recipe, device)
    train, valid = TrainingItems(recipe, "training"), TrainingItems(recipe, "validation")
    plans = recipe["training_batches"]
    first = train.batch(plans[0], 0)
    probe = teacher(first["keys"][0], first["waves"][0])
    projection = None
    if mode == "hint":
        # Private initialization cannot change matched augmentation or updates.
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(recipe["seed"])
            projection = torch.nn.Linear(model.ctc_output[-1].in_features, probe["features"].shape[1]).to(device)
    parameters = list(model.parameters()) + (list(projection.parameters()) if projection is not None else [])
    opt = torch.optim.AdamW(parameters, lr=recipe["learning_rate"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2))
    best = float("inf")
    with (out / "log.jsonl").open("w") as log:
        for step in range(recipe["steps"]):
            batch = train.batch(plans[step % len(plans)], step)
            rows = [teacher(k, w) for k, w in zip(batch["keys"], batch["waves"])]
            loss, terms, counts = batch_objective(model, anchor, rows, batch, mode, recipe["temperature"],
                                                  recipe["weights"], projection)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite matched training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            record = {"step": step + 1, "mode": mode, **counts, **{k: float(v.detach()) for k, v in terms.items()}}
            if (step + 1) % recipe["validate_every"] == 0 or step + 1 == recipe["steps"]:
                # Same CTC selection criterion for both arms; no relationship
                # loss or target incident can select a checkpoint.
                total, frames = 0.0, 0
                with torch.no_grad(), deterministic_features(model):
                    for i in range(len(valid.items)):
                        b = valid.batch([i], 0)
                        lp, lens = model(b["feats"].to(device), b["feat_lens"].to(device))
                        value = F.ctc_loss(lp.transpose(0, 1), b["targets"].to(device), lens,
                                           b["target_lens"].to(device), reduction="sum", zero_infinity=False)
                        total += float(value)
                        frames += int(lens.sum())
                score = total / frames
                if not math.isfinite(score):
                    raise ValueError("Non-finite training-speaker validation score")
                record["validation_ctc"] = score
                if score < best:
                    best = score
                    # The projection is training-only and absent from deployment.
                    torch.save({"model": model.state_dict(), "cfg": ck["cfg"], "state": {"step": step + 1},
                                "relational_recipe": recipe, "mode": mode}, out / "ckpt.pt")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--mode", choices=("relational", "hint"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.recipe).read_text()), args.mode, args.out)
