"""Train both phone-event and equal-capacity CTC arms on training-speaker data only.

Invoke directly inside the supervisor's train image. No Modal client calls are made.
Checkpoint selection uses a deterministic speaker-grouped subset of the *training* split;
the release validation/test splits are read only for exclusion metadata, never for audio or selection.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

import student_train as T
from student import TOKENS, ctc_collapse, edit_distance, phoneme_ids
from student_integrate import acoustic_outputs, attach_heads, decode_events, objective


def development(group):
    return int(hashlib.sha256(("cif-speakers-v1:" + group).encode()).hexdigest()[:8], 16) % 10 == 0


def load_records(hours, seed):
    import pyarrow.parquet as pq

    gold = set(Path("/root/gold-per-v1-exclude.txt").read_text().split())
    metadata, excluded = [], set()
    for path in sorted(Path("/student/s1").glob("*.parquet")):
        rows = pq.read_table(path, columns=["shard", "row", "cluster", "split", "bucket", "n_samples"]).to_pylist()
        for i, row in enumerate(rows):
            if row["cluster"] is None:
                raise ValueError("Missing speaker cluster; cannot construct a speaker-disjoint split")
            group = "quran:" + str(row["cluster"])
            key = f'{row["shard"]}:{row["row"]}'
            if row["split"] != "train" or key in gold:
                excluded.add(group)
            metadata.append((path, i, row, group, key))
    eligible = [(p, i, r, g, k) for p, i, r, g, k in metadata
                if r["split"] == "train" and r["bucket"] == "clean-match" and g not in excluded
                and 16000 <= r["n_samples"] <= 30 * 16000]
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    selected, seconds = {False: [], True: []}, {False: 0.0, True: 0.0}
    for item in eligible:
        dev = development(item[3])
        limit = hours * 3600 * (0.1 if dev else 1)
        if seconds[dev] < limit:
            selected[dev].append(item)
            seconds[dev] += item[2]["n_samples"] / 16000
    selected_by_file = {}
    for dev, items in selected.items():
        for path, i, row, group, key in items:
            selected_by_file.setdefault(path, []).append((dev, i, group, key))
    records = {False: [], True: []}
    for path, items in selected_by_file.items():
        requested = [int(key.rsplit(":", 1)[1]) for _, _, _, key in items]
        rows = pq.read_table(path, columns=["row", "flac", "ref", "n_samples"],
                             filters=[("split", "=", "train"), ("row", "in", requested)]).to_pylist()
        by_row = {r["row"]: r for r in rows}
        for dev, _, group, key in items:
            row = by_row[int(key.rsplit(":", 1)[1])]
            records[dev].append(dict(key=key, group=group, audio=row["flac"], target=row["ref"], n_samples=row["n_samples"]))

    root = Path("/lab/data/help-train")
    ref = json.loads(Path("/app/packages/core/assets/ref-v2.json").read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in ref["ayahs"]}
    help_rows = [json.loads(s) for s in (root / "labels.jsonl").read_text().splitlines() if s.strip()]
    if any(r.get("split") != "train" for r in help_rows):
        raise ValueError("Release-held-out help audio must never enter this runner")
    import soundfile as sf
    for row in help_rows:
        if (row.get("kind") != "none" or row.get("events") or row.get("non_errors") or row.get("unsupported")
                or row.get("excluded") or row.get("exclude") or row.get("scored") is False):
            continue
        group = "help:" + str(row["speaker"])
        wav = (root / row["wav"]).resolve()
        if not wav.is_relative_to(root.resolve()):
            raise ValueError("Help audio path escaped the training directory")
        info = sf.info(wav)
        records[development(group)].append(dict(key="help:" + row["id"], group=group, path=str(wav),
                                                 target="".join(ayahs[a] for a in row["ayahs"]),
                                                 n_samples=round(info.frames * 16000 / info.samplerate)))
    for dev in (False, True):
        if not records[dev]:
            raise ValueError("No eligible training-speaker records for " + ("selection" if dev else "training"))
        # Explicit no-speech examples complement the real pauses retained in full recordings.
        records[dev] += [dict(key=f"silence:{i}", group="silence", target="", n_samples=32000)
                         for i in range(max(1, len(records[dev]) // 10))]
    train_groups = {r["group"] for r in records[False]} - {"silence"}
    dev_groups = {r["group"] for r in records[True]} - {"silence"}
    assert not train_groups & dev_groups and not (train_groups | dev_groups) & excluded
    audit = dict(train=len(records[False]), development=len(records[True]), train_groups=sorted(train_groups),
                 development_groups=sorted(dev_groups), excluded_groups=len(excluded),
                 keys={str(dev): [r["key"] for r in rows] for dev, rows in records.items()})
    return records[False], records[True], audit


class RecordDataset:
    def __init__(self, records, train, aug, seed):
        self.records, self.train, self.aug, self.seed = records, train, aug, seed
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import soundfile as sf
        import torchaudio
        from student import mix_noise

        r = self.records[i]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
        if "audio" in r:
            wave, sr = sf.read(io.BytesIO(r["audio"]), dtype="float32")
        elif "path" in r:
            wave, sr = sf.read(r["path"], dtype="float32")
        else:
            wave, sr = np.zeros(r["n_samples"], np.float32), 16000
        if wave.ndim > 1:
            wave = wave.mean(1)
        if sr != 16000:
            wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
        if self.train:
            if rng.random() < 0.2:
                wave = T.reverb(wave, T.pick(self.aug["rir"], rng))
            if rng.random() < 0.5:
                wave = mix_noise(wave, T.pick(self.aug["noise"], rng), rng.uniform(5, 25), rng)
            wave *= 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0),
                                                num_mel_bins=80, sample_frequency=16000, dither=0,
                                                energy_floor=1e-10, high_freq=-400)
        if self.train:
            feats = T.spec_augment(feats, rng)
        return dict(feats=feats, target=torch.tensor(phoneme_ids(r["target"]), dtype=torch.long))


def collate(records):
    return dict(feats=torch.nn.utils.rnn.pad_sequence([r["feats"] for r in records], batch_first=True, padding_value=-23.0259),
                feat_lens=torch.tensor([len(r["feats"]) for r in records]),
                targets=torch.cat([r["target"] for r in records]),
                target_lens=torch.tensor([len(r["target"]) for r in records]))


def evaluate(model, records, arm, device):
    model.eval()
    model.streaming_eval()
    errors = total = 0
    count_error = 0.0
    ds = RecordDataset(records, False, None, 0)
    with torch.no_grad():
        for i in range(len(ds)):
            b = {k: v.to(device) for k, v in collate([ds[i]]).items()}
            features, lengths = model.encode(b["feats"], b["feat_lens"])
            mass, scores = acoustic_outputs(model, features.float())
            n = int(lengths[0])
            if arm == "cif":
                events = decode_events(mass[0, :n].cpu(), scores[0, :n].cpu())
                hyp = [e["id"] for e in events]
            else:
                hyp = ctc_collapse(scores[0, :n].argmax(-1).tolist())
            target = b["targets"].tolist()
            errors += edit_distance(hyp, target)
            total += len(target)
            count_error += abs(float(mass[0, :n].sum()) - len(target)) / max(1, len(target))
    model.train()
    model.training_mode()
    return dict(per=errors / max(1, total), count_error=count_error / len(ds), edits=errors, phones=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-seconds", type=float, default=120)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    training, selection, audit = load_records(args.hours, args.seed)
    (out / "partitions.json").write_text(json.dumps(audit, indent=1))
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    dataset = RecordDataset(training, True, aug, args.seed)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    size = checkpoint["cfg"]["size"]
    base = attach_heads(T.build_model(size))
    missing, extra = base.load_state_dict(checkpoint["model"], strict=False)
    if set(missing) != {"phone_mass.weight", "phone_mass.bias", "phone_output.weight", "phone_output.bias"} or extra:
        raise ValueError((missing, extra))
    base.phone_output.load_state_dict(base.ctc_output[1].state_dict())
    import copy
    models = {arm: copy.deepcopy(base).cuda() for arm in ("cif", "ctc")}
    opts = {arm: torch.optim.AdamW(m.parameters(), lr=1e-4) for arm, m in models.items()}
    best = float("inf")
    step = 0

    def selection_step():
        nonlocal best
        results = {}
        for arm, model in models.items():
            results[arm] = evaluate(model, selection, arm, "cuda")
            print(json.dumps(dict(step=step, arm=arm, selection=results[arm])), flush=True)
        # A shared selection step guarantees identical actual update counts in the exported pair.
        # Selection sees no practice audio, incident labels, or release-held-out speaker.
        score = sum(r["per"] for r in results.values()) / len(results)
        if score < best:
            best = score
            for arm, model in models.items():
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(dict(model=state, cfg=dict(size=size, arm=arm), step=step,
                                selection=results[arm], arguments=vars(args)), out / f"{arm}.pt")

    selection_step()
    for epoch in range(args.epochs):
        dataset.epoch = epoch
        loader = torch.utils.data.DataLoader(dataset,
            batch_sampler=T.batches(np.array([r["n_samples"] for r in training]), args.batch_seconds, args.seed + epoch),
            collate_fn=collate, num_workers=4)
        for b in loader:
            b = {k: v.cuda() for k, v in b.items()}
            rng = torch.cuda.get_rng_state()
            cpu_rng = torch.get_rng_state()
            for arm, model in models.items():
                torch.cuda.set_rng_state(rng)
                torch.set_rng_state(cpu_rng)
                for module in model.modules():
                    module.batch_count = (checkpoint["state"]["step"] * checkpoint["cfg"]["max_batch_s"]
                                          + step * args.batch_seconds) / 600.0
                opts[arm].zero_grad(set_to_none=True)
                losses = objective(model, b, arm)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError(f"Nonfinite {arm} training loss at {step}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                opts[arm].step()
                if step % 25 == 0:
                    print(json.dumps(dict(step=step, epoch=epoch, arm=arm, **{k: float(v.detach()) for k, v in losses.items()})), flush=True)
            step += 1
            if step % args.eval_every == 0:
                selection_step()
        selection_step()
    (out / "complete.json").write_text(json.dumps(dict(updates_per_arm=step, parameters=sum(p.numel() for p in base.parameters()),
                                                      best=best, seed=args.seed)))


if __name__ == "__main__":
    main()
