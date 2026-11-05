"""Fine-tune the shipped listener only on mined real-phone false accusations and labelled swaps.

Run directly on the supervisor's training image; no Modal client is needed.
The mining JSON comes from sdk/src/training/mine-cli.ts, never from practice labels.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time

from student_accusation_loss import collapse_owned, masked_loss, viterbi_states, word_slots


def training_deps():
    """The supervisor's training image omits the reference-building dependency."""
    import importlib.metadata
    import subprocess
    try:
        version = importlib.metadata.version("quran-transcript")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "0.6.4":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "quran-transcript==0.6.4"], check=True)


def corrected_words(label, reference):
    """Preserve original slot identities through substitutions, omissions and repeats."""
    if label["split"] != "train" or (label.get("device") != "phone" and not any(e["errorClass"] == "wrong_word" for e in label["events"])) or not label.get("scored"):
        raise ValueError("Not an eligible training phone")
    if label.get("excluded") or label.get("exclude") or label.get("unsupported"):
        raise ValueError("Excluded or unsupported label")
    table = {f'{a["s"]}:{a["a"]}': a for a in reference["ayahs"]}
    words = []
    for key in label["ayahs"]:
        a = table[key]
        for i, text in enumerate(a["w"], 1):
            words.append({"key": f'{key}:{i}', "text": text, "surah": a["s"], "ayah": a["a"]})
    edits = {}
    for e in label["events"]:
        r = e["location"]
        key = f'{r["surah"]}:{r["ayah"]}:{r["word"]}'
        cls = e["errorClass"]
        if cls not in {"wrong_word", "omitted_word", "skipped_passage"}:
            raise ValueError("No corrected pronunciation labels")
        if cls == "wrong_word" and not e.get("observed"):
            raise ValueError("Unspecified spoken substitution")
        for i in range(e.get("words", 1)):
            k = f'{r["surah"]}:{r["ayah"]}:{r["word"] + i}'
            if k in edits or k not in {w["key"] for w in words}:
                raise ValueError("Overlapping or out-of-passage edit")
            edits[k] = e.get("observed") if cls == "wrong_word" else None
    repeats = {f'{e["location"]["surah"]}:{e["location"]["ayah"]}:{e["location"]["word"]}' for e in label["non_errors"] if e["kind"] == "repeat"}
    if any(e["kind"] != "repeat" for e in label["non_errors"]):
        raise ValueError("Unsupported non-error label")
    out = []
    for w in words:
        text = edits.get(w["key"], w["text"])
        if text is None:
            continue
        for t in text.split():
            out.append({**w, "text": t})
        if w["key"] in repeats:
            out.append(w.copy())
    if not out:
        raise ValueError("Empty corrected transcript")
    return out


def transcript(words, reference):
    from build_index import word_offsets
    from quran_transcript import MoshafAttributes, quran_phonetizer
    from student import phoneme_ids

    profile = MoshafAttributes(**reference["profile"])
    def phones(text, surah):
        p = quran_phonetizer(text, profile, remove_spaces=True, sura_idx=surah).phonemes
        if not isinstance(p, str):
            raise ValueError("Unsupported phonetizer result")
        phoneme_ids(p)  # inventory guard
        return p
    all_p, all_w = [], []
    for ayah in dict.fromkeys((w["surah"], w["ayah"]) for w in words):
        ws = [w for w in words if (w["surah"], w["ayah"]) == ayah]
        p = phones(" ".join(w["text"] for w in ws), ayah[0])
        iso = [phones(w["text"], ayah[0]) for w in ws]
        offsets = word_offsets(p, iso) + [len(p)]
        for i, w in enumerate(ws):
            all_p.extend(phoneme_ids(p[offsets[i]:offsets[i + 1]]))
            all_w.extend([w["key"]] * (offsets[i + 1] - offsets[i]))
    return collapse_owned(all_p, all_w)


def accusation_words(event, reference):
    """Graph accusations may name several contiguous words; supervise the entire named span."""
    r = event["location"]
    all_words = []
    for a in reference["ayahs"]:
        for i, start in enumerate(a["o"]):
            end = a["o"][i + 1] if i + 1 < len(a["o"]) else len(a["p"])
            all_words.append((f'{a["s"]}:{a["a"]}:{i + 1}', a["p"][start:end]))
    key = f'{r["surah"]}:{r["ayah"]}:{r["word"]}'
    start = next(i for i, (k, _) in enumerate(all_words) if k == key)
    phones, selected = "", set()
    for k, p in all_words[start:]:
        phones += p; selected.add(k)
        if phones == event["expected"]:
            return selected
        if not event["expected"].startswith(phones):
            break
    raise ValueError("Accusation is not a contiguous written span")


def stream(model, feats, device):
    """Actual deployed chunk loop with detached history; gradients stay inside each chunk."""
    import torch
    from student_export import PAD, LOG_EPS

    model.eval(); model.streaming_eval(16, 128)
    states = model.encoder.get_init_states(1, device) + [model.encoder_embed.get_init_states(1, device), torch.zeros(1, dtype=torch.int64, device=device)]
    need = ((len(feats) - 7) // 2 + 1) // 2
    windows = max(1, -(-need // 8))
    total = 45 + (windows - 1) * 32
    if total > len(feats):
        feats = torch.cat([feats, torch.full((total - len(feats), 80), LOG_EPS, device=device)])
    out = []
    for a in range(0, total - (32 + PAD) + 1, 32):
        lens = torch.tensor([32 + PAD], device=device)
        x, lens, pad = model.encoder_embed.streaming_forward(x=feats[None, a:a + 32 + PAD], x_lens=lens, cached_left_pad=states[-2])
        processed = states[-1]
        mask = (processed[:, None] <= torch.arange(128, device=device)[None]).to(torch.int32).flip(1)
        mask = torch.cat([mask, torch.zeros(1, 16, dtype=torch.int32, device=device)], 1)
        encoded, _, st = model.encoder.streaming_forward(x=x.permute(1, 0, 2), x_lens=lens, states=states[:-2], src_key_padding_mask=mask)
        out.append(model.ctc_output(encoded.permute(1, 0, 2)).log_softmax(-1)[0])
        states = st + [pad, processed + lens]
        states = [t.detach() for t in states]
    return torch.cat(out)[:need]


def chunk_masked(model, feats):
    """The shipped recipe's parallel causal chunk mask, with the deployment chunk/context fixed."""
    import torch
    model.eval(); model.streaming_eval(16, 128)
    lp, lens = model(feats[None], torch.tensor([len(feats)], device=feats.device))
    return lp[0, :int(lens[0])]


def eos_batch(samples):
    """EOS padding is valid encoder context, as in the SDK; it never receives a loss.

    A long masked tail in a mixed-length evaluation batch can create fully masked
    attention rows, then NaN gradients even when the real output rows are finite.
    """
    import torch
    from student_export import LOG_EPS

    real = [((len(f) - 7) // 2 + 1) // 2 for f, _ in samples]
    windows = max(1, -(-max(real) // 8))
    total = 45 + (windows - 1) * 32
    padded = torch.stack([torch.cat([f, torch.full((max(0, total - len(f)), 80), LOG_EPS, device=f.device)])[:total] for f, _ in samples])
    return padded, torch.full((len(samples),), total, device=padded.device, dtype=torch.int64), real


def clean_anchor_clips(root=Path("/student/s1"), exclusions_path=Path("/root/gold-per-v1-exclude.txt")):
    """Select exactly the shipped clean-match pool, reading only audio for the KL anchor.

    Teacher top-5 arrays and transcripts have no role in this objective. Avoid
    loading those large columns merely to discard them in ClipDataset.
    """
    import numpy as np
    import pyarrow.parquet as pq
    import student_train as T
    from student import exclusions, held_out

    gold = exclusions(exclusions_path.read_text())
    files = sorted(root.glob("*.parquet"))
    meta = []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["shard", "row", "bucket", "split", "n_samples"]).to_pydict()
        for r, (sh, row, bk, sp, n) in enumerate(zip(t["shard"], t["row"], t["bucket"], t["split"], t["n_samples"])):
            if bk == "clean-match" and sp == "train" and n <= 30.0 * T.SR and not held_out(gold, sh, row, sp):
                meta.append((fi, r, n))
    order = np.random.default_rng(0).permutation(len(meta))
    cum = np.cumsum([meta[i][2] for i in order]) / T.SR / 3600
    order = order[:int(np.searchsorted(cum, 20.0)) + 1]
    by = {}
    for i in sorted(order):
        by.setdefault(meta[i][0], []).append(meta[i][1])
    clips, keys = [], []
    for fi, rows in by.items():
        t = pq.read_table(files[fi], columns=["shard", "row", "flac"]).take(rows).to_pydict()
        clips.extend(t["flac"])
        keys.extend(f"{s}:{r}" for s, r in zip(t["shard"], t["row"]))
    if not clips:
        raise ValueError("No original clean-match anchors")
    return {"audio": clips, "key": keys}


def stable_swoosh_training():
    """Use stable log(1+exp(x)) backward; no-grad inference/export stays original.

    Icefall's evaluation fallback replaces an infinite forward exp result, but
    its unused branch still contributes inf * 0 in autograd. Preserve the exact
    installed activation formula and constants, replacing only that operation.
    """
    import inspect
    import torch
    import scaling

    if hasattr(scaling, "_accusation_original_swoosh"):
        return scaling._accusation_original_swoosh
    originals = {}
    for name in ("SwooshRForward", "SwooshLForward"):
        original = getattr(scaling, name)
        source = inspect.getsource(original)
        needle = "(1.0 + x_offset.exp()).log().to(x.dtype)"
        if source.count(needle) != 1:
            raise ValueError("Installed Swoosh formula changed")
        scope = {}
        exec(compile(source.replace(needle, "torch.nn.functional.softplus(x_offset).to(x.dtype)"),
                     f"<stable-{name}>", "exec"), scaling.__dict__, scope)
        stable = scope[name]
        def forward(x, original=original, stable=stable):
            return stable(x) if torch.is_grad_enabled() and x.requires_grad else original(x)
        originals[name] = original
        setattr(scaling, name, forward)
    scaling._accusation_original_swoosh = originals
    return originals


def main():
    training_deps()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import student_train as T
    from student import mix_noise

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--mined", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max-seconds", type=int, default=3000, help="Save and return before the supervisor timeout; resume to the same fixed step count")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="Trace a non-finite backward operation")
    args = ap.parse_args()
    torch.autograd.set_detect_anomaly(args.diagnose)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    ck = torch.load(Path(args.source) / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = {**T.DEFAULTS, **ck["cfg"]}
    shipped = T.build_model(cfg["size"]).to(device)
    stable_swoosh_training()  # load the encoder's icefall modules before patching
    shipped.load_state_dict(ck["model"]); shipped.eval()
    for p in shipped.parameters():
        p.requires_grad_(False)
    reference = json.loads(Path("/app/packages/core/assets/ref-v2.json").read_text())
    mining = json.loads(Path(args.mined).read_text())
    source_hash = hashlib.sha256((Path(args.source) / "ckpt.pt").read_bytes()).hexdigest()
    mining_hash = hashlib.sha256(Path(args.mined).read_bytes()).hexdigest()
    cache = json.loads((out / "admission.json").read_text()) if (out / "admission.json").exists() else {}
    cached = cache.get("sourceSha256") == source_hash and cache.get("miningSha256") == mining_hash
    cache_rows = {r["id"]: r for r in cache.get("accepted", [])} if cached else {}
    cache_bad = {r["id"]: r for r in cache.get("rejected", [])} if cached else {}
    accepted, rejected = [], []
    for row in mining:
        l = row["label"]
        if l["id"] in cache_bad:
            rejected.append(cache_bad[l["id"]]); continue
        try:
            if l["id"] in cache_rows:
                r = cache_rows[l["id"]]
                path = (Path(args.data) / l["wav"]).resolve()
                if not path.is_relative_to((Path(args.data) / "wav").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != r["audioSha256"]:
                    raise ValueError("Changed cached training audio")
                wave, sr = sf.read(path, dtype="float32")
                if sr != T.SR or wave.ndim != 1:
                    raise ValueError("Changed cached audio format")
                accepted.append({**r, "wave": wave}); continue
            words = corrected_words(l, reference)
            target, owners = transcript(words, reference)
            selected = set().union(*(accusation_words(e, reference) for e in row["mined"]))
            if selected and l.get("device") != "phone":
                raise ValueError("Non-phone negative")
            positive = {f'{e["location"]["surah"]}:{e["location"]["ayah"]}:{e["location"]["word"]}' for e in l["events"] if e["errorClass"] == "wrong_word"}
            bad = set()
            for e in l["events"] + l["non_errors"]:
                r = e["location"]
                bad.update(f'{r["surah"]}:{r["ayah"]}:{r["word"] + i}' for i in range(e.get("words", 1)))
            if selected & bad or not selected <= {w["key"] for w in words}:
                raise ValueError("Mined labelled-error slot")
            path = (Path(args.data) / l["wav"]).resolve()
            if not path.is_relative_to((Path(args.data) / "wav").resolve()):
                raise ValueError("Audio outside training directory")
            wave, sr = sf.read(path, dtype="float32")
            if sr != T.SR or wave.ndim != 1:
                raise ValueError("Expected mono 16 kHz training audio")
            feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave)[None], num_mel_bins=80,
                sample_frequency=T.SR, dither=0.0, energy_floor=1e-10, high_freq=-400)
            with torch.no_grad():
                lp = stream(shipped, feats.to(device), device).cpu().numpy()
            slots = word_slots(viterbi_states(lp, target), target, owners, selected | positive)
            accepted.append({"id": l["id"], "speaker": l["speaker"], "device": l.get("device"), "split": l["split"], "wave": wave, "slots": slots,
                "negative": sorted(selected), "positive": sorted(positive), "audioSha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (ValueError, KeyError, AssertionError, IndexError) as e:
            rejected.append({"id": l["id"], "reason": str(e)})
    audit = {"accepted": [{k: v for k, v in r.items() if k != "wave"} for r in accepted], "rejected": rejected,
        "sourceSha256": source_hash, "miningSha256": mining_hash, "cfg": cfg}
    (out / "admission.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    print(json.dumps({"admitted": len(accepted), "rejected": len(rejected), "negative": sum(len(x["negative"]) for x in accepted),
        "positive": sum(len(x["positive"]) for x in accepted), "cached": cached}, ensure_ascii=False), flush=True)
    focus = [x for x in accepted if x["slots"]]
    help_clean = [x for x in accepted if not x["slots"]]
    if not any(x["negative"] for x in focus) or not any(x["positive"] for x in focus):
        raise ValueError("Missing mined false accusations or protected real substitutions")
    if args.prepare_only:
        return
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    pool_key = {"source": source_hash, "bucket": "clean-match", "split": "train", "hours": 20.0, "max_clip_s": 30.0, "seed": 0,
        "gold": hashlib.sha256(Path("/root/gold-per-v1-exclude.txt").read_bytes()).hexdigest()}
    pool_file = out / "originals-audio-cache.pt"
    pool = torch.load(pool_file, map_location="cpu", weights_only=False) if pool_file.exists() else {}
    if pool.get("key") == pool_key:
        originals = pool["clips"]
    else:
        print("Loading original clean-match audio (no teacher columns)", flush=True)
        legacy_file = out / "originals-cache.pt"
        legacy = torch.load(legacy_file, map_location="cpu", weights_only=False) if legacy_file.exists() else {}
        if legacy.get("key") == pool_key:
            c = legacy["clips"]
            originals = {"audio": [c["flac"][a:b].tobytes() for a, b in zip(c["flac_off"][:-1], c["flac_off"][1:])], "key": c["key"]}
            del c, legacy
        else:
            originals = clean_anchor_clips()
        torch.save({"key": pool_key, "clips": originals}, str(pool_file) + ".tmp")
        Path(str(pool_file) + ".tmp").replace(pool_file)
    print(json.dumps({"originals": len(originals["audio"]), "poolCached": pool.get("key") == pool_key}), flush=True)
    model = T.build_model(cfg["size"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
    start_step = 0
    if (out / "ckpt.pt").exists():
        saved = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        if saved["adaptation"]["miningSha256"] != audit["miningSha256"] or saved["adaptation"]["sourceSha256"] != audit["sourceSha256"]:
            raise ValueError("Changed source or mining input during resume")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        start_step = saved["adaptation"]["steps"]
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torchRng"])
        torch.cuda.set_rng_state_all(saved["cudaRng"])
    def feature(wave):
        if rng.random() < cfg["rir_p"]:
            wave = T.reverb(wave, T.pick(aug["rir"], rng))
        if rng.random() < cfg["noise_p"]:
            wave = mix_noise(wave, T.pick(aug["noise"], rng), rng.uniform(*cfg["snr_db"]), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None], num_mel_bins=80,
            sample_frequency=T.SR, dither=0.0, energy_floor=1e-10, high_freq=-400)
        return (T.spec_augment(f, rng) if cfg["spec_aug"] else f).to(device)
    t0 = time.time()
    log = (out / "log.jsonl").open("a")
    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        item = focus[step % len(focus)]
        samples = [(feature(item["wave"]), item["slots"])]
        if help_clean:
            samples.append((feature(help_clean[step % len(help_clean)]["wave"]), []))
        for _ in range(3 if help_clean else 4):
            wave, sr = sf.read(io.BytesIO(originals["audio"][int(rng.integers(len(originals["audio"])))]), dtype="float32")
            if sr != T.SR or wave.ndim != 1:
                raise ValueError("Invalid original clean-match audio")
            samples.append((feature(wave), []))
        sums = {"ctc": 0.0, "kl": 0.0}
        feats, lengths, real_frames = eos_batch(samples)
        model.eval(); model.streaming_eval(16, 128)
        shipped.eval(); shipped.streaming_eval(16, 128)
        with torch.no_grad():
            anchor, anchor_lens = shipped(feats, lengths)
        lp, out_lens = model(feats, lengths)
        if not torch.equal(anchor_lens, out_lens):
            raise ValueError("Changed output frame grid")
        objective = torch.zeros((), device=device)
        for i, (_, slots) in enumerate(samples):
            n = real_frames[i]
            if n > int(out_lens[i]):
                raise ValueError("Missing real output frames")
            losses = masked_loss(lp[i, :n], anchor[i, :n], slots, kl_weight=10.0)
            if not torch.isfinite(losses["loss"]):
                print(json.dumps({"step": step, "sample": i, "clip": item["id"], "frames": n, "slots": slots,
                    "ctc": float(losses["ctc"].detach()), "kl": float(losses["kl"].detach()),
                    "modelNonfinite": int((~torch.isfinite(lp[i, :n])).sum()),
                    "anchorNonfinite": int((~torch.isfinite(anchor[i, :n])).sum())}), flush=True)
                raise ValueError("Non-finite local objective")
            objective = objective + losses["loss"] / 5
            for k in sums:
                sums[k] += float(losses[k].detach()) / 5
        objective.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            print(json.dumps({"step": step, "event": "nonfinite-gradient", "lengths": [len(s[0]) for s in samples],
                "parameters": [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]}), flush=True)
            raise ValueError("Non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (step + 1) % 50 == 0:
            r = {"step": step + 1, "seconds": time.time() - t0, **sums}
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        stop = time.time() - t0 >= args.max_seconds
        if (step + 1) % 500 == 0 or step + 1 == args.steps or stop:
            torch.save({"model": model.state_dict(), "cfg": cfg, "state": {**ck["state"], "step": ck["state"]["step"] + step + 1},
                "opt": opt.state_dict(), "rng": rng.bit_generator.state, "torchRng": torch.get_rng_state(), "cudaRng": torch.cuda.get_rng_state_all(),
                "adaptation": {"localCTC": True, "klWeight": 10, "mix": [1, 4], "steps": step + 1,
                    "miningSha256": audit["miningSha256"], "sourceSha256": audit["sourceSha256"]}}, out / "ckpt.pt.tmp")
            (out / "ckpt.pt.tmp").replace(out / "ckpt.pt")
        if stop:
            break
    print(json.dumps({"steps": step + 1 if start_step < args.steps else start_step, "complete": (step + 1 if start_step < args.steps else start_step) >= args.steps}), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/app/tools")
    main()
