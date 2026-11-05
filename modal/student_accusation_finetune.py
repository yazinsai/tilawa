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
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    rng = np.random.default_rng(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    ck = torch.load(Path(args.source) / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = {**T.DEFAULTS, **ck["cfg"]}
    shipped = T.build_model(cfg["size"]).to(device)
    shipped.load_state_dict(ck["model"]); shipped.eval()
    for p in shipped.parameters():
        p.requires_grad_(False)
    reference = json.loads(Path("/app/packages/core/assets/ref-v2.json").read_text())
    mining = json.loads(Path(args.mined).read_text())
    accepted, rejected = [], []
    for row in mining:
        l = row["label"]
        try:
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
        "sourceSha256": hashlib.sha256((Path(args.source) / "ckpt.pt").read_bytes()).hexdigest(),
        "miningSha256": hashlib.sha256(Path(args.mined).read_bytes()).hexdigest(), "cfg": cfg}
    (out / "admission.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    print(json.dumps({"admitted": len(accepted), "rejected": len(rejected), "negative": sum(len(x["negative"]) for x in accepted),
        "positive": sum(len(x["positive"]) for x in accepted)}, ensure_ascii=False), flush=True)
    focus = [x for x in accepted if x["slots"]]
    help_clean = [x for x in accepted if not x["slots"]]
    if not any(x["negative"] for x in focus) or not any(x["positive"] for x in focus):
        raise ValueError("Missing mined false accusations or protected real substitutions")
    if args.prepare_only:
        return
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    originals = T.load_clips("clean-match", "train", 20.0, 30.0, 0)
    clean = T.ClipDataset(originals, cfg, train=True, aug=aug)
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
            samples.append((clean[int(rng.integers(len(clean)))]["feats"].to(device), []))
        sums = {"ctc": 0.0, "kl": 0.0}
        feats = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True, padding_value=-23.0259)
        lengths = torch.tensor([len(s[0]) for s in samples], device=device)
        model.eval(); model.streaming_eval(16, 128)
        shipped.eval(); shipped.streaming_eval(16, 128)
        with torch.no_grad():
            anchor, anchor_lens = shipped(feats, lengths)
        lp, out_lens = model(feats, lengths)
        if not torch.equal(anchor_lens, out_lens):
            raise ValueError("Changed output frame grid")
        objective = lp.sum() * 0
        for i, (_, slots) in enumerate(samples):
            n = int(out_lens[i])
            losses = masked_loss(lp[i, :n], anchor[i, :n], slots, kl_weight=10.0)
            if not torch.isfinite(losses["loss"]):
                raise ValueError("Non-finite local objective")
            objective = objective + losses["loss"] / 5
            for k in sums:
                sums[k] += float(losses[k].detach()) / 5
        objective.backward()
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
