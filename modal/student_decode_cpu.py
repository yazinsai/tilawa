"""CPU greedy decode of the clean-match test eval set with a student run; keeps every hypothesis.

    modal run modal/student_decode_cpu.py --run e2-libri-full

Writes `eval/<run>-clean-match-test.jsonl` on `tilawa-student`: key, ref, teacher, student per clip. Same clips,
chunk mask and greedy decode as `evaluate` in `student_train.py`, in fp32 instead of bf16 autocast.
"""

import json

import modal
from student_train import SR, ClipDataset, DEFAULTS, batches, collate, image, load_clips, load_student, vol

app = modal.App("tilawa-student-decode-cpu", image=image.add_local_python_source("student_train"))
PARTS = 10


@app.function(cpu=8, memory=16384, volumes={"/student": vol}, timeout=2 * 60 * 60)
def decode_part(run: str, part: int, chunk: int = 16, left: int = 128) -> list[dict]:
    import torch
    from student import ctc_collapse, ids_phonemes

    torch.set_num_threads(8)
    clips = load_clips("clean-match", "test", 0, 60.0, 1, DEFAULTS["eval_max"])
    model, _ = load_student(run)
    model.eval()
    model.streaming_eval(chunk, left)
    ds = ClipDataset(clips, {**DEFAULTS, "workers": 0}, train=False, aug=None)
    plan = [b for k, b in enumerate(batches(clips["n_samples"], 200.0, 0, shuffle=False)) if k % PARTS == part]
    out = []
    with torch.no_grad():
        for idx in plan:
            b = collate([ds[i] for i in idx])
            lp, lens = model(b["feats"], b["feat_lens"])
            best = lp.argmax(-1)
            for k, i in enumerate(b["i"].tolist()):
                out.append(dict(i=i, key=clips["key"][i], ref=clips["ref"][i], teacher=clips["hyp"][i],
                                student=ids_phonemes(ctc_collapse(best[k, : lens[k]].tolist())),
                                dur_s=int(clips["n_samples"][i]) / SR))
    return out


@app.function(cpu=8, memory=16384, volumes={"/student": vol}, timeout=2 * 60 * 60)
def evidence_part(runs: list[str], items: list[tuple[str, bytes]], chunk: int = 16, left: int = 128) -> dict:
    """Tracker evidence (`<clip>.evidence.json` body, as modal/decode_audio.py writes) for each audio file and run."""
    import subprocess

    import numpy as np
    import torch
    import torchaudio
    from student import TOKENS

    torch.set_num_threads(8)
    feats = {}
    for name, data in items:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                             input=data, capture_output=True, check=True).stdout
        w = torch.from_numpy(np.frombuffer(pcm, dtype=np.float32).copy())
        feats[name] = torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=SR, dither=0.0,
                                                        energy_floor=1e-10, high_freq=-400)
    out = {}
    for run in runs:
        model, _ = load_student(run)
        model.eval()
        model.streaming_eval(chunk, left)
        out[run] = {}
        with torch.no_grad():
            for name, f in feats.items():
                lp, lens = model(f.unsqueeze(0), torch.tensor([f.shape[0]]))
                p = lp[0, : lens[0]].float().exp()
                top_p, top_i = p.topk(5, dim=-1)
                phonemes, prev = [], 0
                for t, (i, c) in enumerate(zip(top_i[:, 0].tolist(), top_p[:, 0].tolist())):
                    if i != 0 and i == prev:
                        phonemes[-1]["end"] = t
                        phonemes[-1]["conf"] = max(phonemes[-1]["conf"], round(c, 4))
                    elif i != 0:
                        phonemes.append({"p": TOKENS[i - 1], "start": t, "end": t, "conf": round(c, 4)})
                    prev = i
                out[run][name] = {"frameMs": 40, "nFrames": int(p.shape[0]), "phonemes": phonemes,
                                  "posteriors": {"k": 5, "ids": top_i.flatten().tolist(), "probs": [round(x, 4) for x in top_p.flatten().tolist()]}}
    return out


@app.local_entrypoint()
def gold(runs: str = "e5-top1-lite,e5-soft5-lite,e2-scratch-full,e2-libri-full", main_checkout: str = "../tilawa", per_part: int = 40):
    """Decode the gold PER set and every help eval clip: writes `<main>/.scratch/gold/<run>/<clip>.evidence.json`."""
    from pathlib import Path

    main_dir = Path(main_checkout).resolve()
    help_dir = main_dir / ".scratch/help"
    manifest = json.loads((Path(__file__).resolve().parent.parent / "docs/benchmarks/gold-per-v1.json").read_text())
    files = {}
    for row in manifest["slices"]["gold"] + manifest["slices"]["silver"]:
        base = main_dir / ".scratch/v1-roadmap" if row["source"] == "tlog-bucket-review" else help_dir
        files[row["id"].replace(":", "-")] = base / row["audio"]
    for line in (help_dir / "labels.jsonl").read_text().splitlines():
        lab = json.loads(line)
        if lab["split"] == "eval":
            files[lab["id"]] = help_dir / lab["wav"]
    index = json.loads((main_dir / "packages/core/assets/ref-v2.json").read_text())
    names = sorted(files)
    parts = [[(n, files[n].read_bytes()) for n in names[k : k + per_part]] for k in range(0, len(names), per_part)]
    run_list = runs.split(",")
    print(f"{len(names)} clips in {len(parts)} parts x {len(run_list)} runs")
    for res in evidence_part.map([run_list] * len(parts), parts):
        for run, clips in res.items():
            d = main_dir / ".scratch/gold" / run
            d.mkdir(parents=True, exist_ok=True)
            for name, ev in clips.items():
                (d / f"{name}.evidence.json").write_text(json.dumps({"inventory": index["inventory"]["id"], **ev}, ensure_ascii=False, separators=(",", ":")))
    print("done")


@app.local_entrypoint()
def main(run: str = "e2-libri-full"):
    rows = sorted((r for part in decode_part.starmap([(run, p) for p in range(PARTS)]) for r in part), key=lambda r: r["i"])
    dest = f"/tmp/{run}-clean-match-test.jsonl"
    with open(dest, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), dest)
