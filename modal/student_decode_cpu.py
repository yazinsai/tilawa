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


@app.local_entrypoint()
def main(run: str = "e2-libri-full"):
    rows = sorted((r for part in decode_part.starmap([(run, p) for p in range(PARTS)]) for r in part), key=lambda r: r["i"])
    dest = f"/tmp/{run}-clean-match-test.jsonl"
    with open(dest, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), dest)
