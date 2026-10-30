# 05 Decode test set v1 on Modal

Status: resolved
Type: task
Blocked by: 03

Decode each session (studio and phone-degraded) with the main-line listener `spliced-clean-int8` (streaming SDK, 20 ms pushes, posteriors on) and with the teacher. Evidence goes onto the lab volume, keyed by test set version, listener and model hash. One command decodes any listener checkpoint, so listener experiments can use it later.

## Comments

- Done. `modal run modal/lab_decode.py --listener <name> --model <dir under /lab/data>` decodes any student checkpoint; `--listener teacher` runs `obadx/muaalem-model-v3_2`. It runs one container per session × variant and skips files already present. It refuses to mix models under one listener name without `--force`.
- Evidence is at `/lab/data/lab-v1/sessions/<id>.<spliced-clean-int8|teacher>.<studio|phone>.evidence.json`, 80 + 80 files. Every frame count matches its label within 0.1 s.
- `decode-v1.json` records per listener: model file sha256s (or HF revision `01a1ef9f`), flags, source commit, run id, and per-file sha256, frames and seconds. Details and the timing table are in `docs/benchmarks/lab-v1.md#decode`.
- Student `spliced-clean-int8` (model.onnx `714e98ad…`) ran through the SDK `decode-cli` on onnxruntime-node CPU (1 thread), with 20 ms pushes and top-k posteriors at 40 ms. It took 228 s wall and 3148 container-s for 22,418 s of audio. Run `decode-spliced-clean-int8-d407c89-20261028T123012Z`.
- The teacher used 30/25 s windows, bf16 and top-5. It took 437 GPU-s on L4 and 600 s wall, because Modal gave it one L4. Total cost was under $1.
