# 12 Calibrate the independent audio opinion

Status: ready-for-agent
Type: task
Blocked by: 07

Spec: `../spec.md` decision 17; `docs/agents/audio-review.md`.

Run `modal/review_asr.py` on the 40 cases from ticket 07 before showing it the labels. Report its agreement with your verdicts. At 95% or better agreement it pre-screens later test set versions and training-data samples, and you hear only disagreements plus a 10% spot check. Below that, record why and keep listening fully manual.
