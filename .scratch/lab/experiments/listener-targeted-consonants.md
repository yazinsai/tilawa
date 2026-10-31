# Targeted consonant evidence training

Status: running

Frozen experiment: same 50-hour, seed-0 clean-match training pool and shipped checkpoint as full phone-channel adaptation; 200 updates, LR 0.0015, CTC plus KD2, 75% phone augmentation, same batch budget. Before updates, compare clean and fixed phone degraded greedy consonant strings using the shipped checkpoint. Weight clips containing insertions/deletions/substitutions 3x, otherwise short consonant-string (<=8) and adjacent-repeat controls 2x, others 1x. Apply weights to both CTC and KD, normalized by weighted frame count. No replacement transcript, synthetic canonical correction, beam search or threshold changes. No mined benchmark material.

The whole training pool is mined. A separate 600-clip validation pool is independently decoded and retained for reporting; its results do not select weights or hyperparameters. Source split and existing gold-exclusion rules remain active. The full-phone historical checkpoint supplies a fixed-compute reference (identical pool, initialization, schedule and channel recipe; uniform weight). Mainline practice is the promotion comparison; phone and studio are mandatory, with recovered/newly missed mistake IDs and new/removed warnings. All genuine benchmark mistakes remain held-out controls. Short/repeat mining controls are acoustic proxies, not independently reviewed self-correction labels. No assertion of genuine self-correction training labels is made.

Target: four fewer listener and overall wrong calls. Any new false-warning incidents or newly missed genuine mistakes require explicit disclosure; aggregate improvement alone does not earn promotion. No exam evaluation, fold or discard.

Preflight: Python compilation and diff whitespace checks pass. Tiny Modal loss-invariant test `ap-pUraROUozH0I41ibSPARVA` confirms all-one weighted CTC and KD reproduce the uniform objectives, explicit 3:1 weighting matches the independently assembled weighted KD mean, and gradients remain finite. First test invocation lacked a container source mount and was stopped; no training results were read or altered. Training source commit `ee0e6a3`, app `ap-e7oSY9RMBJUy4hLX9RdHrg`.
