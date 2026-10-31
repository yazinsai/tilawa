# Matched continuation control

Status: tracing before training. No fold or discard.

Compare uniform continuation with phone_p 0 versus .75; same original spliced-clean checkpoint (step 30287), ordered 50-hour pool, seed 0, CTC1 + KD2, spec augmentation and original generic noise/reverb, 200 updates, one epoch cap, LR .0015, padded batch budget 800 seconds. All clip weights are one. Retain final scheduled checkpoint only. No exam scoring or checkpoint tuning.

Both arms use deterministic per-clip augmentation RNG and consume the phone-choice draw. Previously phone_p=0 used an unseeded generator, so a fresh phone arm is run alongside this control. Removing the added phone channel leaves the original generic augmentation in place; this is not a fully unaugmented arm.

Freeze ten phone warnings present in uniform, broad targeted, sparse selected and sparse random, absent from shipped main under the existing same-word/six-second policy. Existing oracle-position attribution: seven listener, three tracker. Trace as-said word alignment, phoneme changes and tracker/checker event histories before training.

Predictions: (1) If added phone augmentation causes the shared regression, clean continuation should remove most recurring sites while phone continuation reproduces them. (2) If continued optimization or fixed targets drive the failure, both arms should retain the sites. (3) If the warnings principally reflect tracker position, oracle-position replay should remove them despite usable local evidence. No inference of target validity from teacher agreement alone.

Expected score deltas are zero, expressing a diagnostic experiment rather than a preregistered improvement claim. Evaluate both arms on all 100 frozen lab-v1.5 practice sessions, phone and studio; compare incidents and scores to each other, shipped main and historical uniform. Preserve real-mistake and recorded correction controls.
