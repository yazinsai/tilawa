# Pause-mask-trained streaming student

Status: implementing. The card is unchanged.

## Prediction and premise

The card predicts listener −2 and overall −1, roughly two recoveries and one
loss. Its premise is that cutting pre-pause attention at inference put the
shipped student outside its training regime. Train that regime before asking
whether it can recover the planted post-pause consonants.

The registered seven incidents are checked acoustically in paired full-session
streams. Already having two planted consonants is not a new recovery. Record
ordered consonant counts, changes, and new two-consonant recoveries separately.
If fewer than three targets satisfy the acoustic premise, stop without practice
scoring. If three or more targets are won, request practice scoring before
judging losses. No practice-selected training recipe or constant is allowed.

## Built mechanism

`modal/student_pause_mask.py` shares the same masked streaming computation
between training and ONNX export. The deployed geometry remains chunk 16,
left cache 128, 45 fbank frames per window, 32-frame shift and 40 ms CTC frames.
Cutoffs are absolute on the native 20 ms embedding clock. Each attention query
masks keys earlier than its cutoff before softmax. Both depthwise convolution
branches and the ConvNeXt frontend pad mask earlier input samples. State and
positions advance normally. A pause starts at its labelled onset, so its own
frames remain available as warm-up. The downsampled stages use the corresponding
native clock with the cutoff rounded up to their frame boundary.

The training catalog uses only clean-match/train recordings from supplied s1,
with the existing gold exclusion function. A generic canonical-text matcher
requires consecutive complete ayahs inside one recording. No recordings are
joined and no audio is spliced. A frozen shipped streaming decode supplies the
existing 300 ms, blank-posterior-at-least-one-half pause labels. Canonical CTC
Viterbi associates those pauses with ayah seams; it does not label silence from
text alone. Each such pause is masked with probability 0.5, preserving its frames.

Fixed recipe: seed 2026, at most 20 hours of eligible recordings, 800 AdamW
updates at LR 0.00005, canonical CTC plus shipped top-five KD (weight 2), and
KL(shipped || candidate), weight 4, on unmasked items. Retain the checkpoint's
RIR → noise → gain augmentation and SpecAugment. The direct streaming path has
carried caches and the deployed lookahead, with state gradients detached every
eight chunks. The ordinary checkpoint architecture and MatMul int8 quantizer
remain. Training does not read practice audio or labels.

The SDK mask input and seam selector are generic engine code. A labelled pause
at an ayah or waqf-marked inner boundary opens the span, and the unchanged
primary tracker's next in-order word step closes it. Primary model feeds remain
unchanged when the manifest has no mask input. Primary strings and tracker rules
are unchanged; deployment of the candidate evidence is pending the acoustic check.

## Validation

Pending Modal smoke, corpus census, unit tests, training, export and paired target
streams. No practice-score calls have been made.

## Limits

Blank pause labels can include weak speech the shipped model missed. Canonical
training and the original KD teacher still carry their original supervision;
this experiment changes the context regime rather than adding insertion labels.
The finite-resolution stages quantize a pause cutoff on their native clock.
Teacher evidence is correlated model evidence, not an independent listening review.
