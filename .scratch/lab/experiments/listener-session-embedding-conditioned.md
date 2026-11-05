# Causal session embedding conditioned student

Status: implementation and validation in progress. The experiment card is unchanged.

## Prediction and continuation check

Predict three of seven target slots recovered, about one caught swap lost:
listener −2, overall −1. Continue to practice scoring only if the built int8
student, decoded as a full session, spells at least three target slots as the
written word with zero hard edits under the existing soft costs. Warning
removal and written-word recovery are separate measurements.

## Mechanism

A frozen copy of the shipped spliced-clean student supplies stack 3 features,
after upsampling and residual combination, on the 20 ms grid. A small
LayerNorm/MLP maps each frame to 64 dimensions. The running mean includes every
completed chunk, including silence and opening recitation. The next chunk
receives that mean once at least one second has accumulated. There is no
reference text, tracker output, gradient update or fitted session parameter in
this input. The chunk scheduling makes the condition independent of the
caller's audio push size.

Each BiasNorm in the acoustic student has a bias-free linear projection of the
session vector to scale and offset. Its weights start at zero, so initialization
is identity; a zero vector also makes every FiLM operation identity after
training. The acoustic weights may still change in training.

Training uses clean-match/train clips, at most 30 seconds long, sampled to
20 hours with seed zero. Every voice cluster containing a gold-excluded clip
is excluded entirely. A target's context always comes from a different clip
of the same remaining cluster, with a prefix duration sampled between one
second and the clip's full duration. Both clips use the same seed through the
shipped noise/RIR/gain/SpecAugment chain and the parent's saved probabilities.
The supervision is canonical collapsed-phone CTC and the original teacher's
KD. Every fourth item zeros the condition and adds KL-to-shipped at weight
10. AdamW uses 0.00003 for acoustic weights and 0.0003 for FiLM and the frame
encoder, for a fixed 3,000 steps; checkpoint resumes preserve optimizer and
random states. No training setting is selected using practice.

The graph exports the frozen and conditioned streaming branches, their
separate recurrent states and the learned frame features. Export uses the
existing streaming window (45 frames), shift (32), context (128), and dynamic
MatMul int8 quantization. The SDK maintains the running sum and count per
listener, excluding EOS padding and overlapping lookahead. The checker,
tracker and their constants are unchanged.

## Validation and limits

Supervisor run 1 verified availability of the shipped parent and the s1
schema, including `cluster`, canonical reference, teacher posteriors and
16 kHz FLAC audio. tlog has no verified speaker identities: its clusters are
approximate voice groups, so same-cluster pairing is not proof of same-person
pairing. This is a limitation of this training test. The synthetic augmentation
chain is unchanged; this experiment adds no new phone transform or corpus.

The frozen branch increases inference compute and recurrent-state memory.
Mobile latency and footprint need measurement before a release claim. A KL
anchor discourages output drift but does not guarantee unchanged strings.
No independent listening review or label correction is part of this experiment.
