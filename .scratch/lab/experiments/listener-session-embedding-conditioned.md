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

Runs 2 and 3 failed in Modal's volume infrastructure (`too many layers in
volume`); neither returned a usable training/test result. Run 4 reached the
six-shard build but found a pre-existing TypeScript error in `reference.ts`:
`filter(Boolean)` retained `undefined` in the inferred type of optional plain
onset forms. The explicit type predicate preserves that filter's runtime
behavior and changes no pronunciation form.

Run 6 passed core and SDK builds and all six validation shards: 737 core tests
and 17 SDK tests passed, with one existing optional SDK test skipped. The
unchanged test wrapper accepted the known Vitest RPC timeout only after all
core tests in that shard passed. The new SDK case checks the one-second gate,
completed-chunk running mean, silence inclusion, push-size independence,
per-listener reset and rejection of missing feature output.

Run 6 also read the persistent pool audit: 8,445 clips from 90 training voice
clusters, sampled to 20.0002 hours, after excluding all 48 clusters linked to
the gold clip exclusions. At that snapshot only data preparation was visible;
no completed optimization checkpoint was yet reported.

Run 7 passed all three Python tests, including a small actual Zipformer stream:
64-dimensional frozen features on the 20 ms grid, unchanged initial logits,
separate recurrent states, frozen parameters, zero-vector identity after
nonzero FiLM weights, and both BiasNorm batch layouts. Run 8 read a valid
2,000-step checkpoint from the persistent scratch area. Open log files were
not visible in that concurrent container's snapshot.

Run 5 completed the fixed 3,000-step training run with finite objectives and
gradients. Optimization took 1,378.2 seconds; the supervisor command took
1,855.2 seconds including preparation. GPU cost was $2.04 (plus $0.50 spent
on the earlier infrastructure failure), leaving $117.46. The final minibatch
logged CTC 0.09661, KD 0.24614, embedding norm 0.54325 and FiLM weight norm
12.1902. Logs are sampled every 50 steps and therefore miss the every-fourth
zero-conditioned items; their printed KL values are zero and must not be read
as an aggregate KL measurement. The zero-conditioned term is computed on the
intervening items by the fixed training loop.

Parent checkpoint SHA-256:
`bcd41fcffe1b298cb4da36ea0e3df87d1033e209b8bba1cbff7d0a654b749b9f`.
No intermediate checkpoint was selected using target or practice results.

Run 9's export preflight read the final 3,000-step checkpoint. It produced an
int8 graph but failed its Torch/float-ONNX parity assertion (maximum log-prob
error 4.1–5.9 and feature error 0.025–0.040). Inspection identified a validation
mode bug: the outer `Streaming` wrapper defaulted to training mode, so ONNX's
mode restoration re-enabled dropout on the reused Torch oracle after export.
The inference wrapper now starts in evaluation mode; the small-model test
asserts that every submodule stays in evaluation mode. Parity state arrays
also receive independent copies rather than sharing Torch memory. This is an
export/validation fix, not a new training arm or a target-driven adjustment.
